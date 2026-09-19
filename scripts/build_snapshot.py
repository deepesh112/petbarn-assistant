"""Build the curated catalog and the offline fallback snapshot.

Run this to (re)ingest data from Petbarn:

    python scripts/build_snapshot.py            # ingest, then verify
    python scripts/build_snapshot.py --verify   # verify what is already on disk
    python scripts/build_snapshot.py --count 12 # ingest a different catalog size

Product selection is automated rather than hand-picked, because the constraints
that matter are mechanical:

* **Enough written reviews.** A product with six reviews cannot support a
  conversation about sentiment, so candidates are drawn in descending order of
  review count and rejected below :data:`MIN_TEXT_REVIEWS`.
* **A real price.** Petbarn models multi-size products as a ``ProductGroup``
  whose parent has no price of its own; only the variants do. Requiring a
  resolved price filters those out automatically.
* **A spread.** Caps on products per brand and per category stop the catalog
  collapsing into "ten flea treatments", which would make comparison questions
  dull and unrepresentative.

Everything fetched is also written to ``data/raw/`` gzipped, so the parsers can
be re-run and audited later without touching the network again.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petbarn import config, reviews as review_api  # noqa: E402
from petbarn.http import FetchError, get_session, now_iso  # noqa: E402
from petbarn.models import CatalogEntry, ProductDetails  # noqa: E402
from petbarn.scraper import ProductParseError, fetch_product  # noqa: E402
from petbarn.textutils import clean_text  # noqa: E402

TARGET_COUNT = 10
MIN_TEXT_REVIEWS = 40
REVIEWS_PER_PRODUCT = 150

#: How many Bazaarvoice candidates to consider. Generous, because the brand and
#: category caps reject a lot of the top of the list (Petbarn's most-reviewed
#: products are dominated by a handful of flea-treatment ranges).
CANDIDATE_POOL = 120

#: One product per brand. Petbarn's most-reviewed list is full of near-clones
#: (the same flea chew in five dog-weight bands), and a catalog of clones makes
#: for dull comparisons. One per brand buys maximum variety for ten slots.
MAX_PER_BRAND = 1
MAX_PER_CATEGORY = 2

#: A trailing size token, stripped to produce a shorter conversational alias:
#: people ask about "the Black Hawk lamb and rice", not the 20kg bag. Anchored
#: to the end of the string on purpose -- an unanchored pattern chews through
#: "Simparica Trio 20.1-40kg Dog Flea Tick & Worm Chew" from the "-40kg" onwards
#: and leaves behind the nonsense alias "Simparica Trio 20.1".
_SIZE_SUFFIX_RE = re.compile(
    r"\s*[-–]?\s*\d+(?:\.\d+)?\s*(?:kg|g|ml|l|litre|liter)\b"
    r"\s*(?:x\s*\d+)?\s*(?:pack|pk)?\s*$",
    re.IGNORECASE,
)

#: Matches any letter, used to tell a real word from a leftover number.
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")

_PET_KEYWORDS = (
    ("dog", "Dog"),
    ("puppy", "Dog"),
    ("cat", "Cat"),
    ("kitten", "Cat"),
    ("fish", "Fish"),
    ("bird", "Bird"),
    ("rabbit", "Small pet"),
    ("guinea", "Small pet"),
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_raw(name: str, body: str) -> None:
    """Archive a raw response gzipped, for later auditing and re-parsing."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(config.RAW_DIR / f"{name}.gz", "wt", encoding="utf-8") as handle:
        handle.write(body)


def _infer_pet(*texts: str | None) -> str | None:
    haystack = " ".join(t.lower() for t in texts if t)
    for needle, label in _PET_KEYWORDS:
        if needle in haystack:
            return label
    return None


def _is_useful_alias(alias: str, product: ProductDetails) -> bool:
    """Reject aliases that are truncated, ambiguous or too generic to match on."""
    alias = alias.strip()
    if len(alias) < 6:
        return False
    words = alias.split()
    # Needs two real words, so a bare size or stray number cannot become an alias.
    if sum(1 for word in words if _HAS_LETTER_RE.search(word)) < 2:
        return False
    # A final token with no letters means a size range was cut in half.
    if not _HAS_LETTER_RE.search(words[-1]):
        return False
    # An alias equal to the category would match every sibling product too.
    if product.category and alias.casefold() == product.category.casefold():
        return False
    return alias.casefold() != product.name.casefold()


def _build_aliases(product: ProductDetails) -> list[str]:
    """Generate the shorter names a shopper is likely to type."""
    candidates: set[str] = set()
    short = _SIZE_SUFFIX_RE.sub("", product.name).strip(" -–")
    candidates.add(short)

    # The same name without its brand prefix, since people say either
    # "Black Hawk lamb and rice" or just "lamb and rice".
    if product.brand:
        pattern = re.compile(rf"^{re.escape(product.brand)}\s*(?:for\s+)?", re.IGNORECASE)
        for candidate in (product.name, short):
            candidates.add(pattern.sub("", candidate).strip())

    return sorted(alias for alias in candidates if _is_useful_alias(alias, product))


def _candidate_rows(session: Any) -> list[dict[str, Any]]:
    print(f"Listing Petbarn's {CANDIDATE_POOL} most-reviewed products from Bazaarvoice...")
    rows = review_api.fetch_top_reviewed_products(limit=CANDIDATE_POOL, session=session)
    usable = [
        row
        for row in rows
        if row.get("Id")
        and row.get("Name")
        and row.get("ProductPageUrl")
        and not row.get("Disabled")
    ]
    print(f"  {len(rows)} returned, {len(usable)} usable (named, linked, enabled)\n")
    return usable


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def _clear_previous_output() -> None:
    """Delete prior snapshot and raw files so a rerun cannot leave orphans.

    Without this, a product dropped from the catalog keeps its snapshot files on
    disk, and the fallback layer would happily keep serving a product the
    assistant no longer lists.
    """
    removed = 0
    for pattern, directory in (
        ("product_*.json", config.SNAPSHOT_DIR),
        ("reviews_*.json", config.SNAPSHOT_DIR),
        ("*.gz", config.RAW_DIR),
    ):
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            path.unlink()
            removed += 1
    if removed:
        print(f"Cleared {removed} file(s) from a previous run")



def ingest(target_count: int = TARGET_COUNT) -> list[CatalogEntry]:
    session = get_session()
    _clear_previous_output()
    candidates = _candidate_rows(session)

    selected: list[CatalogEntry] = []
    brand_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    rejected: list[str] = []

    for row in candidates:
        if len(selected) >= target_count:
            break

        sku = str(row["Id"]).strip()
        bv_reviews = int((row.get("ReviewStatistics") or {}).get("TotalReviewCount") or 0)
        label = clean_text(row.get("Name")) or sku

        if bv_reviews < MIN_TEXT_REVIEWS:
            # The pool is sorted descending, so everything after this is worse.
            print(f"  stopping: {label} has only {bv_reviews} reviews")
            break

        try:
            product, page = fetch_product(row["ProductPageUrl"], prefer_sku=sku, session=session)
        except (FetchError, ProductParseError) as exc:
            rejected.append(f"{label}: {type(exc).__name__} {exc}")
            continue

        if product.price is None:
            rejected.append(f"{label}: no resolvable price (parent product?)")
            continue

        if product.is_bundle:
            # Multi-buy bundles share their single-unit sibling's reviews, so
            # including both would double-count the same customer feedback.
            rejected.append(f"{label}: multi-buy bundle")
            continue

        brand_key = (product.brand or "unknown").lower()
        category_key = (product.category or "unknown").lower()
        if brand_counts.get(brand_key, 0) >= MAX_PER_BRAND:
            rejected.append(f"{label}: brand cap reached for {product.brand}")
            continue
        if category_counts.get(category_key, 0) >= MAX_PER_CATEGORY:
            rejected.append(f"{label}: category cap reached for {product.category}")
            continue

        try:
            bundle, _ = review_api.fetch_reviews(
                product.sku, limit=REVIEWS_PER_PRODUCT, session=session
            )
        except review_api.ReviewFetchError as exc:
            rejected.append(f"{label}: review fetch failed ({exc})")
            continue

        text_reviews = len(bundle.with_text_only())
        if text_reviews < MIN_TEXT_REVIEWS:
            rejected.append(f"{label}: only {text_reviews} reviews with text")
            continue

        # Persist the product, its reviews and the raw responses behind both.
        _write_json(config.SNAPSHOT_DIR / f"product_{product.sku}.json", product.to_dict())
        _write_json(config.SNAPSHOT_DIR / f"reviews_{product.sku}.json", bundle.to_dict())
        _write_raw(f"product_{product.sku}.html", page.body)
        _write_raw(f"reviews_{product.sku}.json", json.dumps(bundle.to_dict(), ensure_ascii=False))

        brand_counts[brand_key] = brand_counts.get(brand_key, 0) + 1
        category_counts[category_key] = category_counts.get(category_key, 0) + 1

        selected.append(
            CatalogEntry(
                sku=product.sku,
                name=product.name,
                url=product.url,
                brand=product.brand,
                category=product.category,
                pet=_infer_pet(product.category, product.name),
                price=product.price,
                average_rating=bundle.stats.average_rating or product.average_rating,
                review_count=bundle.stats.total_reviews or product.review_count,
                text_review_count=text_reviews,
                aliases=_build_aliases(product),
            )
        )
        print(
            f"  [{len(selected):2d}/{target_count}] {product.sku:>7}  "
            f"${product.price:>7.2f}  {bundle.stats.average_rating}*  "
            f"{text_reviews:>3d} text reviews  {product.name}"
        )

    _write_json(
        config.CATALOG_PATH,
        {
            "generated_at": now_iso(),
            "source": config.SITE_ROOT,
            "review_platform": f"Bazaarvoice ({config.BV_CLIENT}/{config.BV_SITE})",
            "products": [entry.to_dict() for entry in selected],
        },
    )

    if rejected:
        print(f"\nSkipped {len(rejected)} candidates:")
        for line in rejected[:25]:
            print(f"  - {line}")
    return selected


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def verify() -> bool:
    """Re-read everything from disk and assert it is usable. Returns success."""
    from petbarn.models import ReviewBundle

    if not config.CATALOG_PATH.exists():
        print(f"FAIL: {config.CATALOG_PATH} does not exist -- run without --verify first")
        return False

    catalog = json.loads(config.CATALOG_PATH.read_text(encoding="utf-8"))
    entries = [CatalogEntry.from_dict(row) for row in catalog.get("products", [])]
    print(f"Catalog generated {catalog.get('generated_at')} with {len(entries)} products\n")

    header = f"{'SKU':>7}  {'PRICE':>9}  {'RATING':>6}  {'ALL':>5}  {'TEXT':>5}  PRODUCT"
    print(header)
    print("-" * len(header))

    ok = bool(entries)
    for entry in entries:
        problems: list[str] = []
        product_path = config.SNAPSHOT_DIR / f"product_{entry.sku}.json"
        reviews_path = config.SNAPSHOT_DIR / f"reviews_{entry.sku}.json"

        if not product_path.exists():
            problems.append("product snapshot missing")
        if not reviews_path.exists():
            problems.append("reviews snapshot missing")

        text_count = 0
        if reviews_path.exists():
            bundle = ReviewBundle.from_dict(json.loads(reviews_path.read_text(encoding="utf-8")))
            text_count = len(bundle.with_text_only())
            if text_count < MIN_TEXT_REVIEWS:
                problems.append(f"only {text_count} text reviews")
        if product_path.exists():
            product = ProductDetails.from_dict(json.loads(product_path.read_text(encoding="utf-8")))
            if product.price is None:
                problems.append("no price")
            if not product.description:
                problems.append("no description")

        price = f"${entry.price:.2f}" if entry.price is not None else "-"
        print(
            f"{entry.sku:>7}  {price:>9}  {str(entry.average_rating):>6}  "
            f"{str(entry.review_count):>5}  {text_count:>5}  {entry.name}"
        )
        if problems:
            ok = False
            print(f"{'':>7}  !! {'; '.join(problems)}")

    categories = sorted({e.category or "?" for e in entries})
    brands = sorted({e.brand or "?" for e in entries})
    print(f"\n{len(categories)} categories: {', '.join(categories)}")
    print(f"{len(brands)} brands: {', '.join(brands)}")
    print(f"\n{'PASS' if ok else 'FAIL'}: catalog and snapshot are {'usable' if ok else 'INCOMPLETE'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="only verify what is on disk")
    parser.add_argument("--count", type=int, default=TARGET_COUNT, help="how many products to select")
    args = parser.parse_args()

    if not args.verify:
        selected = ingest(target_count=args.count)
        print(f"\nIngested {len(selected)} products into {config.DATA_DIR}\n")

    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
