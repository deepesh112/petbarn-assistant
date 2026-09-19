"""Exercise every tool, live and offline, without involving the language model.

    python scripts/smoke_test.py

This is the test that matters most for this project. The agent loop is only as
good as the tools beneath it, and the failure modes worth guarding against are
all in the tools: a scraper that silently returns no price, a snapshot that
cannot satisfy a filter the live API can, a payload missing the provenance the
UI renders. Running it needs no API key and costs nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petbarn import config, tools  # noqa: E402
from petbarn.catalog import get_catalog  # noqa: E402

failures: list[str] = []
checks = 0


def check(condition: bool, description: str) -> bool:
    """Record one assertion without aborting the run."""
    global checks
    checks += 1
    if not condition:
        failures.append(description)
        print(f"    FAIL  {description}")
    return bool(condition)


def run(name: str, /, **arguments: Any) -> tools.ToolResult:
    result = tools.execute(name, arguments)
    status = "ok" if result.ok else f"ERROR: {result.error}"
    source = f" via {result.source}" if result.source else ""
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    print(f"  {name}({args}) -> {status}{source} in {result.duration_ms}ms")
    return result


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #


def test_catalog() -> list[str]:
    section("Catalog")
    catalog = get_catalog()
    print(f"  {len(catalog)} products, generated {catalog.generated_at}")
    check(len(catalog) >= 8, f"catalog should hold at least 8 products, has {len(catalog)}")
    check(len(catalog.categories()) >= 4, "catalog should span at least 4 categories")
    for entry in catalog:
        check(bool(entry.sku and entry.name and entry.url), f"{entry.sku}: sku/name/url present")
        check(entry.price is not None, f"{entry.sku}: has a price")
        check((entry.text_review_count or 0) >= 40, f"{entry.sku}: has 40+ written reviews")
        # Checked for every product, not just the two exercised below: one
        # product in ten stored a site-relative URL, its live fetch failed with
        # "No scheme supplied", and the snapshot fallback hid it completely.
        check(
            entry.url.startswith("https://www.petbarn.com.au/"),
            f"{entry.sku}: URL is absolute, not site-relative ({entry.url})",
        )
    # Two products from different categories, for the comparison path.
    picks = [catalog.entries[0].sku, catalog.entries[-1].sku]
    print(f"  using SKUs {picks} for tool tests")
    return picks


def test_search() -> None:
    section("search_catalog")
    result = run("search_catalog")
    check(result.ok and len(result.payload.get("products", [])) >= 8, "bare call lists the catalog")

    catalog = get_catalog()
    target = catalog.entries[0]
    result = run("search_catalog", query=target.name)
    check(
        result.ok and result.payload["matches"][0]["sku"] == target.sku,
        f"exact name resolves to {target.sku}",
    )

    brand_word = (target.brand or target.name).split()[0]
    result = run("search_catalog", query=f"what do people think of the {brand_word}?")
    check(
        result.ok and result.payload["matches"][0]["sku"] == target.sku,
        f"conversational mention of {brand_word!r} resolves to {target.sku}",
    )

    result = run("search_catalog", query="dyson cordless vacuum")
    check(result.ok and not result.payload.get("matches"), "an off-catalog product returns no match")
    check(
        result.ok and bool(result.payload.get("available_products")),
        "a miss still offers what the catalog does carry",
    )


def test_details(sku: str) -> None:
    section(f"get_product_details({sku})")
    result = run("get_product_details", sku=sku)
    if not check(result.ok, "details call succeeds"):
        return
    payload = result.payload
    print(f"    {payload['name']}  [{payload['brand']} / {payload['category']}]")
    print(f"    price ${payload['price']['regular']} regular, "
          f"${payload['price']['loyalty_member']} member, {payload['availability']}")
    print(f"    rating {payload['average_rating']} over {payload['review_count']} reviews")
    print(f"    description: {len((payload.get('description') or ''))} chars")

    check(payload["sku"] == sku, "returns the SKU that was asked for")
    check(payload["price"]["regular"] is not None, "has a regular price")
    check(payload["price"]["currency"] == "AUD", "price is in AUD")
    check(bool(payload.get("description")), "has a description")
    check(payload.get("source") in {"live", "cache", "snapshot"}, "carries a provenance source")
    check(bool(payload.get("fetched_at")), "carries a fetched_at timestamp")
    check(str(payload.get("url", "")).startswith("https://www.petbarn.com.au/"), "links to Petbarn")


def test_reviews(sku: str) -> None:
    section(f"get_product_reviews({sku})")
    result = run("get_product_reviews", sku=sku, limit=5)
    if not check(result.ok, "reviews call succeeds"):
        return
    payload = result.payload
    summary = payload["rating_summary"]
    print(f"    average {summary['average_rating']} over {summary['total_reviews']} reviews "
          f"({summary['reviews_with_written_text']} written)")
    print(f"    distribution {summary['rating_distribution']}")
    print(f"    aspect averages {summary.get('aspect_rating_averages')}")
    print(f"    would recommend {summary.get('would_recommend_percent')}%")
    for review in payload["reviews"][:2]:
        print(f"    - {review['rating']}* {review['date']} [{review['author']}] "
              f"{(review['title'] or '')[:40]}: {review['text'][:90]}")

    check(len(payload["reviews"]) == 5, "honours the requested limit")
    check(all(r["text"] for r in payload["reviews"]), "every returned review has text")
    check(summary["average_rating"] is not None, "reports an average rating")
    check(sum(summary["rating_distribution"].values()) > 0, "reports a star distribution")
    check(payload.get("source") in {"live", "cache", "snapshot"}, "carries a provenance source")

    # The filter path is what makes "what are the complaints" answerable.
    critical = run("get_product_reviews", sku=sku, max_rating=2, sort="lowest_rating", limit=5)
    if critical.ok:
        ratings = [r["rating"] for r in critical.payload["reviews"]]
        print(f"    critical-only ratings: {ratings}")
        check(all(r <= 2 for r in ratings), "max_rating filter excludes happy reviews")

    capped = run("get_product_reviews", sku=sku, limit=999)
    if capped.ok:
        check(
            len(capped.payload["reviews"]) <= tools.MAX_REVIEW_LIMIT,
            f"an absurd limit is capped at {tools.MAX_REVIEW_LIMIT}",
        )


def test_sentiment(sku: str) -> None:
    section(f"analyze_review_sentiment({sku})")
    result = run("analyze_review_sentiment", sku=sku)
    if not check(result.ok, "sentiment call succeeds"):
        return
    payload = result.payload
    overall = payload["sample_statistics"]
    print(f"    {payload['reviews_analysed']} reviews analysed; mean "
          f"{overall['mean_stars_in_sample']}* polarity {overall['mean_text_polarity']}; "
          f"{overall['star_vs_text_disagreements']} star/text disagreements")
    for aspect in payload["aspects"]:
        print(f"    {aspect['aspect']:26} n={aspect['mentions']:3d} "
              f"+{aspect['positive']:3d}/-{aspect['negative']:3d} "
              f"pos%={aspect['positive_percent']} stars={aspect['mean_stars_of_mentioning_reviews']}")
    print(f"    PROS: {payload['pros']}")
    print(f"    CONS: {payload['cons']}")

    check(payload["reviews_analysed"] >= 40, "analyses a meaningful number of reviews")
    check(bool(payload["aspects"]), "finds at least one aspect")
    check(
        any(a["positive_quotes"] or a["negative_quotes"] for a in payload["aspects"]),
        "supports at least one aspect with a verbatim quote",
    )
    labels = [a["aspect"] for a in payload["aspects"]]
    check(len(labels) == len(set(labels)), "aspects are not duplicated")
    overlap = {p.split(" —")[0] for p in payload["pros"]} & {c.split(" —")[0] for c in payload["cons"]}
    check(not overlap, f"no aspect is both a pro and a con (overlap: {overlap})")


def test_error_handling() -> None:
    section("Error handling")
    result = run("get_product_details", sku="does-not-exist")
    check(not result.ok, "an unknown SKU fails rather than inventing data")
    check("search_catalog" in (result.error or ""), "the error tells the model how to recover")

    result = run("no_such_tool")
    check(not result.ok, "an unknown tool name fails cleanly")

    result = run("get_product_details")
    check(not result.ok, "a missing required argument fails cleanly")

    payload = json.loads(result.to_json())
    check("error" in payload, "a failed result serialises to an error object for the model")


def test_offline(sku: str) -> None:
    section("Offline mode (PETBARN_LIVE=0): every tool must still answer")
    for name, kwargs in (
        ("get_product_details", {"sku": sku}),
        ("get_product_reviews", {"sku": sku, "limit": 5}),
        ("get_product_reviews", {"sku": sku, "max_rating": 3, "sort": "lowest_rating"}),
        ("analyze_review_sentiment", {"sku": sku}),
    ):
        result = run(name, **kwargs)
        if check(result.ok, f"{name} works offline"):
            check(result.source == "snapshot", f"{name} reports 'snapshot' as its source")


def main() -> int:
    print(f"Live fetching enabled: {config.live_fetch_enabled()}")
    picks = test_catalog()
    test_search()
    for sku in picks:
        test_details(sku)
        test_reviews(sku)
        test_sentiment(sku)
    test_error_handling()

    # Force the fallback path by disabling live fetches for the rest of the run.
    import os

    os.environ["PETBARN_LIVE"] = "0"
    test_offline(picks[0])

    section("Result")
    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
