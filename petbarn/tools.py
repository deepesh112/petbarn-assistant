"""The tools the assistant can invoke, and the data plumbing behind them.

This module is the boundary between the language model and everything real. It
owns two responsibilities:

**The contract.** :data:`TOOL_SCHEMAS` are the function definitions handed to the
model. Their descriptions are part of the system's behaviour, not documentation:
they are what stops the model inventing a SKU or asking for a product Petbarn
does not sell.

**The fallback chain.** Each tool resolves data in the same order -- recent local
cache, then a live fetch, then the snapshot committed to the repo. A blocked
request, a rotated Bazaarvoice passkey or an outright outage therefore degrades
the *freshness* of an answer rather than the ability to answer at all. Which
layer served a call travels back in the payload as ``source``, so neither the
model nor the reader has to guess.

Four tools are exposed where the brief asks for two. ``search_catalog`` exists
because every other tool is keyed by SKU and the model must never guess one.
``analyze_review_sentiment`` is separate from ``get_product_reviews`` so that
"what are the pros and cons" is answered from counted evidence rather than from
the model's impression of a wall of text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config, reviews as review_api
from .catalog import Catalog, get_catalog, pretty_brand
from .http import FetchError, get_session
from .models import ProductDetails, Review, ReviewBundle, Source
from .scraper import ProductParseError, fetch_product
from .sentiment import analyse
from .textutils import truncate

#: Default and maximum number of individual reviews returned to the model.
DEFAULT_REVIEW_LIMIT = 12
MAX_REVIEW_LIMIT = config.MAX_REVIEWS_TO_MODEL

#: How many reviews to pull for analysis. Sentiment wants the whole picture even
#: when only a handful of quotes are shown.
ANALYSIS_REVIEW_LIMIT = 150


class ToolError(RuntimeError):
    """A tool could not complete. The message is shown to the model verbatim."""


@dataclass(slots=True)
class ToolResult:
    """The outcome of one tool call, including how it was served."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    source: Source | None = None
    duration_ms: int = 0
    error: str | None = None

    def to_json(self) -> str:
        """Serialise for the ``tool`` message sent back to the model."""
        if self.ok:
            return json.dumps(self.payload, ensure_ascii=False, default=str)
        return json.dumps({"error": self.error}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Data access with fallback
# --------------------------------------------------------------------------- #


def _snapshot_path(kind: str, sku: str):
    return config.SNAPSHOT_DIR / f"{kind}_{sku}.json"


def _read_snapshot(kind: str, sku: str) -> dict[str, Any] | None:
    path = _snapshot_path(kind, sku)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_product(sku: str, *, allow_live: bool | None = None) -> ProductDetails:
    """Fetch product details, falling back to the bundled snapshot."""
    catalog = get_catalog()
    entry = catalog.get(sku)
    if entry is None:
        raise ToolError(
            f"SKU {sku!r} is not in this assistant's catalog. "
            f"Call search_catalog first to find a valid SKU."
        )

    live = config.live_fetch_enabled() if allow_live is None else allow_live
    if live:
        try:
            product, _ = fetch_product(entry.url, prefer_sku=sku, session=get_session())
            return product
        except (FetchError, ProductParseError):
            # Fall through to the snapshot: a slightly stale price is a far
            # better answer than an apology.
            pass

    payload = _read_snapshot("product", sku)
    if payload is None:
        raise ToolError(f"no product data available for SKU {sku} (live fetch failed, no snapshot)")
    product = ProductDetails.from_dict(payload)
    product.source = "snapshot"
    return product


def load_reviews(
    sku: str,
    *,
    limit: int = ANALYSIS_REVIEW_LIMIT,
    sort: str = review_api.DEFAULT_SORT,
    min_rating: int | None = None,
    max_rating: int | None = None,
    allow_live: bool | None = None,
) -> ReviewBundle:
    """Fetch reviews, falling back to the bundled snapshot.

    Filtering and sorting are re-applied locally after a snapshot read, so the
    offline path supports the same questions as the live one ("show me the
    one-star reviews") instead of only ever returning the newest.
    """
    catalog = get_catalog()
    if not catalog.contains(sku):
        raise ToolError(
            f"SKU {sku!r} is not in this assistant's catalog. "
            f"Call search_catalog first to find a valid SKU."
        )

    live = config.live_fetch_enabled() if allow_live is None else allow_live
    if live:
        try:
            bundle, _ = review_api.fetch_reviews(
                sku,
                limit=limit,
                sort=sort,
                min_rating=min_rating,
                max_rating=max_rating,
                session=get_session(),
            )
            return bundle
        except (FetchError, review_api.ReviewFetchError):
            pass

    payload = _read_snapshot("reviews", sku)
    if payload is None:
        raise ToolError(f"no review data available for SKU {sku} (live fetch failed, no snapshot)")
    bundle = ReviewBundle.from_dict(payload)
    bundle.source = "snapshot"
    bundle.reviews = _filter_and_sort(
        bundle.reviews, limit=limit, sort=sort, min_rating=min_rating, max_rating=max_rating
    )
    return bundle


def _filter_and_sort(
    reviews: list[Review],
    *,
    limit: int,
    sort: str,
    min_rating: int | None,
    max_rating: int | None,
) -> list[Review]:
    """Apply the review tool's filters to an in-memory list."""
    selected = [review for review in reviews if review.has_text]
    if min_rating is not None:
        selected = [r for r in selected if (r.rating or 0) >= min_rating]
    if max_rating is not None:
        selected = [r for r in selected if (r.rating or 0) <= max_rating]

    if sort == "highest_rating":
        selected.sort(key=lambda r: (r.rating or 0), reverse=True)
    elif sort == "lowest_rating":
        selected.sort(key=lambda r: (r.rating or 6))
    elif sort == "most_helpful":
        selected.sort(key=lambda r: r.helpful_votes, reverse=True)
    elif sort == "oldest":
        selected.sort(key=lambda r: r.submitted_at or "")
    else:  # most_recent
        selected.sort(key=lambda r: r.submitted_at or "", reverse=True)
    return selected[:limit]


# --------------------------------------------------------------------------- #
# Payload shaping
# --------------------------------------------------------------------------- #


def _provenance(source: Source | None, fetched_at: str | None) -> dict[str, Any]:
    return {
        "source": source,
        "fetched_at": fetched_at,
        "note": {
            "live": "fetched from petbarn.com.au just now",
            "cache": "served from a recent local cache of petbarn.com.au",
            "snapshot": "served from the bundled offline snapshot; may be out of date",
        }.get(source or "", None),
    }


def _product_payload(product: ProductDetails) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sku": product.sku,
        "name": product.name,
        "brand": pretty_brand(product.brand),
        "category": product.category,
        "url": product.url,
        "price": {
            "currency": product.currency,
            "regular": product.price,
            "loyalty_member": product.member_price,
        },
        "availability": product.availability,
        "size": product.size,
        "barcode_gtin": product.gtin,
        "average_rating": product.average_rating,
        "review_count": product.review_count,
        "description": product.description,
        **_provenance(product.source, product.fetched_at),
    }
    if product.variants:
        payload["other_sizes"] = [
            {
                "sku": variant.sku,
                "size": variant.size,
                "price": variant.price,
                "loyalty_member_price": variant.member_price,
                "availability": variant.availability,
            }
            for variant in product.variants
            if variant.sku != product.sku
        ]
    if product.shipping:
        payload["delivery_options"] = [
            {"name": option.name, "price": option.price} for option in product.shipping
        ]
    return payload


def _review_payload(review: Review) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rating": review.rating,
        "title": review.title,
        "text": truncate(review.text, config.MAX_REVIEW_CHARS),
        "author": review.author,
        "date": (review.submitted_at or "")[:10] or None,
        "verified_purchaser": review.is_verified_purchaser,
    }
    if review.is_recommended is not None:
        payload["would_recommend"] = review.is_recommended
    if review.helpful_votes:
        payload["helpful_votes"] = review.helpful_votes
    if review.secondary_ratings:
        payload["aspect_ratings"] = review.secondary_ratings
    return payload


def _stats_payload(bundle: ReviewBundle) -> dict[str, Any]:
    stats = bundle.stats
    payload: dict[str, Any] = {
        "average_rating": stats.average_rating,
        "total_reviews": stats.total_reviews,
        "reviews_with_written_text": stats.text_review_count,
        "rating_distribution": {
            f"{stars}_star": count for stars, count in review_api.summarise_distribution(stats.rating_distribution)
        },
        "newest_review_date": (stats.last_review_at or "")[:10] or None,
        "oldest_review_date": (stats.first_review_at or "")[:10] or None,
    }
    if stats.secondary_rating_averages:
        payload["aspect_rating_averages"] = stats.secondary_rating_averages
    share = stats.recommend_share
    if share is not None:
        payload["would_recommend_percent"] = round(share * 100, 1)
    return payload


# --------------------------------------------------------------------------- #
# The tools
# --------------------------------------------------------------------------- #


def tool_search_catalog(query: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Resolve a product mention to catalog SKUs, or list the whole catalog."""
    catalog: Catalog = get_catalog()

    if not query or not query.strip():
        return {
            "catalog_size": len(catalog),
            "note": "Full catalog. This assistant can only discuss these products.",
            "products": catalog.overview(),
        }

    limit = max(1, min(int(limit or 5), len(catalog)))
    matches = catalog.search(query, limit=limit)
    if not matches:
        return {
            "query": query,
            "matches": [],
            "catalog_size": len(catalog),
            "note": (
                "No product in this assistant's catalog matches that. Tell the user it is not "
                "covered and offer what is available."
            ),
            "available_products": [
                {"sku": entry.sku, "name": entry.name} for entry in catalog.entries
            ],
        }

    return {
        "query": query,
        "matches": [
            {
                "sku": match.entry.sku,
                "name": match.entry.name,
                "brand": pretty_brand(match.entry.brand),
                "category": match.entry.category,
                "pet": match.entry.pet,
                "price_aud": match.entry.price,
                "average_rating": match.entry.average_rating,
                "review_count": match.entry.review_count,
                "confidence": match.score,
                "matched_on": match.matched_on,
            }
            for match in matches
        ],
        "guidance": (
            "Use the top match when its confidence is clearly highest. If two matches are close, "
            "ask the user which one they mean rather than guessing."
        ),
    }


def tool_get_product_details(sku: str) -> dict[str, Any]:
    """Specifications, pricing, brand and availability for one product."""
    return _product_payload(load_product(str(sku).strip()))


def tool_get_product_reviews(
    sku: str,
    limit: int = DEFAULT_REVIEW_LIMIT,
    sort: str = review_api.DEFAULT_SORT,
    min_rating: int | None = None,
    max_rating: int | None = None,
) -> dict[str, Any]:
    """Individual customer reviews plus the aggregate rating picture."""
    sku = str(sku).strip()
    limit = max(1, min(int(limit or DEFAULT_REVIEW_LIMIT), MAX_REVIEW_LIMIT))
    if sort not in review_api.SORT_ORDERS:
        sort = review_api.DEFAULT_SORT

    # Ask the source for more than we will show, so filters applied to a snapshot
    # have a decent pool to work from.
    bundle = load_reviews(
        sku,
        limit=max(limit, ANALYSIS_REVIEW_LIMIT if (min_rating or max_rating) else limit),
        sort=sort,
        min_rating=min_rating,
        max_rating=max_rating,
    )
    entry = get_catalog().get(sku)
    shown = _filter_and_sort(
        bundle.reviews, limit=limit, sort=sort, min_rating=min_rating, max_rating=max_rating
    )

    return {
        "sku": sku,
        "product_name": entry.name if entry else bundle.product_name,
        "product_url": entry.url if entry else bundle.product_url,
        "filters_applied": {
            "sort": sort,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "returned": len(shown),
        },
        "rating_summary": _stats_payload(bundle),
        "reviews": [_review_payload(review) for review in shown],
        **_provenance(bundle.source, bundle.fetched_at),
    }


def tool_analyze_review_sentiment(sku: str) -> dict[str, Any]:
    """Aspect-level sentiment, pros and cons, computed from the review text."""
    sku = str(sku).strip()
    bundle = load_reviews(sku, limit=ANALYSIS_REVIEW_LIMIT)
    entry = get_catalog().get(sku)
    report = analyse(
        bundle,
        product_name=entry.name if entry else None,
        category=entry.category if entry else None,
    )

    return {
        "sku": report.sku,
        "product_name": report.product_name,
        "reviews_analysed": report.reviews_analysed,
        "method": (
            "VADER sentence-level polarity with a pet-retail lexicon extension, bucketed into "
            "aspects by keyword. Star ratings are reported separately, not blended in."
        ),
        # Every key here is named for the *sample*, not the product. A field
        # called "mean_stars" gets quoted back as the product's rating, which it
        # is not: it is the mean of the reviews analysed here. The authoritative
        # product-wide average lives in get_product_reviews' rating_summary.
        "sample_statistics": {
            "note": (
                "These describe only the reviews analysed above, not the product's lifetime "
                "figures. For the official average rating and total review count, use "
                "get_product_reviews."
            ),
            "mean_stars_in_sample": report.mean_stars,
            "mean_text_polarity": report.mean_polarity,
            "positive_reviews_in_sample": report.positive,
            "neutral_reviews_in_sample": report.neutral,
            "negative_reviews_in_sample": report.negative,
            "star_vs_text_disagreements": report.mixed_signal_count,
        },
        "aspects": [
            {
                "aspect": item.label,
                "mentions": item.mentions,
                "weak_evidence": item.weak_evidence,
                "positive": item.positive,
                "neutral": item.neutral,
                "negative": item.negative,
                "positive_share": item.positive_share,
                "mean_stars_of_mentioning_reviews": item.mean_stars,
                "mean_polarity": item.mean_polarity,
                "positive_quotes": [quote.to_dict() for quote in item.supporting_quotes],
                "negative_quotes": [quote.to_dict() for quote in item.critical_quotes],
            }
            for item in report.aspects
            if item.mentions
        ],
        "pros": report.pros,
        "cons": report.cons,
        "guidance": (
            "Quote the reviewers' own words when summarising. An aspect with few mentions is weak "
            "evidence -- say so rather than presenting it as a finding."
        ),
        **_provenance(report.source, report.fetched_at),
    }


# --------------------------------------------------------------------------- #
# Schemas and dispatch
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Find which catalog products a user's wording refers to, and get their SKUs. "
                "ALWAYS call this before any other tool, because the other tools require a SKU "
                "and SKUs must never be guessed. Call it once per product mentioned. Omit the "
                "query to list the entire catalog, e.g. when the user asks what you can help with."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The product as the user described it, e.g. 'Black Hawk lamb and rice' "
                            "or 'the kangaroo roll'. Omit to list everything."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum candidates to return (default 5).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": (
                "Get specifications, current price (including the Petbarn loyalty member price), "
                "brand, category, size, availability, barcode, description and overall rating for "
                "one product. Use for questions about what a product is, what it costs, or whether "
                "it is in stock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "A SKU returned by search_catalog.",
                    }
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_reviews",
            "description": (
                "Get individual customer reviews for one product, plus the aggregate picture: "
                "average rating, star distribution, percentage who would recommend, and Petbarn's "
                "per-aspect rating averages for Quality, Value for money and Pet satisfaction. "
                "Use when the user wants to know what reviewers actually said, or wants specific "
                "kinds of review (e.g. only the critical ones)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "A SKU returned by search_catalog."},
                    "limit": {
                        "type": "integer",
                        "description": f"How many reviews to return, 1-{MAX_REVIEW_LIMIT} (default {DEFAULT_REVIEW_LIMIT}).",
                    },
                    "sort": {
                        "type": "string",
                        "enum": sorted(review_api.SORT_ORDERS),
                        "description": "Ordering. Use 'most_recent' for recent feedback, 'lowest_rating' to investigate complaints.",
                    },
                    "min_rating": {
                        "type": "integer",
                        "description": "Only reviews with at least this many stars (1-5).",
                    },
                    "max_rating": {
                        "type": "integer",
                        "description": "Only reviews with at most this many stars (1-5). Use 2 to find complaints.",
                    },
                },
                "required": ["sku"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_review_sentiment",
            "description": (
                "Analyse all available review text for one product and return sentiment broken "
                "down by theme -- price and value, quality and ingredients, taste, effectiveness, "
                "pet health, delivery and packaging, size -- each with mention counts, positive "
                "share, average stars and representative quotes, plus ranked pros and cons. "
                "Use this for 'what are people saying about the price/quality', for pros and cons, "
                "and as the basis for comparing two products' feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "A SKU returned by search_catalog."}
                },
                "required": ["sku"],
            },
        },
    },
]

#: Name -> implementation. Kept beside the schemas so the two cannot drift.
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_catalog": tool_search_catalog,
    "get_product_details": tool_get_product_details,
    "get_product_reviews": tool_get_product_reviews,
    "analyze_review_sentiment": tool_analyze_review_sentiment,
}

assert {schema["function"]["name"] for schema in TOOL_SCHEMAS} == set(TOOL_FUNCTIONS), (
    "TOOL_SCHEMAS and TOOL_FUNCTIONS are out of sync"
)


def execute(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    review_cap: int | None = None,
) -> ToolResult:
    """Run one tool by name, capturing timing, provenance and any failure.

    Errors are returned rather than raised: the model needs to be told that a
    lookup failed so it can say so, and an exception escaping here would end the
    turn with a stack trace instead of an answer.

    ``review_cap`` trims how many reviews a payload may carry. A local 8B model
    has a fraction of a hosted model's context, and overflowing it silently drops
    the *start* of the conversation -- including the instructions that keep
    answers grounded -- so the caller narrows the payload instead.
    """
    arguments = dict(arguments or {})
    started = time.perf_counter()

    if review_cap is not None and name == "get_product_reviews":
        try:
            requested = int(arguments.get("limit") or DEFAULT_REVIEW_LIMIT)
        except (TypeError, ValueError):
            requested = DEFAULT_REVIEW_LIMIT
        arguments["limit"] = max(1, min(requested, review_cap))

    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=False,
            error=f"unknown tool {name!r}; available tools: {', '.join(sorted(TOOL_FUNCTIONS))}",
            duration_ms=0,
        )

    try:
        payload = function(**arguments)
        return ToolResult(
            name=name,
            arguments=arguments,
            ok=True,
            payload=payload,
            source=payload.get("source") if isinstance(payload, dict) else None,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except ToolError as exc:
        error = str(exc)
    except TypeError as exc:
        error = f"invalid arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - the model must hear about any failure
        error = f"{type(exc).__name__} while running {name}: {exc}"

    return ToolResult(
        name=name,
        arguments=arguments,
        ok=False,
        error=error,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
