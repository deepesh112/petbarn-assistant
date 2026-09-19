"""Customer review retrieval via Bazaarvoice.

Petbarn renders reviews client-side through Bazaarvoice, so they are absent from
the product page HTML. The reviews live behind the Bazaarvoice Conversations
display API -- the same public, read-only endpoint the product page itself calls
from the visitor's browser, keyed by Petbarn's Magento SKU.

The display *passkey* that authorises those reads ships inside Petbarn's own
browser bundle. It is not a secret (every visitor receives it), but it can be
rotated, so :func:`resolve_passkey` re-discovers it at runtime and only falls
back to a hard-coded value if discovery fails.

What comes back is richer than star ratings alone: Petbarn collects secondary
ratings for **Quality**, **Value for money** and **Pet satisfaction**, which is
exactly the breakdown needed to answer questions about price versus quality.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from . import config
from .http import FetchError, FetchResult, PoliteSession, get_session
from .models import Review, ReviewBundle, ReviewStats
from .textutils import clean_text

#: Bazaarvoice display passkeys are long alphanumeric strings beginning "ca".
_PASSKEY_RE = re.compile(r"\bca[A-Za-z0-9]{35,60}\b")

#: Sort orders we expose to the agent, mapped to Bazaarvoice sort expressions.
SORT_ORDERS = {
    "most_recent": "SubmissionTime:desc",
    "oldest": "SubmissionTime:asc",
    "highest_rating": "Rating:desc",
    "lowest_rating": "Rating:asc",
    "most_helpful": "Helpfulness:desc",
}
DEFAULT_SORT = "most_recent"

_passkey_cache: str | None = None


class ReviewFetchError(RuntimeError):
    """Reviews could not be retrieved from the review platform."""


# --------------------------------------------------------------------------- #
# Passkey discovery
# --------------------------------------------------------------------------- #


def resolve_passkey(session: PoliteSession | None = None, *, refresh: bool = False) -> str:
    """Return a usable Bazaarvoice display passkey.

    Resolution order: explicit ``BV_PASSKEY`` override, then the key embedded in
    Petbarn's live Bazaarvoice bundle, then the last-known-good constant.
    """
    global _passkey_cache

    override = config.bv_passkey_override()
    if override:
        return override
    if _passkey_cache and not refresh:
        return _passkey_cache

    session = session or get_session()
    try:
        # The loader changes rarely, so cache it for a full day rather than
        # re-downloading on every cold start.
        result = session.get(config.BV_LOADER_JS_URL, ttl_seconds=86_400)
        match = _PASSKEY_RE.search(result.body)
        if match:
            _passkey_cache = match.group(0)
            return _passkey_cache
    except FetchError:
        pass

    _passkey_cache = config.BV_FALLBACK_PASSKEY
    return _passkey_cache


# --------------------------------------------------------------------------- #
# Field coercion
# --------------------------------------------------------------------------- #


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value)
    return default if number is None else int(round(number))


def _is_verified(node: dict[str, Any]) -> bool:
    """Detect the verified-purchaser badge, which BV exposes two ways."""
    order = node.get("BadgesOrder")
    if isinstance(order, list) and any("verifiedpurchaser" in str(b).lower() for b in order):
        return True
    badges = node.get("Badges")
    if isinstance(badges, dict):
        return any("verifiedpurchaser" in str(k).lower() for k in badges)
    return False


def _secondary_ratings(node: dict[str, Any]) -> dict[str, float]:
    """Normalise per-review aspect ratings to ``{friendly label: stars}``."""
    raw = node.get("SecondaryRatings")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        score = _to_float(value.get("Value")) if isinstance(value, dict) else _to_float(value)
        if score is None:
            continue
        out[config.BV_SECONDARY_RATING_LABELS.get(key, key)] = score
    return out


def _parse_review(node: dict[str, Any]) -> Review:
    return Review(
        review_id=str(node.get("Id")) if node.get("Id") is not None else None,
        title=clean_text(node.get("Title")) or None,
        text=clean_text(node.get("ReviewText")),
        rating=_to_int(node.get("Rating"), default=0) or None,
        author=clean_text(node.get("UserNickname")) or None,
        submitted_at=node.get("SubmissionTime"),
        is_verified_purchaser=_is_verified(node),
        is_recommended=node.get("IsRecommended"),
        helpful_votes=_to_int(node.get("TotalPositiveFeedbackCount")),
        unhelpful_votes=_to_int(node.get("TotalNegativeFeedbackCount")),
        secondary_ratings=_secondary_ratings(node),
    )


def _parse_stats(raw: Any, *, text_review_count: int | None = None) -> ReviewStats:
    """Convert a Bazaarvoice ``ReviewStatistics`` block into :class:`ReviewStats`."""
    stats = ReviewStats()
    if not isinstance(raw, dict):
        if text_review_count is not None:
            stats.text_review_count = text_review_count
        return stats

    average = _to_float(raw.get("AverageOverallRating"))
    stats.average_rating = None if average is None else round(average, 2)
    stats.total_reviews = _to_int(raw.get("TotalReviewCount"))
    stats.recommended_count = _to_int(raw.get("RecommendedCount"))
    stats.not_recommended_count = _to_int(raw.get("NotRecommendedCount"))
    stats.first_review_at = raw.get("FirstSubmissionTime")
    stats.last_review_at = raw.get("LastSubmissionTime")

    distribution = raw.get("RatingDistribution")
    if isinstance(distribution, list):
        stats.rating_distribution = {
            str(_to_int(bucket.get("RatingValue"))): _to_int(bucket.get("Count"))
            for bucket in distribution
            if isinstance(bucket, dict)
        }

    averages = raw.get("SecondaryRatingsAverages")
    if isinstance(averages, dict):
        for key, value in averages.items():
            score = (
                _to_float(value.get("AverageRating")) if isinstance(value, dict) else _to_float(value)
            )
            if score is None:
                continue
            label = config.BV_SECONDARY_RATING_LABELS.get(key, key)
            stats.secondary_rating_averages[label] = round(score, 2)

    if text_review_count is not None:
        stats.text_review_count = text_review_count
    else:
        ratings_only = _to_int(raw.get("RatingsOnlyReviewCount"))
        stats.text_review_count = max(stats.total_reviews - ratings_only, 0)
    return stats


def _product_statistics(payload: dict[str, Any], sku: str) -> tuple[Any, str | None]:
    """Extract the included product's review statistics and display name."""
    products = (payload.get("Includes") or {}).get("Products")
    if not isinstance(products, dict):
        return None, None
    node = products.get(sku)
    if not isinstance(node, dict):
        # BV occasionally keys the include by a different identifier; with a
        # single-product query there is only one entry, so take it.
        candidates = [v for v in products.values() if isinstance(v, dict)]
        node = candidates[0] if candidates else None
    if not isinstance(node, dict):
        return None, None
    return node.get("ReviewStatistics"), node.get("Name")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def fetch_reviews(
    sku: str,
    *,
    limit: int = 150,
    text_only: bool = True,
    sort: str = DEFAULT_SORT,
    min_rating: int | None = None,
    max_rating: int | None = None,
    session: PoliteSession | None = None,
    use_cache: bool = True,
) -> tuple[ReviewBundle, FetchResult]:
    """Fetch reviews for ``sku``, paging until ``limit`` is reached.

    ``text_only`` drops star-only ratings, which are the majority on popular
    products and carry nothing to summarise.
    """
    session = session or get_session()
    passkey = resolve_passkey(session)
    sort_expression = SORT_ORDERS.get(sort, SORT_ORDERS[DEFAULT_SORT])

    filters = [f"ProductId:{sku}"]
    if text_only:
        filters.append("IsRatingsOnly:false")
    if min_rating is not None:
        filters.append(f"Rating:gte:{int(min_rating)}")
    if max_rating is not None:
        filters.append(f"Rating:lte:{int(max_rating)}")

    collected: list[Review] = []
    stats = ReviewStats()
    product_name: str | None = None
    last_result: FetchResult | None = None
    total_matching: int | None = None
    offset = 0

    while len(collected) < limit:
        page_size = min(config.BV_MAX_PAGE_LIMIT, limit - len(collected))
        params: dict[str, Any] = {
            "ApiVersion": config.BV_API_VERSION,
            "PassKey": passkey,
            "Filter": filters,
            "Sort": sort_expression,
            "Limit": page_size,
            "Offset": offset,
        }
        if offset == 0:
            # Statistics only need requesting once, on the first page.
            params["Include"] = "Products"
            params["Stats"] = "Reviews"

        try:
            payload, last_result = session.get_json(
                f"{config.BV_API_ROOT}/reviews.json",
                params=params,
                use_cache=use_cache,
            )
        except FetchError as exc:
            if collected:
                break  # partial data is still useful; stop paging and return it
            raise ReviewFetchError(f"could not fetch reviews for SKU {sku}: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("HasErrors"):
            errors = payload.get("Errors") if isinstance(payload, dict) else None
            raise ReviewFetchError(f"Bazaarvoice rejected the request for SKU {sku}: {errors}")

        if offset == 0:
            total_matching = _to_int(payload.get("TotalResults"))
            raw_stats, product_name = _product_statistics(payload, str(sku))
            stats = _parse_stats(raw_stats, text_review_count=total_matching if text_only else None)

        results = payload.get("Results") or []
        if not results:
            break
        collected.extend(_parse_review(node) for node in results if isinstance(node, dict))

        offset += len(results)
        if total_matching is not None and offset >= total_matching:
            break

    if stats.total_reviews == 0 and total_matching:
        stats.total_reviews = total_matching
    if stats.average_rating is None:
        rated = [r.rating for r in collected if r.rating]
        if rated:
            stats.average_rating = round(sum(rated) / len(rated), 2)

    bundle = ReviewBundle(
        sku=str(sku),
        stats=stats,
        reviews=collected[:limit],
        product_name=product_name,
        source=last_result.source if last_result else "live",
        fetched_at=last_result.fetched_at if last_result else None,
    )
    return bundle, last_result  # type: ignore[return-value]


def fetch_top_reviewed_products(
    *,
    limit: int = 100,
    session: PoliteSession | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """List Petbarn's most-reviewed products, newest statistics included.

    Used by the ingestion script to pick a catalog worth talking about: a
    product with four reviews cannot support a conversation about sentiment.
    """
    session = session or get_session()
    passkey = resolve_passkey(session)

    products: list[dict[str, Any]] = []
    offset = 0
    while len(products) < limit:
        page_size = min(config.BV_MAX_PAGE_LIMIT, limit - len(products))
        payload, _ = session.get_json(
            f"{config.BV_API_ROOT}/products.json",
            params={
                "ApiVersion": config.BV_API_VERSION,
                "PassKey": passkey,
                "Sort": "TotalReviewCount:desc",
                "Stats": "Reviews",
                "Limit": page_size,
                "Offset": offset,
            },
            use_cache=use_cache,
        )
        if not isinstance(payload, dict) or payload.get("HasErrors"):
            raise ReviewFetchError(f"Bazaarvoice product listing failed: {payload}")

        results = payload.get("Results") or []
        if not results:
            break
        products.extend(r for r in results if isinstance(r, dict))
        offset += len(results)

    return products[:limit]


def summarise_distribution(distribution: dict[str, int]) -> Iterable[tuple[int, int]]:
    """Yield ``(stars, count)`` from 5 down to 1, filling in absent buckets."""
    for stars in range(5, 0, -1):
        yield stars, int(distribution.get(str(stars), 0))
