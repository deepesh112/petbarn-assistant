"""Product detail extraction from Petbarn product pages.

Petbarn's storefront is a Next.js app, so most of the page is assembled in the
browser -- but it publishes complete schema.org JSON-LD server-side. Parsing
that is far more stable than scraping rendered markup: it is a published
contract aimed at search engines, and it carries everything the brief asks for
(specifications, pricing, brand, basic details) including Petbarn's loyalty
member price.

Two page shapes exist and both are handled here:

``@type: "Product"``
    A single-size item. One ``offers`` object with the price.

``@type: "ProductGroup"``
    A multi-size item (e.g. cat litter in 10L and 30L). There is **no
    top-level price**; each entry in ``hasVariant`` carries its own SKU, size
    and price. Missing this is the fastest way to end up with a catalog full of
    products that have no price.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from . import config
from .http import FetchResult, PoliteSession, get_session
from .models import Offer, ProductDetails, ShippingOption, Source
from .textutils import clean_text

_JSONLD_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

#: Petbarn tags its own DOM nodes with ``data-identifier-id``. These are the
#: nodes that hold the long marketing description, most specific first.
_DESCRIPTION_MARKERS = (
    "description-text",
    "description-content-wrapper",
    "accordion-item-description",
)

_AVAILABILITY_LABELS = {
    "instock": "In stock",
    "outofstock": "Out of stock",
    "onlineonly": "Online only",
    "instoreonly": "In store only",
    "preorder": "Pre-order",
    "backorder": "On back-order",
    "limitedavailability": "Limited availability",
    "soldout": "Sold out",
    "discontinued": "Discontinued",
}


class ProductParseError(RuntimeError):
    """The page loaded but contained no usable product data."""


# --------------------------------------------------------------------------- #
# JSON-LD plumbing
# --------------------------------------------------------------------------- #


def iter_jsonld(html: str) -> list[dict[str, Any]]:
    """Return every JSON-LD object on the page, flattening ``@graph`` wrappers."""
    nodes: list[dict[str, Any]] = []
    for match in _JSONLD_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(node, dict):
                continue
            graph = node.get("@graph")
            if isinstance(graph, list):
                nodes.extend(n for n in graph if isinstance(n, dict))
            else:
                nodes.append(node)
    return nodes


def _node_types(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type") or []
    values = raw if isinstance(raw, list) else [raw]
    return {str(v).lower() for v in values}


def find_product_node(html: str) -> dict[str, Any]:
    """Locate the ``Product`` or ``ProductGroup`` node on a product page."""
    nodes = iter_jsonld(html)
    for wanted in ("product", "productgroup"):
        for node in nodes:
            if wanted in _node_types(node):
                return node
    raise ProductParseError("no Product or ProductGroup JSON-LD found on page")


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


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return None if number is None else int(round(number))


def _to_rating(value: Any) -> float | None:
    """Round a star rating to two places.

    Petbarn publishes ratings to four decimals (``4.6139``). Passing that through
    to the model invites answers like "rated 4.6139 stars", and it would disagree
    with the review platform's own rounded figure for the same product.
    """
    number = _to_float(value)
    return None if number is None else round(number, 2)


def _first(value: Any) -> Any:
    """Schema.org fields are routinely either a value or a list of values."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _brand_name(node: dict[str, Any]) -> str | None:
    brand = _first(node.get("brand"))
    if isinstance(brand, dict):
        return brand.get("name")
    return brand if isinstance(brand, str) else None


def _availability_label(raw: Any) -> str | None:
    token = str(_first(raw) or "").rsplit("/", 1)[-1]
    if not token:
        return None
    return _AVAILABILITY_LABELS.get(token.lower(), token)


def _split_prices(offer: dict[str, Any]) -> tuple[float | None, float | None]:
    """Separate the everyday price from Petbarn's loyalty member price.

    Both arrive in ``priceSpecification``; the member price is the entry tagged
    with a ``validForMemberTier``. We fall back to ``offers.price`` because
    single-tier products omit the specification list entirely.
    """
    standard = _to_float(offer.get("price"))
    member: float | None = None

    specs = offer.get("priceSpecification")
    for spec in specs if isinstance(specs, list) else [specs]:
        if not isinstance(spec, dict):
            continue
        amount = _to_float(spec.get("price"))
        if amount is None:
            continue
        if spec.get("validForMemberTier"):
            member = amount if member is None else min(member, amount)
        elif standard is None:
            standard = amount

    # A "member price" that isn't actually cheaper is noise, not a discount.
    if member is not None and standard is not None and member >= standard:
        member = None
    return standard, member


def _parse_shipping(offer: dict[str, Any]) -> list[ShippingOption]:
    options: list[ShippingOption] = []
    details = offer.get("shippingDetails")
    for detail in details if isinstance(details, list) else [details]:
        if not isinstance(detail, dict):
            continue
        rate = detail.get("shippingRate") or {}
        options.append(
            ShippingOption(
                name=detail.get("name") or "Delivery",
                price=_to_float(rate.get("value")) if isinstance(rate, dict) else None,
                currency=(rate.get("currency") if isinstance(rate, dict) else None) or "AUD",
            )
        )
    return options


def _is_bundle(node: dict[str, Any]) -> bool:
    """Read Petbarn's ``is_bundle`` flag out of ``additionalProperty``."""
    props = node.get("additionalProperty")
    for prop in props if isinstance(props, list) else [props]:
        if not isinstance(prop, dict):
            continue
        if str(prop.get("name", "")).strip().lower() == "is_bundle":
            return str(prop.get("value", "")).strip().lower() in {"true", "1", "yes"}
    return False


def _offer_of(node: dict[str, Any]) -> dict[str, Any]:
    offer = _first(node.get("offers"))
    return offer if isinstance(offer, dict) else {}


def _variant_to_offer(node: dict[str, Any]) -> Offer:
    offer = _offer_of(node)
    standard, member = _split_prices(offer)
    return Offer(
        sku=str(node.get("sku") or "").strip(),
        name=node.get("name"),
        size=node.get("size"),
        price=standard,
        member_price=member,
        currency=offer.get("priceCurrency") or "AUD",
        availability=_availability_label(offer.get("availability")),
        gtin=node.get("gtin") or node.get("gtin13"),
        url=node.get("url") or offer.get("url"),
    )


# --------------------------------------------------------------------------- #
# Description
# --------------------------------------------------------------------------- #


def extract_description(html: str, *, max_chars: int = 2000) -> str | None:
    """Pull the marketing description out of the server-rendered accordion.

    The page is well over a megabyte, so rather than parse all of it we slice a
    window around the description node and let BeautifulSoup clean up just that
    fragment. ``html.parser`` is used deliberately: it is pure Python, so there
    is no compiled dependency to go wrong on a deployment host.
    """
    for marker in _DESCRIPTION_MARKERS:
        index = html.find(f'data-identifier-id="{marker}"')
        if index == -1:
            continue
        # Resume after the enclosing tag's ">" -- otherwise the attribute text
        # we just matched on has no leading "<" and the fragment parser reads
        # it as body text, prefixing the description with raw markup.
        start = html.find(">", index)
        if start == -1:
            continue
        window = html[start + 1 : start + 24_000]
        text = clean_text(BeautifulSoup(window, "html.parser").get_text(" ", strip=True))
        # The node's own heading leads the text; drop it so the description
        # reads as prose rather than starting with "Description".
        text = re.sub(r"^(Description|Ingredients|Features and Benefits)\s*", "", text)
        if len(text) > 80:
            return text[:max_chars].strip()
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_product(
    html: str,
    *,
    url: str,
    prefer_sku: str | None = None,
    source: Source = "live",
    fetched_at: str | None = None,
) -> ProductDetails:
    """Turn a product page into :class:`ProductDetails`.

    ``prefer_sku`` selects a specific variant of a multi-size product; without
    it the first variant that actually has a price wins.
    """
    node = find_product_node(html)
    types = _node_types(node)

    variants: list[Offer] = []
    if "productgroup" in types:
        raw_variants = node.get("hasVariant")
        raw_variants = raw_variants if isinstance(raw_variants, list) else []
        variants = [_variant_to_offer(v) for v in raw_variants if isinstance(v, dict)]
        chosen_node = _pick_variant_node(raw_variants, variants, prefer_sku)
    else:
        chosen_node = node

    offer = _offer_of(chosen_node)
    standard, member = _split_prices(offer)
    rating = node.get("aggregateRating") or chosen_node.get("aggregateRating") or {}

    sku = str(chosen_node.get("sku") or node.get("sku") or prefer_sku or "").strip()
    if not sku:
        # Without a SKU we cannot look up reviews, which makes the product
        # useless to this assistant -- fail loudly rather than half-load it.
        raise ProductParseError(f"no SKU resolvable for {url}")

    return ProductDetails(
        sku=sku,
        name=clean_text(chosen_node.get("name") or node.get("name")) or sku,
        url=chosen_node.get("url") or node.get("url") or url,
        brand=_brand_name(chosen_node) or _brand_name(node),
        category=node.get("category") or chosen_node.get("category"),
        description=extract_description(html) or clean_text(node.get("description")) or None,
        image=_first(chosen_node.get("image")) or _first(node.get("image")),
        gtin=chosen_node.get("gtin") or chosen_node.get("gtin13") or node.get("gtin"),
        size=chosen_node.get("size"),
        price=standard,
        member_price=member,
        currency=offer.get("priceCurrency") or "AUD",
        availability=_availability_label(offer.get("availability")),
        is_bundle=_is_bundle(chosen_node) or _is_bundle(node) or sku.lower().startswith("bundle"),
        average_rating=_to_rating(rating.get("ratingValue")) if isinstance(rating, dict) else None,
        review_count=_to_int(rating.get("reviewCount")) if isinstance(rating, dict) else None,
        variants=variants,
        shipping=_parse_shipping(offer),
        source=source,
        fetched_at=fetched_at,
    )


def _pick_variant_node(
    raw_variants: list[Any],
    offers: list[Offer],
    prefer_sku: str | None,
) -> dict[str, Any]:
    """Choose which variant of a product group this page represents."""
    nodes = [v for v in raw_variants if isinstance(v, dict)]
    if not nodes:
        raise ProductParseError("ProductGroup had no usable variants")

    if prefer_sku:
        for node in nodes:
            if str(node.get("sku") or "").strip() == str(prefer_sku).strip():
                return node

    for node, offer in zip(nodes, offers):
        if offer.has_price:
            return node
    return nodes[0]


def fetch_product(
    url: str,
    *,
    prefer_sku: str | None = None,
    session: PoliteSession | None = None,
    use_cache: bool = True,
) -> tuple[ProductDetails, FetchResult]:
    """Fetch and parse a Petbarn product page."""
    session = session or get_session()
    result = session.get(url, use_cache=use_cache)
    product = parse_product(
        result.body,
        url=url,
        prefer_sku=prefer_sku,
        source=result.source,
        fetched_at=result.fetched_at,
    )
    return product, result


def product_url_for_sku(slug: str, sku: str) -> str:
    """Build the canonical variant URL Petbarn uses for a specific SKU."""
    return f"{config.SITE_ROOT}{config.PRODUCT_PATH_PREFIX}{slug.strip('/')}/{sku}"
