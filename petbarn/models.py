"""Domain types shared by the scrapers, the tools and the UI.

These dataclasses are the contract between layers. They round-trip cleanly to
JSON in both directions, which is what lets a live fetch and a snapshot read
produce byte-identical tool payloads.

Every object that a tool can return carries its own provenance (``source`` and
``fetched_at``) so the assistant -- and the person reading its answer -- always
knows where a number came from and how old it is.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

#: Where a payload came from. ``live`` means we just fetched it, ``cache`` means
#: a recent local copy, ``snapshot`` means the dataset committed to the repo.
Source = Literal["live", "cache", "snapshot"]

T = TypeVar("T")


def _coerce(cls: type[T], payload: dict[str, Any]) -> T:
    """Build a dataclass from a dict, ignoring keys the class doesn't declare.

    Snapshot files outlive code changes; tolerating unknown and missing keys
    means an older snapshot still loads instead of crashing the app.
    """
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in payload.items() if k in known})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Product side
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Offer:
    """A buyable variant: one size/pack of a product, with its own price."""

    sku: str
    name: str | None = None
    size: str | None = None
    price: float | None = None
    member_price: float | None = None
    currency: str = "AUD"
    availability: str | None = None
    gtin: str | None = None
    url: str | None = None

    @property
    def has_price(self) -> bool:
        return self.price is not None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Offer:
        return _coerce(cls, payload)


@dataclass(slots=True)
class ShippingOption:
    name: str
    price: float | None = None
    currency: str = "AUD"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShippingOption:
        return _coerce(cls, payload)


@dataclass(slots=True)
class ProductDetails:
    """Specifications, pricing and identity for a single Petbarn product."""

    sku: str
    name: str
    url: str
    brand: str | None = None
    category: str | None = None
    description: str | None = None
    image: str | None = None
    gtin: str | None = None
    size: str | None = None

    price: float | None = None
    member_price: float | None = None
    currency: str = "AUD"
    availability: str | None = None

    average_rating: float | None = None
    review_count: int | None = None

    #: Petbarn sells multi-buy bundles ("... 30L x 3") as their own products.
    #: They duplicate a single-unit item's reviews, so the catalog excludes them.
    is_bundle: bool = False

    #: Other sizes of the same product, when Petbarn models it as a group.
    variants: list[Offer] = field(default_factory=list)
    shipping: list[ShippingOption] = field(default_factory=list)

    source: Source = "live"
    fetched_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProductDetails:
        obj = _coerce(cls, payload)
        obj.variants = [Offer.from_dict(v) for v in (payload.get("variants") or [])]
        obj.shipping = [ShippingOption.from_dict(s) for s in (payload.get("shipping") or [])]
        return obj


# --------------------------------------------------------------------------- #
# Review side
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Review:
    """One customer review, as published on the product page."""

    review_id: str | None = None
    title: str | None = None
    text: str = ""
    rating: int | None = None
    author: str | None = None
    submitted_at: str | None = None
    is_verified_purchaser: bool = False
    is_recommended: bool | None = None
    helpful_votes: int = 0
    unhelpful_votes: int = 0
    #: Per-aspect stars the reviewer gave, e.g. ``{"Quality": 5, "Value": 4}``.
    secondary_ratings: dict[str, float] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Review:
        return _coerce(cls, payload)


@dataclass(slots=True)
class ReviewStats:
    """Aggregate review figures straight from the review platform."""

    average_rating: float | None = None
    #: Every review, including star-only ratings with no written text.
    total_reviews: int = 0
    #: Reviews that actually contain prose -- the ones worth summarising.
    text_review_count: int = 0
    #: Star value -> number of reviews, keyed by string for JSON safety.
    rating_distribution: dict[str, int] = field(default_factory=dict)
    recommended_count: int | None = None
    not_recommended_count: int | None = None
    #: Mean of each aspect rating, e.g. ``{"Value for money": 4.61}``.
    secondary_rating_averages: dict[str, float] = field(default_factory=dict)
    first_review_at: str | None = None
    last_review_at: str | None = None

    @property
    def recommend_share(self) -> float | None:
        """Fraction of reviewers who said they'd recommend the product."""
        yes, no = self.recommended_count, self.not_recommended_count
        if yes is None or no is None or (yes + no) == 0:
            return None
        return yes / (yes + no)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReviewStats:
        return _coerce(cls, payload)


@dataclass(slots=True)
class ReviewBundle:
    """Reviews for one product, plus the aggregates that give them context."""

    sku: str
    stats: ReviewStats = field(default_factory=ReviewStats)
    reviews: list[Review] = field(default_factory=list)
    product_name: str | None = None
    product_url: str | None = None
    source: Source = "live"
    fetched_at: str | None = None

    def with_text_only(self) -> list[Review]:
        return [r for r in self.reviews if r.has_text]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReviewBundle:
        obj = _coerce(cls, payload)
        obj.stats = ReviewStats.from_dict(payload.get("stats") or {})
        obj.reviews = [Review.from_dict(r) for r in (payload.get("reviews") or [])]
        return obj


# --------------------------------------------------------------------------- #
# Sentiment side
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Quote:
    """A short excerpt used as evidence for an aspect's sentiment."""

    text: str
    rating: int | None = None
    author: str | None = None
    submitted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class AspectSummary:
    """What reviewers said about one theme, e.g. price or quality."""

    aspect: str
    label: str
    mentions: int = 0
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    #: Share of mentions that read as positive, 0.0-1.0.
    positive_share: float | None = None
    #: Mean star rating of the reviews that touched on this aspect.
    mean_stars: float | None = None
    #: Mean VADER compound polarity of the matching sentences, -1.0 to 1.0.
    mean_polarity: float | None = None
    #: True when too few people mentioned this for it to support a claim. Given
    #: to the model explicitly rather than left for it to infer from the count,
    #: because a small model will happily turn three comments into "customers say".
    weak_evidence: bool = True
    supporting_quotes: list[Quote] = field(default_factory=list)
    critical_quotes: list[Quote] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class SentimentReport:
    """Aspect-level sentiment for a product, computed from its review text."""

    sku: str
    product_name: str | None = None
    reviews_analysed: int = 0
    #: Overall label counts derived from star ratings.
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    mean_stars: float | None = None
    mean_polarity: float | None = None
    #: Where star rating and text polarity disagree -- worth surfacing honestly.
    mixed_signal_count: int = 0
    aspects: list[AspectSummary] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    source: Source = "live"
    fetched_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Catalog side
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CatalogEntry:
    """A product in the curated catalog this assistant is scoped to."""

    sku: str
    name: str
    url: str
    brand: str | None = None
    category: str | None = None
    pet: str | None = None
    price: float | None = None
    average_rating: float | None = None
    review_count: int | None = None
    text_review_count: int | None = None
    #: Extra names a person might use for this product, for fuzzy matching.
    aliases: list[str] = field(default_factory=list)

    def search_terms(self) -> list[str]:
        terms = [self.name, *self.aliases]
        if self.brand:
            terms.append(self.brand)
        return [t for t in terms if t]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CatalogEntry:
        return _coerce(cls, payload)
