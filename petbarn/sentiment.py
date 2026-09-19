"""Aspect-based sentiment analysis over customer reviews.

The brief asks for sentiment, and the sample questions ask for it *per theme*
("the price and quality of X", "pros and cons for Z"). Those are different
problems: an overall polarity score cannot tell you that shoppers love a food's
quality while calling it expensive -- which is exactly the pattern in Petbarn's
data, where Value for money consistently averages below Quality.

So sentiment is **computed here rather than left to the language model**:

* VADER scores polarity per sentence. It is rule-based, so the same review
  always yields the same number, and a reviewer's words drive the result rather
  than the model's impression of them.
* Sentences are bucketed into aspects by keyword, so "great food but pricey"
  contributes positively to quality and negatively to price.
* Star ratings are reported *alongside* polarity instead of being blended into
  it, so the two signals can be compared -- and disagreement between them is
  counted explicitly as ``mixed_signal_count``.

The model's job is then to narrate evidence it has been handed, not to guess at
sentiment from a wall of text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .models import AspectSummary, Quote, Review, ReviewBundle, SentimentReport
from .textutils import split_sentences, truncate

#: Polarity beyond which a sentence counts as clearly positive or negative.
#: Anything in between is reported as neutral rather than being forced either
#: way by the reviewer's star rating.
POLARITY_THRESHOLD = 0.15

#: A star rating and the words in the review pointing opposite ways. Worth
#: counting: it is usually a happy customer with one specific complaint.
CONFLICT_POLARITY = 0.35

MAX_QUOTES_PER_ASPECT = 3
QUOTE_MAX_CHARS = 220

#: Minimum mentions before an aspect is allowed to become a "pro" or a "con".
#: Two people mentioning dust is an anecdote, not a finding.
MIN_MENTIONS_FOR_VERDICT = 3


@dataclass(frozen=True, slots=True)
class Aspect:
    """A theme reviewers talk about, and the words that signal it."""

    key: str
    label: str
    keywords: tuple[str, ...]
    #: Substrings of a product's category this aspect is meaningful for. Empty
    #: means "all products". Some themes simply do not apply everywhere: taste is
    #: a real signal for food and a category error for cat litter, and no amount
    #: of keyword tuning fixes that, because the words genuinely appear
    #: ("my cat eats the pellets"). Gating by category removes the nonsense at
    #: source rather than relying on the model to disregard it.
    categories: tuple[str, ...] = ()

    @property
    def pattern(self) -> re.Pattern[str]:
        return _compile_aspect(self.keywords)

    def applies_to(self, category: str | None) -> bool:
        if not self.categories:
            return True
        if not category:
            return True  # unknown category: measure it and let the counts speak
        folded = category.casefold()
        return any(needle in folded for needle in self.categories)


#: The themes that actually recur in Australian pet-retail reviews. Keyword
#: lists favour the words shoppers use over formal product vocabulary.
ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        "price_value",
        "Price & value for money",
        (
            "price", "priced", "pricey", "expensive", "cheap", "cheaper", "cost", "costly",
            "value", "worth", "affordable", "bargain", "overpriced", "budget", "dollar",
            "money", "sale", "discount", "deal", "rrp",
        ),
    ),
    Aspect(
        "quality",
        "Quality & ingredients",
        (
            "quality", "ingredient", "ingredients", "formula", "nutrition", "nutritious",
            "protein", "grain", "filler", "corn", "wheat", "natural", "premium", "additive",
            "preservative", "durable", "sturdy", "flimsy", "well made", "wellmade",
        ),
    ),
    Aspect(
        "palatability",
        "Taste & palatability",
        # Deliberately free of generic affection words. "loves" and "love it"
        # were in this list and matched 36 cat-litter reviews ("my cat loves
        # this product"), reporting taste findings for a product nothing eats.
        # A sentence that is merely warm belongs to no aspect: it still counts
        # towards overall sentiment, just not towards a specific claim.
        # "smell" is likewise excluded -- on litter it means odour control, which
        # the effectiveness aspect already covers.
        (
            "taste", "tasty", "flavour", "flavor", "palatable", "fussy", "picky",
            "gobble", "gobbles", "devour", "devours", "refuses", "refused",
            "wont eat", "won't eat", "eats", "eating", "ate", "appetite",
            "yummy", "delicious", "fussy eater", "mealtime",
        ),
        # Only things a pet actually eats: food, treats, and the chewable
        # parasite tablets, which reviewers routinely complain about refusing.
        categories=("food", "treat", "chew", "milk", "biscuit", "flea", "worm", "dental"),
    ),
    Aspect(
        "effectiveness",
        "Effectiveness & results",
        (
            "works", "worked", "working", "effective", "ineffective", "result", "results",
            "flea", "fleas", "tick", "ticks", "worm", "worms", "protection", "prevent",
            "absorbent", "absorb", "clump", "clumps", "clumping", "odour", "odor", "control",
        ),
    ),
    Aspect(
        "pet_health",
        "Coat, skin & digestion",
        (
            "coat", "fur", "shiny", "skin", "itch", "itchy", "allergy", "allergies",
            "digest", "digestion", "digestive", "stomach", "tummy", "stool", "stools",
            "poo", "diarrhoea", "diarrhea", "vomit", "vomited", "sick", "energy", "weight",
            "healthy", "health", "teeth", "joints",
        ),
    ),
    Aspect(
        "delivery_packaging",
        "Delivery & packaging",
        (
            "delivery", "delivered", "shipping", "shipped", "postage", "courier", "arrived",
            "packaging", "packaged", "box", "bag", "sealed", "leaked", "leaking", "damaged",
            "torn", "split", "resealable", "zip", "order", "ordered", "dispatch",
        ),
    ),
    Aspect(
        "size_quantity",
        "Size & quantity",
        (
            "size", "sized", "kilo", "kilos", "kg", "litre", "liter", "big", "small",
            "large", "tiny", "huge", "portion", "quantity", "lasts", "last", "lasted",
            "bulk", "pack", "amount", "kibble size",
        ),
    ),
)


#: Retail vocabulary VADER's general-purpose lexicon does not carry. Scores use
#: VADER's own -4..+4 valence scale. Applied with ``setdefault`` so VADER's
#: tuned values are never overwritten -- we only fill genuine gaps.
_DOMAIN_LEXICON: dict[str, float] = {
    # price
    "expensive": -1.5, "pricey": -1.5, "overpriced": -2.5, "affordable": 1.8,
    "bargain": 2.2, "costly": -1.5, "rrp": 0.0,
    # quality
    "filler": -1.4, "fillers": -1.4, "additives": -1.0, "preservatives": -0.8,
    "premium": 1.6, "flimsy": -1.8, "sturdy": 1.6, "durable": 1.8,
    "digestible": 1.4,
    # palatability
    "fussy": -1.2, "picky": -1.0, "palatable": 1.5, "gobbles": 1.6,
    "devours": 1.6, "refuses": -1.8, "refused": -1.8, "unappealing": -1.6,
    # effectiveness / litter
    "absorbent": 1.8, "clumping": 1.0, "clumps": 0.8, "dusty": -1.6,
    "ineffective": -2.0, "odour": -0.5, "odourless": 1.4,
    # health
    "itchy": -1.6, "diarrhoea": -2.4, "diarrhea": -2.4, "vomited": -2.4,
    "bloated": -1.6, "glossy": 1.6, "lethargic": -1.6,
    # logistics
    "leaked": -2.0, "leaking": -1.8, "resealable": 1.0, "undamaged": 1.0,
    # intent
    "repurchase": 1.8, "restock": 1.0, "reordered": 1.4,
}


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return the shared VADER analyser, extended with retail vocabulary."""
    analyzer = SentimentIntensityAnalyzer()
    for word, score in _DOMAIN_LEXICON.items():
        analyzer.lexicon.setdefault(word, score)
    return analyzer


@lru_cache(maxsize=32)
def _compile_aspect(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Build a word-boundary alternation for one aspect's keywords."""
    # Sort longest-first so "won't eat" is preferred over a bare "eat", and
    # allow flexible whitespace inside multi-word keywords.
    ordered = sorted(keywords, key=len, reverse=True)
    parts = [re.escape(k).replace(r"\ ", r"\s+") for k in ordered]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def polarity(text: str) -> float:
    """Compound VADER polarity for ``text``, from -1.0 to +1.0."""
    if not text.strip():
        return 0.0
    return float(get_analyzer().polarity_scores(text)["compound"])


def label_for_polarity(score: float) -> str:
    if score >= POLARITY_THRESHOLD:
        return "positive"
    if score <= -POLARITY_THRESHOLD:
        return "negative"
    return "neutral"


def label_for_stars(stars: int | None) -> str | None:
    if stars is None:
        return None
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"


@dataclass(slots=True)
class _Mention:
    """One sentence in one review that touched on a given aspect."""

    sentence: str
    score: float
    review: Review


def _summarise_aspect(aspect: Aspect, mentions: list[_Mention]) -> AspectSummary:
    summary = AspectSummary(aspect=aspect.key, label=aspect.label, mentions=len(mentions))
    if not mentions:
        return summary

    for mention in mentions:
        bucket = label_for_polarity(mention.score)
        setattr(summary, bucket, getattr(summary, bucket) + 1)

    decisive = summary.positive + summary.negative
    summary.positive_share = round(summary.positive / decisive, 3) if decisive else None
    summary.mean_polarity = round(sum(m.score for m in mentions) / len(mentions), 3)

    # Star ratings belong to whole reviews, so count each review once even when
    # it mentioned the aspect in several sentences.
    stars_by_review = {
        id(m.review): m.review.rating for m in mentions if m.review.rating is not None
    }
    if stars_by_review:
        values = list(stars_by_review.values())
        summary.mean_stars = round(sum(values) / len(values), 2)

    summary.weak_evidence = len(mentions) < MIN_MENTIONS_FOR_VERDICT

    ranked = sorted(mentions, key=lambda m: m.score, reverse=True)
    summary.supporting_quotes = [
        _to_quote(m) for m in ranked[:MAX_QUOTES_PER_ASPECT] if m.score >= POLARITY_THRESHOLD
    ]
    summary.critical_quotes = [
        _to_quote(m)
        for m in reversed(ranked[-MAX_QUOTES_PER_ASPECT:])
        if m.score <= -POLARITY_THRESHOLD
    ]
    return summary


def _to_quote(mention: _Mention) -> Quote:
    return Quote(
        text=truncate(mention.sentence, QUOTE_MAX_CHARS),
        rating=mention.review.rating,
        author=mention.review.author,
        submitted_at=mention.review.submitted_at,
    )


#: An aspect must be this positive to be sold as a pro, and no more positive
#: than :data:`CON_MAX_SHARE` to be called out as a con. The gap between the two
#: is deliberate: an aspect landing in it is genuinely mixed, and belongs in
#: neither list rather than in both.
PRO_MIN_SHARE = 0.8
CON_MAX_SHARE = 0.75


def _verdicts(aspects: list[AspectSummary]) -> tuple[list[str], list[str]]:
    """Rank aspects into pros and cons, with the evidence behind each.

    An aspect is never returned as both. Mixed aspects are omitted from both
    lists; they remain visible in ``report.aspects`` with their full counts.
    """
    pros: list[tuple[int, str]] = []
    cons: list[tuple[int, str]] = []

    for item in aspects:
        if item.mentions < MIN_MENTIONS_FOR_VERDICT:
            continue
        stars = f", avg {item.mean_stars}/5 stars" if item.mean_stars is not None else ""
        share = item.positive_share
        if item.positive >= 2 and share is not None and share >= PRO_MIN_SHARE:
            pros.append(
                (item.positive, f"{item.label} — {item.positive} of {item.mentions} mentions positive{stars}")
            )
        elif item.negative >= 2 and (share is None or share <= CON_MAX_SHARE):
            cons.append(
                (item.negative, f"{item.label} — {item.negative} of {item.mentions} mentions negative{stars}")
            )

    pros.sort(key=lambda pair: pair[0], reverse=True)
    cons.sort(key=lambda pair: pair[0], reverse=True)
    return [text for _, text in pros], [text for _, text in cons]


def analyse(
    bundle: ReviewBundle,
    *,
    product_name: str | None = None,
    category: str | None = None,
) -> SentimentReport:
    """Build an aspect-level sentiment report from a bundle of reviews.

    ``category`` gates which aspects are measured at all -- see
    :attr:`Aspect.categories`. Omitting it measures everything.
    """
    reviews = bundle.with_text_only()
    report = SentimentReport(
        sku=bundle.sku,
        product_name=product_name or bundle.product_name,
        reviews_analysed=len(reviews),
        source=bundle.source,
        fetched_at=bundle.fetched_at,
    )
    if not reviews:
        return report

    aspects = tuple(aspect for aspect in ASPECTS if aspect.applies_to(category))
    mentions: dict[str, list[_Mention]] = {aspect.key: [] for aspect in aspects}
    review_polarities: list[float] = []
    stars: list[int] = []

    for review in reviews:
        body = " ".join(part for part in (review.title, review.text) if part)
        review_score = polarity(body)
        review_polarities.append(review_score)

        star_label = label_for_stars(review.rating)
        if star_label:
            setattr(report, star_label, getattr(report, star_label) + 1)
        if review.rating is not None:
            stars.append(review.rating)

        # A glowing star rating next to negative prose (or the reverse) is a
        # signal in itself, so count it instead of silently averaging it away.
        if review.rating is not None and (
            (review.rating >= 4 and review_score <= -CONFLICT_POLARITY)
            or (review.rating <= 2 and review_score >= CONFLICT_POLARITY)
        ):
            report.mixed_signal_count += 1

        for sentence in split_sentences(body):
            sentence_score: float | None = None
            for aspect in aspects:
                if not aspect.pattern.search(sentence):
                    continue
                if sentence_score is None:
                    sentence_score = polarity(sentence)
                mentions[aspect.key].append(_Mention(sentence, sentence_score, review))

    report.mean_polarity = round(sum(review_polarities) / len(review_polarities), 3)
    if stars:
        report.mean_stars = round(sum(stars) / len(stars), 2)

    report.aspects = sorted(
        (_summarise_aspect(aspect, mentions[aspect.key]) for aspect in aspects),
        key=lambda item: item.mentions,
        reverse=True,
    )
    report.pros, report.cons = _verdicts(report.aspects)
    return report
