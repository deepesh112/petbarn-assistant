"""The curated catalog, and resolving what a shopper said to a SKU.

Every other tool takes a SKU. People do not talk in SKUs -- they say "the Black
Hawk lamb and rice", "that kangaroo roll", "royal canin indoor cat". So this
module owns the translation, and owns it *deterministically*: if the assistant
were left to guess SKUs it would invent them, and a fabricated SKU produces a
confident answer about a product that does not exist.

Matching is scored rather than boolean so the assistant can be handed ranked
candidates and ask a clarifying question when two products are close, instead of
silently picking one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from . import config
from .models import CatalogEntry

#: Words that carry no product signal. Stripped from queries so "what are people
#: saying about the price of the Black Hawk" scores on "black hawk" alone.
_STOPWORDS = frozenset(
    """
    a an the and or of for to in on at is are was were be been do does did
    about what how why which who whom this that these those there here
    i me my we our you your it its they them their
    petbarn product products item items thing stuff
    review reviews reviewer reviewers rating ratings rated star stars
    people customer customers buyer buyers shopper shoppers say saying said
    tell show give list compare comparison versus vs between both
    price priced pricing cost costs quality good bad best worst
    pro pros con cons feedback sentiment opinion opinions think thoughts
    please can could would should like want need any some more most
    main recent latest new old
    """.split()
)

#: Keep letters and digits; everything else is a separator. Tokens like "20kg"
#: and "85gx12" survive intact, which matters because sizes disambiguate.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Below this a match is too weak to offer as a candidate at all.
MIN_SCORE = 0.32

#: At or above this, a single match is confident enough to use without asking.
STRONG_SCORE = 0.72


def tokenize(text: str) -> list[str]:
    """Split text into lowercase content tokens, dropping stopwords.

    Digit-only tokens are discarded. Product names are full of fragmentary
    numbers -- "15.1 - 30kg" tokenises to ``15``, ``1``, ``30kg`` -- and matching
    on those alone means "iphone 15" scores against a flea treatment. Sizes that
    carry their unit ("30kg", "750g") contain letters and so survive.
    """
    tokens = (match.group(0).lower() for match in _TOKEN_RE.finditer(text or ""))
    return [
        token
        for token in tokens
        if len(token) > 1 and not token.isdigit() and token not in _STOPWORDS
    ]


#: How alike two tokens must be to count as the same word. Tuned so "royl"
#: matches "royal" but "liver" does not match "litter".
TOKEN_SIMILARITY = 0.85

#: A whole-name similarity below this is coincidence, not a typo. Short queries
#: against long product names produce ratios around 0.4 purely by chance, and
#: accepting those invents matches for products the catalog does not carry.
MIN_NAME_SIMILARITY = 0.6


def pretty_brand(brand: str | None) -> str | None:
    """Soften the all-caps brand names Petbarn's catalog stores.

    "ROYAL CANIN" reads as shouting in an answer; "Royal Canin" does not. Brands
    that are already mixed case are left exactly as the retailer writes them.
    """
    if not brand:
        return None
    return brand.title() if brand.isupper() else brand


def _token_coverage(
    query_tokens: set[str], entry_tokens: set[str]
) -> tuple[set[str], float]:
    """Fraction of query tokens this entry accounts for, allowing near-misses.

    Exact hits are free; a query token with no exact match is compared against
    similarly-sized entry tokens so "royl canin" still finds Royal Canin. Returns
    the query tokens that matched, and their share of the query.
    """
    if not query_tokens:
        return set(), 0.0

    matched: set[str] = set()
    for token in query_tokens:
        if token in entry_tokens:
            matched.add(token)
            continue
        if len(token) < 4:
            continue  # too short for a similarity check to mean anything
        for candidate in entry_tokens:
            if abs(len(candidate) - len(token)) > 2:
                continue
            if SequenceMatcher(None, token, candidate).ratio() >= TOKEN_SIMILARITY:
                matched.add(token)
                break

    return matched, len(matched) / len(query_tokens)


@dataclass(slots=True)
class Match:
    """A catalog entry that plausibly answers a query, and how well it fits."""

    entry: CatalogEntry
    score: float
    matched_on: str

    @property
    def is_strong(self) -> bool:
        return self.score >= STRONG_SCORE


class Catalog:
    """The fixed set of products this assistant can talk about."""

    def __init__(self, entries: list[CatalogEntry], *, generated_at: str | None = None) -> None:
        self._entries = entries
        self.generated_at = generated_at
        self._by_sku = {entry.sku: entry for entry in entries}
        # Precomputed token sets, so scoring a query is pure set arithmetic.
        self._tokens: dict[str, set[str]] = {}
        for entry in entries:
            haystack = " ".join(
                part
                for part in (entry.name, *entry.aliases, entry.brand, entry.category, entry.pet)
                if part
            )
            self._tokens[entry.sku] = set(tokenize(haystack))

    # ----------------------------------------------------------------- #
    # Access
    # ----------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> list[CatalogEntry]:
        return list(self._entries)

    @property
    def skus(self) -> list[str]:
        return [entry.sku for entry in self._entries]

    def get(self, sku: str) -> CatalogEntry | None:
        return self._by_sku.get(str(sku).strip())

    def contains(self, sku: str) -> bool:
        return str(sku).strip() in self._by_sku

    def categories(self) -> list[str]:
        return sorted({entry.category for entry in self._entries if entry.category})

    # ----------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------- #

    def search(self, query: str, *, limit: int = 5) -> list[Match]:
        """Rank catalog entries against a free-text product mention."""
        raw = (query or "").strip()
        if not raw:
            return []

        # A bare SKU is unambiguous, so short-circuit before any fuzzy work.
        exact = self.get(raw)
        if exact is not None:
            return [Match(exact, 1.0, "sku")]

        needle = raw.casefold()
        query_tokens = set(tokenize(raw))
        matches: list[Match] = []

        for entry in self._entries:
            score, reason = self._score(entry, needle, query_tokens)
            if score >= MIN_SCORE:
                matches.append(Match(entry, round(score, 3), reason))

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def resolve(self, query: str) -> tuple[CatalogEntry | None, list[Match]]:
        """Return a confidently-matched entry, plus the ranked alternatives.

        The entry is only returned when the best match is strong *and* clearly
        ahead of the runner-up. Otherwise the caller gets candidates and should
        ask which one was meant.
        """
        matches = self.search(query)
        if not matches:
            return None, []
        best = matches[0]
        runner_up = matches[1].score if len(matches) > 1 else 0.0
        if best.is_strong and best.score - runner_up >= 0.1:
            return best.entry, matches
        return None, matches

    # ----------------------------------------------------------------- #
    # Scoring
    # ----------------------------------------------------------------- #

    def _score(self, entry: CatalogEntry, needle: str, query_tokens: set[str]) -> tuple[float, str]:
        """Score one entry against a prepared query. Returns (score, reason)."""
        names = [entry.name, *entry.aliases]

        # A verbatim name or alias is as good as it gets short of a SKU.
        for name in names:
            if needle == name.casefold():
                return 1.0, "exact name"

        # The product name appearing inside a longer sentence is near-certain.
        for name in names:
            folded = name.casefold()
            if len(folded) > 8 and folded in needle:
                return 0.95, "name in query"

        if not query_tokens:
            return 0.0, "no signal"

        entry_tokens = self._tokens[entry.sku]
        # Coverage: how much of what the shopper said this product accounts for.
        matched, coverage = _token_coverage(query_tokens, entry_tokens)

        brand_tokens = set(tokenize(entry.brand or ""))
        brand_hit = bool(brand_tokens and brand_tokens <= query_tokens)

        score = coverage
        reason = "token overlap"
        if brand_hit:
            # Naming the brand is a strong signal, and with one product per brand
            # in this catalog it is very nearly decisive on its own.
            score = max(score, 0.55) + 0.25
            reason = "brand + tokens" if matched - brand_tokens else "brand"

        # Typo tolerance, as a floor rather than a boost: "royl canin" should
        # still land, but never outrank a genuine token match.
        if score < STRONG_SCORE:
            best_ratio = max(
                SequenceMatcher(None, needle, name.casefold()).ratio() for name in names
            )
            if best_ratio >= MIN_NAME_SIMILARITY and best_ratio > score:
                score, reason = best_ratio, "approximate name"

        return min(score, 0.99), reason

    # ----------------------------------------------------------------- #
    # Presentation
    # ----------------------------------------------------------------- #

    def overview(self) -> list[dict[str, object]]:
        """Compact rows describing the catalog, for the UI and for the model."""
        return [
            {
                "sku": entry.sku,
                "name": entry.name,
                "brand": pretty_brand(entry.brand),
                "category": entry.category,
                "pet": entry.pet,
                "price_aud": entry.price,
                "average_rating": entry.average_rating,
                "review_count": entry.review_count,
                "url": entry.url,
            }
            for entry in self._entries
        ]


def load_catalog(path: Path | None = None) -> Catalog:
    """Read the catalog from disk."""
    path = path or config.CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"catalog not found at {path} -- run 'python scripts/build_snapshot.py' to build it"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [CatalogEntry.from_dict(row) for row in payload.get("products", [])]
    return Catalog(entries, generated_at=payload.get("generated_at"))


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Return the process-wide catalog, loading it on first use."""
    return load_catalog()
