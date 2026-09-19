"""Central configuration for the Petbarn assistant.

Every endpoint, credential lookup and tuning knob lives here so that the rest of
the package contains no magic strings. Anything that might plausibly need to
change in a deployed environment is overridable via an environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
SNAPSHOT_DIR = DATA_DIR / "snapshot"
RAW_DIR = DATA_DIR / "raw"

#: Runtime HTTP cache. Ephemeral and gitignored -- safe to delete at any time.
CACHE_DIR = Path(os.environ.get("PETBARN_CACHE_DIR", PROJECT_ROOT / ".cache"))

# --------------------------------------------------------------------------- #
# Petbarn storefront
# --------------------------------------------------------------------------- #

SITE_ROOT = "https://www.petbarn.com.au"
SITEMAP_INDEX_URL = f"{SITE_ROOT}/media/sitemap/au/sitemap.xml"

#: Product detail pages live at ``/p/<slug>`` (and ``/p/<slug>/<sku>`` for a
#: specific variant of a multi-size product). robots.txt permits both.
PRODUCT_PATH_PREFIX = "/p/"

# --------------------------------------------------------------------------- #
# Bazaarvoice (Petbarn's review platform)
# --------------------------------------------------------------------------- #

BV_CLIENT = "petbarn-au"
BV_SITE = "main_site"
BV_LOCALE = "en_AU"

BV_API_ROOT = "https://api.bazaarvoice.com/data"
BV_API_VERSION = "5.4"

#: The storefront's own browser-side Bazaarvoice bundle. It embeds the public
#: *display* passkey, which we re-discover at runtime so a key rotation on
#: Petbarn's side does not break this app.
BV_LOADER_JS_URL = (
    f"https://display.ugc.bazaarvoice.com/static/{BV_CLIENT}/{BV_SITE}/{BV_LOCALE}/bvapi.js"
)

#: Last-known-good display passkey, used if discovery fails. Not a secret: it is
#: served to every visitor's browser and grants read-only access to the same
#: reviews the product page renders.
BV_FALLBACK_PASSKEY = "caYoBHPBvrSBq8gDNiuznQOZJk5sBqoCKyjxQrCYcYeGo"

#: Bazaarvoice rejects Limit values above this.
BV_MAX_PAGE_LIMIT = 100

#: Secondary ("aspect") ratings Petbarn collects alongside the overall star
#: rating. Mapped to friendlier labels for display.
BV_SECONDARY_RATING_LABELS = {
    "Quality": "Quality",
    "Value": "Value for money",
    "Petsatisfaction": "Pet satisfaction",
}


def bv_passkey_override() -> str | None:
    """Return an operator-supplied Bazaarvoice passkey, if one is configured."""
    return _clean_env("BV_PASSKEY")


# --------------------------------------------------------------------------- #
# HTTP behaviour
# --------------------------------------------------------------------------- #

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = float(os.environ.get("PETBARN_TIMEOUT", "15"))
MAX_RETRIES = int(os.environ.get("PETBARN_MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = 0.8
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

#: Minimum seconds between requests to the same host. Keeps us to roughly one
#: request per second per host, which is deliberately gentle.
MIN_REQUEST_INTERVAL = float(os.environ.get("PETBARN_MIN_INTERVAL", "1.0"))

#: How long a cached HTTP response stays fresh. Product prices and review counts
#: move slowly, so six hours keeps the app responsive without going stale.
CACHE_TTL_SECONDS = int(os.environ.get("PETBARN_CACHE_TTL", str(6 * 60 * 60)))


# --------------------------------------------------------------------------- #
# Data sourcing policy
# --------------------------------------------------------------------------- #

def live_fetch_enabled() -> bool:
    """Whether tools may reach out to the network.

    Set ``PETBARN_LIVE=0`` to pin the app to its bundled snapshot -- useful for
    offline demos, and for avoiding surprises during a live walkthrough.
    """
    return _env_flag("PETBARN_LIVE", default=True)


# --------------------------------------------------------------------------- #
# Language model
# --------------------------------------------------------------------------- #

#: Where the local Ollama server listens. The ``/v1`` suffix is its
#: OpenAI-compatible endpoint, which is what this app speaks.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

#: Which backend to use when nothing says otherwise. Ollama is the default so a
#: fresh clone runs fully offline with no account, no key and no cost.
_DEFAULT_PROVIDER = "ollama"


def default_provider() -> str:
    """The model backend to start with: ``ollama`` or ``groq``.

    A deployed app sets ``PETBARN_PROVIDER=groq``, because a local Ollama server
    is not reachable from a hosted container.
    """
    return (_clean_env("PETBARN_PROVIDER") or _DEFAULT_PROVIDER).lower()


#: Ceiling on agent tool-call rounds per user turn, so a confused model cannot
#: loop indefinitely (and cannot burn through a free-tier quota).
MAX_TOOL_ITERATIONS = 5

#: Caps on how much review text is handed back to the model. Tool payloads are
#: trimmed rather than truncated mid-thought, keeping turns cheap and focused.
#: Providers narrow this further -- a local 8B model has far less context to
#: spend than a hosted 70B.
MAX_REVIEWS_TO_MODEL = 25
MAX_REVIEW_CHARS = 420

LLM_TEMPERATURE = float(os.environ.get("PETBARN_TEMPERATURE", "0.2"))

#: Seconds to wait on a model response. Generous, because a local model on a
#: laptop GPU is far slower than a hosted one and a timeout mid-answer is worse
#: than a wait.
LLM_TIMEOUT = float(os.environ.get("PETBARN_LLM_TIMEOUT", "180"))


def api_key_for(provider: str) -> str | None:
    """Read a provider's API key from the environment.

    Streamlit secrets are layered on top of this in the UI, so a deployed app
    works from platform secrets while a visitor can still supply their own key.
    """
    return _clean_env(f"{provider.upper()}_API_KEY")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _clean_env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return default
