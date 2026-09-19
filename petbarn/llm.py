"""Language-model providers: a local Ollama server, or Groq's hosted API.

Both are reached through the **OpenAI-compatible** chat-completions interface,
which Ollama and Groq each expose. That means one client type, one request
shape and one error taxonomy for both, and the agent loop in :mod:`petbarn.agent`
never learns which provider it is talking to.

The two exist for different jobs:

``ollama``
    Runs entirely on your machine. No API key, no per-request cost, no data
    leaving the computer, and it works with the network unplugged. The catch is
    that a local model cannot be reached from a hosted deployment, and small
    models are noticeably weaker at choosing tools than a 70B hosted one.

``groq``
    Needs a free API key and an internet connection, but is fast, reliable at
    tool calling, and reachable from a deployed app.

Model lists for Ollama are **discovered from the running server** rather than
hard-coded, because what is installed is a property of the machine and any
baked-in list would start rotting immediately.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass(frozen=True, slots=True)
class Provider:
    """Everything that differs between one model backend and another."""

    name: str
    label: str
    base_url: str
    requires_api_key: bool
    #: Used when the provider needs no real credential but the SDK insists on one.
    placeholder_key: str = "not-needed"
    default_model: str = ""
    #: Shown in the UI when no model list can be discovered.
    suggested_models: tuple[str, ...] = ()
    key_url: str = ""
    #: Reviews handed to the model in one tool payload. Local models run with
    #: far less context than a hosted 70B, so they get a smaller helping.
    max_reviews_to_model: int = config.MAX_REVIEWS_TO_MODEL
    #: Context window to request from Ollama. Ignored by hosted providers.
    context_tokens: int | None = None
    notes: str = ""

    @property
    def is_local(self) -> bool:
        return self.name == "ollama"


#: Suggested Ollama models that support tool calling and fit a laptop GPU.
#: Only a starting point -- the UI lists whatever is actually installed.
OLLAMA_SUGGESTED = (
    "granite4.1:8b",
    "granite4.1:3b",
    "qwen3:8b",
    "llama3.1:8b",
)

PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        name="ollama",
        label="Ollama (local, offline)",
        base_url=config.OLLAMA_BASE_URL,
        requires_api_key=False,
        placeholder_key="ollama",
        default_model=OLLAMA_SUGGESTED[0],
        suggested_models=OLLAMA_SUGGESTED,
        max_reviews_to_model=8,
        context_tokens=16384,
        notes="Runs on this machine. Needs Ollama installed and a tool-capable model pulled.",
    ),
    "groq": Provider(
        name="groq",
        label="Groq (hosted)",
        base_url="https://api.groq.com/openai/v1",
        requires_api_key=True,
        default_model="llama-3.3-70b-versatile",
        suggested_models=(
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
        ),
        key_url="https://console.groq.com/keys",
        notes="Fast and reliable at tool use, and reachable from a deployed app.",
    ),
}

DEFAULT_PROVIDER = config.default_provider()


def get_provider(name: str | None) -> Provider:
    """Look up a provider, falling back to the configured default."""
    return PROVIDERS.get((name or "").strip().lower()) or PROVIDERS[DEFAULT_PROVIDER]


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #


def build_client(provider: Provider, *, api_key: str | None = None, base_url: str | None = None):
    """Create an OpenAI-compatible client pointed at ``provider``."""
    from openai import OpenAI  # imported lazily so tests can stub the client

    key = (api_key or "").strip() or provider.placeholder_key
    if provider.requires_api_key and not (api_key or "").strip():
        raise ValueError(f"{provider.label} requires an API key")

    return OpenAI(
        api_key=key,
        base_url=base_url or provider.base_url,
        timeout=config.LLM_TIMEOUT,
        max_retries=1,
    )


def extra_request_options(provider: Provider) -> dict[str, Any]:
    """Provider-specific request additions.

    Ollama defaults to a small context window, which silently truncates the
    conversation -- and a truncated tool result is worse than none, because the
    model answers from half a payload without knowing anything is missing. The
    context size is requested explicitly; hosted providers ignore the field.
    """
    if provider.is_local and provider.context_tokens:
        return {"extra_body": {"options": {"num_ctx": provider.context_tokens}}}
    return {}


# --------------------------------------------------------------------------- #
# Health and discovery
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Health:
    """Whether a provider is usable right now, and why not if it isn't."""

    ok: bool
    detail: str
    models: list[str] = field(default_factory=list)


def _ollama_root(base_url: str) -> str:
    """Strip the OpenAI-compat suffix to reach Ollama's native API."""
    return base_url.rstrip("/").removesuffix("/v1")


def list_local_models(base_url: str | None = None, *, timeout: float = 3.0) -> list[str]:
    """Return the model tags installed on the local Ollama server.

    Uses ``urllib`` rather than ``requests`` so this stays usable even if the
    shared session is misconfigured, and so a dead server fails in milliseconds
    instead of going through the retry policy built for scraping.
    """
    url = f"{_ollama_root(base_url or config.OLLAMA_BASE_URL)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed localhost URL
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return []

    models = payload.get("models")
    if not isinstance(models, list):
        return []
    return sorted(
        str(model.get("name")) for model in models if isinstance(model, dict) and model.get("name")
    )


def check(provider: Provider, *, api_key: str | None = None, base_url: str | None = None) -> Health:
    """Report whether ``provider`` can serve a request right now."""
    if provider.is_local:
        models = list_local_models(base_url or provider.base_url)
        if models:
            return Health(True, f"{len(models)} model(s) installed", models)
        return Health(
            False,
            "No Ollama server reachable at "
            f"{_ollama_root(base_url or provider.base_url)}. Start it with 'ollama serve'.",
        )

    if provider.requires_api_key and not (api_key or "").strip():
        return Health(False, f"{provider.label} needs an API key")
    return Health(True, "ready", list(provider.suggested_models))


def describe_error(exc: Exception, provider: Provider) -> str:
    """Turn an SDK exception into something worth showing a user."""
    text = str(exc)
    lowered = text.lower()

    if provider.is_local:
        if "connection" in lowered or "connect" in lowered or "refused" in lowered:
            return (
                f"Could not reach Ollama at {_ollama_root(provider.base_url)}. "
                "Start it with `ollama serve`, then try again."
            )
        if "not found" in lowered or "404" in text:
            return (
                f"Ollama does not have that model. Pull it first: "
                f"`ollama pull {provider.default_model}`."
            )
        if "does not support tools" in lowered or "tools" in lowered and "support" in lowered:
            return (
                "That local model cannot call tools, so it cannot look anything up. "
                f"Try a tool-capable one, e.g. `ollama pull {provider.default_model}`."
            )
    else:
        if "authentication" in lowered or "invalid api key" in lowered or "401" in text:
            return "That API key was rejected. Check the key and try again."
        if "rate limit" in lowered or "429" in text:
            return (
                "The provider's rate limit has been reached. Wait a moment and resend, "
                "or switch model in the sidebar."
            )

    if "model" in lowered and ("not found" in lowered or "decommissioned" in lowered):
        return "That model is unavailable. Pick another in the sidebar."
    if "connection" in lowered or "timeout" in lowered or "timed out" in lowered:
        return f"Could not reach {provider.label}. Check the connection and try again."
    if "context" in lowered and ("length" in lowered or "window" in lowered):
        return (
            "The conversation outgrew the model's context window. Clear the conversation, "
            "or use a model with more context."
        )
    return f"The model call failed ({type(exc).__name__}): {text}"
