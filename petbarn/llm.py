"""Language-model providers: a local Ollama server, or four hosted APIs.

Every backend is presented to :mod:`petbarn.agent` as a chat-completions
client, so the agent loop never learns which one it is talking to. Ollama, Groq,
OpenAI and Gemini all speak that dialect natively and need only a base URL.

``ollama``
    Runs entirely on your machine. No API key, no per-request cost, no data
    leaving the computer, and it works with the network unplugged. The catch is
    that a local model cannot be reached from a hosted deployment, and small
    models are noticeably weaker at choosing tools than a large hosted one.

``groq`` / ``gemini``
    Free tiers, reliable tool calling, reachable from a deployed app.

``openai`` / ``anthropic``
    Paid, and the strongest at tool use.

**Claude is the exception, deliberately.** Anthropic publishes an
OpenAI-compatible endpoint, but it is a compatibility shim with reduced feature
support and Anthropic's own guidance is to use the real SDK. So Claude goes
through the official ``anthropic`` client, and the cost of that -- translating
between two message formats -- is paid once in
:class:`_AnthropicMessagesAdapter` rather than leaking into the agent loop.

Model lists for Ollama are **discovered from the running server** rather than
hard-coded, because what is installed is a property of the machine and any
baked-in list would start rotting immediately.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace
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
    "openai": Provider(
        name="openai",
        label="OpenAI (hosted)",
        base_url="https://api.openai.com/v1",
        requires_api_key=True,
        default_model="gpt-5.6-terra",
        suggested_models=(
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-6-astra",
        ),
        key_url="https://platform.openai.com/api-keys",
        notes="Paid only, no free tier. Very reliable at tool use. 'luna' is the cheapest.",
    ),
    "anthropic": Provider(
        name="anthropic",
        label="Anthropic Claude (hosted)",
        # Unused: Claude goes through the official anthropic SDK, not the
        # OpenAI-compatible shim. See _AnthropicMessagesAdapter.
        base_url="",
        requires_api_key=True,
        default_model="claude-opus-5",
        suggested_models=(
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ),
        key_url="https://console.anthropic.com/settings/keys",
        notes="Paid only. Strongest tool use of the five; uses the official Anthropic SDK.",
    ),
    "gemini": Provider(
        name="gemini",
        label="Google Gemini (hosted)",
        # Gemini speaks OpenAI's chat-completions dialect at this path, which is
        # why it needs no client of its own. Google describes the compatibility
        # layer as beta, so it is the likeliest of the three to shift.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        requires_api_key=True,
        default_model="gemini-3.8-flash",
        suggested_models=(
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-2.5-flash",
            "gemini-3.5-flash-lite",
        ),
        key_url="https://aistudio.google.com/apikey",
        notes="Large free tier and strong tool use. Also reachable from a deployed app.",
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
    """Create a client for ``provider``, exposing the chat-completions surface."""
    key = (api_key or "").strip() or provider.placeholder_key
    if provider.requires_api_key and not (api_key or "").strip():
        raise ValueError(f"{provider.label} requires an API key")

    if provider.name == "anthropic":
        return _build_anthropic_client(key)

    from openai import OpenAI  # imported lazily so tests can stub the client

    return OpenAI(
        api_key=key,
        base_url=base_url or provider.base_url,
        timeout=config.LLM_TIMEOUT,
        max_retries=1,
    )


def _build_anthropic_client(api_key: str):
    """Wrap the official Anthropic SDK in the chat-completions shape.

    Anthropic does publish an OpenAI-compatible endpoint, but it is explicitly a
    compatibility shim with reduced feature support, and Anthropic's own guidance
    is to use the real SDK. So Claude gets the official client, and the cost of
    that decision -- translating between two message formats -- is paid once,
    here, rather than leaking into the agent loop.
    """
    import anthropic  # imported lazily: only this provider needs it

    return _AnthropicMessagesAdapter(
        anthropic.Anthropic(api_key=api_key, timeout=config.LLM_TIMEOUT, max_retries=1)
    )


class _AnthropicMessagesAdapter:
    """Speaks ``chat.completions.create`` on the outside, Messages API within.

    The two formats differ in three ways that matter here:

    * The system prompt is a top-level argument, not a message with a role.
    * A tool call is a ``tool_use`` content block on the assistant turn, and its
      result is a ``tool_result`` block inside a **user** turn -- and all results
      from one assistant turn must arrive in a single user message, or the model
      learns to stop making parallel calls.
    * ``temperature`` is rejected outright by the current Claude models, so it is
      dropped rather than forwarded.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,  # noqa: ARG002 - deliberately unused
        **_ignored: Any,
    ) -> Any:
        system, converted = _to_anthropic_messages(messages)

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": converted,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [_to_anthropic_tool(tool) for tool in tools]
            if tool_choice == "auto":
                request["tool_choice"] = {"type": "auto"}

        return _from_anthropic_response(self._client.messages.create(**request))


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI tool schema into Anthropic's shape."""
    function = tool.get("function", tool)
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split out the system prompt and convert the rest to Messages format."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            system_parts.append(str(message.get("content") or ""))

        elif role == "tool":
            # Results belong in a user turn, and every result from the same
            # assistant turn has to share one message.
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": str(message.get("content") or ""),
            }
            if converted and converted[-1]["role"] == "user" and isinstance(
                converted[-1]["content"], list
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})

        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if text := str(message.get("content") or "").strip():
                content.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                function = call["function"]
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                )
            # An assistant turn with neither text nor tool calls is not sendable.
            if content:
                converted.append({"role": "assistant", "content": content})

        else:
            converted.append({"role": "user", "content": str(message.get("content") or "")})

    return "\n\n".join(part for part in system_parts if part), converted


def _from_anthropic_response(response: Any) -> Any:
    """Reshape a Messages response into what the agent loop reads."""
    text_parts: list[str] = []
    tool_calls: list[SimpleNamespace] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name, arguments=json.dumps(block.input)
                    ),
                )
            )

    message = SimpleNamespace(
        content="".join(text_parts),
        tool_calls=tool_calls or None,
    )
    usage = getattr(response, "usage", None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        ),
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
        # Each provider words the same failure differently: Groq says "invalid
        # api key", Gemini says "API key not valid" and "RESOURCE_EXHAUSTED".
        # Matching on all the phrasings keeps one readable message per cause.
        if (
            "authentication" in lowered
            or "invalid api key" in lowered
            or "api key not valid" in lowered
            or "api_key_invalid" in lowered
            or "invalid x-api-key" in lowered
            or "401" in text
        ):
            return f"That {provider.label} API key was rejected. Check the key and try again."
        if (
            "rate limit" in lowered
            or "quota" in lowered
            or "resource_exhausted" in lowered
            or "credit balance" in lowered
            or "429" in text
        ):
            return (
                f"{provider.label} has hit a rate or quota limit. Wait a moment and resend, "
                "switch model in the sidebar, or use a different backend."
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
