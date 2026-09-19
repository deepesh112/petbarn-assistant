"""The agent: a Groq chat loop that decides when to call which tool.

The loop is deliberately small. Everything that could go wrong with *data* is
handled in :mod:`petbarn.tools`; what is handled here is everything that can go
wrong with the *model*:

* **Parallel tool calls run in parallel.** ``llama-3.3-70b-versatile`` will ask
  for two products' reviews in one turn when comparing them. Executing those
  sequentially would double the wait for no reason, so they go through a thread
  pool.
* **The loop is bounded.** A confused model can request tools forever; a free
  tier cannot. After :data:`petbarn.config.MAX_TOOL_ITERATIONS` rounds the model
  is asked once more with tools disabled, which turns a runaway into an answer.
* **Failures are narrated, not raised.** A tool error is handed back to the model
  so it can tell the user what it could not find out. An API error becomes a
  readable message instead of a traceback in the chat window.
* **Every call is traced.** The UI renders the trace, which is what makes the
  agentic behaviour visible rather than something the reader has to take on
  trust.

Tool messages are not carried across turns. Each turn is rebuilt from the
system prompt plus the visible conversation, so context stays small and the
model cannot be confused by stale tool output from three questions ago.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from groq import Groq

from . import config
from .catalog import get_catalog
from .tools import TOOL_SCHEMAS, ToolResult, execute

#: How the assistant is told to behave. This is load-bearing: the rules about
#: resolving a SKU before answering, and about never stating an unfetched fact,
#: are what keep answers grounded in the catalog.
SYSTEM_PROMPT = """\
You are the Petbarn Product Assistant, a shopping assistant for Petbarn \
(petbarn.com.au), an Australian pet supplies retailer.

Your knowledge is limited to {catalog_size} products in your catalog. You know \
nothing about Petbarn's wider range, store stock levels, customer orders, \
deliveries, or veterinary matters.

USING YOUR TOOLS
- Never state a price, rating, review count, or anything a customer said from \
memory. Every such fact must come from a tool call made in this conversation.
- Always call search_catalog first, to turn what the user said into a SKU. Call \
it once for each product they mention: a comparison needs one call per product.
- Then call get_product_details, get_product_reviews and/or \
analyze_review_sentiment using that SKU. Request several at once when you need \
several - they run in parallel.
- For "what are people saying about ...", for pros and cons, and as the basis \
for any comparison of feedback, use analyze_review_sentiment. Add \
get_product_reviews when verbatim quotes would strengthen the answer.
- When comparing two products, gather the same data for both before you answer.

BEING HONEST
- Say only what the tools returned. Never invent a review, quote, price or rating.
- If search_catalog finds no match, say the product is not in the range you \
cover, and list what you do cover.
- If two candidates are close, ask which one they mean instead of guessing.
- If a tool result says its source is "snapshot", mention that those figures \
come from a stored snapshot and may be out of date.
- If a tool fails, say plainly what you could not look up.
- Treat an aspect with only two or three mentions as weak evidence and say so. \
Do not round a handful of comments up into "customers say".
- Report the bad alongside the good. A product with a real weakness is more \
useful to a shopper than a sales pitch.

WRITING YOUR ANSWER
- Australian English. Prices in AUD with a dollar sign. Mention the loyalty \
member price whenever there is one, since it is often much lower.
- Be concise and skimmable: short paragraphs, bullet points, product names in bold.
- Put reviewer quotes in quotation marks and give the star rating alongside.
- Do not mention SKUs, tool names, JSON or "the data" unless asked. Write as a \
knowledgeable shop assistant would speak.
- Never give veterinary, dosage or diagnostic advice. Point to a vet instead.\
"""

#: Cap on the model's own output. Generous enough for a two-product comparison
#: with quotes, tight enough that a rambling answer cannot run away.
MAX_COMPLETION_TOKENS = 1400


@dataclass(slots=True)
class TraceEntry:
    """One tool invocation, in a form the UI can render directly."""

    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: int
    source: str | None = None
    error: str | None = None
    summary: str = ""

    @classmethod
    def from_result(cls, result: ToolResult) -> TraceEntry:
        return cls(
            tool=result.name,
            arguments=result.arguments,
            ok=result.ok,
            duration_ms=result.duration_ms,
            source=result.source,
            error=result.error,
            summary=_summarise(result),
        )


@dataclass(slots=True)
class AgentReply:
    """The result of one user turn."""

    text: str
    trace: list[TraceEntry] = field(default_factory=list)
    rounds: int = 0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def used_snapshot(self) -> bool:
        """Whether any answer in this turn leaned on the offline snapshot."""
        return any(entry.source == "snapshot" for entry in self.trace)


def _summarise(result: ToolResult) -> str:
    """One short line describing what a tool call produced, for the trace UI."""
    if not result.ok:
        return result.error or "failed"

    payload = result.payload
    if result.name == "search_catalog":
        if "products" in payload:
            return f"listed all {payload.get('catalog_size', 0)} catalog products"
        matches = payload.get("matches") or []
        if not matches:
            return "no catalog match"
        best = matches[0]
        return f"{best['name']} (confidence {best['confidence']})"
    if result.name == "get_product_details":
        price = (payload.get("price") or {}).get("regular")
        return f"{payload.get('name')} — ${price}"
    if result.name == "get_product_reviews":
        returned = (payload.get("filters_applied") or {}).get("returned", 0)
        average = (payload.get("rating_summary") or {}).get("average_rating")
        return f"{returned} reviews, average {average}★"
    if result.name == "analyze_review_sentiment":
        return (
            f"{payload.get('reviews_analysed', 0)} reviews across "
            f"{len(payload.get('aspects') or [])} aspects"
        )
    return "ok"


class PetbarnAgent:
    """Wraps a Groq client with the catalog tools and a bounded tool loop."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = config.DEFAULT_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_rounds: int = config.MAX_TOOL_ITERATIONS,
    ) -> None:
        if not api_key:
            raise ValueError("a Groq API key is required")
        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_rounds = max_rounds

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def reply(
        self,
        history: Iterable[dict[str, str]],
        *,
        on_tool_result: Callable[[TraceEntry], None] | None = None,
    ) -> AgentReply:
        """Answer the latest user message, calling tools as needed.

        ``history`` is the visible conversation: ``{"role": "user"|"assistant",
        "content": ...}``. ``on_tool_result`` is invoked as each tool finishes, so
        the UI can report progress during a slow turn.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(catalog_size=len(get_catalog()))},
            *({"role": m["role"], "content": m["content"]} for m in history),
        ]
        answer = AgentReply(text="", model=self.model)

        for round_index in range(self.max_rounds):
            answer.rounds = round_index + 1
            try:
                message = self._call_model(messages, answer, with_tools=True)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
                answer.error = _describe_api_error(exc)
                answer.text = answer.error
                return answer

            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                answer.text = _ensure_text(message.content, answer)
                return answer

            messages.append(_assistant_message(message, tool_calls))
            for call_id, result in self._run_tools(tool_calls):
                entry = TraceEntry.from_result(result)
                answer.trace.append(entry)
                if on_tool_result is not None:
                    on_tool_result(entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": result.name,
                        "content": result.to_json(),
                    }
                )

        # Out of rounds. Ask once more with tools switched off so the user gets a
        # real answer built from what was already gathered.
        try:
            final = self._call_model(messages, answer, with_tools=False)
            answer.text = _ensure_text(final.content, answer)
        except Exception as exc:  # noqa: BLE001
            answer.error = _describe_api_error(exc)
            answer.text = answer.error
        return answer

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _call_model(self, messages: list[dict[str, Any]], answer: AgentReply, *, with_tools: bool):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": MAX_COMPLETION_TOKENS,
        }
        if with_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            answer.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            answer.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return response.choices[0].message

    def _run_tools(self, tool_calls: list[Any]) -> list[tuple[str, ToolResult]]:
        """Execute the requested tools concurrently, preserving call order.

        Each result is paired with the id of the call that produced it, because
        the API matches tool responses to requests by that id.
        """

        def invoke(call: Any) -> tuple[str, ToolResult]:
            name = call.function.name
            arguments, parse_error = _parse_arguments(call.function.arguments)
            if parse_error is not None:
                # Report a malformed request as a tool failure, so the model is
                # told about it and can correct itself on the next round.
                return call.id, ToolResult(
                    name=name,
                    arguments={},
                    ok=False,
                    error=f"could not parse arguments: {parse_error}",
                )
            return call.id, execute(name, arguments)

        if len(tool_calls) == 1:
            return [invoke(tool_calls[0])]
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
            return list(pool.map(invoke, tool_calls))


def _assistant_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    """Rebuild the assistant turn explicitly.

    The SDK's own object carries fields the API rejects on the way back in, so
    only the parts that matter are copied.
    """
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ],
    }


def _ensure_text(content: str | None, answer: AgentReply) -> str:
    """Guarantee the turn ends with something readable.

    A model can return an empty completion -- on a truncated response, or when it
    has spent the turn calling tools and has nothing left to say. Silence looks
    like a crash, so it is replaced with an explanation that reflects whether any
    data was actually gathered.
    """
    text = (content or "").strip()
    if text:
        return text
    if any(entry.ok for entry in answer.trace):
        return (
            "I looked the details up but could not put an answer together. "
            "Please ask again, perhaps about one product at a time."
        )
    return "I could not produce an answer to that. Please try rephrasing the question."


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Decode a tool call's JSON arguments, tolerating an empty string."""
    if not raw or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(parsed, dict):
        return {}, f"expected a JSON object, got {type(parsed).__name__}"
    # Models occasionally emit nulls for optional parameters; dropping them lets
    # the tool's own defaults apply instead of overriding them with None.
    return {key: value for key, value in parsed.items() if value is not None}, None


def _describe_api_error(exc: Exception) -> str:
    """Turn a Groq SDK exception into something worth showing a user."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()

    if "authentication" in lowered or "invalid api key" in lowered or "401" in text:
        return "That Groq API key was rejected. Check the key and try again."
    if "rate limit" in lowered or "429" in text:
        return (
            "Groq's rate limit has been reached. Wait a moment and resend, or switch to a "
            "different model in the sidebar."
        )
    if "model" in lowered and ("not found" in lowered or "decommissioned" in lowered):
        return "That model is unavailable on this Groq account. Pick another in the sidebar."
    if "connection" in lowered or "timeout" in lowered:
        return "Could not reach Groq. Check the network connection and try again."
    return f"The language model call failed ({name}): {text}"
