"""Test the agent's tool loop against a stubbed model. No API key needed.

    python scripts/loop_test.py

The loop has behaviours that are hard to observe through a real model, because
you cannot make a real model reliably produce them on demand: two tool calls in
one round, malformed tool arguments, an unbounded request for more tools, an
authentication failure. Each is scripted here and asserted.

The tools themselves are real -- only the model is replaced -- so this also
confirms that tool results are packaged into `tool` messages the API would
accept, with the call ids matched up correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petbarn import config  # noqa: E402
from petbarn.agent import PetbarnAgent  # noqa: E402
from petbarn.catalog import get_catalog  # noqa: E402

failures: list[str] = []
checks = 0


def check(condition: bool, description: str) -> None:
    global checks
    checks += 1
    if condition:
        print(f"    ok    {description}")
    else:
        failures.append(description)
        print(f"    FAIL  {description}")


# --------------------------------------------------------------------------- #
# A stub standing in for groq.Groq
# --------------------------------------------------------------------------- #


def tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class ScriptedClient:
    """Replays a scripted sequence of model responses and records requests."""

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        step = self._script.pop(0) if self._script else SimpleNamespace(content="fallback", tool_calls=None)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(
            choices=[SimpleNamespace(message=step)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25),
        )

    @property
    def tool_messages(self) -> list[dict]:
        """Every `tool` message the loop sent back, across all requests."""
        if not self.requests:
            return []
        return [m for m in self.requests[-1]["messages"] if m.get("role") == "tool"]


def agent_with(script: list[object], **kwargs) -> tuple[PetbarnAgent, ScriptedClient]:
    agent = PetbarnAgent("gsk_stub_key", **kwargs)
    client = ScriptedClient(script)
    agent._client = client  # noqa: SLF001 - substituting the model is the point
    return agent, client


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_parallel_tool_calls() -> None:
    section("Two tool calls in one round run in parallel and both come back")
    skus = get_catalog().skus[:2]
    agent, client = agent_with(
        [
            SimpleNamespace(
                content="",
                tool_calls=[
                    tool_call("call_a", "get_product_details", f'{{"sku": "{skus[0]}"}}'),
                    tool_call("call_b", "get_product_details", f'{{"sku": "{skus[1]}"}}'),
                ],
            ),
            SimpleNamespace(content="Here is the comparison.", tool_calls=None),
        ]
    )
    reply = agent.reply([{"role": "user", "content": "compare these two"}])

    check(reply.text == "Here is the comparison.", "final answer is returned")
    check(len(reply.trace) == 2, f"both calls are traced (got {len(reply.trace)})")
    check(reply.rounds == 2, f"took 2 rounds (got {reply.rounds})")
    check(all(entry.ok for entry in reply.trace), "both tool calls succeeded")
    check(reply.total_tokens == 250, f"token usage accumulates across rounds ({reply.total_tokens})")

    tool_messages = client.tool_messages
    check(len(tool_messages) == 2, "two tool messages were sent back to the model")
    check(
        [m["tool_call_id"] for m in tool_messages] == ["call_a", "call_b"],
        "tool results keep their call ids, in request order",
    )
    check(
        all(m.get("name") and m.get("content") for m in tool_messages),
        "every tool message carries a name and a content payload",
    )
    # The assistant turn that requested the tools must be replayed verbatim.
    assistant = [m for m in client.requests[-1]["messages"] if m.get("role") == "assistant"]
    check(len(assistant) == 1 and len(assistant[0]["tool_calls"]) == 2,
          "the assistant's tool-call turn is echoed back with both calls")


def test_malformed_arguments() -> None:
    section("Malformed tool arguments are reported to the model, not raised")
    agent, client = agent_with(
        [
            SimpleNamespace(
                content="",
                tool_calls=[tool_call("call_x", "get_product_details", "{not valid json")],
            ),
            SimpleNamespace(content="I could not look that up.", tool_calls=None),
        ]
    )
    reply = agent.reply([{"role": "user", "content": "tell me about something"}])

    check(len(reply.trace) == 1, "the bad call is still traced")
    check(not reply.trace[0].ok, "the bad call is marked as failed")
    check("parse" in (reply.trace[0].error or "").lower(), "the failure names the parse problem")
    check("error" in client.tool_messages[0]["content"], "the model is told about the failure")
    check(reply.text == "I could not look that up.", "the turn still produces an answer")


def test_unknown_tool() -> None:
    section("A hallucinated tool name fails cleanly")
    agent, client = agent_with(
        [
            SimpleNamespace(
                content="", tool_calls=[tool_call("call_y", "check_stock_levels", "{}")]
            ),
            SimpleNamespace(content="I cannot check stock.", tool_calls=None),
        ]
    )
    reply = agent.reply([{"role": "user", "content": "is it in stock"}])
    check(not reply.trace[0].ok, "the unknown tool is marked as failed")
    check("unknown tool" in (reply.trace[0].error or ""), "the error says the tool is unknown")
    check(reply.text == "I cannot check stock.", "the turn still produces an answer")


def test_round_limit() -> None:
    section("A model that never stops calling tools is cut off and forced to answer")
    sku = get_catalog().skus[0]
    forever = [
        SimpleNamespace(
            content="",
            tool_calls=[tool_call(f"call_{i}", "get_product_details", f'{{"sku": "{sku}"}}')],
        )
        for i in range(10)
    ]
    agent, client = agent_with(
        forever[:3] + [SimpleNamespace(content="Forced answer.", tool_calls=None)], max_rounds=3
    )
    reply = agent.reply([{"role": "user", "content": "loop please"}])

    check(reply.rounds == 3, f"stopped at the round limit (got {reply.rounds})")
    check(len(reply.trace) == 3, f"traced exactly 3 calls (got {len(reply.trace)})")
    check(reply.text == "Forced answer.", "a final answer was still produced")
    final_request = client.requests[-1]
    check("tools" not in final_request, "the final request disables tools so it cannot loop again")


def test_never_answers_with_silence() -> None:
    section("An empty completion is replaced with an explanation, not silence")
    sku = get_catalog().skus[0]

    agent, _ = agent_with([SimpleNamespace(content="", tool_calls=None)])
    reply = agent.reply([{"role": "user", "content": "hello"}])
    check(bool(reply.text.strip()), "an empty first response still yields text")
    check("rephrasing" in reply.text, "with no data gathered, it suggests rephrasing")

    agent, _ = agent_with(
        [
            SimpleNamespace(
                content="",
                tool_calls=[tool_call("call_1", "get_product_details", f'{{"sku": "{sku}"}}')],
            ),
            SimpleNamespace(content="   ", tool_calls=None),
        ]
    )
    reply = agent.reply([{"role": "user", "content": "tell me about it"}])
    check(bool(reply.text.strip()), "an empty response after a good tool call still yields text")
    check("looked the details up" in reply.text, "it acknowledges the lookup succeeded")


def test_api_error() -> None:
    section("An API failure becomes a readable message, not a traceback")
    for error, expected in (
        (Exception("Error code: 401 - invalid api key provided"), "rejected"),
        (Exception("Error code: 429 - rate limit reached for model"), "rate limit"),
        (Exception("connection error while contacting host"), "reach Groq"),
    ):
        agent, _ = agent_with([error])
        reply = agent.reply([{"role": "user", "content": "hello"}])
        check(reply.error is not None, f"{str(error)[:34]!r} -> reported as an error")
        check(expected.lower() in reply.text.lower(), f"{str(error)[:34]!r} -> message mentions {expected!r}")
        check(reply.text == reply.error, "the user-facing text is the explanation, not a stack trace")


def test_system_prompt() -> None:
    section("The system prompt reaches the model with the catalog size filled in")
    agent, client = agent_with([SimpleNamespace(content="hi", tool_calls=None)])
    agent.reply([{"role": "user", "content": "hello"}])

    messages = client.requests[0]["messages"]
    check(messages[0]["role"] == "system", "the first message is the system prompt")
    check(str(len(get_catalog())) in messages[0]["content"], "the catalog size is interpolated")
    check("{catalog_size}" not in messages[0]["content"], "no placeholder was left unfilled")
    check(messages[1] == {"role": "user", "content": "hello"}, "the user turn follows it")
    check(client.requests[0]["tools"] and len(client.requests[0]["tools"]) == 4,
          "all four tools are offered")
    check(client.requests[0]["model"] == config.DEFAULT_MODEL, "the configured model is requested")


def main() -> int:
    test_parallel_tool_calls()
    test_malformed_arguments()
    test_unknown_tool()
    test_round_limit()
    test_never_answers_with_silence()
    test_api_error()
    test_system_prompt()

    section("Result")
    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
