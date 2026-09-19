"""Run the brief's sample questions through the real agent, end to end.

    python scripts/agent_test.py                 # all sample questions
    python scripts/agent_test.py "your question" # just this one
    python scripts/agent_test.py --offline       # force the snapshot fallback

Needs a Groq key, taken from ``GROQ_API_KEY`` or from ``.streamlit/secrets.toml``
so the key only ever has to live in one place.

What this checks that :mod:`scripts.smoke_test` cannot: whether the *model*
chooses the right tools. It prints the tool trace for every question, so the
important behaviours are visible -- that a product is resolved before it is
looked up, that a comparison fetches both products, and that an off-catalog
product is refused rather than invented.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from petbarn import config, llm  # noqa: E402
from petbarn.agent import PetbarnAgent  # noqa: E402
from petbarn.catalog import get_catalog  # noqa: E402

SAMPLE_QUESTIONS = [
    # The three question shapes named in the brief.
    "What are people saying about the price and quality of the Breeders Choice cat litter?",
    "Can you compare the reviews between the Black Hawk lamb and rice and the Prime100 kangaroo roll?",
    "List the main pros and cons based on recent customer feedback for the NexGard Spectra.",
    # Plus the awkward cases a demo tends to hit.
    "How much is the Royal Canin indoor cat food, and is it in stock?",
    "What do the one-star reviews of the Bravecto say?",
    "Do you sell Whiskas dry cat food?",
    "What products can you help me with?",
]


def load_api_key(provider: llm.Provider) -> str | None:
    """Read a provider's key from the environment or Streamlit's secrets file.

    Ollama needs no key at all, so this returns ``None`` for it immediately.
    """
    if not provider.requires_api_key:
        return None
    if key := config.api_key_for(provider.name):
        return key

    secrets = config.PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets.exists():
        return None
    # A two-line TOML file does not justify a TOML dependency here.
    name = re.escape(f"{provider.name.upper()}_API_KEY")
    match = re.search(
        rf'^\s*{name}\s*=\s*["\']([^"\']+)["\']', secrets.read_text(encoding="utf-8"), re.M
    )
    return match.group(1) if match else None


def ask(agent: PetbarnAgent, question: str) -> bool:
    print(f"\n{'=' * 78}\nQ: {question}\n{'=' * 78}")
    started = time.perf_counter()
    reply = agent.reply([{"role": "user", "content": question}])
    elapsed = time.perf_counter() - started

    print(f"\n--- {len(reply.trace)} tool call(s) over {reply.rounds} round(s) ---")
    for index, entry in enumerate(reply.trace, start=1):
        arguments = ", ".join(f"{k}={v!r}" for k, v in entry.arguments.items())
        status = "ok" if entry.ok else f"FAILED: {entry.error}"
        source = f" [{entry.source}]" if entry.source else ""
        print(f"  {index}. {entry.tool}({arguments}) -> {status}{source} {entry.duration_ms}ms")
        if entry.ok:
            print(f"     {entry.summary}")

    print(f"\n--- answer ({elapsed:.1f}s, {reply.total_tokens:,} tokens) ---")
    print(reply.text or "(empty)")

    if reply.error:
        print(f"\n!! ERROR: {reply.error}")
        return False
    if not reply.text.strip():
        print("\n!! ERROR: empty answer")
        return False
    if not reply.trace:
        print("\n!! WARNING: answered with no tool calls -- the answer may be ungrounded")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="a single question to ask")
    parser.add_argument("--offline", action="store_true", help="force the snapshot fallback")
    parser.add_argument(
        "--provider",
        default=config.default_provider(),
        choices=sorted(llm.PROVIDERS),
        help="model backend to use",
    )
    parser.add_argument("--model", default=None, help="model name (defaults to the backend's)")
    args = parser.parse_args()

    if args.offline:
        os.environ["PETBARN_LIVE"] = "0"

    provider = llm.get_provider(args.provider)
    key = load_api_key(provider)

    health = llm.check(provider, api_key=key)
    if not health.ok:
        print(f"{provider.label} is not ready: {health.detail}\n", file=sys.stderr)
        if provider.is_local:
            print(
                "Install Ollama from https://ollama.com/download, then pull a tool-capable "
                f"model:\n  ollama pull {provider.default_model}",
                file=sys.stderr,
            )
        else:
            name = f"{provider.name.upper()}_API_KEY"
            secrets = config.PROJECT_ROOT / ".streamlit" / "secrets.toml"
            print(
                f"Set {name} in the environment or in {secrets}.\n"
                f"Free keys: {provider.key_url}",
                file=sys.stderr,
            )
        return 2

    model = args.model or provider.default_model
    if provider.is_local and health.models:
        if args.model and args.model not in health.models:
            print(
                f"Model {args.model!r} is not installed. Available: {', '.join(health.models)}",
                file=sys.stderr,
            )
            return 2
        if not args.model and model not in health.models:
            # Prefer something actually installed over the suggested default.
            model = health.models[0]

    print(
        f"provider={provider.name}  model={model}  "
        f"live={config.live_fetch_enabled()}  catalog={len(get_catalog())}"
    )
    agent = PetbarnAgent(key, provider=provider, model=model)

    questions = [" ".join(args.question)] if args.question else SAMPLE_QUESTIONS
    failures = sum(0 if ask(agent, question) else 1 for question in questions)

    print(f"\n{'=' * 78}")
    print(f"{len(questions) - failures}/{len(questions)} questions answered")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
