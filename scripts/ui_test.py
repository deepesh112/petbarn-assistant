"""Drive the Streamlit UI through Streamlit's own AppTest harness.

    python scripts/ui_test.py

No model and no API key needed -- nothing here sends a request.

This file exists because of a bug it would have caught. The sidebar's API-key box
used one session-state key for every backend, so a key typed for one provider was
handed to whichever provider you switched to next: paste a Gemini key, switch to
Groq, and Groq was sent the Gemini key and rejected it. The tool and agent-loop
suites were both green throughout -- the fault was entirely in widget state, and
nothing was exercising it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

from petbarn import llm  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

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


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def launch(**secrets: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=90)
    for name, value in secrets.items():
        app.secrets[name] = value
    return app.run()


def backend_of(app: AppTest) -> str:
    return app.sidebar.selectbox[0].value


def model_of(app: AppTest) -> str:
    return app.sidebar.selectbox[1].value if len(app.sidebar.selectbox) > 1 else ""


def key_box(app: AppTest) -> str:
    return (app.sidebar.text_input[0].value or "") if app.sidebar.text_input else ""


# --------------------------------------------------------------------------- #


def test_renders() -> None:
    section("The app renders, and offers every backend")
    app = launch()
    check(not app.exception, f"no exception on load: {[str(e.value)[:120] for e in app.exception]}")
    check(len(app.sidebar.selectbox[0].options) == len(llm.PROVIDERS),
          f"all {len(llm.PROVIDERS)} backends are listed")
    check(len(app.sidebar.dataframe[0].value) >= 8, "the catalog table is populated")
    check(bool(app.title), "the page has a title")


def test_keys_do_not_leak_between_backends() -> None:
    section("A key typed for one backend is never sent to another")
    app = launch()

    app.sidebar.selectbox[0].set_value("gemini").run()
    check(backend_of(app) == "gemini", "switching the backend selector works")
    app.sidebar.text_input[0].set_value("AIza_GEMINI_key").run()

    app.sidebar.selectbox[0].set_value("groq").run()
    check(backend_of(app) == "groq", "switched to groq")
    check(not key_box(app).startswith("AIza"),
          f"groq's key box does not hold the Gemini key (got {key_box(app)!r})")
    check(key_box(app) == "", "groq's key box starts empty")

    # And each backend should remember its own.
    app.sidebar.text_input[0].set_value("gsk_GROQ_key").run()
    app.sidebar.selectbox[0].set_value("gemini").run()
    check(key_box(app) == "AIza_GEMINI_key", "gemini remembers its own key on return")
    app.sidebar.selectbox[0].set_value("groq").run()
    check(key_box(app) == "gsk_GROQ_key", "groq remembers its own key on return")


def test_model_follows_the_backend() -> None:
    section("The model list follows the backend, with no stale selection")
    app = launch()
    seen: dict[str, str] = {}

    for name, spec in llm.PROVIDERS.items():
        app.sidebar.selectbox[0].set_value(name).run()
        check(not app.exception,
              f"{name}: switching raised nothing: {[str(e.value)[:120] for e in app.exception]}")
        if not app.exception and len(app.sidebar.selectbox) > 1:
            options, chosen = app.sidebar.selectbox[1].options, model_of(app)
            seen[name] = chosen
            check(chosen in options, f"{name}: selected model {chosen!r} is one of its own options")
            if not spec.is_local:
                check(chosen == spec.default_model, f"{name}: defaults to {spec.default_model}")

    hosted = [m for n, m in seen.items() if not llm.PROVIDERS[n].is_local]
    check(len(set(hosted)) == len(hosted), "no two hosted backends ended up on the same model")


def test_backend_is_derived_when_not_pinned() -> None:
    section("The backend is derived from what the environment can do")
    app = launch(PETBARN_PROVIDER="gemini", GEMINI_API_KEY="AIza_fake")
    check(backend_of(app) == "gemini", "an explicit PETBARN_PROVIDER wins")
    check(bool(app.chat_input), "the chat opens once a backend is usable")

    app = launch()
    check(backend_of(app) == "ollama", "with Ollama reachable locally, it is preferred")


def test_starter_questions() -> None:
    section("The starter questions are wired up")
    app = launch(PETBARN_PROVIDER="groq", GROQ_API_KEY="gsk_fake")
    check(len(app.button) >= 4, f"starter buttons are rendered (got {len(app.button)})")
    check(bool(app.chat_input), "the chat input is present")


def main() -> int:
    test_renders()
    test_keys_do_not_leak_between_backends()
    test_model_follows_the_backend()
    test_backend_is_derived_when_not_pinned()
    test_starter_questions()

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
