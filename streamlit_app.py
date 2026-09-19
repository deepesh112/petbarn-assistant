"""Petbarn Product Assistant -- Streamlit chat UI.

Entry point for local runs and for Streamlit Community Cloud.

Three decisions shape this file:

**The tool trace is part of the interface, not debug output.** Every answer ships
with an expander listing the tools the model chose, the arguments it passed, how
long each took, and whether the data came from petbarn.com.au live or from the
bundled snapshot. An agentic app whose tool use is invisible is indistinguishable
from one that made the numbers up, so the trace is always available.

**The model backend is a choice, made here.** Ollama runs the whole thing
offline on your own machine with no key and no cost; Groq is there because a
hosted deployment cannot reach your laptop. Neither is hard-coded.

**Nothing is assumed to be working.** The sidebar probes the chosen backend
before the chat opens, so a stopped Ollama server or a missing model produces
the instruction that fixes it rather than an exception mid-answer.
"""

from __future__ import annotations

import os

import streamlit as st

from petbarn import config, llm
from petbarn.agent import AgentReply, PetbarnAgent, TraceEntry
from petbarn.catalog import get_catalog, pretty_brand

PAGE_TITLE = "Petbarn Product Assistant"

#: The brief's own sample questions, wired to buttons so the app demonstrates
#: itself without the reader having to think of something to ask.
STARTER_QUESTIONS = [
    "What are people saying about the price and quality of the Breeders Choice cat litter?",
    "Can you compare the reviews between the Black Hawk lamb and rice and the Prime100 kangaroo roll?",
    "List the main pros and cons based on recent customer feedback for the NexGard Spectra.",
    "Which dog food is better value, and what do reviewers complain about most?",
]

SOURCE_BADGES = {
    "live": ("🟢", "Live from petbarn.com.au"),
    "cache": ("🔵", "Recent local cache"),
    "snapshot": ("🟡", "Offline snapshot — may be out of date"),
}


# --------------------------------------------------------------------------- #
# Backend plumbing
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=10, show_spinner=False)
def discover_ollama_models(base_url: str) -> list[str]:
    """List models installed on the local Ollama server.

    Cached briefly because Streamlit reruns the whole script on every widget
    interaction, and probing a socket on each keystroke would be wasteful. Ten
    seconds is short enough that a freshly pulled model shows up promptly.
    """
    return llm.list_local_models(base_url)


def read_secret(name: str) -> str | None:
    """Read a Streamlit secret, tolerating there being no secrets file at all."""
    try:
        value = str(st.secrets.get(name, "") or "").strip()
    except Exception:  # noqa: BLE001 - no secrets configured is normal locally
        return None
    return value or None


def choose_default_provider() -> str:
    """Pick the backend to start on, based on what this machine can actually do.

    An explicit ``PETBARN_PROVIDER`` always wins. Failing that the app works it
    out: a hosted container can never reach a laptop's Ollama server, so a
    deployment that has an API key but no local server should open on the hosted
    backend rather than greeting its first visitor with Ollama install
    instructions. Deriving this beats relying on a config line someone has to
    remember to set.
    """
    explicit = read_secret("PETBARN_PROVIDER") or os.environ.get("PETBARN_PROVIDER", "").strip()
    if explicit and explicit.lower() in llm.PROVIDERS:
        return explicit.lower()

    if llm.list_local_models():
        return "ollama"
    for name, spec in llm.PROVIDERS.items():
        if spec.requires_api_key and (read_secret(f"{name.upper()}_API_KEY") or config.api_key_for(name)):
            return name
    return config.default_provider()


def api_key_widget_key(provider: llm.Provider) -> str:
    """Session-state key for this provider's sidebar API-key box.

    Per-provider on purpose. A single shared key meant the value typed for one
    backend was handed to the next one you switched to -- paste a Gemini key,
    switch to Groq, and Groq was sent the Gemini key and rejected it. Separate
    keys also let the sidebar remember a key per backend, so switching back and
    forth does not mean retyping.
    """
    return f"api_key_{provider.name}"


def model_widget_key(provider: llm.Provider) -> str:
    """Session-state key for this provider's model picker."""
    return f"model_{provider.name}"


def resolve_api_key(provider: llm.Provider) -> tuple[str | None, str]:
    """Find an API key for ``provider``. Returns the key and where it came from.

    A key typed into the sidebar wins, so a visitor to a deployed app can always
    use their own quota rather than the owner's.
    """
    if not provider.requires_api_key:
        return None, "not needed"

    typed = (st.session_state.get(api_key_widget_key(provider)) or "").strip()
    if typed:
        return typed, "sidebar"

    if secret := read_secret(f"{provider.name.upper()}_API_KEY"):
        return secret, "secrets"

    if env := config.api_key_for(provider.name):
        return env, "environment"
    return None, "missing"


def apply_live_setting(enabled: bool) -> None:
    """Push the sidebar's live/offline choice into the config layer."""
    os.environ["PETBARN_LIVE"] = "1" if enabled else "0"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_trace(trace: list[TraceEntry], *, reply: AgentReply | None = None) -> None:
    """Show which tools ran, with their arguments, timing and data source."""
    if not trace:
        return

    label = f"🔧 {len(trace)} tool call{'s' if len(trace) != 1 else ''}"
    if reply is not None and reply.rounds > 1:
        label += f" over {reply.rounds} rounds"
    if reply is not None and reply.total_tokens:
        label += f" · {reply.total_tokens:,} tokens"

    with st.expander(label, expanded=False):
        for index, entry in enumerate(trace, start=1):
            arguments = ", ".join(f"{k}={v!r}" for k, v in entry.arguments.items()) or "no arguments"
            icon = "✅" if entry.ok else "⚠️"
            st.markdown(f"**{index}. {icon} `{entry.tool}`** — {entry.duration_ms} ms")
            st.caption(f"`{arguments}`")
            if entry.ok:
                badge, description = SOURCE_BADGES.get(entry.source or "", ("", ""))
                detail = f"{badge} {description}" if badge else ""
                st.caption(f"{entry.summary}{'  ·  ' + detail if detail else ''}")
            else:
                st.caption(f":red[{entry.error}]")


def render_backend_controls() -> tuple[llm.Provider, str | None, str, llm.Health]:
    """Draw the backend picker and probe it. Returns provider, key, model, health."""
    names = list(llm.PROVIDERS)
    preferred = choose_default_provider()
    default_index = names.index(preferred) if preferred in names else 0
    provider = llm.PROVIDERS[
        st.selectbox(
            "Model backend",
            options=names,
            index=default_index,
            format_func=lambda name: llm.PROVIDERS[name].label,
            help="Ollama runs on this machine with no key. Groq is hosted and needs a free key.",
        )
    ]
    st.caption(provider.notes)

    api_key: str | None = None
    if provider.requires_api_key:
        api_key, origin = resolve_api_key(provider)
        # persist_state keeps each backend's key for the session, so switching
        # away and back does not clear the box.
        field = dict(
            label=f"{provider.label} API key",
            key=api_key_widget_key(provider),
            type="password",
            persist_state="session",
            placeholder="paste your key",
        )
        if origin in {"secrets", "environment"}:
            st.success(f"API key loaded from {origin}", icon="🔑")
            with st.expander("Use your own key instead"):
                st.text_input(**field)
        else:
            st.text_input(**field)
            if not api_key and provider.key_url:
                st.caption(f"Get a key at [{provider.key_url}]({provider.key_url})")
        api_key, _ = resolve_api_key(provider)

    if provider.is_local:
        installed = discover_ollama_models(provider.base_url)
        health = llm.Health(bool(installed), f"{len(installed)} model(s) installed", installed)
        if not installed:
            health = llm.check(provider)
    else:
        health = llm.check(provider, api_key=api_key)

    options = health.models or list(provider.suggested_models)
    model = ""
    if options:
        widget_key = model_widget_key(provider)
        # A remembered model that no longer exists -- an Ollama model deleted
        # since it was picked -- would make the selectbox raise. Drop it and let
        # the provider's default take over.
        if st.session_state.get(widget_key) not in options:
            st.session_state.pop(widget_key, None)
        preferred = provider.default_model if provider.default_model in options else options[0]
        model = st.selectbox(
            "Model",
            options=options,
            index=options.index(preferred),
            key=widget_key,
            persist_state="session",
        )

    if provider.is_local and health.ok:
        st.caption(f"🟢 Ollama reachable · {health.detail}")

    return provider, api_key, model, health


def render_sidebar() -> tuple[llm.Provider, str | None, str, llm.Health]:
    """Draw the whole sidebar."""
    with st.sidebar:
        st.subheader("Settings")
        provider, api_key, model, health = render_backend_controls()

        live = st.toggle(
            "Fetch live data",
            value=config.live_fetch_enabled(),
            help=(
                "On: scrape petbarn.com.au and Bazaarvoice on demand. "
                "Off: answer from the snapshot committed with the app."
            ),
        )
        apply_live_setting(live)

        st.divider()
        catalog = get_catalog()
        st.subheader(f"Catalog · {len(catalog)} products")
        st.caption(f"Ingested {(catalog.generated_at or '')[:10]} from petbarn.com.au")
        st.dataframe(
            [
                {
                    "Product": entry.name,
                    "Brand": pretty_brand(entry.brand),
                    "Price": f"${entry.price:.2f}" if entry.price else "—",
                    "Rating": f"{entry.average_rating}★ ({entry.review_count})",
                    "Link": entry.url,
                }
                for entry in catalog
            ],
            hide_index=True,
            width="stretch",
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="View")},
        )

        st.divider()
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

        with st.expander("How this works"):
            st.markdown(
                "Product details come from the **schema.org JSON-LD** on each Petbarn product "
                "page. Reviews come from **Bazaarvoice**, the platform Petbarn renders its "
                "reviews with. Sentiment is computed here with **VADER** plus a pet-retail "
                "lexicon, per theme, rather than being guessed by the language model.\n\n"
                "Each tool tries a recent cache, then a live fetch, then the snapshot committed "
                "with the app — so an answer degrades in freshness rather than disappearing."
            )

    return provider, api_key, model, health


def render_backend_help(provider: llm.Provider, health: llm.Health) -> None:
    """Explain how to make an unavailable backend work."""
    if provider.is_local:
        st.warning(f"**Ollama is not ready.** {health.detail}", icon="🦙")
        st.markdown(
            f"""
Ollama runs the assistant entirely on this machine — no API key, no cost, and it
works with the network unplugged.

1. **Install it** from [ollama.com/download](https://ollama.com/download).
2. **Pull a model that can call tools** — this app is useless without tool support:
   ```
   ollama pull {provider.default_model}
   ```
   Other good options: {", ".join(f"`{m}`" for m in provider.suggested_models[1:])}
3. Ollama serves automatically once installed. If not, run `ollama serve`.

Then reload this page. Or switch the backend to **Groq** in the sidebar to use a
hosted model instead.
"""
        )
    else:
        st.info(
            f"**An API key is needed to chat.** Add one in the sidebar — free keys are at "
            f"[{provider.key_url}]({provider.key_url}).",
            icon="🔑",
        )

    st.markdown(
        "Everything except the conversation works regardless. The catalog in the sidebar was "
        "scraped from petbarn.com.au, and the tools behind this assistant can be exercised "
        "directly with `python scripts/smoke_test.py`."
    )


def render_welcome() -> None:
    catalog = get_catalog()
    st.markdown(
        f"Ask about any of **{len(catalog)} real Petbarn products** — pricing, specifications, "
        "or what customers actually say in their reviews. "
        "I look everything up as you ask, and show you every lookup I make."
    )
    st.caption("Try one of these:")
    columns = st.columns(2)
    for index, question in enumerate(STARTER_QUESTIONS):
        with columns[index % 2]:
            if st.button(question, key=f"starter_{index}", width="stretch"):
                st.session_state.pending_prompt = question
                st.rerun()


def answer_turn(agent: PetbarnAgent) -> None:
    """Run one agent turn, reporting tool progress into a status container."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]

    with st.chat_message("assistant", avatar="🐾"):
        hint = " (a local model can take a while)" if agent.provider.is_local else ""
        status = st.status(f"Looking this up…{hint}", expanded=True)

        def on_tool_result(entry: TraceEntry) -> None:
            icon = "✅" if entry.ok else "⚠️"
            status.write(f"{icon} `{entry.tool}` — {entry.summary} ({entry.duration_ms} ms)")

        reply = agent.reply(history, on_tool_result=on_tool_result)

        if reply.error:
            status.update(label="Something went wrong", state="error", expanded=False)
        else:
            calls = len(reply.trace)
            status.update(
                label=f"Answered using {calls} tool call{'s' if calls != 1 else ''}",
                state="complete",
                expanded=False,
            )

        st.markdown(reply.text or "_No answer was produced._")
        render_trace(reply.trace, reply=reply)
        if reply.used_snapshot:
            st.caption("🟡 Some figures above came from the offline snapshot, not a live fetch.")
        if not reply.error and not reply.trace:
            st.caption(
                "⚠️ This answer used no tools, so it may not be grounded in Petbarn's data. "
                "Smaller local models sometimes skip tool calls — try a larger one."
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply.text, "trace": reply.trace, "reply": reply}
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🐾", layout="centered")
    st.session_state.setdefault("messages", [])

    try:
        get_catalog()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    provider, api_key, model, health = render_sidebar()

    st.title("🐾 Petbarn Product Assistant")

    if not health.ok or not model:
        render_backend_help(provider, health)
        st.stop()

    if not st.session_state.messages:
        render_welcome()

    for message in st.session_state.messages:
        avatar = "🐾" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_trace(message.get("trace") or [], reply=message.get("reply"))

    prompt = st.chat_input("Ask about a product, its price, or its reviews…")
    if not prompt:
        prompt = st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        agent = PetbarnAgent(api_key, provider=provider, model=model)
    except ValueError as exc:
        st.error(str(exc))
        return

    answer_turn(agent)


if __name__ == "__main__":
    main()
