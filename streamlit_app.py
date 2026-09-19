"""Petbarn Product Assistant -- Streamlit chat UI.

Entry point for Streamlit Community Cloud.

Two decisions shape this file:

**The tool trace is part of the interface, not debug output.** Every answer ships
with an expander listing the tools the model chose, the arguments it passed, how
long each took, and whether the data came from petbarn.com.au live or from the
bundled snapshot. An agentic app whose tool use is invisible is indistinguishable
from one that made the numbers up, so the trace is shown by default.

**The app stays usable without the owner's API key.** It reads a key from
Streamlit secrets when deployed, and otherwise offers a sidebar input, so the
hosted demo keeps working when its free-tier quota runs out and a reviewer can
try it with their own key straight away.
"""

from __future__ import annotations

import os

import streamlit as st

from petbarn import config
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
# Configuration plumbing
# --------------------------------------------------------------------------- #


def resolve_api_key() -> tuple[str | None, str]:
    """Find a Groq key. Returns the key and where it came from.

    A key typed into the sidebar wins, so a visitor can always override the
    deployment's own key -- with their own quota rather than the owner's.
    """
    typed = (st.session_state.get("api_key_input") or "").strip()
    if typed:
        return typed, "sidebar"

    try:
        secret = str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:  # noqa: BLE001 - no secrets file configured at all
        secret = ""
    if secret:
        return secret, "secrets"

    env = config.groq_api_key()
    if env:
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


def render_sidebar() -> tuple[str | None, str, str]:
    """Draw the sidebar. Returns the API key, its origin and the chosen model."""
    with st.sidebar:
        st.subheader("Settings")

        api_key, origin = resolve_api_key()
        if origin in {"secrets", "environment"}:
            st.success(f"Groq key loaded from {origin}", icon="🔑")
            with st.expander("Use your own key instead"):
                st.text_input(
                    "Groq API key",
                    key="api_key_input",
                    type="password",
                    placeholder="gsk_...",
                    help="Overrides the deployed key, so requests count against your quota.",
                )
        else:
            st.text_input(
                "Groq API key",
                key="api_key_input",
                type="password",
                placeholder="gsk_...",
                help="Free keys are available at console.groq.com/keys",
            )
            if not api_key:
                st.caption("Get a free key at [console.groq.com/keys](https://console.groq.com/keys)")
        api_key, origin = resolve_api_key()

        model = st.selectbox(
            "Model",
            options=list(config.GROQ_MODELS),
            index=0,
            help="The default supports parallel tool calls, which speeds up comparisons.",
        )

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

    return api_key, origin, model


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
    """Run one agent turn, streaming tool progress into a status container."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]

    with st.chat_message("assistant", avatar="🐾"):
        status = st.status("Looking this up…", expanded=True)

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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply.text,
            "trace": reply.trace,
            "reply": reply,
        }
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🐾", layout="centered")
    st.session_state.setdefault("messages", [])

    api_key, origin, model = render_sidebar()

    st.title("🐾 Petbarn Product Assistant")

    try:
        get_catalog()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()

    if not api_key:
        st.info(
            "**A Groq API key is needed to chat.** Add one in the sidebar — free keys are "
            "available at [console.groq.com/keys](https://console.groq.com/keys).",
            icon="🔑",
        )
        st.markdown(
            "Everything except the conversation works without a key. The catalog in the sidebar "
            "was scraped from petbarn.com.au, and the tools behind this assistant can be "
            "exercised directly with `python scripts/smoke_test.py`."
        )
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
        agent = PetbarnAgent(api_key, model=model)
    except ValueError as exc:
        st.error(str(exc))
        return

    answer_turn(agent)


if __name__ == "__main__":
    main()
