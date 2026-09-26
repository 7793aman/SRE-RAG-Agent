"""Streamlit demo UI (issue #34): auth, a query console with every retrieval
flag exposed, the SQL-approval flow, and preset example questions.

Talks to the FastAPI service over plain HTTP (`requests`), the same way any
other client would — no FastAPI internals imported here. Requires the API
already running (`make api`) at `settings.streamlit_api_base_url`.

The layout (persistent sidebar + transcript), the theme, the route-glyph
system, the inspector, and the flag-disabling rule are design decisions
recorded in spec.md's "Demo UI (issue #34)" section, made from throwaway HTML
prototypes — this file is the real implementation of that decision, not a
port of the prototype's code. Two adaptations from the prototype: entries
render as Streamlit's native `st.chat_message` bubbles rather than the
prototype's flat log-entry style, and presets ask their question immediately
on click rather than only prefilling it — both are Streamlit-idiomatic
choices that keep this maintainable rather than fighting the framework for
pixel parity with a throwaway mockup. The inspector is a per-message
`st.expander` directly under that answer (a persistent shared right-hand
column was tried and reverted after live testing — one growing column
whose content depended on whatever was last clicked made it unclear which
answer's detail was showing; an expander scoped to its own message has no
such shared state to confuse).
"""

from __future__ import annotations

import secrets
from typing import Any

import requests
import streamlit as st

from app.config import settings

st.set_page_config(page_title="Query Console", page_icon="🛰️", layout="wide")

# Reading text (the actual questions and answers) gets Anthropic's serif,
# same as claude.ai's own message text — everything else (buttons, labels,
# captions) stays the sans set globally via .streamlit/config.toml's `font`.
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

[data-testid="stChatMessageContent"] p {
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 1.05rem;
    line-height: 1.65;
}

/* st.chat_input's textarea inherits Streamlit's small default widget font
   (0.875rem) instead of the body text size, so what you're typing reads
   noticeably smaller than the transcript above it. */
[data-testid="stChatInput"] textarea {
    font-size: 1.05rem;
}

/* st.divider()'s <hr> ships with its own 32px top+bottom margin *on top of*
   the sidebar's own gap between every element — two spacing systems
   stacking, which is why a divider stood out with a much bigger gap than
   anywhere else. Zeroing the hr's own margin makes the single flex `gap`
   below the only source of spacing in the sidebar, so every gap — around a
   divider or not — is the same size, comfortably larger than the old 16px
   default without needing a scroll. */
[data-testid="stSidebar"] hr {
    margin: 0;
}
/* A section heading (st.subheader) gets deliberately *more* space above it
   than a gap between two plain elements — that's what actually signals
   "new section starts here" instead of everything reading as one
   undifferentiated list. st.subheader's own default padding-top did this
   by accident (an arbitrary 12px nobody chose); this replaces it with a
   fixed, intentional amount on top of the uniform gap. */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    padding-top: 0.75rem;
    margin-top: 0;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* Pushes the Log out button (wrapped in st.container(key="sidebar_footer")
   in render_sidebar) to the bottom of the sidebar instead of sitting right
   under the retrieval settings with empty space below it — the sidebar's
   own vertical block is already a flex column stretched to the sidebar's
   full height, so this is the standard flexbox "stick to the end" trick,
   not a fixed/absolute position hack. */
.st-key-sidebar_footer {
    margin-top: auto;
}
</style>
"""

_API = settings.streamlit_api_base_url

# `st.session_state` is tied to the browser tab's live connection — a reload
# opens a new one with empty state, which is why reloading used to bounce
# back to the login screen every time. This is a process-lifetime, in-memory
# map from a random session id (never the JWT itself) to the logged-in
# token/username; the id lives in the URL's query string (`?sid=...`), which
# *does* survive a reload, so `_restore_session` below can look the real
# token back up instead of putting the token itself in the URL.
#
# A plain module-level dict won't do — Streamlit re-executes this whole
# script top to bottom on *every* rerun (not just every reload), so a bare
# `_SESSIONS = {}` here would reinitialize to empty on the very next rerun
# after login, wiping the entry it just wrote. `st.cache_resource` is
# Streamlit's own way to get a value that's actually built once and shared
# for the life of the process, across every rerun and every session.
@st.cache_resource
def _session_store() -> dict[str, dict[str, str]]:
    return {}

_PRESETS = [
    "How do I debug a crashing pod?",
    "Which cluster had the most P1 incidents last month?",
    "Show P1 incidents on prod-us-east and the fix for each alert type",
]

# route -> (glyph, label). ● = single-source resolved, ◐ = hybrid (merged
# sources), ○ = unresolved/awaiting approval, ✕ = the SQL path didn't
# complete. Unknown routes fall back to a plain label in `_route_status`.
_ROUTE_LABELS: dict[str, tuple[str, str]] = {
    "rag": ("●", "documentation"),
    "hybrid": ("◐", "hybrid — docs + database"),
    "sql": ("●", "database only"),
    "sql_pending": ("○", "awaiting approval"),
    "sql_rejected": ("✕", "rejected"),
    "sql_refused": ("✕", "refused — not a safe read-only query"),
    "sql_error": ("✕", "failed"),
    "rag_general_knowledge": ("●", "general knowledge"),
}

# Only these routes actually retrieved anything — everything else (SQL-only,
# general-knowledge skip, a halted SQL path) has no relevance score or
# sources to show, and showing a "0% relevant" meter for them would be
# misleading rather than honest.
_RETRIEVAL_ROUTES = {"rag", "hybrid"}

# Flags only affect a `sql`-intent question if it never resolves purely as
# SQL (i.e. never for these two) — see the flag x intent liveness table on
# issue #34. Told to the user instead of silently disabling the sidebar,
# since the intent isn't known until the response comes back.
_ROUTE_NOTES: dict[str, str] = {
    "sql": "Answered from the database — retrieval settings weren't used for this question.",
    "rag_general_knowledge": (
        "Answered from general knowledge — no documents were searched, "
        "so retrieval settings weren't used."
    ),
}


# --- API client ------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


def api_post(path: str, body: dict[str, Any], auth: bool = True) -> dict[str, Any] | None:
    """POST to the API; on any failure, shows `st.error` and returns None so
    callers can just check for None instead of handling exceptions."""
    try:
        response = requests.post(
            f"{_API}{path}", json=body, headers=_auth_headers() if auth else {}, timeout=120
        )
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the API at {_API}: {exc}")
        return None

    # A 401 on an authenticated call means the token itself is invalid or
    # expired (JWTs here last settings.jwt_expiration_minutes, 60 by
    # default) — no retry will ever succeed, so force a real login instead
    # of leaving the chat silently failing forever. Login/register itself
    # (auth=False) still falls through to the plain error below, since a
    # 401 there just means a wrong password, not a dead session.
    if auth and response.status_code == 401:
        _end_session()
        st.session_state["session_expired_notice"] = True
        st.rerun()

    if not response.ok:
        st.error(_error_detail(response))
        return None
    return response.json()  # type: ignore[no-any-return]


# --- auth --------------------------------------------------------------------


def _start_session(token: str, username: str) -> None:
    sid = secrets.token_urlsafe(24)
    _session_store()[sid] = {"token": token, "username": username}
    st.session_state.token = token
    st.session_state.username = username
    st.query_params["sid"] = sid


def _restore_session() -> None:
    session = _session_store().get(st.query_params.get("sid", ""))
    if session is None:
        return
    st.session_state.token = session["token"]
    st.session_state.username = session["username"]


def _end_session() -> None:
    _session_store().pop(st.query_params.get("sid", ""), None)
    st.query_params.clear()
    st.session_state.clear()


def _log_in_or_register(path: str, username: str, password: str) -> None:
    verb = "Logging in…" if path.endswith("login") else "Registering…"
    with st.spinner(verb):
        result = api_post(path, {"username": username, "password": password}, auth=False)
    if result is None:
        return
    _start_session(result["token"], username)
    st.rerun()


def render_auth_gate() -> None:
    # A full-width st.title + bare form used to leave most of the screen
    # empty — nothing here needs the wide layout the signed-in app uses, so
    # this reads as an actual sign-in screen (a centered, bounded card with
    # a mark above the title) instead of an unstyled form floating in a
    # mostly-blank page.
    st.markdown(
        """
        <style>
        .st-key-auth_card {
            max-width: 420px;
            margin: 8vh auto 0;
            padding: 2.5rem 2rem 2rem;
            background-color: #F4F3EE;
            border: 1px solid #E4E2DB;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="auth_card"):
        if st.session_state.pop("session_expired_notice", False):
            st.info("Your session expired — log in again to keep asking questions.")
        st.markdown(
            "<div style='text-align:center; font-size:2.75rem; line-height:1;'>🛰️</div>"
            "<h1 style='text-align:center; margin:0.5rem 0 0;'>Query Console</h1>"
            "<p style='text-align:center; color:#87867F; "
            "margin:0.35rem 0 1.5rem;'>Sign in to ask the Kubernetes ops assistant "
            "a question.</p>",
            unsafe_allow_html=True,
        )
        login_tab, register_tab = st.tabs(["Log in", "Register"])

        with login_tab, st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                _log_in_or_register("/auth/login", username, password)

        with register_tab, st.form("register_form"):
            username = st.text_input("Username", key="register_username")
            password = st.text_input(
                "Password", type="password", key="register_password", help="At least 8 characters."
            )
            if st.form_submit_button("Register", type="primary", use_container_width=True):
                _log_in_or_register("/auth/register", username, password)


# --- sidebar: retrieval controls + presets ------------------------------------


def _current_flags() -> dict[str, Any]:
    return {
        "search_mode": st.session_state.search_mode,
        "enable_rerank": st.session_state.enable_rerank,
        "enable_hyde": st.session_state.enable_hyde,
        "enable_crag": st.session_state.enable_crag,
        "enable_self_reflective": st.session_state.enable_self_reflective,
        "enable_adaptive_retrieval": st.session_state.enable_adaptive_retrieval,
        "top_k": st.session_state.top_k,
    }


def render_sidebar() -> None:
    with st.sidebar:
        # A plain content element, not wrapped in its own padded container —
        # the uniform 1rem gap CSS below spaces it from "Signed in as..."
        # exactly like every other pair of elements in the sidebar, instead
        # of stacking extra margin of its own on top of that gap.
        # A magnifying glass, not an arbitrary icon — this tool exists to
        # investigate incidents (debug a pod, find which cluster, trace a
        # root cause), so the mark matches what the app actually does.
        st.markdown(
            "<div style='display:flex; align-items:center; gap:0.4rem; "
            "font-size:0.75rem; font-weight:600; letter-spacing:0.12em; "
            "color:#D97757;'>"
            "<span style='font-size:1rem;'>🔍</span>QUERY CONSOLE</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Signed in as **{st.session_state.username}**")

        st.subheader("Try a question")
        for question in _PRESETS:
            if st.button(question, key=f"preset_{question}", use_container_width=True):
                send_question(question)
                st.rerun()

        st.divider()
        st.subheader("Retrieval settings")

        # The one flag conflict knowable before a question is even sent:
        # HyDE always searches densely internally, ignoring search_mode.
        # Every other case (a question turning out to be SQL-only, adaptive
        # retrieval skipping the corpus) can't be predicted from the
        # question text, so it's disclosed on the response instead — see
        # `_ROUTE_NOTES` — rather than guessed at here.
        hyde_on = st.session_state.get("enable_hyde", False)
        st.selectbox(
            "Search mode",
            ["dense", "sparse", "hybrid"],
            key="search_mode",
            disabled=hyde_on,
            help=(
                "Ignored while HyDE is on — HyDE always searches densely."
                if hyde_on
                else "How the corpus is searched."
            ),
        )
        st.toggle("Rerank", key="enable_rerank", help="Cross-encoder re-scores the top chunks.")
        st.toggle(
            "HyDE",
            key="enable_hyde",
            help="Searches with a hypothetical answer's embedding instead of the search mode above.",
        )
        st.toggle(
            "CRAG grading",
            key="enable_crag",
            value=True,
            help="Falls back to a web search if retrieval is weak. On by default.",
        )
        st.toggle(
            "Self-RAG reflect",
            key="enable_self_reflective",
            help="Critiques and retries a weak answer once.",
        )
        st.toggle(
            "Adaptive retrieval",
            key="enable_adaptive_retrieval",
            help="Skips the corpus search entirely when the question doesn't need it.",
        )
        st.number_input("top_k", min_value=1, max_value=50, value=5, key="top_k")

        # Pinned to the bottom of the sidebar (see the sidebar_footer CSS
        # rule) instead of sitting up top where a fixed-height viewport
        # otherwise leaves it stranded next to a lot of empty space below
        # the settings — logging out is the one action that belongs at the
        # edge of the screen, not mixed in with the retrieval controls.
        with st.container(key="sidebar_footer"):
            st.divider()
            if st.button("Log out"):
                _end_session()
                st.rerun()


# --- response rendering --------------------------------------------------------


def _clamped(score: float) -> float:
    """`st.progress` requires 0.0-1.0; nothing in the response schema
    actually guarantees a chunk score stays in that range (a reranker's raw
    score, in particular, isn't bounded), so clamp defensively rather than
    let a real answer crash the page over a display detail."""
    return max(0.0, min(1.0, score))


def _route_status(route: str) -> str:
    glyph, label = _ROUTE_LABELS.get(route, ("●", route))
    return f"{glyph} {label}"


def _resolve_sql(index: int, entry: dict[str, Any], query_id: str, approved: bool) -> None:
    verb = "Running the query…" if approved else "Rejecting…"
    with st.spinner(verb):
        result = api_post("/query/sql/execute", {"query_id": query_id, "approved": approved})
    if result is None:
        return
    entry["response"] = result
    st.rerun()


def render_pending_sql(index: int, entry: dict[str, Any], pending: dict[str, Any]) -> None:
    st.write(pending["explanation"])
    st.code(pending["sql"], language="sql")
    # `st.columns` always splits the row into equal-width tracks and a button
    # doesn't stretch to fill its track, so any fixed ratio still leaves each
    # button sitting at the *left* of its own too-wide track — which reads as
    # mismatched sizes with a gap between them, worse the wider the chat
    # bubble is. Scoping this row's columns to shrink to their buttons'
    # actual content width (instead of splitting available space) is what
    # actually puts them flush next to each other.
    st.markdown(
        """
        <style>
        [class*="st-key-sql_actions_"] [data-testid="stHorizontalBlock"] {
            width: fit-content;
            gap: 0.6rem;
        }
        [class*="st-key-sql_actions_"] [data-testid="stColumn"] {
            width: fit-content !important;
            flex: none !important;
            min-width: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Keyed per query, not a shared literal key — two pending-SQL turns
    # showing at once would otherwise collide on the same container key.
    with st.container(key=f"sql_actions_{pending['query_id']}"):
        approve_col, reject_col = st.columns(2)
        if approve_col.button(
            "Approve & run", key=f"approve_{pending['query_id']}", type="primary"
        ):
            _resolve_sql(index, entry, pending["query_id"], approved=True)
        if reject_col.button("Reject", key=f"reject_{pending['query_id']}"):
            _resolve_sql(index, entry, pending["query_id"], approved=False)


def render_response(index: int, entry: dict[str, Any]) -> None:
    response = entry["response"]
    metadata = response["metadata"]
    pending = response.get("pending_sql")

    st.caption(_route_status(metadata["route"]))

    if pending:
        render_pending_sql(index, entry, pending)
        return

    st.write(response["answer"])

    if metadata.get("used_web_fallback"):
        st.info(
            "This answer used a web search — the documentation corpus didn't have enough to go on."
        )

    note = _ROUTE_NOTES.get(metadata["route"])
    if note:
        st.caption(note)

    if metadata["route"] in _RETRIEVAL_ROUTES:
        st.progress(
            _clamped(response["retrieval_score"]),
            text=f"relevance {response['retrieval_score']:.2f}",
        )
        if response["sources"]:
            tags = [
                f"`[{'sql' if 'sql' in source.lower() else 'doc'}]` {source}"
                for source in response["sources"]
            ]
            st.caption(", ".join(tags))

    # Scoped to this one message — an `st.expander` per answer, not a shared
    # panel — so there's never a question of whose detail is on screen.
    with st.expander("Inspect response"):
        formatted_tab, raw_tab = st.tabs(["Formatted", "Raw JSON"])
        with formatted_tab:
            st.write(f"**Cache:** {'hit' if response['cache_hit'] else 'miss'}")
            # Toggling HyDE or Rerank otherwise has no visible effect on the
            # response — this is that confirmation. None when retrieval
            # never ran at all (adaptive retrieval's general-knowledge skip,
            # a pure SQL answer), so there's nothing to show.
            if metadata.get("retrieval_path"):
                st.write(f"**Retrieval:** {metadata['retrieval_path']}")
            reflection_note = f"{metadata['reflection_iterations']} iteration(s)"
            if metadata.get("reflection_score") is not None:
                reflection_note += f", score {metadata['reflection_score']:.2f}"
            st.write(f"**Reflection:** {reflection_note}")

            chunks = metadata.get("retrieved_chunks", [])
            if not chunks:
                st.caption("No chunks retrieved for this turn.")
            for chunk in chunks:
                st.progress(
                    _clamped(chunk["score"]), text=f"{chunk['source']} — {chunk['score']:.2f}"
                )
                # Retrieved chunk text is arbitrary corpus content, not
                # markdown we wrote — st.caption/st.write would parse a
                # stray "#" as a heading. st.text renders it literally.
                st.text(chunk["text"])
        with raw_tab:
            st.json(response)


# --- transcript + composer -----------------------------------------------------


def send_question(question: str) -> None:
    # Only queue the question here — appending it with `response: None` lets
    # the *next* rerun paint the user bubble immediately, with a spinner in
    # the assistant bubble below it (see `render_transcript`), instead of
    # blocking on the API call before the question ever reaches the screen.
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"question": question, "response": None})


def _fetch_response(index: int, entry: dict[str, Any]) -> None:
    with st.spinner("Thinking…"):
        response = api_post("/query", {"question": entry["question"], **_current_flags()})
    if response is None:
        # api_post already showed *why* via st.error, but that render is
        # about to be discarded by the rerun below — dropping the message
        # here used to make a failed question vanish with no trace at all,
        # question gone, no error, nothing. Keeping it and marking it
        # failed means the transcript actually shows what happened.
        entry["failed"] = True
        st.rerun()
        return
    entry["response"] = response
    st.rerun()


def render_transcript() -> None:
    for index, entry in enumerate(st.session_state.messages):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            if entry.get("failed"):
                st.error("Couldn't get an answer for this one — try asking again.")
            elif entry["response"] is None:
                _fetch_response(index, entry)
            else:
                render_response(index, entry)


def render_composer() -> None:
    # Streamlit only docks `st.chat_input` to the bottom of the viewport
    # (the ChatGPT/Claude-style "input stays put, transcript scrolls"
    # behavior) when it's called at the top level — nested inside
    # `st.columns`/`st.container`, as it used to be at one point, it just
    # renders inline instead and scrolls away with the rest of the page.
    question = st.chat_input("Ask about your clusters…")
    if question:
        send_question(question)
        st.rerun()


# --- entry point ---------------------------------------------------------------


def main() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("messages", [])

    if not st.session_state.token:
        _restore_session()

    if not st.session_state.token:
        render_auth_gate()
        return

    render_sidebar()
    st.title("Query Console")
    st.caption("Ask about your clusters, or approve a generated query.")

    render_transcript()
    render_composer()


if __name__ == "__main__":
    main()
