"""Session-state helpers for Streamlit UI."""

from __future__ import annotations

import streamlit as st


DEFAULT_MESSAGES = [
    {
        "role": "assistant",
        "content": "Hi! I am KairoAI. Ask me anything, or teach me from the sidebar.",
        "sources": [],
    }
]


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = list(DEFAULT_MESSAGES)
    if "memory_search_results" not in st.session_state:
        st.session_state.memory_search_results = []
    if "learned_sources" not in st.session_state:
        st.session_state.learned_sources = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit-session"


def append_user_message(content: str) -> None:
    st.session_state.messages.append({"role": "user", "content": content, "sources": []})


def append_assistant_message(content: str, sources: list[dict] | None = None) -> None:
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "sources": sources or []}
    )
