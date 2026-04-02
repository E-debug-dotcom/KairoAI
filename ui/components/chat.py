"""Chat interface component."""

from __future__ import annotations

import streamlit as st

from ui.components.api_client import APIClient, APIClientError
from ui.components.state import append_assistant_message, append_user_message
from ui.utils.streaming import stream_text


def render_chat(client: APIClient) -> None:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))

    prompt = st.chat_input("Ask KairoAI...")
    if not prompt:
        return

    append_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = client.send_chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                )
            except APIClientError as e:
                error_msg = f"Sorry, I couldn't get a response. {e}"
                st.error(error_msg)
                append_assistant_message(error_msg, [])
                return

        data = result.get("data", {})
        answer = data.get("answer", "No answer returned.")
        sources = data.get("retrieved_memory", [])

        stream_text(answer)
        _render_sources(sources)
        append_assistant_message(answer, sources)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for idx, item in enumerate(sources, start=1):
            st.markdown(f"**{idx}. {item.get('source', 'unknown')}** ({item.get('category', 'general')})")
            st.code(item.get("text", ""), language="markdown")
            if item.get("distance") is not None:
                st.caption(f"Similarity distance: {item['distance']:.4f}")
