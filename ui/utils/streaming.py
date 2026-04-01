"""UI utility helpers for response rendering."""

from __future__ import annotations

import time
import streamlit as st


def stream_text(text: str, delay: float = 0.008) -> None:
    """Render text with a small typing effect for better UX."""
    placeholder = st.empty()
    rendered = ""
    for token in text.split(" "):
        rendered += token + " "
        placeholder.markdown(rendered)
        time.sleep(delay)
