"""Streamlit entrypoint for KairoAI interactive UI."""

from __future__ import annotations

import os

import streamlit as st

from ui.components.api_client import APIClient
from ui.components.chat import render_chat
from ui.components.sidebar import render_sidebar
from ui.components.state import init_state


st.set_page_config(page_title="KairoAI – Teachable AI Assistant", page_icon="🤖", layout="wide")


def main() -> None:
    st.title("KairoAI – Teachable AI Assistant")
    st.caption("Chat, teach new knowledge, and inspect memory grounding.")

    base_url = os.getenv("KAIRO_API_BASE_URL", "http://localhost:8000")
    api_url = st.text_input("Backend API URL", value=base_url, help="FastAPI base URL")

    init_state()
    client = APIClient(base_url=api_url)

    render_sidebar(client)
    render_chat(client)


if __name__ == "__main__":
    main()
