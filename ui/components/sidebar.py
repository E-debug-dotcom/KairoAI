"""Sidebar learning tools component."""

from __future__ import annotations

import streamlit as st

from ui.components.api_client import APIClient, APIClientError


CATEGORY_OPTIONS = ["general", "security", "iam", "automation", "troubleshooting"]


def render_sidebar(client: APIClient) -> None:
    with st.sidebar:
        st.header("Teach KairoAI")

        with st.expander("Teach via Text", expanded=True):
            category_text = st.selectbox("Category", CATEGORY_OPTIONS, key="teach_text_category")
            teach_input = st.text_area("Knowledge text", height=140, key="teach_text_input")
            if st.button("Save Knowledge", key="teach_text_btn", use_container_width=True):
                if not teach_input.strip():
                    st.warning("Please enter text before saving.")
                else:
                    with st.spinner("Saving knowledge..."):
                        try:
                            result = client.teach_text(teach_input, category_text)
                            stored = result.get("data", {}).get("stored_chunks", 0)
                            st.success(f"Saved successfully ({stored} chunks).")
                        except APIClientError as e:
                            st.error(str(e))

        with st.expander("Teach via Document"):
            category_doc = st.selectbox("Document category", CATEGORY_OPTIONS, key="teach_doc_category")
            upload = st.file_uploader(
                "Upload file (.pdf, .docx, .txt)",
                type=["pdf", "docx", "txt"],
                key="teach_doc_file",
            )
            if st.button("Upload & Learn", key="teach_doc_btn", use_container_width=True):
                if upload is None:
                    st.warning("Please choose a file.")
                else:
                    with st.spinner("Ingesting document..."):
                        try:
                            result = client.upload_document(
                                file_name=upload.name,
                                file_bytes=upload.read(),
                                category=category_doc,
                            )
                            stored = result.get("data", {}).get("stored_chunks", 0)
                            st.success(f"Document ingested ({stored} chunks).")
                        except APIClientError as e:
                            st.error(str(e))

        with st.expander("Search Memory"):
            search_query = st.text_input("Memory query", key="memory_query")
            search_category = st.selectbox(
                "Search category",
                ["all"] + CATEGORY_OPTIONS,
                key="memory_search_category",
            )
            if st.button("Search", key="memory_search_btn", use_container_width=True):
                if not search_query.strip():
                    st.warning("Enter a query first.")
                else:
                    with st.spinner("Searching memory..."):
                        try:
                            result = client.search_memory(
                                query=search_query,
                                category=None if search_category == "all" else search_category,
                            )
                            st.session_state.memory_search_results = result.get("data", {}).get("matches", [])
                        except APIClientError as e:
                            st.error(str(e))
                            st.session_state.memory_search_results = []

            _render_search_results(st.session_state.memory_search_results)


def _render_search_results(matches: list[dict]) -> None:
    if not matches:
        st.caption("No memory results yet.")
        return

    st.markdown("**Results**")
    for idx, match in enumerate(matches, start=1):
        st.markdown(f"**{idx}. {match.get('source', 'unknown')}** ({match.get('category', 'general')})")
        st.code(match.get("text", ""), language="markdown")
