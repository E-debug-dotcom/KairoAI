"""HTTP client layer for the Streamlit UI.

This module is the only place where the UI makes API calls.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx


class APIClientError(Exception):
    """Raised for user-facing API errors in the UI."""


class APIClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def send_chat(self, message: str, session_id: str = "streamlit-session", category: Optional[str] = None) -> dict[str, Any]:
        payload = {
            "message": message,
            "session_id": session_id,
            "category": category,
            "top_k": 3,
        }
        return self._post_json("/api/v1/assistant/chat", payload)

    def teach_text(self, text: str, category: str = "general") -> dict[str, Any]:
        payload = {
            "text": text,
            "source": "streamlit_text_input",
            "category": category,
            "tags": [],
        }
        return self._post_json("/api/v1/learn/text", payload)

    def upload_document(self, file_name: str, file_bytes: bytes, category: str = "general") -> dict[str, Any]:
        return self._post_file(
            "/api/v1/learn/upload",
            file_name=file_name,
            file_bytes=file_bytes,
            category=category,
        )

    def search_memory(self, query: str, category: Optional[str] = None, top_k: int = 5) -> dict[str, Any]:
        payload = {
            "query": query,
            "category": category,
            "top_k": top_k,
        }
        return self._post_json("/api/v1/learn/search", payload)

    def list_sources(self, category: Optional[str] = None, limit: int = 200) -> dict[str, Any]:
        payload = {"category": category, "limit": limit}
        return self._post_json("/api/v1/learn/sources", payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            detail = self._extract_detail(e.response)
            raise APIClientError(f"Request failed ({e.response.status_code}): {detail}") from e
        except httpx.RequestError as e:
            raise APIClientError(f"Cannot reach backend at {self.base_url}. Is the API running?") from e

    def _post_file(self, path: str, file_name: str, file_bytes: bytes, category: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        files = {"file": (file_name, file_bytes)}
        data = {"category": category, "tags": "streamlit-upload"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            detail = self._extract_detail(e.response)
            raise APIClientError(f"Upload failed ({e.response.status_code}): {detail}") from e
        except httpx.RequestError as e:
            raise APIClientError(f"Cannot reach backend at {self.base_url}. Is the API running?") from e

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            return body.get("detail") or body.get("error") or str(body)
        except Exception:
            return response.text[:400]
