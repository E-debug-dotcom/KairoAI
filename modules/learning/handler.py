"""
modules/learning/handler.py — Teachable memory module.

Supports ingesting user notes/documents and searching stored knowledge.
"""

from __future__ import annotations

import time
import uuid

from core.output_formatter import formatter
from parsers.document_parser import document_parser
from storage.database import db
from storage.vector_store import vector_store
from utils.helpers import truncate_text
from utils.logger import get_logger

logger = get_logger(__name__)


class LearningHandler:
    """Ingest and retrieve teachable knowledge artifacts."""

    async def handle(self, payload: dict) -> dict:
        sub_task = payload.get("sub_task", "teach_text")
        dispatch = {
            "teach_text": self._teach_text,
            "teach_document": self._teach_document,
            "search": self._search,
            "list_sources": self._list_sources,
        }
        if sub_task not in dispatch:
            return formatter.error("learning", f"Unknown sub_task '{sub_task}'.")
        return await dispatch[sub_task](payload)

    async def _teach_text(self, payload: dict) -> dict:
        text = payload.get("text", "").strip()
        source = payload.get("source", "user_note")
        category = payload.get("category", "general")
        tags = payload.get("tags", [])

        if not text:
            return formatter.error("learning", "Field 'text' is required.")

        start = time.time()
        chunks = self._chunk_text(text)
        doc_hash = vector_store.document_hash(text)
        if vector_store.has_document(doc_hash=doc_hash, source=source):
            return formatter.success(
                "learning",
                {
                    "stored_chunks": 0,
                    "source": source,
                    "category": category,
                    "preview": truncate_text(text, 200),
                    "skipped_existing": True,
                },
                meta={"duration_seconds": 0.0},
            )

        ids = vector_store.add_chunks(
            chunks,
            source=source,
            category=category,
            tags=tags,
            extra_metadata={"doc_hash": doc_hash, "doc_id": str(uuid.uuid4())},
        )
        duration = time.time() - start

        db.save_task(
            task_type="learning",
            input_summary={"sub_task": "teach_text", "source": source, "category": category},
            result=formatter.success("learning", {"stored_chunks": len(ids)}),
            duration_seconds=duration,
        )

        return formatter.success(
            "learning",
            {
                "stored_chunks": len(ids),
                "source": source,
                "category": category,
                "preview": truncate_text(text, 200),
            },
            meta={"duration_seconds": round(duration, 2)},
        )

    async def _teach_document(self, payload: dict) -> dict:
        filename = payload.get("filename", "")
        content = payload.get("content")
        category = payload.get("category", "general")
        tags = payload.get("tags", [])

        if not filename or content is None:
            return formatter.error("learning", "Fields 'filename' and 'content' are required.")

        try:
            text = document_parser.parse_upload(filename=filename, content=content)
        except Exception as e:
            return formatter.error("learning", f"Document parse failed: {str(e)}")

        chunks = self._chunk_text(text)
        doc_hash = vector_store.document_hash(text)
        if vector_store.has_document(doc_hash=doc_hash, source=filename):
            return formatter.success(
                "learning",
                {
                    "stored_chunks": 0,
                    "source": filename,
                    "category": category,
                    "skipped_existing": True,
                },
            )

        ids = vector_store.add_chunks(
            chunks,
            source=filename,
            category=category,
            tags=tags,
            extra_metadata={"doc_hash": doc_hash, "doc_id": str(uuid.uuid4())},
        )
        return formatter.success(
            "learning",
            {
                "stored_chunks": len(ids),
                "source": filename,
                "category": category,
                "document_hash": doc_hash,
            },
        )

    async def _search(self, payload: dict) -> dict:
        query = payload.get("query", "").strip()
        category = payload.get("category")
        top_k = int(payload.get("top_k", 5))

        if not query:
            return formatter.error("learning", "Field 'query' is required.")

        items = vector_store.query(query_text=query, top_k=top_k, category=category)
        return formatter.success("learning", {"matches": items, "query": query})

    async def _list_sources(self, payload: dict) -> dict:
        category = payload.get("category")
        limit = int(payload.get("limit", 200))
        items = vector_store.list_sources(category=category, limit=limit)
        return formatter.success("learning", {"sources": items, "count": len(items)})

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 90) -> list[str]:
        """Simple sliding-window chunking for long ingested documents."""
        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks


learning_handler = LearningHandler()
