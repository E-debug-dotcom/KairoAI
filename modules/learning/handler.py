"""
modules/learning/handler.py — Teachable memory module.

Supports ingesting user notes/documents and searching stored knowledge.
"""

from __future__ import annotations

import time

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
            "teach_dataset": self._teach_dataset,
            "search": self._search,
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
        ids = vector_store.add_chunks(chunks, source=source, category=category, tags=tags)
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
        ids = vector_store.add_chunks(chunks, source=filename, category=category, tags=tags)
        return formatter.success(
            "learning",
            {
                "stored_chunks": len(ids),
                "source": filename,
                "category": category,
            },
        )

    async def _teach_dataset(self, payload: dict) -> dict:
        items = payload.get("items")
        dataset_name = payload.get("dataset_name", "dataset")
        default_category = payload.get("category", "general")

        if not items or not isinstance(items, list):
            return formatter.error(
                "learning",
                "Field 'items' is required and must be a list of dataset documents.",
            )

        if len(items) == 0:
            return formatter.error("learning", "Dataset contains no items to ingest.")

        start = time.time()
        total_chunks = 0
        summary = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            content = (item.get("content") or item.get("text") or "").strip()
            if not content:
                continue

            source = item.get("source") or item.get("title") or f"{dataset_name}_{index+1}"
            category = item.get("category") or default_category or "general"
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                tags = [str(tags)]

            chunks = self._chunk_text(content)
            ids = vector_store.add_chunks(chunks, source=source, category=category, tags=tags)
            total_chunks += len(ids)

            summary.append(
                {
                    "source": source,
                    "category": category,
                    "stored_chunks": len(ids),
                }
            )

        if total_chunks == 0:
            return formatter.error(
                "learning",
                "No valid dataset items were ingested. Ensure each item has non-empty content.",
            )

        duration = time.time() - start
        db.save_task(
            task_type="learning",
            input_summary={
                "sub_task": "teach_dataset",
                "dataset_name": dataset_name,
                "items_ingested": len(summary),
            },
            result=formatter.success("learning", {"stored_chunks": total_chunks}),
            duration_seconds=duration,
        )

        return formatter.success(
            "learning",
            {
                "dataset_name": dataset_name,
                "items_ingested": len(summary),
                "total_chunks": total_chunks,
                "summary": summary,
            },
            meta={"duration_seconds": round(duration, 2)},
        )

    async def _search(self, payload: dict) -> dict:
        query = payload.get("query", "").strip()
        category = payload.get("category")
        top_k = int(payload.get("top_k", 5))

        if not query:
            return formatter.error("learning", "Field 'query' is required.")

        items = vector_store.query(query_text=query, top_k=top_k, category=category)
        return formatter.success("learning", {"matches": items, "query": query})

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
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
