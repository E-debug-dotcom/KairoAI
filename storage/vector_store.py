"""
storage/vector_store.py — Persistent vector memory using ChromaDB.

Provides simple ingest and semantic retrieval methods used by assistant and learning modules.
"""

from __future__ import annotations

import uuid
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Wrapper around ChromaDB for teachable memory operations."""

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._initialized = False

    def init(self) -> None:
        """Initialize the persistent Chroma client and collection lazily."""
        if self._initialized:
            return

        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            ) from e

        self._client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        self._collection = self._client.get_or_create_collection(
            name=settings.VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._initialized = True
        logger.info(
            "Vector store initialized at %s (collection=%s)",
            settings.VECTOR_DB_PATH,
            settings.VECTOR_COLLECTION_NAME,
        )

    def add_text(
        self,
        text: str,
        source: str,
        category: str = "general",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Add a text chunk to memory and return its generated record id."""
        self.init()

        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Cannot store empty text in vector memory")

        record_id = str(uuid.uuid4())
        self._collection.add(
            ids=[record_id],
            documents=[cleaned],
            metadatas=[
                {
                    "source": source,
                    "category": category,
                    "tags": ",".join(tags or []),
                }
            ],
        )
        return record_id

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        category: str = "general",
        tags: Optional[list[str]] = None,
    ) -> list[str]:
        """Bulk insert multiple text chunks with shared metadata."""
        self.init()
        valid_chunks = [c.strip() for c in chunks if c and c.strip()]
        if not valid_chunks:
            return []

        ids = [str(uuid.uuid4()) for _ in valid_chunks]
        self._collection.add(
            ids=ids,
            documents=valid_chunks,
            metadatas=[
                {
                    "source": source,
                    "category": category,
                    "tags": ",".join(tags or []),
                }
                for _ in valid_chunks
            ],
        )
        return ids

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search over stored memory with optional category filter."""
        self.init()
        if not query_text.strip():
            return []

        where = {"category": category} if category else None
        results = self._collection.query(
            query_texts=[query_text],
            n_results=max(1, top_k),
            where=where,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        response: list[dict] = []
        for idx, doc in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            distance = distances[idx] if idx < len(distances) else None
            response.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "category": meta.get("category", "general"),
                    "tags": (meta.get("tags", "") or "").split(",") if meta.get("tags") else [],
                    "distance": distance,
                }
            )
        return response


vector_store = VectorStore()
