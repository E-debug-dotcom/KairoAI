"""
storage/vector_store.py — Persistent vector memory using ChromaDB.

Provides simple ingest and semantic retrieval methods used by assistant and learning modules.
"""

from __future__ import annotations

import os
import uuid
from hashlib import sha256
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

        os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
        os.environ.setdefault(
            "CHROMA_PRODUCT_TELEMETRY_IMPL",
            "storage.chroma_telemetry.NoOpProductTelemetryClient",
        )
        os.environ.setdefault(
            "CHROMA_TELEMETRY_IMPL",
            "storage.chroma_telemetry.NoOpProductTelemetryClient",
        )
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            ) from e

        try:
            from chromadb.config import Settings as ChromaSettings
            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                chroma_telemetry_impl="storage.chroma_telemetry.NoOpProductTelemetryClient",
                chroma_product_telemetry_impl="storage.chroma_telemetry.NoOpProductTelemetryClient",
            )
            self._client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
                settings=chroma_settings,
            )
        except (ImportError, TypeError):
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
        extra_metadata: Optional[dict] = None,
    ) -> list[str]:
        """Bulk insert multiple text chunks with shared metadata."""
        self.init()
        valid_chunks = [c.strip() for c in chunks if c and c.strip()]
        if not valid_chunks:
            return []

        ids = [str(uuid.uuid4()) for _ in valid_chunks]
        base_metadata = {
            "source": source,
            "category": category,
            "tags": ",".join(tags or []),
        }
        if extra_metadata:
            for key, value in extra_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    base_metadata[key] = value

        self._collection.add(
            ids=ids,
            documents=valid_chunks,
            metadatas=[
                {
                    **base_metadata,
                    "chunk_index": idx,
                }
                for idx, _ in enumerate(valid_chunks)
            ],
        )
        return ids

    def document_hash(self, text: str) -> str:
        """Stable hash used for deduplication."""
        return sha256(text.encode("utf-8")).hexdigest()

    def has_document(self, doc_hash: str, source: Optional[str] = None) -> bool:
        """Check whether a source has already ingested this document hash."""
        self.init()
        if not doc_hash:
            return False

        where = {"doc_hash": doc_hash}
        if source:
            where = {"$and": [{"doc_hash": doc_hash}, {"source": source}]}

        result = self._collection.get(where=where, limit=1, include=[])
        return bool(result.get("ids"))

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

    def list_sources(self, category: Optional[str] = None, limit: int = 200) -> list[dict]:
        """List deduplicated sources currently stored in memory."""
        self.init()
        where = {"category": category} if category else None
        page_size = 500
        offset = 0
        metadatas: list[dict] = []

        while True:
            result = self._collection.get(
                where=where,
                include=["metadatas"],
                limit=page_size,
                offset=offset,
            )
            batch = result.get("metadatas", []) or []
            if not batch:
                break
            metadatas.extend(batch)
            offset += len(batch)
            if len({(m.get("source", "unknown"), m.get("category", "general")) for m in metadatas}) >= max(1, limit):
                break

        agg: dict[tuple[str, str], dict] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            source_category = meta.get("category", "general")
            key = (source, source_category)
            if key not in agg:
                agg[key] = {
                    "source": source,
                    "category": source_category,
                    "chunk_count": 0,
                    "doc_id": meta.get("doc_id"),
                }
            agg[key]["chunk_count"] += 1

        return sorted(agg.values(), key=lambda x: x["chunk_count"], reverse=True)[: max(1, limit)]


vector_store = VectorStore()
