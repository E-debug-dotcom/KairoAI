"""
core/session_memory.py — Short-term conversational memory manager with caching support.
"""

import time
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class SessionMemory:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_messages))
        self._session_metadata: Dict[str, dict] = defaultdict(dict)

    def append_user(self, session_id: str, text: str) -> None:
        if not session_id or not text:
            return
        self._sessions[session_id].append({"role": "user", "text": text, "ts": time.time()})

    def append_assistant(self, session_id: str, text: str) -> None:
        if not session_id or not text:
            return
        self._sessions[session_id].append({"role": "assistant", "text": text, "ts": time.time()})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def prune_low_value(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return

        clean_queue = deque(maxlen=self.max_messages)
        for event in self._sessions[session_id]:
            text = event.get("text", "").strip().lower()
            if text in ("ok", "thanks", "thank you", "got it"):
                continue
            clean_queue.append(event)

        self._sessions[session_id] = deque(clean_queue, maxlen=self.max_messages)

    # ─── Caching support ──────────────────────────────────────────────────────

    def set_cached_context(self, session_id: str, context_key: str, context_value: any) -> None:
        """Store cached context (e.g., user profile, task definitions)."""
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {}
        self._session_metadata[session_id][context_key] = {
            "value": context_value,
            "ts": time.time(),
        }
        logger.debug("Cached context for session %s: %s", session_id, context_key)

    def get_cached_context(self, session_id: str, context_key: str) -> Optional[any]:
        """Retrieve cached context."""
        metadata = self._session_metadata.get(session_id, {})
        context = metadata.get(context_key)
        return context["value"] if context else None

    def compute_cache_key(self, session_id: str, stable_fields: dict) -> str:
        """
        Compute a cache key hash from session metadata and stable fields.

        This hash represents the "cacheable context" for prompt caching.
        If this hash doesn't change, the prompt context is stable and can be cached.

        Args:
            session_id: Session identifier
            stable_fields: Dict of fields that identify the cache slot

        Returns:
            SHA256 hash of the combined fields
        """
        import json

        metadata = self._session_metadata.get(session_id, {})
        cache_data = {
            "session_id": session_id,
            "stable_fields": stable_fields,
            "metadata_keys": sorted(metadata.keys()),
        }

        cache_json = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_json.encode()).hexdigest()[:16]

    def get_cacheable_context(self, session_id: str) -> dict:
        """
        Export all cacheable context for this session.

        Returns a dict suitable for caching with Claude prompt caching.
        """
        return {
            "session_id": session_id,
            "history_summary": f"Session with {len(self.get_history(session_id))} messages",
            "metadata": self._session_metadata.get(session_id, {}),
        }

    def reset_session(self, session_id: str) -> None:
        """Clear all data for a session (useful for testing)."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_metadata:
            del self._session_metadata[session_id]


# singleton
session_memory = SessionMemory()
