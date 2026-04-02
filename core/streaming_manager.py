"""
core/streaming_manager.py — Manages real-time streaming responses via SSE.

Handles server-sent events (SSE) for token-by-token LLM response delivery.
"""

import asyncio
import json
from typing import AsyncGenerator, Callable, Optional
from datetime import datetime, timezone

from utils.logger import get_logger

logger = get_logger(__name__)


class StreamingSession:
    """Manages a single streaming response session."""

    def __init__(self, session_id: str, task_type: str):
        self.session_id = session_id
        self.task_type = task_type
        self.created_at = datetime.now(timezone.utc)
        self.tokens: list[str] = []
        self.is_complete = False
        self.error: Optional[str] = None
        self._subscribers: list[asyncio.Queue] = []

    def add_subscriber(self, queue: asyncio.Queue) -> None:
        """Register a queue to receive token updates."""
        self._subscribers.append(queue)

    async def emit_token(self, token: str) -> None:
        """Broadcast a token to all subscribers."""
        self.tokens.append(token)
        for queue in self._subscribers:
            try:
                await queue.put({"type": "token", "data": token})
            except asyncio.QueueFull:
                logger.warning("Queue full for session %s; dropping token", self.session_id)

    async def emit_done(self, data: Optional[dict] = None) -> None:
        """Signal that streaming is complete."""
        self.is_complete = True
        for queue in self._subscribers:
            try:
                await queue.put({"type": "done", "data": data or {}})
            except asyncio.QueueFull:
                logger.warning("Queue full for session %s; dropping done signal", self.session_id)

    async def emit_error(self, error_msg: str) -> None:
        """Broadcast an error to all subscribers."""
        self.error = error_msg
        for queue in self._subscribers:
            try:
                await queue.put({"type": "error", "data": error_msg})
            except asyncio.QueueFull:
                logger.warning("Queue full for session %s; dropping error", self.session_id)

    def full_text(self) -> str:
        """Return the full accumulated text."""
        return "".join(self.tokens)

    def to_dict(self) -> dict:
        """Export session metadata."""
        return {
            "session_id": self.session_id,
            "task_type": self.task_type,
            "created_at": self.created_at.isoformat(),
            "tokens": len(self.tokens),
            "is_complete": self.is_complete,
            "error": self.error,
            "full_text": self.full_text(),
        }


class StreamingManager:
    """Manages active streaming sessions."""

    def __init__(self, max_sessions: int = 100):
        self.sessions: dict[str, StreamingSession] = {}
        self.max_sessions = max_sessions

    def create_session(self, session_id: str, task_type: str) -> StreamingSession:
        """Create a new streaming session."""
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest = min(self.sessions.values(), key=lambda s: s.created_at)
            logger.warning(
                "Max streaming sessions reached; removing oldest: %s",
                oldest.session_id,
            )
            del self.sessions[oldest.session_id]

        session = StreamingSession(session_id, task_type)
        self.sessions[session_id] = session
        logger.debug("Created streaming session %s for task %s", session_id, task_type)
        return session

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Retrieve an active session."""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove a completed session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug("Removed streaming session %s", session_id)

    def active_sessions(self) -> list[str]:
        """Return list of active session IDs."""
        return list(self.sessions.keys())

    def session_count(self) -> int:
        """Return number of active sessions."""
        return len(self.sessions)


async def stream_with_sinks(
    generator: AsyncGenerator[str, None],
    on_token: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_done: Optional[Callable[[], None]] = None,
    chunk_size: int = 10,
) -> str:
    """
    Consume a token-generating async generator with callbacks for batching.

    Args:
        generator: Async generator that yields tokens
        on_token: Called for each individual token
        on_chunk: Called when chunk_size tokens are accumulated
        on_done: Called when generator is exhausted
        chunk_size: Number of tokens to batch before calling on_chunk

    Returns:
        Full concatenated text
    """
    full_text = []
    chunk = []

    try:
        async for token in generator:
            full_text.append(token)
            chunk.append(token)

            if on_token:
                on_token(token)

            if len(chunk) >= chunk_size:
                chunk_text = "".join(chunk)
                if on_chunk:
                    on_chunk(chunk_text)
                chunk = []

        # Flush remaining chunk
        if chunk and on_chunk:
            on_chunk("".join(chunk))

        if on_done:
            on_done()

    except Exception as e:
        logger.error("Error during streaming: %s", str(e))
        raise

    return "".join(full_text)


# ─── Module-level singleton ───────────────────────────────────────────────────
streaming_manager = StreamingManager()
