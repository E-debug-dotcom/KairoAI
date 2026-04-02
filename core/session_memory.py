"""
core/session_memory.py — Short-term conversational memory manager.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class SessionMemory:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_messages))

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


# singleton
session_memory = SessionMemory()
