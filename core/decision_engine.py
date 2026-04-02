"""
core/decision_engine.py — Decide whether to use memory/tool/LLM pipeline for a task.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from storage.vector_store import vector_store
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionOutcome:
    decision_type: str
    reason: str
    memory_score: float
    use_memory: bool
    use_llm: bool
    use_tools: bool


class DecisionEngine:
    MEMORY_THRESHOLD = 0.75
    SHORT_QUERY_THRESHOLD = 120

    def evaluate(self, task_type: str, payload: Dict[str, Any]) -> DecisionOutcome:
        start = time.time()

        task_type = task_type.strip().lower()
        query_text = payload.get("query") or payload.get("question") or payload.get("input") or ""
        session_id = payload.get("session_id")

        # 1) Basic validation rules
        if task_type == "resume":
            if not payload.get("job_description"):
                raise ValueError("resume tasks require job_description")

        if payload.get("force_llm"):
            outcome = DecisionOutcome(
                decision_type="llm_only",
                reason="forced llm",
                memory_score=0.0,
                use_memory=False,
                use_llm=True,
                use_tools=False,
            )
            self._log(outcome, start)
            return outcome

        if payload.get("force_tools"):
            outcome = DecisionOutcome(
                decision_type="tools_llm",
                reason="forced tools",
                memory_score=0.0,
                use_memory=False,
                use_llm=True,
                use_tools=True,
            )
            self._log(outcome, start)
            return outcome

        # 2) Memory similarity scoring
        memory_score = 0.0
        if query_text.strip():
            try:
                results = vector_store.query(query_text=query_text, top_k=3)
                if results:
                    best = results[0]
                    distance = best.get("distance")
                    if distance is not None and isinstance(distance, (int, float)):
                        memory_score = 1.0 - min(max(distance, 0.0), 1.0)
            except Exception as e:
                logger.warning("DecisionEngine memory query failed: %s", str(e))

        if task_type in ["assistant", "learning"] and memory_score >= self.MEMORY_THRESHOLD:
            outcome = DecisionOutcome(
                decision_type="memory_only",
                reason=f"high memory score {memory_score:.3f}",
                memory_score=memory_score,
                use_memory=True,
                use_llm=False,
                use_tools=False,
            )
            self._log(outcome, start)
            return outcome

        if not query_text.strip() or len(query_text) < self.SHORT_QUERY_THRESHOLD:
            outcome = DecisionOutcome(
                decision_type="llm_only",
                reason="short query or empty input",
                memory_score=memory_score,
                use_memory=False,
                use_llm=True,
                use_tools=False,
            )
            self._log(outcome, start)
            return outcome

        # default to full pipeline
        outcome = DecisionOutcome(
            decision_type="full_pipeline",
            reason="default route",
            memory_score=memory_score,
            use_memory=True,
            use_llm=True,
            use_tools=False,
        )
        self._log(outcome, start)
        return outcome

    def _log(self, outcome: DecisionOutcome, start_time: float) -> None:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.debug(
            "span_decision_engine | decision_type=%s memory_score=%.3f reason=%s latency_ms=%.2f",
            outcome.decision_type,
            outcome.memory_score,
            outcome.reason,
            latency_ms,
        )
        logger.info(
            "span_decision_engine_agg | decision_type=%s memory_score=%.3f latency_ms=%.2f",
            outcome.decision_type,
            outcome.memory_score,
            latency_ms,
        )


# singleton
decision_engine = DecisionEngine()
