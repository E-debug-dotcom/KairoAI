"""
modules/assistant/handler.py — General-purpose AI assistant module.

Handles:
- Technical Q&A (IT, cybersecurity, scripting)
- Concept explanations at configurable depth levels
- Multi-turn context (caller passes history manually)
"""

import time
from typing import Optional

from core.llm_service import llm_service
from core.output_formatter import formatter
from prompts.prompt_manager import prompt_manager
from prompts.assistant_prompts import ASSISTANT_SYSTEM_PROMPT
from storage.database import db
from storage.vector_store import vector_store
from utils.logger import get_logger
from utils.helpers import truncate_text

logger = get_logger(__name__)


class AssistantHandler:
    """
    General-purpose assistant routing to query or explain sub-tasks.

    Expected payload keys:
        sub_task (str): "query" | "explain"
        question (str): The user's question
        topic (str): For "explain" sub-task — the concept to explain
        level (str): "beginner" | "intermediate" | "expert" (default: intermediate)
        context (str, optional): Additional background context
        history (list, optional): Previous turn [{role, content}, ...] for multi-turn
    """

    async def handle(self, payload: dict) -> dict:
        sub_task = payload.get("sub_task", "query")
        logger.info("Assistant module handling sub_task='%s'", sub_task)

        dispatch = {
            "query": self._query,
            "explain": self._explain,
            "chat": self._chat,
        }

        if sub_task not in dispatch:
            return formatter.error(
                "assistant",
                f"Unknown sub_task '{sub_task}'. Available: {list(dispatch.keys())}",
            )

        return await dispatch[sub_task](payload)

    # ─── Sub-task: Query ──────────────────────────────────────────────────────

    async def _query(self, payload: dict) -> dict:
        question = payload.get("question", "").strip()
        if not question:
            return formatter.error("assistant", "Field 'question' is required.")

        context = payload.get("context", "").strip()
        history = payload.get("history", [])

        # Build context block if provided
        context_block = f"\nADDITIONAL CONTEXT:\n{truncate_text(context, 500)}" if context else ""

        # Build multi-turn history prefix if provided
        history_block = self._format_history(history)

        try:
            prompt = prompt_manager.render(
                "assistant", "query",
                question=question,
                context_block=context_block,
            )
        except Exception as e:
            return formatter.error("assistant", f"Prompt error: {str(e)}")

        # Prepend history to prompt for multi-turn support
        if history_block:
            prompt = f"{history_block}\n\n{prompt}"

        start = time.time()
        try:
            response = llm_service.complete(
                prompt=prompt,
                system_prompt=ASSISTANT_SYSTEM_PROMPT,
            )
        except Exception as e:
            return formatter.error("assistant", f"LLM request failed: {str(e)}")
        duration = time.time() - start

        db.save_task(
            task_type="assistant",
            input_summary={"sub_task": "query", "question": question[:200]},
            result=formatter.success("assistant", {"answer": response}),
            duration_seconds=duration,
            model_used=llm_service.model,
        )

        return formatter.success(
            "assistant",
            {"answer": response, "question": question},
            meta={"duration_seconds": round(duration, 2), "model": llm_service.model},
        )

    # ─── Sub-task: Explain ────────────────────────────────────────────────────

    async def _explain(self, payload: dict) -> dict:
        topic = payload.get("topic", "").strip()
        if not topic:
            return formatter.error("assistant", "Field 'topic' is required for explain sub-task.")

        level = payload.get("level", "intermediate")
        if level not in ("beginner", "intermediate", "expert"):
            level = "intermediate"

        context = payload.get("context", "").strip()
        context_block = f"\nCONTEXT:\n{context}" if context else ""

        try:
            prompt = prompt_manager.render(
                "assistant", "explain",
                topic=topic,
                level=level,
                context_block=context_block,
            )
        except Exception as e:
            return formatter.error("assistant", f"Prompt error: {str(e)}")

        start = time.time()
        try:
            response = llm_service.complete(
                prompt=prompt,
                system_prompt=ASSISTANT_SYSTEM_PROMPT,
            )
        except Exception as e:
            return formatter.error("assistant", f"LLM request failed: {str(e)}")
        duration = time.time() - start

        return formatter.success(
            "assistant",
            {"explanation": response, "topic": topic, "level": level},
            meta={"duration_seconds": round(duration, 2), "model": llm_service.model},
        )


    async def _chat(self, payload: dict) -> dict:
        message = payload.get("message", "").strip()
        if not message:
            return formatter.error("assistant", "Field 'message' is required.")

        session_id = payload.get("session_id", "default")
        category = payload.get("category")
        top_k = int(payload.get("top_k", 4))

        memory_items = vector_store.query(query_text=message, top_k=top_k, category=category)
        memory_block = "\n".join(
            f"- ({item['category']} | {item['source']}) {truncate_text(item['text'], 400)}"
            for item in memory_items
        )

        prompt = prompt_manager.render(
            "assistant",
            "chat",
            message=message,
            memory_block=memory_block or "No prior knowledge retrieved.",
        )

        start = time.time()
        answer = llm_service.complete(prompt=prompt, system_prompt=ASSISTANT_SYSTEM_PROMPT)
        duration = time.time() - start

        db.save_task(
            task_type="assistant",
            input_summary={"sub_task": "chat", "session_id": session_id, "message": message[:200]},
            result=formatter.success("assistant", {"answer": answer}),
            duration_seconds=duration,
            model_used=llm_service.model,
        )

        return formatter.success(
            "assistant",
            {
                "answer": answer,
                "session_id": session_id,
                "retrieved_memory": memory_items,
            },
            meta={"duration_seconds": round(duration, 2), "model": llm_service.model},
        )

    # ─── Helper ───────────────────────────────────────────────────────────────

    def _format_history(self, history: list[dict]) -> str:
        """
        Format conversation history for inclusion in prompt.
        Expects: [{"role": "user"|"assistant", "content": "..."}, ...]
        """
        if not history:
            return ""

        lines = ["=== PREVIOUS CONVERSATION ==="]
        for turn in history[-6:]:  # Cap at last 6 turns to manage context size
            role = turn.get("role", "user").upper()
            content = truncate_text(turn.get("content", ""), 300)
            lines.append(f"{role}: {content}")
        lines.append("=== END PREVIOUS CONVERSATION ===")
        return "\n".join(lines)


# ─── Singleton ────────────────────────────────────────────────────────────────
assistant_handler = AssistantHandler()
