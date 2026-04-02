"""
core/task_router.py — Central dispatcher for all task types.

Modules register themselves here. The router accepts a task_type string
and routes the request to the correct handler. Adding a new module
requires only one line in TASK_REGISTRY.
"""

import time
from typing import Any, Callable, Optional

from config import settings
from utils.logger import get_logger
from utils.helpers import span

logger = get_logger(__name__)


class TaskRouter:
    """
    Routes incoming task requests to registered module handlers.

    Each handler is a callable that accepts a dict payload and returns
    a structured result dict. This keeps the core layer clean of any
    module-specific logic.

    Usage:
        router = TaskRouter()
        result = await router.dispatch("resume", payload)
    """

    TASK_ALIASES = {
        "job": "job_application",
        "job_app": "job_application",
        "job_application": "job_application",
        "assistant": "assistant",
        "resume": "resume",
        "resume_coach": "resume_coach",
        "code": "code",
        "code_gen": "code_gen",
        "learning": "learning",
    }

    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, task_type: str, handler: Callable) -> None:
        """
        Register a task handler.

        Args:
            task_type: Unique string key (e.g., "resume", "assistant").
            handler: Async callable that processes the task payload.
        """
        if task_type in self._registry:
            logger.warning("Overwriting existing handler for task_type='%s'", task_type)
        self._registry[task_type] = handler
        logger.debug("Registered handler for task_type='%s'", task_type)

    def _normalize_task_type(self, task_type: str) -> str:
        requested = task_type.strip().lower()
        return self.TASK_ALIASES.get(requested, requested)

    async def dispatch(self, task_type: str, payload: dict) -> dict:
        """
        Route a request to the appropriate module handler.

        Args:
            task_type: The type of task to execute.
            payload: All input data for the task.

        Returns:
            Structured result dict from the module handler.

        Raises:
            TaskNotFoundError: If task_type has no registered handler.
        """
        dispatch_start = time.time()
        normalized = self._normalize_task_type(task_type)

        lookup_start = time.time()
        handler = self._registry.get(normalized)
        lookup_time_ms = round((time.time() - lookup_start) * 1000, 2)

        if handler is None:
            available = sorted(self._registry.keys())
            logger.error("Unknown task_type='%s'. Available: %s", task_type, available)
            raise TaskNotFoundError(
                f"No handler registered for task_type='{task_type}'. "
                f"Available tasks: {available}"
            )

        logger.debug(
            "span_task_router_dispatch | task_type=%s normalized_key=%s lookup_time_ms=%.2f",
            task_type,
            normalized,
            lookup_time_ms,
        )

        handler_start = time.time()
        try:
            result = await handler(payload)
            handler_time_ms = round((time.time() - handler_start) * 1000, 2)
            sub_task = payload.get("sub_task", "")
            logger.info(
                "span_module_handler | task_type=%s sub_task=%s handler_time_ms=%.2f",
                normalized,
                sub_task,
                handler_time_ms,
            )
            return result
        except Exception as e:
            logger.error("Task '%s' failed: %s", normalized, str(e), exc_info=True)
            raise
        finally:
            total_time_ms = round((time.time() - dispatch_start) * 1000, 2)
            logger.debug(
                "span_task_router_total | task_type=%s total_time_ms=%.2f",
                normalized,
                total_time_ms,
            )

    async def route_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """
        Route a tool call (from LLM) to the appropriate handler.

        Maps tool_name to task_type and invokes dispatch.

        Args:
            tool_name: Name of the tool to invoke (e.g., "resume_coach").
            tool_input: Tool arguments dict.

        Returns:
            Structured result dict from the module handler.

        Raises:
            TaskNotFoundError: If tool is not registered.
        """
        normalized = self._normalize_task_type(tool_name)

        if not settings.ENABLE_TOOL_USE:
            raise RuntimeError("Tool use is disabled (ENABLE_TOOL_USE=False)")

        logger.debug(
            "span_route_tool_call | tool_name=%s tool_args_keys=%s",
            tool_name,
            list(tool_input.keys()),
        )

        # Invoke dispatch with tool_input as payload
        return await self.dispatch(normalized, tool_input)

    def available_tasks(self) -> list[str]:
        """Return a sorted list of all registered task types."""
        return sorted(self._registry.keys())

    def is_registered(self, task_type: str) -> bool:
        return task_type.strip().lower() in self._registry


class TaskNotFoundError(Exception):
    """Raised when the router receives an unknown task_type."""
    pass


# ─── Singleton router — all routes import this and register against it ────────
router = TaskRouter()
