"""
core/task_router.py — Central dispatcher for all task types.

Modules register themselves here. The router accepts a task_type string
and routes the request to the correct handler. Adding a new module
requires only one line in TASK_REGISTRY.
"""

from typing import Any, Callable, Optional

from utils.logger import get_logger

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
        normalized = task_type.strip().lower()

        if normalized not in self._registry:
            available = sorted(self._registry.keys())
            logger.error("Unknown task_type='%s'. Available: %s", task_type, available)
            raise TaskNotFoundError(
                f"No handler registered for task_type='{task_type}'. "
                f"Available tasks: {available}"
            )

        handler = self._registry[normalized]
        logger.info("Dispatching task_type='%s'", normalized)

        try:
            result = await handler(payload)
            logger.info("Task '%s' completed successfully", normalized)
            return result
        except Exception as e:
            logger.error("Task '%s' failed: %s", normalized, str(e), exc_info=True)
            raise

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
