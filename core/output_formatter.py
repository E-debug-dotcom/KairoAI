"""
core/output_formatter.py — Standardized response structure for all modules.

Every module returns its raw output through this formatter, ensuring
consistent JSON structure that downstream consumers can rely on.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"       # Completed with warnings
    ERROR = "error"


class OutputFormatter:
    """
    Wraps module outputs into a standardized envelope.

    Standard response shape:
    {
        "status": "success" | "partial" | "error",
        "task_type": "resume",
        "timestamp": "2025-01-15T10:30:00Z",
        "data": { ... },          <- main result payload
        "meta": { ... },          <- timing, model, token counts
        "warnings": [ ... ],      <- non-fatal issues
        "error": null | "..."     <- set only on error
    }
    """

    @staticmethod
    def success(
        task_type: str,
        data: dict,
        meta: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
    ) -> dict:
        """Build a successful response envelope."""
        return {
            "status": TaskStatus.SUCCESS,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": data,
            "meta": meta or {},
            "warnings": warnings or [],
            "error": None,
        }

    @staticmethod
    def partial(
        task_type: str,
        data: dict,
        warnings: list[str],
        meta: Optional[dict] = None,
    ) -> dict:
        """Build a partial-success response (completed with non-fatal issues)."""
        return {
            "status": TaskStatus.PARTIAL,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": data,
            "meta": meta or {},
            "warnings": warnings,
            "error": None,
        }

    @staticmethod
    def error(
        task_type: str,
        message: str,
        meta: Optional[dict] = None,
    ) -> dict:
        """Build an error response envelope."""
        logger.error("Formatting error response for task='%s': %s", task_type, message)
        return {
            "status": TaskStatus.ERROR,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": {},
            "meta": meta or {},
            "warnings": [],
            "error": message,
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def to_json(result: dict, indent: int = 2) -> str:
        """Serialize a formatted result to a JSON string."""
        return json.dumps(result, indent=indent, default=str)


# ─── Module-level singleton ───────────────────────────────────────────────────
formatter = OutputFormatter()
