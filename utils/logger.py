"""
utils/logger.py — Centralized logging for all system components.
Uses Python's standard logging with rotating file handler.
"""

import logging
import os
import contextvars
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import settings

# Context variable for per-request tracing
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, "request_id", request_id_ctx.get(""))
        return True


def set_request_id(request_id: str) -> None:
    request_id_ctx.set(request_id)


def clear_request_id() -> None:
    request_id_ctx.set("")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Returns a named logger configured with both console and rotating file handlers.
    Call this at the top of any module: logger = get_logger(__name__)
    """
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Attach request id filter to all handlers
    request_filter = RequestIdFilter()

    # ─── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)
    logger.addHandler(console_handler)

    # ─── Rotating file handler ─────────────────────────────────────────────────
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=settings.LOG_FILE,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_filter)
    logger.addHandler(file_handler)

    return logger


# Module-level convenience logger for quick imports
logger = get_logger("ai_system")
