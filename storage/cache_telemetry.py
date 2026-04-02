"""
storage/cache_telemetry.py — Tracks prompt caching metrics.

Records cache hit/miss rates, token savings, and latency improvements.
Helps measure ROI of prompt caching across sessions.
"""

import time
from dataclasses import dataclass, asdict
from collections import deque
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheMetrics:
    """Single cache operation metrics."""
    timestamp: float
    session_id: str
    task_type: str
    cache_key_hash: str
    hit: bool
    tokens_input: int
    tokens_if_no_cache: int
    tokens_saved: int
    latency_ms: float
    first_token_latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class CacheTelemetry:
    """Collects and aggregates cache metrics."""

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.metrics: deque = deque(maxlen=max_records)
        self._hit_count = 0
        self._miss_count = 0
        self._total_tokens_saved = 0

    def record(self, metrics: CacheMetrics) -> None:
        """Record a cache operation."""
        self.metrics.append(metrics)
        if metrics.hit:
            self._hit_count += 1
        else:
            self._miss_count += 1
        self._total_tokens_saved += metrics.tokens_saved
        
        logger.debug(
            "span_cache_metric | session=%s task=%s hit=%s tokens_saved=%d latency_ms=%.2f",
            metrics.session_id,
            metrics.task_type,
            metrics.hit,
            metrics.tokens_saved,
            metrics.latency_ms,
        )

    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def total_tokens_saved(self) -> int:
        """Return total tokens saved by caching."""
        return self._total_tokens_saved

    def average_latency_ms(self, hit_only: bool = False) -> float:
        """Return average latency for cache operations."""
        if not self.metrics:
            return 0.0

        if hit_only:
            hit_metrics = [m for m in self.metrics if m.hit]
            if not hit_metrics:
                return 0.0
            return sum(m.latency_ms for m in hit_metrics) / len(hit_metrics)

        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)

    def stats_by_task(self) -> Dict[str, dict]:
        """Return aggregated stats per task type."""
        stats = {}
        for metric in self.metrics:
            task = metric.task_type
            if task not in stats:
                stats[task] = {
                    "hit_count": 0,
                    "miss_count": 0,
                    "tokens_saved": 0,
                    "total_latency_ms": 0.0,
                    "call_count": 0,
                }
            stats[task]["hit_count"] += 1 if metric.hit else 0
            stats[task]["miss_count"] += 0 if metric.hit else 1
            stats[task]["tokens_saved"] += metric.tokens_saved
            stats[task]["total_latency_ms"] += metric.latency_ms
            stats[task]["call_count"] += 1

        # Compute averages
        for task, data in stats.items():
            data["hit_rate"] = (
                data["hit_count"] / data["call_count"] if data["call_count"] > 0 else 0.0
            )
            data["avg_latency_ms"] = (
                data["total_latency_ms"] / data["call_count"] if data["call_count"] > 0 else 0.0
            )

        return stats

    def reset(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._total_tokens_saved = 0

    def to_dict(self) -> dict:
        """Export telemetry as a dict."""
        return {
            "hit_rate": self.hit_rate(),
            "total_hits": self._hit_count,
            "total_misses": self._miss_count,
            "total_tokens_saved": self._total_tokens_saved,
            "avg_latency_ms": self.average_latency_ms(),
            "avg_hit_latency_ms": self.average_latency_ms(hit_only=True),
            "stats_by_task": self.stats_by_task(),
            "total_operations": len(self.metrics),
        }


# ─── Module-level singleton ───────────────────────────────────────────────────
cache_telemetry = CacheTelemetry()
