import time
from typing import Dict, Any


class Metrics:
    """
    Lightweight metrics for RAG API monitoring.
    Tracks request latency and cache performance.
    """

    def __init__(self):
        self.calls = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    def start_timer(self) -> float:
        """Return timestamp for latency tracking."""
        return time.time()

    def end_timer(self, start: float) -> float:
        """Record latency measurement."""
        latency = time.time() - start
        self.calls += 1
        self.total_latency += latency
        return latency

    def record_cache_hit(self):
        """Track cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Track cache miss."""
        self.cache_misses += 1

    def stats(self) -> Dict[str, Any]:
        """Return aggregated metrics for monitoring dashboards."""
        avg_latency = self.total_latency / self.calls if self.calls else 0

        total_queries = self.cache_hits + self.cache_misses

        hit_rate = (
            self.cache_hits / total_queries
            if total_queries > 0
            else 0
        )

        return {
            "calls": self.calls,
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(hit_rate, 3)
        }
