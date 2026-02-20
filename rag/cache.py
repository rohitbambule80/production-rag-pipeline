from typing import Dict, List, Optional
from datetime import datetime, timedelta


class SimpleCache:
    """
    In-memory LRU-style cache for RAG queries.
    Reduces latency/cost for repeated queries.
    """

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        self.store: Dict[str, dict] = {}
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes

    def _is_expired(self, entry: dict) -> bool:
        """Check if cached result expired."""
        if 'expires_at' not in entry:
            return True
        return datetime.now() > entry['expires_at']

    def get(self, query: str) -> Optional[List[str]]:
        """Retrieve cached context chunks for query."""
        if query not in self.store:
            return None

        entry = self.store[query]
        if self._is_expired(entry):
            del self.store[query]
            return None

        return entry['context']

    def set(self, query: str, result: List[str]):
        """Cache query result with TTL."""
        if len(self.store) >= self.max_size:
            # Simple eviction: remove oldest (first key)
            oldest_key = next(iter(self.store))
            del self.store[oldest_key]

        expires_at = datetime.now() + timedelta(minutes=self.ttl_minutes)
        self.store[query] = {
            'context': result,
            'expires_at': expires_at,
            'cached_at': datetime.now()
        }

    def stats(self) -> dict:
        """Cache statistics."""
        active = sum(1 for entry in self.store.values()
                     if not self._is_expired(entry))
        return {
            'total_entries': len(self.store),
            'active_entries': active,
            'hit_rate': 'N/A'  # Requires tracking
        }
