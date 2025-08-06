"""Cache implementation for search results."""

import hashlib
import time
from typing import Any

from .config import CACHE_SIZE, CACHE_TTL


class SearchCache:
    """LRU cache for search results with TTL support."""

    def __init__(self, max_size: int = CACHE_SIZE, ttl: int = CACHE_TTL):
        """
        Initialize the search cache.

        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds for cached entries
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, tuple[float, Any]] = {}
        self._access_order: list[str] = []

    def _make_key(
        self,
        query_embedding: list[float],
        k: int,
        type_filter: str | None = None,
        crate_filter: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
    ) -> str:
        """Generate a cache key from search parameters."""
        # Use a subset of the embedding for efficiency
        # Full embedding would be too large for a key
        key_parts = [
            str(query_embedding[:10]),  # Use first 10 dimensions for efficiency
            str(k),
            str(type_filter) if type_filter else "none",
            str(crate_filter) if crate_filter else "none",
            str(has_examples) if has_examples is not None else "none",
            str(min_doc_length) if min_doc_length else "none",
            str(visibility) if visibility else "none",
            str(deprecated) if deprecated is not None else "none",
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        query_embedding: list[float],
        k: int,
        type_filter: str | None = None,
        crate_filter: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
    ) -> list[tuple[float, str, str, str]] | None:
        """
        Retrieve cached results if available and not expired.

        Returns:
            Cached results or None if not found or expired
        """
        key = self._make_key(
            query_embedding,
            k,
            type_filter,
            crate_filter,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
        )

        if key in self._cache:
            timestamp, results = self._cache[key]

            # Check if entry has expired
            if time.time() - timestamp > self.ttl:
                # Remove expired entry
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return results

        return None

    def set(
        self,
        query_embedding: list[float],
        k: int,
        results: list[tuple[float, str, str, str]],
        type_filter: str | None = None,
        crate_filter: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
    ) -> None:
        """
        Store search results in cache.

        Args:
            query_embedding: The query embedding vector
            k: Number of results
            results: Search results to cache
            type_filter: Optional type filter
            crate_filter: Optional crate filter
            has_examples: Optional examples filter
            min_doc_length: Optional minimum doc length filter
            visibility: Optional visibility filter
            deprecated: Optional deprecated filter
        """
        key = self._make_key(
            query_embedding,
            k,
            type_filter,
            crate_filter,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
        )

        # Remove least recently used if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

        # Store with timestamp
        self._cache[key] = (time.time(), results)

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = sum(
            1
            for timestamp, _ in self._cache.values()
            if current_time - timestamp <= self.ttl
        )

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


# Global cache instance
_search_cache = SearchCache()


def get_search_cache() -> SearchCache:
    """Get the global search cache instance."""
    return _search_cache
