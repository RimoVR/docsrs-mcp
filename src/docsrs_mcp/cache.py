"""Cache implementation for search results."""

import hashlib
import time
from typing import Any

from .config import (
    CACHE_ADAPTIVE_TTL_ENABLED,
    CACHE_SIZE,
    CACHE_TTL,
    RANKING_DIVERSITY_LAMBDA,
    RANKING_DIVERSITY_WEIGHT,
)


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
        self.adaptive_ttl_enabled = CACHE_ADAPTIVE_TTL_ENABLED
        self._cache: dict[str, tuple[float, Any]] = {}
        self._access_order: list[str] = []
        self._hit_count: dict[str, int] = {}  # Track hit counts for popular queries

    def _make_key(
        self,
        query_embedding: list[float],
        k: int,
        type_filter: str | None = None,
        crate_filter: str | None = None,
        module_path: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
        stability_filter: str | None = None,
    ) -> str:
        """Generate a cache key from search parameters including diversity settings."""
        # Use a subset of the embedding for efficiency
        # Full embedding would be too large for a key
        key_parts = [
            str(query_embedding[:10]),  # Use first 10 dimensions for efficiency
            str(k),
            str(type_filter) if type_filter else "none",
            str(crate_filter) if crate_filter else "none",
            str(module_path) if module_path else "none",
            str(has_examples) if has_examples is not None else "none",
            str(min_doc_length) if min_doc_length else "none",
            str(visibility) if visibility else "none",
            str(deprecated) if deprecated is not None else "none",
            str(stability_filter) if stability_filter else "none",
            # Include diversity parameters in cache key
            f"lambda_{RANKING_DIVERSITY_LAMBDA:.2f}",
            f"weight_{RANKING_DIVERSITY_WEIGHT:.2f}",
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_adaptive_ttl(
        self,
        query_embedding: list[float],
        type_filter: str | None = None,
        crate_filter: str | None = None,
        module_path: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
        stability_filter: str | None = None,
    ) -> int:
        """
        Calculate adaptive TTL based on query complexity and popularity.

        Simple queries get longer TTL, complex queries get shorter TTL.
        Popular queries get extended TTL based on hit rate.

        Returns:
            TTL in seconds
        """
        if not self.adaptive_ttl_enabled:
            return self.ttl

        # Calculate query complexity
        complexity = 0

        # Base complexity from embedding (simple heuristic: more non-zero values = more complex)
        embedding_complexity = sum(1 for v in query_embedding[:50] if abs(v) > 0.1)
        complexity += embedding_complexity / 10  # Normalize to 0-5 range

        # Add complexity for each filter
        if type_filter:
            complexity += 1
        if crate_filter:
            complexity += 1
        if module_path:
            complexity += 2  # Module path is more specific
        if has_examples is not None:
            complexity += 1
        if min_doc_length:
            complexity += 1
        if visibility:
            complexity += 1
        if deprecated is not None:
            complexity += 1
        if stability_filter:
            complexity += 1

        # Determine base TTL based on complexity
        if complexity <= 2:
            base_ttl = 3600  # 1 hour for simple queries
        elif complexity <= 5:
            base_ttl = 1800  # 30 minutes for moderate queries
        else:
            base_ttl = 900  # 15 minutes for complex queries

        # Check hit count for popularity adjustment
        key = self._make_key(
            query_embedding,
            0,  # k doesn't matter for key
            type_filter,
            crate_filter,
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
            stability_filter,
        )

        hit_count = self._hit_count.get(key, 0)

        # Extend TTL for popular queries (hit more than 3 times)
        if hit_count > 3:
            popularity_multiplier = min(2.0, 1.0 + (hit_count * 0.1))
            base_ttl = int(base_ttl * popularity_multiplier)

        # Cap at maximum of 2 hours
        return min(7200, base_ttl)

    def get(
        self,
        query_embedding: list[float],
        k: int,
        type_filter: str | None = None,
        crate_filter: str | None = None,
        module_path: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
        stability_filter: str | None = None,
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
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
            stability_filter,
        )

        if key in self._cache:
            cache_entry = self._cache[key]
            # Handle both old (timestamp, results) and new (timestamp, results, ttl) formats
            if len(cache_entry) == 2:
                timestamp, results = cache_entry
                ttl = self.ttl  # Use default TTL for old entries
            else:
                timestamp, results, ttl = cache_entry

            # Check if entry has expired
            if time.time() - timestamp > ttl:
                # Remove expired entry
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                if key in self._hit_count:
                    del self._hit_count[key]
                return None

            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # Track hit count for popularity-based TTL
            self._hit_count[key] = self._hit_count.get(key, 0) + 1

            return results

        return None

    def set(
        self,
        query_embedding: list[float],
        k: int,
        results: list[tuple[float, str, str, str]],
        type_filter: str | None = None,
        crate_filter: str | None = None,
        module_path: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
        stability_filter: str | None = None,
    ) -> None:
        """
        Store search results in cache.

        Args:
            query_embedding: The query embedding vector
            k: Number of results
            results: Search results to cache
            type_filter: Optional type filter
            crate_filter: Optional crate filter
            module_path: Optional module path filter
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
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
            stability_filter,
        )

        # Remove least recently used if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
                if lru_key in self._hit_count:
                    del self._hit_count[lru_key]

        # Calculate adaptive TTL for this query
        ttl = self.get_adaptive_ttl(
            query_embedding,
            type_filter,
            crate_filter,
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
            stability_filter,
        )

        # Store with timestamp and adaptive TTL
        self._cache[key] = (time.time(), results, ttl)

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        self._hit_count.clear()

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

    def get_cache_stats(self) -> dict[str, int]:
        """Wrapper for consistency with PopularCratesManager interface."""
        return self.get_stats()


# Global cache instances
_search_cache = SearchCache()


def get_search_cache() -> SearchCache:
    """Get the global search cache instance."""
    return _search_cache


class CrateInfoCache:
    """LRU cache for crate info with TTL support."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the crate info cache.

        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds for cached entries (default 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, tuple[float, Any]] = {}
        self._access_order: list[str] = []

    def get(self, crate_name: str) -> Any | None:
        """
        Retrieve cached crate info if available and not expired.

        Returns:
            Cached crate info or None if not found or expired
        """
        if crate_name in self._cache:
            timestamp, crate_info = self._cache[crate_name]

            # Check if entry has expired
            if time.time() - timestamp > self.ttl:
                # Remove expired entry
                del self._cache[crate_name]
                if crate_name in self._access_order:
                    self._access_order.remove(crate_name)
                return None

            # Move to end (most recently used)
            if crate_name in self._access_order:
                self._access_order.remove(crate_name)
            self._access_order.append(crate_name)

            return crate_info

        return None

    def set(self, crate_name: str, crate_info: Any) -> None:
        """
        Store crate info in cache.

        Args:
            crate_name: The crate name
            crate_info: Crate info to cache
        """
        # Remove least recently used if at capacity
        if len(self._cache) >= self.max_size and crate_name not in self._cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

        # Store with timestamp
        self._cache[crate_name] = (time.time(), crate_info)

        # Update access order
        if crate_name in self._access_order:
            self._access_order.remove(crate_name)
        self._access_order.append(crate_name)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()


# Global crate info cache instance
_crate_info_cache = CrateInfoCache()


def get_crate_info_cache() -> CrateInfoCache:
    """Get the global crate info cache instance."""
    return _crate_info_cache
