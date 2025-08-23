"""
Dependency filter for docsrs-mcp.

This module provides efficient filtering of dependency modules to reduce
noise in search results and documentation.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DependencyFilter:
    """
    Filter for identifying and excluding dependency modules.

    Uses a simple set-based approach for compatibility.
    Future enhancement: Use pybloomfiltermmap3 when Python 3.13 support is available.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        capacity: int = 100000,
        error_rate: float = 0.001,
    ):
        """
        Initialize the dependency filter.

        Args:
            cache_path: Optional path to persist filter state
            capacity: Expected number of items (for future bloom filter)
            error_rate: Acceptable false positive rate (for future bloom filter)
        """
        self.cache_path = cache_path or Path("/tmp/docsrs_deps.json")
        self.capacity = capacity
        self.error_rate = error_rate

        # Use a set for now (memory-efficient for moderate sizes)
        self._dependencies: set[str] = set()
        self._crate_dependencies: dict[str, set[str]] = {}

        # Load existing cache if available
        self._load_cache()

    def _load_cache(self) -> None:
        """Load dependency filter from cache if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    self._dependencies = set(data.get("dependencies", []))
                    self._crate_dependencies = {
                        k: set(v) for k, v in data.get("crate_dependencies", {}).items()
                    }
                logger.info(f"Loaded {len(self._dependencies)} dependencies from cache")
            except Exception as e:
                logger.warning(f"Failed to load dependency cache: {e}")

    def _save_cache(self) -> None:
        """Save dependency filter to cache."""
        try:
            # Ensure directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "dependencies": list(self._dependencies),
                "crate_dependencies": {
                    k: list(v) for k, v in self._crate_dependencies.items()
                },
            }

            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._dependencies)} dependencies to cache")
        except Exception as e:
            logger.warning(f"Failed to save dependency cache: {e}")

    def add_dependency(self, module_path: str, crate_name: str | None = None) -> None:
        """
        Add a dependency module to the filter.

        Args:
            module_path: Full module path (e.g., "serde::de::Deserialize")
            crate_name: Optional crate that contains this dependency
        """
        self._dependencies.add(module_path)

        if crate_name:
            if crate_name not in self._crate_dependencies:
                self._crate_dependencies[crate_name] = set()
            self._crate_dependencies[crate_name].add(module_path)

        # Periodically save cache
        if len(self._dependencies) % 1000 == 0:
            self._save_cache()

    def is_dependency(self, module_path: str) -> bool:
        """
        Check if a module is a dependency.

        Args:
            module_path: Module path to check

        Returns:
            True if the module is a dependency
        """
        return module_path in self._dependencies

    def add_dependencies_from_rustdoc(
        self, rustdoc_json: dict, target_crate: str
    ) -> None:
        """
        Extract and add dependencies from rustdoc JSON.

        This identifies modules that are external to the target crate
        and marks them as dependencies.

        Args:
            rustdoc_json: Parsed rustdoc JSON
            target_crate: Name of the main crate being documented
        """
        try:
            # Get the index from rustdoc
            index = rustdoc_json.get("index", {})
            paths = rustdoc_json.get("paths", {})

            # Track external crates
            external_crates = set()

            # Find external crate references
            for item_id, item_data in index.items():
                if isinstance(item_data, dict):
                    # Check if this is an external item
                    crate_id = item_data.get("crate_id")
                    if crate_id and crate_id != 0:  # 0 is typically the local crate
                        # This is from an external crate
                        if item_id in paths:
                            path_info = paths[item_id]
                            if isinstance(path_info, dict):
                                path = path_info.get("path", [])
                                if path and path[0] != target_crate:
                                    # This is a dependency module
                                    module_path = "::".join(path)
                                    self.add_dependency(module_path, target_crate)
                                    external_crates.add(path[0])

            if external_crates:
                logger.info(
                    f"Identified {len(external_crates)} external crates as dependencies: "
                    f"{', '.join(sorted(external_crates))}"
                )

            # Also check for common dependency patterns
            # Many crates re-export items from their dependencies
            for item_id, item_data in index.items():
                if isinstance(item_data, dict):
                    # Look for re-exports (items with import information)
                    if "import" in item_data or "glob_import" in item_data:
                        # This might be a re-export from a dependency
                        name = item_data.get("name", "")
                        if name and not name.startswith(target_crate):
                            self.add_dependency(name, target_crate)

        except Exception as e:
            logger.warning(f"Failed to extract dependencies from rustdoc: {e}")

    def filter_items(self, items: list[dict], crate_name: str) -> list[dict]:
        """
        Filter out dependency items from a list of documentation items.

        Args:
            items: List of documentation items
            crate_name: Main crate name

        Returns:
            Filtered list with dependencies removed
        """
        filtered = []
        removed_count = 0

        for item in items:
            # Check various fields that might indicate a dependency
            item_path = item.get("item_path", "")
            module_path = item.get("module_path", "")

            # Check if this item is from a dependency
            is_dep = False

            # Check full paths
            if item_path and self.is_dependency(item_path):
                is_dep = True
            elif module_path and self.is_dependency(module_path):
                is_dep = True

            # Check if the item is from an external crate
            # (not starting with the target crate name)
            if not is_dep and item_path:
                # Split the path and check the root
                path_parts = item_path.split("::")
                if path_parts and path_parts[0] != crate_name:
                    # Check common dependency crate names
                    common_deps = {
                        "std",
                        "core",
                        "alloc",  # stdlib
                        "serde",
                        "tokio",
                        "async_trait",  # common deps
                        "futures",
                        "bytes",
                        "log",
                        "tracing",
                        "anyhow",
                        "thiserror",
                    }
                    if path_parts[0] in common_deps:
                        is_dep = True
                        self.add_dependency(item_path, crate_name)

            if not is_dep:
                filtered.append(item)
            else:
                removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Filtered out {removed_count} dependency items, "
                f"kept {len(filtered)} items from {crate_name}"
            )

        return filtered

    def get_statistics(self) -> dict:
        """
        Get statistics about the dependency filter.

        Returns:
            Dictionary with filter statistics
        """
        return {
            "total_dependencies": len(self._dependencies),
            "crates_tracked": len(self._crate_dependencies),
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
        }

    def clear(self) -> None:
        """Clear all dependency information."""
        self._dependencies.clear()
        self._crate_dependencies.clear()
        logger.info("Cleared dependency filter")

    def save(self) -> None:
        """Explicitly save the filter to cache."""
        self._save_cache()


# Global filter instance
_global_filter: DependencyFilter | None = None


def get_dependency_filter() -> DependencyFilter:
    """Get or create the global dependency filter instance."""
    global _global_filter
    if _global_filter is None:
        from .config import (
            BLOOM_FILTER_CAPACITY,
            BLOOM_FILTER_ERROR_RATE,
            BLOOM_FILTER_PATH,
        )

        _global_filter = DependencyFilter(
            cache_path=Path(BLOOM_FILTER_PATH),
            capacity=BLOOM_FILTER_CAPACITY,
            error_rate=BLOOM_FILTER_ERROR_RATE,
        )
    return _global_filter
