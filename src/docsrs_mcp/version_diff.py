"""Version diff engine for comparing Rust crate documentation between versions.

This module provides semantic diff capabilities optimized for Rust coding agents,
focusing on API changes, breaking changes detection, and migration hints.
"""

import hashlib
import logging
import time
from collections import defaultdict

from .database import (
    compute_item_hash,
    get_all_items_for_version,
)
from .ingest import ingest_crate
from .models import (
    ChangeCategory,
    ChangeDetails,
    ChangeType,
    CompareVersionsRequest,
    DiffSummary,
    ItemChange,
    ItemKind,
    ItemSignature,
    MigrationHint,
    Severity,
    VersionDiffResponse,
)

logger = logging.getLogger(__name__)


class DiffCache:
    """LRU cache for version comparisons to improve performance."""

    def __init__(self, max_size: int = 100):
        """Initialize cache with maximum size limit."""
        self.max_size = max_size
        self.cache: dict[str, tuple[float, VersionDiffResponse]] = {}
        self.access_order: list[str] = []

    def _make_key(self, crate_name: str, version_a: str, version_b: str) -> str:
        """Generate cache key from comparison parameters."""
        # Sort versions to ensure cache hits regardless of order
        versions = sorted([version_a, version_b])
        key_str = f"{crate_name}|{versions[0]}|{versions[1]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(
        self, crate_name: str, version_a: str, version_b: str
    ) -> VersionDiffResponse | None:
        """Get cached comparison result if available and not expired."""
        key = self._make_key(crate_name, version_a, version_b)

        if key in self.cache:
            timestamp, result = self.cache[key]
            # Cache expires after 1 hour
            if time.time() - timestamp < 3600:
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                # Mark as cached
                result.cached = True
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)

        return None

    async def set(
        self,
        crate_name: str,
        version_a: str,
        version_b: str,
        result: VersionDiffResponse,
    ) -> None:
        """Store comparison result in cache."""
        key = self._make_key(crate_name, version_a, version_b)

        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

        # Store with timestamp
        self.cache[key] = (time.time(), result)

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


class VersionDiffEngine:
    """Main engine for comparing crate versions."""

    def __init__(self):
        """Initialize the diff engine with cache."""
        self.cache = DiffCache()

    async def compare_versions(
        self, request: CompareVersionsRequest
    ) -> VersionDiffResponse:
        """Compare two versions of a crate and return structured diff.

        Args:
            request: Comparison request with crate name and versions

        Returns:
            VersionDiffResponse with categorized changes and migration hints
        """
        start_time = time.time()

        # Check cache first
        cached_result = await self.cache.get(
            request.crate_name, request.version_a, request.version_b
        )
        if cached_result:
            logger.info(
                f"Cache hit for {request.crate_name} {request.version_a} vs {request.version_b}"
            )
            return cached_result

        # Ensure both versions are ingested
        logger.info(
            f"Comparing {request.crate_name} versions {request.version_a} vs {request.version_b}"
        )

        db_path_a = await ingest_crate(request.crate_name, request.version_a)
        db_path_b = await ingest_crate(request.crate_name, request.version_b)

        # Get all items from both versions
        items_a = await get_all_items_for_version(db_path_a)
        items_b = await get_all_items_for_version(db_path_b)

        # Compute item hashes for change detection
        hashes_a = {}
        hashes_b = {}

        for path, item in items_a.items():
            hashes_a[path] = await compute_item_hash(item)

        for path, item in items_b.items():
            hashes_b[path] = await compute_item_hash(item)

        # Detect changes
        paths_a = set(items_a.keys())
        paths_b = set(items_b.keys())

        added_paths = paths_b - paths_a
        removed_paths = paths_a - paths_b
        common_paths = paths_a & paths_b

        # Categorize changes
        changes: dict[str, list[ItemChange]] = defaultdict(list)

        # Process removed items
        for path in removed_paths:
            item = items_a[path]
            change = await self._create_removed_change(path, item)
            changes[ChangeCategory.REMOVED.value].append(change)
            if change.severity == Severity.BREAKING:
                changes[ChangeCategory.BREAKING.value].append(change)

        # Process added items
        for path in added_paths:
            item = items_b[path]
            change = await self._create_added_change(path, item)
            changes[ChangeCategory.ADDED.value].append(change)

        # Process modified items
        for path in common_paths:
            if hashes_a[path] != hashes_b[path]:
                item_a = items_a[path]
                item_b = items_b[path]
                change = await self._create_modified_change(path, item_a, item_b)
                changes[ChangeCategory.MODIFIED.value].append(change)

                # Check if it's a breaking change
                if change.severity == Severity.BREAKING:
                    changes[ChangeCategory.BREAKING.value].append(change)

                # Check if it became deprecated
                if not item_a.get("deprecated") and item_b.get("deprecated"):
                    changes[ChangeCategory.DEPRECATED.value].append(change)

        # Generate migration hints for breaking changes
        migration_hints = await self._generate_migration_hints(
            changes.get(ChangeCategory.BREAKING.value, [])
        )

        # Create summary
        summary = DiffSummary(
            total_changes=sum(len(items) for items in changes.values()),
            breaking_changes=len(changes.get(ChangeCategory.BREAKING.value, [])),
            deprecated_items=len(changes.get(ChangeCategory.DEPRECATED.value, [])),
            added_items=len(changes.get(ChangeCategory.ADDED.value, [])),
            removed_items=len(changes.get(ChangeCategory.REMOVED.value, [])),
            modified_items=len(changes.get(ChangeCategory.MODIFIED.value, [])),
            migration_hints_available=len(migration_hints),
        )

        # Filter by requested categories
        filtered_changes = {}
        for category in request.categories:
            if category.value in changes:
                filtered_changes[category.value] = changes[category.value][
                    : request.max_results
                ]

        # Build response
        computation_time_ms = (time.time() - start_time) * 1000

        response = VersionDiffResponse(
            crate_name=request.crate_name,
            version_a=request.version_a,
            version_b=request.version_b,
            summary=summary,
            changes=filtered_changes,
            migration_hints=migration_hints[:10],  # Limit hints to top 10
            computation_time_ms=computation_time_ms,
            cached=False,
        )

        # Cache the result
        await self.cache.set(
            request.crate_name, request.version_a, request.version_b, response
        )

        logger.info(
            f"Comparison completed in {computation_time_ms:.2f}ms: "
            f"{summary.total_changes} changes, {summary.breaking_changes} breaking"
        )

        return response

    async def _create_removed_change(self, path: str, item: dict) -> ItemChange:
        """Create change object for removed item."""
        return ItemChange(
            path=path,
            kind=self._map_item_type(item.get("item_type", "unknown")),
            change_type=ChangeType.REMOVED,
            severity=Severity.BREAKING
            if item.get("visibility") == "public"
            else Severity.MINOR,
            details=ChangeDetails(
                before=ItemSignature(
                    raw_signature=item.get("signature") or "",
                    visibility=item.get("visibility", "public"),
                    deprecated=item.get("deprecated", False),
                ),
                after=None,
                semantic_changes=["Item removed from API"],
            ),
        )

    async def _create_added_change(self, path: str, item: dict) -> ItemChange:
        """Create change object for added item."""
        return ItemChange(
            path=path,
            kind=self._map_item_type(item.get("item_type", "unknown")),
            change_type=ChangeType.ADDED,
            severity=Severity.MINOR,
            details=ChangeDetails(
                before=None,
                after=ItemSignature(
                    raw_signature=item.get("signature") or "",
                    visibility=item.get("visibility", "public"),
                    deprecated=item.get("deprecated", False),
                ),
                semantic_changes=["New item added to API"],
            ),
        )

    async def _create_modified_change(
        self, path: str, item_a: dict, item_b: dict
    ) -> ItemChange:
        """Create change object for modified item."""
        semantic_changes = []
        severity = Severity.PATCH

        # Check signature changes
        if item_a.get("signature") != item_b.get("signature"):
            semantic_changes.append("Function/type signature changed")
            severity = Severity.BREAKING

        # Check visibility changes
        if item_a.get("visibility") != item_b.get("visibility"):
            if item_b.get("visibility") == "private":
                semantic_changes.append("Changed from public to private")
                severity = Severity.BREAKING
            else:
                semantic_changes.append(
                    f"Visibility changed to {item_b.get('visibility')}"
                )

        # Check deprecation
        if not item_a.get("deprecated") and item_b.get("deprecated"):
            semantic_changes.append("Item marked as deprecated")
            if severity == Severity.PATCH:
                severity = Severity.MINOR

        # Check generic changes
        if item_a.get("generic_params") != item_b.get("generic_params"):
            semantic_changes.append("Generic parameters changed")
            severity = Severity.BREAKING

        # Check trait bounds
        if item_a.get("trait_bounds") != item_b.get("trait_bounds"):
            semantic_changes.append("Trait bounds changed")
            severity = Severity.BREAKING

        return ItemChange(
            path=path,
            kind=self._map_item_type(item_a.get("item_type", "unknown")),
            change_type=ChangeType.MODIFIED,
            severity=severity,
            details=ChangeDetails(
                before=ItemSignature(
                    raw_signature=item_a.get("signature") or "",
                    visibility=item_a.get("visibility", "public"),
                    deprecated=item_a.get("deprecated", False),
                    generics=item_a.get("generic_params"),
                ),
                after=ItemSignature(
                    raw_signature=item_b.get("signature") or "",
                    visibility=item_b.get("visibility", "public"),
                    deprecated=item_b.get("deprecated", False),
                    generics=item_b.get("generic_params"),
                ),
                semantic_changes=semantic_changes,
            ),
        )

    async def _generate_migration_hints(
        self, breaking_changes: list[ItemChange]
    ) -> list[MigrationHint]:
        """Generate migration hints for breaking changes."""
        hints = []

        for change in breaking_changes[:10]:  # Limit to top 10 for performance
            hint = await self._create_migration_hint(change)
            if hint:
                hints.append(hint)

        return hints

    async def _create_migration_hint(self, change: ItemChange) -> MigrationHint | None:
        """Create a migration hint for a specific breaking change."""
        if change.change_type == ChangeType.REMOVED:
            return MigrationHint(
                affected_path=change.path,
                issue=f"'{change.path}' has been removed",
                suggested_fix=f"Find alternative API or remove usage of '{change.path}'",
                severity=Severity.BREAKING,
            )

        elif change.change_type == ChangeType.MODIFIED:
            # Check for signature changes
            if (
                "signature changed"
                in " ".join(
                    x for x in change.details.semantic_changes if x is not None
                ).lower()
            ):
                return MigrationHint(
                    affected_path=change.path,
                    issue="Function signature has changed",
                    suggested_fix="Update function calls to match new signature",
                    severity=Severity.BREAKING,
                    example_before=change.details.before.raw_signature
                    if change.details.before
                    else None,
                    example_after=change.details.after.raw_signature
                    if change.details.after
                    else None,
                )

            # Check for visibility changes
            if (
                "private"
                in " ".join(
                    x for x in change.details.semantic_changes if x is not None
                ).lower()
            ):
                return MigrationHint(
                    affected_path=change.path,
                    issue="Item is no longer publicly accessible",
                    suggested_fix="Use alternative public API or refactor code",
                    severity=Severity.BREAKING,
                )

        return None

    def _map_item_type(self, type_str: str) -> ItemKind:
        """Map database item type to ItemKind enum."""
        mapping = {
            "function": ItemKind.FUNCTION,
            "struct": ItemKind.STRUCT,
            "enum": ItemKind.ENUM,
            "trait": ItemKind.TRAIT,
            "type": ItemKind.TYPE_ALIAS,
            "const": ItemKind.CONST,
            "static": ItemKind.STATIC,
            "module": ItemKind.MODULE,
            "macro": ItemKind.MACRO,
            "impl": ItemKind.IMPL,
        }
        return mapping.get(type_str.lower(), ItemKind.FUNCTION)


# Global engine instance
_engine: VersionDiffEngine | None = None


def get_diff_engine() -> VersionDiffEngine:
    """Get or create the global diff engine instance."""
    global _engine
    if _engine is None:
        _engine = VersionDiffEngine()
    return _engine
