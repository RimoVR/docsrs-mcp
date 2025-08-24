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
    IngestionTier,
    ItemChange,
    ItemKind,
    ItemSignature,
    MigrationHint,
    Severity,
    VersionDiffResponse,
)
from .validation import (
    format_error_message,
    validate_crate_name,
    validate_version_string,
)

logger = logging.getLogger(__name__)


async def get_ingestion_tier(db_path) -> str | None:
    """Get the ingestion tier for a crate from the database.

    Args:
        db_path: Path to the database file

    Returns:
        The ingestion tier string or None if not found
    """
    import aiosqlite

    try:
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("""
                SELECT ingestion_tier 
                FROM ingestion_status 
                WHERE status = 'completed'
                LIMIT 1
            """)
            row = await cursor.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning(f"Failed to get ingestion tier: {e}")
        return None


def is_fallback_tier(tier: str | None) -> bool:
    """Check if the ingestion tier is a fallback tier (not full rustdoc JSON).

    Args:
        tier: The ingestion tier string

    Returns:
        True if this is a fallback tier (source extraction or description only)
    """
    if tier is None:
        # If tier is not tracked, assume it's not fallback (backward compatibility)
        return False

    # Check if it's one of the fallback tiers
    return tier in [
        IngestionTier.SOURCE_EXTRACTION.value,
        IngestionTier.DESCRIPTION_ONLY.value,
    ]


def get_tier_threshold(tier: str | None) -> int:
    """Get the appropriate MIN_ITEMS_THRESHOLD based on ingestion tier.

    Args:
        tier: The ingestion tier string

    Returns:
        The minimum items threshold appropriate for the tier
    """
    if tier == IngestionTier.RUSTDOC_JSON.value:
        # Full rustdoc JSON should have many items
        return 2  # At least 2 items for normal ingestion
    elif tier == IngestionTier.SOURCE_EXTRACTION.value:
        # Source extraction may have fewer items
        return 1  # At least 1 item from source extraction
    elif tier == IngestionTier.DESCRIPTION_ONLY.value:
        # Description-only fallback has minimal items
        return 1  # Accept single crate description
    else:
        # Unknown or None - use default threshold
        return 2  # Default to standard threshold


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

        # Validate inputs before expensive operations
        try:
            validated_crate = validate_crate_name(request.crate_name)
            validated_version_a = validate_version_string(request.version_a)
            validated_version_b = validate_version_string(request.version_b)
        except ValueError as e:
            raise ValueError(f"Invalid input parameters: {e}") from e

        # Ensure both versions are ingested
        logger.info(
            f"Comparing {validated_crate} versions {validated_version_a} vs {validated_version_b}"
        )

        db_path_a = await ingest_crate(validated_crate, validated_version_a)
        db_path_b = await ingest_crate(validated_crate, validated_version_b)

        # Get all items from both versions
        items_a = await get_all_items_for_version(db_path_a)
        items_b = await get_all_items_for_version(db_path_b)

        # Get ingestion tiers for tier-aware validation
        tier_a = await get_ingestion_tier(db_path_a)
        tier_b = await get_ingestion_tier(db_path_b)

        # Verify data was actually ingested - prevent misleading "0 changes"
        # Use tier-aware thresholds based on ingestion method
        threshold_a = get_tier_threshold(tier_a)
        threshold_b = get_tier_threshold(tier_b)

        if not items_a or len(items_a) < threshold_a:
            item_count = len(items_a) if items_a else 0

            # Provide tier-specific error message
            if is_fallback_tier(tier_a):
                error_msg = format_error_message(
                    "custom",
                    message=f"Limited documentation available for {validated_crate} v{validated_version_a} "
                    f"(only {item_count} item{'s' if item_count != 1 else ''} found via {tier_a or 'unknown method'}). "
                    f"This crate was ingested using fallback extraction because rustdoc JSON was unavailable. "
                    f"Version comparison may have limited accuracy.",
                )
            else:
                error_msg = format_error_message(
                    "custom",
                    message=f"Insufficient documentation found for {validated_crate} v{validated_version_a} "
                    f"(only {item_count} item{'s' if item_count != 1 else ''} found). "
                    f"The crate may not be fully ingested. This often happens when a version doesn't exist "
                    f"or rustdoc JSON is unavailable. Try using a valid version number.",
                )
            raise ValueError(error_msg)

        if not items_b or len(items_b) < threshold_b:
            item_count = len(items_b) if items_b else 0

            # Provide tier-specific error message
            if is_fallback_tier(tier_b):
                error_msg = format_error_message(
                    "custom",
                    message=f"Limited documentation available for {validated_crate} v{validated_version_b} "
                    f"(only {item_count} item{'s' if item_count != 1 else ''} found via {tier_b or 'unknown method'}). "
                    f"This crate was ingested using fallback extraction because rustdoc JSON was unavailable. "
                    f"Version comparison may have limited accuracy.",
                )
            else:
                error_msg = format_error_message(
                    "custom",
                    message=f"Insufficient documentation found for {validated_crate} v{validated_version_b} "
                    f"(only {item_count} item{'s' if item_count != 1 else ''} found). "
                    f"The crate may not be fully ingested. This often happens when a version doesn't exist "
                    f"or rustdoc JSON is unavailable. Try using a valid version number.",
                )
            raise ValueError(error_msg)

        # Convert list of items to dict keyed by item_path
        # Add defensive None checks for all fields
        items_dict_a = {}
        items_dict_b = {}
        
        for item in items_a:
            # Defensive check for item_path
            path = item.get("item_path")
            if path:
                items_dict_a[path] = item
            else:
                logger.warning("Item without item_path found in version A, skipping")
        
        for item in items_b:
            # Defensive check for item_path
            path = item.get("item_path")
            if path:
                items_dict_b[path] = item
            else:
                logger.warning("Item without item_path found in version B, skipping")

        # Compute item hashes for change detection
        hashes_a = {}
        hashes_b = {}

        for path, item in items_dict_a.items():
            try:
                hashes_a[path] = await compute_item_hash(item)
            except Exception as e:
                logger.warning(f"Failed to compute hash for {path} in version A: {e}")
                # Use a default hash to allow comparison to continue
                hashes_a[path] = hashlib.md5(path.encode()).hexdigest()

        for path, item in items_dict_b.items():
            try:
                hashes_b[path] = await compute_item_hash(item)
            except Exception as e:
                logger.warning(f"Failed to compute hash for {path} in version B: {e}")
                # Use a default hash to allow comparison to continue
                hashes_b[path] = hashlib.md5(path.encode()).hexdigest()

        # Detect changes
        paths_a = set(items_dict_a.keys())
        paths_b = set(items_dict_b.keys())

        added_paths = paths_b - paths_a
        removed_paths = paths_a - paths_b
        common_paths = paths_a & paths_b

        # Categorize changes
        changes: dict[str, list[ItemChange]] = defaultdict(list)

        # Process removed items
        for path in removed_paths:
            item = items_dict_a[path]
            change = await self._create_removed_change(path, item)
            changes[ChangeCategory.REMOVED.value].append(change)
            if change.severity == Severity.BREAKING:
                changes[ChangeCategory.BREAKING.value].append(change)

        # Process added items
        for path in added_paths:
            item = items_dict_b[path]
            change = await self._create_added_change(path, item)
            changes[ChangeCategory.ADDED.value].append(change)

        # Process modified items
        for path in common_paths:
            if hashes_a.get(path) != hashes_b.get(path):
                item_a = items_dict_a[path]
                item_b = items_dict_b[path]
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
        """Create change object for removed item with defensive None handling."""
        # Defensive None checks for all accessed fields
        item_type = item.get("item_type") if item else None
        visibility = item.get("visibility") if item else None
        signature = item.get("signature") if item else None
        deprecated = item.get("deprecated") if item else None
        
        # Default values for None cases
        if visibility is None:
            visibility = "public"
        if signature is None:
            signature = ""
        if deprecated is None:
            deprecated = False
            
        return ItemChange(
            path=path,
            kind=self._map_item_type(item_type),
            change_type=ChangeType.REMOVED,
            severity=Severity.BREAKING
            if visibility == "public"
            else Severity.MINOR,
            details=ChangeDetails(
                before=ItemSignature(
                    raw_signature=signature,
                    visibility=visibility,
                    deprecated=deprecated,
                ),
                after=None,
                semantic_changes=["Item removed from API"],
            ),
        )

    async def _create_added_change(self, path: str, item: dict) -> ItemChange:
        """Create change object for added item with defensive None handling."""
        # Defensive None checks for all accessed fields
        item_type = item.get("item_type") if item else None
        visibility = item.get("visibility") if item else None
        signature = item.get("signature") if item else None
        deprecated = item.get("deprecated") if item else None
        
        # Default values for None cases
        if visibility is None:
            visibility = "public"
        if signature is None:
            signature = ""
        if deprecated is None:
            deprecated = False
            
        return ItemChange(
            path=path,
            kind=self._map_item_type(item_type),
            change_type=ChangeType.ADDED,
            severity=Severity.MINOR,
            details=ChangeDetails(
                before=None,
                after=ItemSignature(
                    raw_signature=signature,
                    visibility=visibility,
                    deprecated=deprecated,
                ),
                semantic_changes=["New item added to API"],
            ),
        )

    async def _create_modified_change(
        self, path: str, item_a: dict, item_b: dict
    ) -> ItemChange:
        """Create change object for modified item with defensive None handling."""
        semantic_changes = []
        severity = Severity.PATCH

        # Defensive None checks for all fields
        sig_a = item_a.get("signature") if item_a else None
        sig_b = item_b.get("signature") if item_b else None
        vis_a = item_a.get("visibility") if item_a else None
        vis_b = item_b.get("visibility") if item_b else None
        dep_a = item_a.get("deprecated") if item_a else None
        dep_b = item_b.get("deprecated") if item_b else None
        gen_a = item_a.get("generic_params") if item_a else None
        gen_b = item_b.get("generic_params") if item_b else None
        bounds_a = item_a.get("trait_bounds") if item_a else None
        bounds_b = item_b.get("trait_bounds") if item_b else None
        type_a = item_a.get("item_type") if item_a else None

        # Check signature changes
        if sig_a != sig_b:
            semantic_changes.append("Function/type signature changed")
            severity = Severity.BREAKING

        # Check visibility changes
        if vis_a != vis_b:
            if vis_b == "private":
                semantic_changes.append("Changed from public to private")
                severity = Severity.BREAKING
            else:
                vis_b_display = vis_b if vis_b else "unknown"
                semantic_changes.append(
                    f"Visibility changed to {vis_b_display}"
                )

        # Check deprecation
        if not dep_a and dep_b:
            semantic_changes.append("Item marked as deprecated")
            if severity == Severity.PATCH:
                severity = Severity.MINOR

        # Check generic changes
        if gen_a != gen_b:
            semantic_changes.append("Generic parameters changed")
            severity = Severity.BREAKING

        # Check trait bounds
        if bounds_a != bounds_b:
            semantic_changes.append("Trait bounds changed")
            severity = Severity.BREAKING

        # Default values for None cases
        if sig_a is None:
            sig_a = ""
        if sig_b is None:
            sig_b = ""
        if vis_a is None:
            vis_a = "public"
        if vis_b is None:
            vis_b = "public"
        if dep_a is None:
            dep_a = False
        if dep_b is None:
            dep_b = False

        return ItemChange(
            path=path,
            kind=self._map_item_type(type_a),
            change_type=ChangeType.MODIFIED,
            severity=severity,
            details=ChangeDetails(
                before=ItemSignature(
                    raw_signature=sig_a,
                    visibility=vis_a,
                    deprecated=dep_a,
                    generics=gen_a,
                ),
                after=ItemSignature(
                    raw_signature=sig_b,
                    visibility=vis_b,
                    deprecated=dep_b,
                    generics=gen_b,
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

    def _map_item_type(self, type_str: str | None) -> ItemKind:
        """Map database item type to ItemKind enum with defensive handling.

        Args:
            type_str: Database item type string, can be None

        Returns:
            ItemKind enum value, defaults to FUNCTION if type unknown or None
        """
        # Defensive None check following codebase patterns
        if type_str is None:
            logger.warning("Received None for item_type, using FUNCTION as default")
            return ItemKind.FUNCTION

        # Additional type safety check
        if not isinstance(type_str, str):
            logger.warning(
                f"Unexpected type for item_type: {type(type_str).__name__}, using FUNCTION as default"
            )
            return ItemKind.FUNCTION

        # Normalize string for mapping lookup
        type_str_normalized = type_str.lower().strip()

        # Map to ItemKind enum
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

        result = mapping.get(type_str_normalized, ItemKind.FUNCTION)

        # Log unknown types for monitoring (debug level to avoid spam)
        if type_str_normalized not in mapping:
            logger.debug(f"Unknown item type '{type_str}', defaulting to FUNCTION")

        return result


# Global engine instance
_engine: VersionDiffEngine | None = None


def get_diff_engine() -> VersionDiffEngine:
    """Get or create the global diff engine instance."""
    global _engine
    if _engine is None:
        _engine = VersionDiffEngine()
    return _engine
