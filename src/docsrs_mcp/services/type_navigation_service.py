"""Service layer for type system navigation operations."""

import json
import logging
from functools import lru_cache

import aiosqlite

from ..database import DB_TIMEOUT
from ..ingest import ingest_crate

logger = logging.getLogger(__name__)


class TypeNavigationService:
    """Service for type system navigation operations.

    Provides comprehensive trait implementation discovery, method resolution,
    associated items handling, and generic constraint analysis.
    """

    def __init__(self):
        """Initialize the type navigation service with caching."""
        self._trait_cache = lru_cache(maxsize=1000)(
            self._get_trait_implementors_uncached
        )
        self._method_cache = lru_cache(maxsize=1000)(self._resolve_method_uncached)

    async def get_trait_implementors(
        self, crate_name: str, trait_path: str, version: str | None = None
    ) -> dict:
        """Find all types that implement a specific trait.

        Args:
            crate_name: Name of the crate
            trait_path: Full path to the trait (e.g., 'std::fmt::Debug')
            version: Optional version (defaults to latest)

        Returns:
            Dictionary containing implementing types and details
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Get crate ID
        crate_id = await self._get_crate_id(db_path)

        # Query trait implementations
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Try exact match first
            cursor = await db.execute(
                """
                SELECT 
                    impl_type_path,
                    generic_params,
                    where_clauses,
                    is_blanket,
                    impl_signature,
                    stability_level
                FROM trait_implementations
                WHERE crate_id = ? AND trait_path = ?
                ORDER BY is_blanket ASC, impl_type_path ASC
                """,
                (crate_id, trait_path),
            )
            
            results = await cursor.fetchall()
            
            # If no results with FQN, try with bare name (last segment after "::")
            if not results and "::" in trait_path:
                bare_name = trait_path.split("::")[-1]
                cursor = await db.execute(
                    """
                    SELECT 
                        impl_type_path,
                        generic_params,
                        where_clauses,
                        is_blanket,
                        impl_signature,
                        stability_level
                    FROM trait_implementations
                    WHERE crate_id = ? AND trait_path = ?
                    ORDER BY is_blanket ASC, impl_type_path ASC
                    """,
                    (crate_id, bare_name),
                )
                results = await cursor.fetchall()

            implementations = []
            for row in results:
                implementations.append(
                    {
                        "type_path": row[0],
                        "generic_params": json.loads(row[1]) if row[1] else None,
                        "where_clauses": json.loads(row[2]) if row[2] else None,
                        "is_blanket": bool(row[3]),
                        "impl_signature": row[4],
                        "stability": row[5],
                    }
                )

            return {
                "trait_path": trait_path,
                "implementors": implementations,
                "total_count": len(implementations),
            }

    async def get_type_traits(
        self, crate_name: str, type_path: str, version: str | None = None
    ) -> dict:
        """Get all traits implemented by a specific type.

        Args:
            crate_name: Name of the crate
            type_path: Full path to the type (e.g., 'std::vec::Vec')
            version: Optional version (defaults to latest)

        Returns:
            Dictionary containing implemented traits
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Get crate ID
        crate_id = await self._get_crate_id(db_path)

        # Query trait implementations for this type
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            cursor = await db.execute(
                """
                SELECT 
                    trait_path,
                    generic_params,
                    where_clauses,
                    is_negative,
                    impl_signature,
                    stability_level
                FROM trait_implementations
                WHERE crate_id = ? AND impl_type_path = ?
                ORDER BY trait_path ASC
                """,
                (crate_id, type_path),
            )

            traits = []
            async for row in cursor:
                traits.append(
                    {
                        "trait_path": row[0],
                        "generic_params": json.loads(row[1]) if row[1] else None,
                        "where_clauses": json.loads(row[2]) if row[2] else None,
                        "is_negative": bool(row[3]),
                        "impl_signature": row[4],
                        "stability": row[5],
                    }
                )

            return {
                "type_path": type_path,
                "traits": traits,
                "total_count": len(traits),
            }

    async def resolve_method(
        self,
        crate_name: str,
        type_path: str,
        method_name: str,
        version: str | None = None,
    ) -> dict:
        """Resolve method calls to find the correct implementation.

        Args:
            crate_name: Name of the crate
            type_path: Full path to the type
            method_name: Name of the method to resolve
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with method resolution details
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Get crate ID
        crate_id = await self._get_crate_id(db_path)

        # Query method signatures
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # First look for inherent methods
            cursor = await db.execute(
                """
                SELECT 
                    full_signature,
                    generic_params,
                    where_clauses,
                    return_type,
                    is_async,
                    is_unsafe,
                    is_const,
                    visibility,
                    method_kind,
                    trait_source,
                    receiver_type,
                    stability_level
                FROM method_signatures
                WHERE crate_id = ? 
                    AND parent_type_path = ? 
                    AND method_name = ?
                ORDER BY method_kind ASC  -- Prioritize inherent over trait methods
                """,
                (crate_id, type_path, method_name),
            )

            methods = []
            async for row in cursor:
                methods.append(
                    {
                        "signature": row[0],
                        "generic_params": json.loads(row[1]) if row[1] else None,
                        "where_clauses": json.loads(row[2]) if row[2] else None,
                        "return_type": row[3],
                        "is_async": bool(row[4]),
                        "is_unsafe": bool(row[5]),
                        "is_const": bool(row[6]),
                        "visibility": row[7],
                        "method_kind": row[8],
                        "trait_source": row[9],
                        "receiver_type": row[10],
                        "stability": row[11],
                    }
                )

            # Disambiguate if multiple methods found
            disambiguation_hints = []
            if len(methods) > 1:
                disambiguation_hints = self._generate_disambiguation_hints(methods)

            return {
                "type_path": type_path,
                "method_name": method_name,
                "candidates": methods,
                "disambiguation_hints": disambiguation_hints,
                "resolution_status": "unique"
                if len(methods) == 1
                else "ambiguous"
                if len(methods) > 1
                else "not_found",
            }

    async def get_associated_items(
        self,
        crate_name: str,
        container_path: str,
        item_kind: str | None = None,
        version: str | None = None,
    ) -> dict:
        """Get associated items (types, constants, functions) for a trait or type.

        Args:
            crate_name: Name of the crate
            container_path: Path to the containing trait/type
            item_kind: Optional filter by item kind ('type', 'const', 'function')
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with associated items
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Get crate ID
        crate_id = await self._get_crate_id(db_path)

        # Build query with optional kind filter
        query = """
            SELECT 
                item_name,
                item_kind,
                item_signature,
                default_value,
                generic_params,
                where_clauses,
                visibility,
                stability_level
            FROM associated_items
            WHERE crate_id = ? AND container_path = ?
        """
        params = [crate_id, container_path]

        if item_kind:
            query += " AND item_kind = ?"
            params.append(item_kind)

        query += " ORDER BY item_kind ASC, item_name ASC"

        # Query associated items
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            cursor = await db.execute(query, params)

            items = []
            async for row in cursor:
                items.append(
                    {
                        "name": row[0],
                        "kind": row[1],
                        "signature": row[2],
                        "default_value": row[3],
                        "generic_params": json.loads(row[4]) if row[4] else None,
                        "where_clauses": json.loads(row[5]) if row[5] else None,
                        "visibility": row[6],
                        "stability": row[7],
                    }
                )

            # Group by kind for better organization
            grouped = {}
            for item in items:
                kind = item["kind"]
                if kind not in grouped:
                    grouped[kind] = []
                grouped[kind].append(item)

            return {
                "container_path": container_path,
                "associated_items": grouped,
                "total_count": len(items),
            }

    async def get_generic_constraints(
        self, crate_name: str, item_path: str, version: str | None = None
    ) -> dict:
        """Get generic constraints (type bounds, lifetime parameters) for an item.

        Args:
            crate_name: Name of the crate
            item_path: Path to the item
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with generic constraint details
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Get crate ID
        crate_id = await self._get_crate_id(db_path)

        # Query generic constraints
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            cursor = await db.execute(
                """
                SELECT 
                    param_name,
                    param_kind,
                    bounds,
                    default_value,
                    variance,
                    position
                FROM generic_constraints
                WHERE crate_id = ? AND item_path = ?
                ORDER BY position ASC, param_name ASC
                """,
                (crate_id, item_path),
            )

            constraints = []
            async for row in cursor:
                constraints.append(
                    {
                        "name": row[0],
                        "kind": row[1],
                        "bounds": json.loads(row[2]) if row[2] else None,
                        "default": row[3],
                        "variance": row[4],
                        "position": row[5],
                    }
                )

            # Group by kind for clarity
            grouped = {
                "type_params": [],
                "lifetime_params": [],
                "const_params": [],
            }

            for constraint in constraints:
                if constraint["kind"] == "type":
                    grouped["type_params"].append(constraint)
                elif constraint["kind"] == "lifetime":
                    grouped["lifetime_params"].append(constraint)
                elif constraint["kind"] == "const":
                    grouped["const_params"].append(constraint)

            return {
                "item_path": item_path,
                "generic_constraints": grouped,
                "total_params": len(constraints),
            }

    # Helper methods

    async def _get_crate_id(self, db_path: str) -> int:
        """Get the crate ID from the database."""
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            cursor = await db.execute("SELECT id FROM crate_metadata LIMIT 1")
            row = await cursor.fetchone()
            if not row:
                raise ValueError("Crate metadata not found")
            return row[0]

    def _generate_disambiguation_hints(self, methods: list[dict]) -> list[str]:
        """Generate hints for disambiguating between multiple method candidates."""
        hints = []

        # Check for inherent vs trait methods
        inherent = [m for m in methods if m["method_kind"] == "inherent"]
        trait_methods = [m for m in methods if m["method_kind"] == "trait"]

        if inherent and trait_methods:
            hints.append("Multiple methods found: inherent and trait implementations")
            hints.append("Inherent method takes precedence by default")

            # List trait sources
            trait_sources = set(
                m["trait_source"] for m in trait_methods if m["trait_source"]
            )
            if trait_sources:
                hints.append(f"Trait methods from: {', '.join(trait_sources)}")

        # Check for different receiver types
        receiver_types = set(m["receiver_type"] for m in methods if m["receiver_type"])
        if len(receiver_types) > 1:
            hints.append(f"Different receiver types: {', '.join(receiver_types)}")

        # Check for const vs non-const
        const_methods = [m for m in methods if m["is_const"]]
        if const_methods and len(const_methods) < len(methods):
            hints.append("Both const and non-const versions available")

        return hints

    def _get_trait_implementors_uncached(self, *args, **kwargs):
        """Uncached version of get_trait_implementors for LRU caching."""
        # This is wrapped by the LRU cache
        return None

    def _resolve_method_uncached(self, *args, **kwargs):
        """Uncached version of resolve_method for LRU caching."""
        # This is wrapped by the LRU cache
        return None

    # Phase 5: Code Intelligence Methods

    async def get_item_intelligence(
        self, crate_name: str, item_path: str, version: str | None = None
    ) -> dict:
        """Get complete code intelligence for an item.

        Args:
            crate_name: Name of the crate
            item_path: Full path to the item (e.g., 'tokio::spawn')
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with comprehensive intelligence data including:
            - enhanced_signature: Complete signature with generics
            - error_types: List of error types from Result patterns
            - safety_info: Safety information and requirements
            - feature_requirements: Required feature flags
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Query database for intelligence data with fallback path resolution
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Try different path formats to handle various input types
            search_paths = []
            
            # If item_path already includes crate prefix, use as-is
            if item_path.startswith(f"{crate_name}::"):
                search_paths.append(item_path)
                # Also try without prefix in case of storage inconsistency
                search_paths.append(item_path[len(f"{crate_name}::"):])
            else:
                # Try with crate prefix first
                search_paths.append(f"{crate_name}::{item_path}")
                # Then try without prefix
                search_paths.append(item_path)
            
            row = None
            found_path = None
            
            for search_path in search_paths:
                cursor = await db.execute(
                    """
                    SELECT 
                        item_path,
                        signature,
                        safety_info,
                        error_types,
                        feature_requirements,
                        is_safe,
                        generic_params,
                        trait_bounds,
                        visibility,
                        deprecated,
                        content
                    FROM embeddings
                    WHERE item_path = ?
                    LIMIT 1
                    """,
                    (search_path,),
                )
                
                row = await cursor.fetchone()
                if row:
                    found_path = search_path
                    break
            
            if not row:
                # Try a final fallback with LIKE pattern for partial matches
                cursor = await db.execute(
                    """
                    SELECT 
                        item_path,
                        signature,
                        safety_info,
                        error_types,
                        feature_requirements,
                        is_safe,
                        generic_params,
                        trait_bounds,
                        visibility,
                        deprecated,
                        content
                    FROM embeddings
                    WHERE item_path LIKE ?
                    LIMIT 1
                    """,
                    (f"%::{item_path.split('::')[-1]}",),
                )
                row = await cursor.fetchone()
                if row:
                    found_path = row[0]
            
            if not row:
                return {
                    "item_path": item_path,
                    "error": "Item not found",
                    "searched_paths": search_paths,
                }

            # Parse JSON fields
            safety_info = json.loads(row[2]) if row[2] else {}
            error_types = json.loads(row[3]) if row[3] else []
            feature_requirements = json.loads(row[4]) if row[4] else []

            return {
                "item_path": found_path or row[0],  # Return the actual found path
                "requested_path": item_path,  # Keep original request for reference
                "enhanced_signature": row[1],
                "safety_info": safety_info,
                "error_types": error_types,
                "feature_requirements": feature_requirements,
                "is_safe": bool(row[5]),
                "generic_params": row[6],
                "trait_bounds": row[7],
                "visibility": row[8],
                "deprecated": bool(row[9]),
                "documentation": row[10][:500] if row[10] else None,  # First 500 chars
            }

    async def search_by_safety(
        self,
        crate_name: str,
        is_safe: bool = True,
        include_reasons: bool = False,
        version: str | None = None,
    ) -> dict:
        """Find items by safety status.

        Args:
            crate_name: Name of the crate
            is_safe: Whether to search for safe (True) or unsafe (False) items
            include_reasons: Include detailed safety information
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with items matching the safety criteria
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Query database using partial index for performance
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Build query based on safety filter
            query = """
                SELECT 
                    item_path,
                    item_type,
                    signature,
                    safety_info,
                    content
                FROM embeddings
                WHERE is_safe = ?
                ORDER BY item_path ASC
                LIMIT 100
            """

            cursor = await db.execute(query, (1 if is_safe else 0,))

            items = []
            async for row in cursor:
                item = {
                    "item_path": row[0],
                    "item_type": row[1],
                    "signature": row[2],
                }

                if include_reasons and row[3]:
                    safety_info = json.loads(row[3])
                    item["safety_info"] = safety_info
                    item["documentation_excerpt"] = row[4][:200] if row[4] else None

                items.append(item)

            return {
                "crate_name": crate_name,
                "safety_filter": "safe" if is_safe else "unsafe",
                "items": items,
                "total_count": len(items),
            }

    async def get_error_catalog(
        self, crate_name: str, pattern: str | None = None, version: str | None = None
    ) -> dict:
        """Get catalog of all error types in crate.

        Args:
            crate_name: Name of the crate
            pattern: Optional pattern to filter error types
            version: Optional version (defaults to latest)

        Returns:
            Dictionary with error type catalog and occurrence counts
        """
        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Query database for error types
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Get all items with error types
            cursor = await db.execute(
                """
                SELECT 
                    item_path,
                    error_types,
                    signature
                FROM embeddings
                WHERE error_types IS NOT NULL AND error_types != '[]'
                """,
            )

            # Aggregate error types
            error_catalog = {}
            items_by_error = {}

            async for row in cursor:
                item_path = row[0]
                error_types = json.loads(row[1]) if row[1] else []
                signature = row[2]

                for error_type in error_types:
                    # Apply pattern filter if provided
                    if pattern and pattern.lower() not in error_type.lower():
                        continue

                    if error_type not in error_catalog:
                        error_catalog[error_type] = 0
                        items_by_error[error_type] = []

                    error_catalog[error_type] += 1
                    items_by_error[error_type].append(
                        {"item_path": item_path, "signature": signature}
                    )

            # Sort by frequency
            sorted_errors = sorted(
                error_catalog.items(), key=lambda x: x[1], reverse=True
            )

            return {
                "crate_name": crate_name,
                "pattern_filter": pattern,
                "error_types": [
                    {
                        "error_type": error_type,
                        "occurrence_count": count,
                        "example_items": items_by_error[error_type][
                            :3
                        ],  # First 3 examples
                    }
                    for error_type, count in sorted_errors
                ],
                "total_error_types": len(sorted_errors),
            }
