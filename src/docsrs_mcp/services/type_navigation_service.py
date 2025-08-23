"""Service layer for type system navigation operations."""

import json
import logging
from functools import lru_cache
from typing import Any

import aiosqlite

from ..database import DB_TIMEOUT, execute_with_retry, get_db_path
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

            implementations = []
            async for row in cursor:
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
            hints.append(f"Inherent method takes precedence by default")

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
