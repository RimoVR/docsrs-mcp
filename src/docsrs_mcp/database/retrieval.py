"""Database retrieval operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ..config import DB_TIMEOUT

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def get_discovered_reexports(
    db_path: Path,
    crate_name: str,
    version: str | None = None,
    include_crossrefs: bool = False,
) -> dict[str, str]:
    """Get auto-discovered re-exports and optionally cross-references for a crate."""
    import aiosqlite

    reexport_map = {}

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Query re-exports for this crate
            if include_crossrefs:
                query = """
                    SELECT r.alias_path, r.actual_path, r.is_glob, r.link_type
                    FROM reexports r
                    JOIN crate_metadata c ON r.crate_id = c.id
                    WHERE c.name = ?
                """
            else:
                query = """
                    SELECT r.alias_path, r.actual_path, r.is_glob
                    FROM reexports r
                    JOIN crate_metadata c ON r.crate_id = c.id
                    WHERE c.name = ? AND r.link_type = 'reexport'
                """

            cursor = await db.execute(query, (crate_name,))

            async for row in cursor:
                if include_crossrefs and len(row) == 4:
                    alias_path, actual_path, is_glob, link_type = row
                else:
                    alias_path, actual_path, is_glob = row

                # Store mapping
                reexport_map[alias_path] = actual_path

                # Also store without crate prefix for convenience
                if alias_path.startswith(f"{crate_name}::"):
                    short_alias = alias_path[len(crate_name) + 2 :]
                    reexport_map[short_alias] = actual_path

            if reexport_map:
                logger.debug(f"Loaded {len(reexport_map)} mappings for {crate_name}")
    except Exception as e:
        logger.warning(f"Error loading mappings for {crate_name}: {e}")

    return reexport_map


async def get_cross_references(
    db_path: Path, item_path: str, direction: str = "both"
) -> dict[str, list[dict]]:
    """Get cross-references for a specific item."""
    import aiosqlite

    result = {}

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Get outgoing cross-references (this item links to others)
            if direction in ["from", "both"]:
                cursor = await db.execute(
                    """
                    SELECT actual_path, link_text, confidence_score
                    FROM reexports
                    WHERE alias_path = ? AND link_type = 'crossref'
                    """,
                    (item_path,),
                )

                from_refs = []
                async for row in cursor:
                    actual_path, link_text, confidence = row
                    from_refs.append(
                        {
                            "target": actual_path,
                            "link_text": link_text,
                            "confidence": confidence,
                        }
                    )

                result["from"] = from_refs

            # Get incoming cross-references (others link to this item)
            if direction in ["to", "both"]:
                cursor = await db.execute(
                    """
                    SELECT alias_path, link_text, confidence_score
                    FROM reexports
                    WHERE actual_path = ? AND link_type = 'crossref'
                    """,
                    (item_path,),
                )

                to_refs = []
                async for row in cursor:
                    alias_path, link_text, confidence = row
                    to_refs.append(
                        {
                            "source": alias_path,
                            "link_text": link_text,
                            "confidence": confidence,
                        }
                    )

                result["to"] = to_refs

    except Exception as e:
        logger.warning(f"Error getting cross-references for {item_path}: {e}")

    return result


async def get_module_tree(db_path: Path, crate_id: int | None = None) -> list[dict]:
    """Get the module hierarchy tree for a crate."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        if crate_id:
            # Get modules for specific crate with recursive CTE
            query = """
                WITH RECURSIVE module_tree AS (
                    -- Anchor: root modules (no parent)
                    SELECT id, name, path, parent_id, depth, item_count, 0 as level
                    FROM modules
                    WHERE crate_id = ? AND parent_id IS NULL

                    UNION ALL

                    -- Recursive: child modules
                    SELECT m.id, m.name, m.path, m.parent_id, m.depth, m.item_count, mt.level + 1
                    FROM modules m
                    INNER JOIN module_tree mt ON m.parent_id = mt.id
                )
                SELECT * FROM module_tree ORDER BY path
            """
            cursor = await db.execute(query, (crate_id,))
        else:
            # Get all modules
            query = """
                SELECT id, name, path, parent_id, depth, item_count
                FROM modules
                ORDER BY path
            """
            cursor = await db.execute(query)

        rows = await cursor.fetchall()

        # Convert to list of dicts
        modules = []
        for row in rows:
            modules.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "path": row[2],
                    "parent_id": row[3],
                    "depth": row[4],
                    "item_count": row[5],
                }
            )

        return modules


async def get_module_by_path(db_path: Path, module_path: str) -> dict | None:
    """Get a specific module by its path."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            "SELECT id, name, path, parent_id, depth, item_count FROM modules WHERE path = ?",
            (module_path,),
        )
        row = await cursor.fetchone()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "path": row[2],
                "parent_id": row[3],
                "depth": row[4],
                "item_count": row[5],
            }

        return None


async def get_all_items_for_version(db_path: Path) -> list[dict[str, Any]]:
    """Get all items for a specific version from the database."""
    import aiosqlite

    items = []
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            SELECT item_path, header, content, item_type, signature, parent_id, examples
            FROM embeddings
            """
        )
        async for row in cursor:
            items.append(
                {
                    "item_path": row[0],
                    "header": row[1],
                    "content": row[2],
                    "item_type": row[3],
                    "signature": row[4],
                    "parent_id": row[5],
                    "examples": row[6],
                }
            )
    return items


async def get_item_signatures_batch(
    db_path: Path, item_paths: list[str]
) -> dict[str, str]:
    """Get signatures for multiple items at once."""
    import aiosqlite

    signatures = {}
    if not item_paths:
        return signatures

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Use parameterized query with IN clause
        placeholders = ",".join("?" * len(item_paths))
        query = f"""
            SELECT item_path, signature
            FROM embeddings
            WHERE item_path IN ({placeholders})
              AND signature IS NOT NULL
        """

        cursor = await db.execute(query, item_paths)
        async for row in cursor:
            signatures[row[0]] = row[1]

    return signatures


async def get_module_items(db_path: Path, module_path: str) -> list[dict[str, Any]]:
    """Get all items within a specific module."""
    import aiosqlite

    items = []
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Query items that belong to this module
        query = """
            SELECT item_path, header, content, item_type, signature
            FROM embeddings
            WHERE item_path LIKE ? OR item_path = ?
            ORDER BY item_path
        """

        cursor = await db.execute(query, (f"{module_path}::%", module_path))

        async for row in cursor:
            items.append(
                {
                    "item_path": row[0],
                    "header": row[1],
                    "content": row[2],
                    "item_type": row[3],
                    "signature": row[4],
                }
            )

    return items


async def query_trait_implementations(
    db_path: Path,
    crate_id: int,
    trait_path: str | None = None,
    type_path: str | None = None,
) -> list[dict[str, Any]]:
    """Query trait implementations from the database.

    Args:
        db_path: Path to the database
        crate_id: ID of the crate
        trait_path: Optional trait path to filter by
        type_path: Optional type path to filter by

    Returns:
        List of trait implementation records
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Build query based on filters
        query_parts = ["SELECT * FROM trait_implementations WHERE crate_id = ?"]
        params = [crate_id]

        if trait_path:
            query_parts.append("AND trait_path = ?")
            params.append(trait_path)

        if type_path:
            query_parts.append("AND impl_type_path = ?")
            params.append(type_path)

        query = " ".join(query_parts) + " ORDER BY trait_path, impl_type_path"

        cursor = await db.execute(query, params)
        columns = [col[0] for col in cursor.description]

        results = []
        async for row in cursor:
            results.append(dict(zip(columns, row)))

        return results


async def query_method_signatures(
    db_path: Path, crate_id: int, parent_type_path: str, method_name: str | None = None
) -> list[dict[str, Any]]:
    """Query method signatures from the database.

    Args:
        db_path: Path to the database
        crate_id: ID of the crate
        parent_type_path: Parent type path
        method_name: Optional method name to filter by

    Returns:
        List of method signature records
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        query = """
            SELECT * FROM method_signatures 
            WHERE crate_id = ? AND parent_type_path = ?
        """
        params = [crate_id, parent_type_path]

        if method_name:
            query += " AND method_name = ?"
            params.append(method_name)

        query += " ORDER BY method_kind, method_name"

        cursor = await db.execute(query, params)
        columns = [col[0] for col in cursor.description]

        results = []
        async for row in cursor:
            results.append(dict(zip(columns, row)))

        return results


async def query_associated_items(
    db_path: Path, crate_id: int, container_path: str, item_kind: str | None = None
) -> list[dict[str, Any]]:
    """Query associated items from the database.

    Args:
        db_path: Path to the database
        crate_id: ID of the crate
        container_path: Container trait/type path
        item_kind: Optional item kind filter

    Returns:
        List of associated item records
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        query = """
            SELECT * FROM associated_items 
            WHERE crate_id = ? AND container_path = ?
        """
        params = [crate_id, container_path]

        if item_kind:
            query += " AND item_kind = ?"
            params.append(item_kind)

        query += " ORDER BY item_kind, item_name"

        cursor = await db.execute(query, params)
        columns = [col[0] for col in cursor.description]

        results = []
        async for row in cursor:
            results.append(dict(zip(columns, row)))

        return results


async def query_generic_constraints(
    db_path: Path, crate_id: int, item_path: str
) -> list[dict[str, Any]]:
    """Query generic constraints from the database.

    Args:
        db_path: Path to the database
        crate_id: ID of the crate
        item_path: Item path to get constraints for

    Returns:
        List of generic constraint records
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            SELECT * FROM generic_constraints 
            WHERE crate_id = ? AND item_path = ?
            ORDER BY position, param_name
            """,
            (crate_id, item_path),
        )

        columns = [col[0] for col in cursor.description]

        results = []
        async for row in cursor:
            results.append(dict(zip(columns, row)))

        return results


async def get_all_item_paths(db_path: Path) -> list[str]:
    """Get all item paths from the database for fuzzy matching."""
    import aiosqlite

    if not db_path.exists():
        return []

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            "SELECT DISTINCT item_path FROM embeddings WHERE item_path IS NOT NULL"
        )
        paths = [row[0] for row in await cursor.fetchall()]

        logger.debug(f"Retrieved {len(paths)} unique item paths from database")
        return paths


__all__ = [
    "get_discovered_reexports",
    "get_cross_references",
    "get_module_tree",
    "get_module_by_path",
    "get_all_items_for_version",
    "get_item_signatures_batch",
    "get_module_items",
    "get_all_item_paths",
]
