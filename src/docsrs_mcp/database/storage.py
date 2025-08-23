"""Database storage operations for crate metadata and relationships."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..config import DB_TIMEOUT

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def store_crate_metadata(
    db_path: Path,
    name: str,
    version: str,
    description: str,
    repository: str | None = None,
    documentation: str | None = None,
) -> int:
    """Store crate metadata and return the crate ID."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            INSERT INTO crate_metadata (name, version, description, repository, documentation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, version, description, repository, documentation),
        )
        await db.commit()
        return cursor.lastrowid


async def store_reexports(db_path: Path, crate_id: int, reexports: list[dict]) -> None:
    """Store discovered re-export mappings and cross-references for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the parent crate
        reexports: List of re-export/crossref dicts with alias_path, actual_path, is_glob,
                  and optionally link_text, link_type, target_item_id, confidence_score
    """
    if not reexports:
        return

    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Prepare batch data
        reexport_data = [
            (
                crate_id,
                reexport["alias_path"],
                reexport["actual_path"],
                reexport.get("is_glob", False),
                reexport.get("link_text", None),
                reexport.get("link_type", "reexport"),
                reexport.get("target_item_id", None),
                reexport.get("confidence_score", 1.0),
            )
            for reexport in reexports
        ]

        # Batch insert with IGNORE for duplicates
        await db.executemany(
            """
            INSERT OR IGNORE INTO reexports (
                crate_id, alias_path, actual_path, is_glob,
                link_text, link_type, target_item_id, confidence_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            reexport_data,
        )

        await db.commit()

        # Count by type for logging
        crossref_count = sum(1 for r in reexports if r.get("link_type") == "crossref")
        reexport_count = len(reexports) - crossref_count

        if crossref_count > 0:
            logger.info(
                f"Stored {crossref_count} cross-references and {reexport_count} re-export mappings"
            )
        else:
            logger.info(f"Stored {reexport_count} re-export mappings")


async def store_modules(db_path: Path, crate_id: int, modules: dict) -> None:
    """Store module hierarchy for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the parent crate
        modules: Dict of module_id -> module info
    """
    if not modules:
        return

    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Build module ID mapping (rustdoc ID -> database ID)
        id_mapping = {}

        # First, insert all modules without parent_id
        for module_id, module_info in modules.items():
            cursor = await db.execute(
                """
                INSERT INTO modules (name, path, crate_id, depth, item_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    module_info["name"],
                    module_info["path"],
                    crate_id,
                    module_info["depth"],
                    module_info["item_count"],
                ),
            )
            id_mapping[module_id] = cursor.lastrowid

        # Update parent_id references
        for module_id, module_info in modules.items():
            parent_rustdoc_id = module_info.get("parent_id")
            if parent_rustdoc_id and parent_rustdoc_id in id_mapping:
                await db.execute(
                    """
                    UPDATE modules SET parent_id = ? WHERE id = ?
                    """,
                    (id_mapping[parent_rustdoc_id], id_mapping[module_id]),
                )

        await db.commit()
        logger.info(f"Stored {len(modules)} modules with hierarchy")


__all__ = [
    "store_crate_metadata",
    "store_reexports",
    "store_modules",
]
