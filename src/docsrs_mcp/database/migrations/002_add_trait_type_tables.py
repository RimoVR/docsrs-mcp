"""Migration to add trait/type system navigation tables for Phase 4.

This migration adds comprehensive support for trait implementations, method resolution,
associated items, and generic constraints to enable deep type system navigation.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import structlog

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def upgrade(db_path: Path) -> None:
    """Add trait/type system navigation tables."""
    async with aiosqlite.connect(db_path) as db:
        logger.info("Adding trait/type system navigation tables for Phase 4")

        # Create trait_implementations table for trait-to-type mappings
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trait_implementations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                trait_path TEXT NOT NULL,
                impl_type_path TEXT NOT NULL,
                generic_params TEXT DEFAULT NULL,
                where_clauses TEXT DEFAULT NULL,
                is_blanket BOOLEAN DEFAULT 0,
                is_negative BOOLEAN DEFAULT 0,
                impl_signature TEXT,
                source_location TEXT,
                stability_level TEXT DEFAULT 'stable',
                item_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, trait_path, impl_type_path, generic_params)
            )
        """)

        # Create method_signatures table for method resolution
        await db.execute("""
            CREATE TABLE IF NOT EXISTS method_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                parent_type_path TEXT NOT NULL,
                method_name TEXT NOT NULL,
                full_signature TEXT NOT NULL,
                generic_params TEXT DEFAULT NULL,
                where_clauses TEXT DEFAULT NULL,
                return_type TEXT,
                is_async BOOLEAN DEFAULT 0,
                is_unsafe BOOLEAN DEFAULT 0,
                is_const BOOLEAN DEFAULT 0,
                visibility TEXT DEFAULT 'pub',
                method_kind TEXT NOT NULL CHECK(method_kind IN ('inherent', 'trait', 'static')),
                trait_source TEXT DEFAULT NULL,
                receiver_type TEXT DEFAULT NULL,
                stability_level TEXT DEFAULT 'stable',
                item_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, parent_type_path, method_name, full_signature)
            )
        """)

        # Create associated_items table for associated types/constants/functions
        await db.execute("""
            CREATE TABLE IF NOT EXISTS associated_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                container_path TEXT NOT NULL,
                item_name TEXT NOT NULL,
                item_kind TEXT NOT NULL CHECK(item_kind IN ('type', 'const', 'function')),
                item_signature TEXT NOT NULL,
                default_value TEXT DEFAULT NULL,
                generic_params TEXT DEFAULT NULL,
                where_clauses TEXT DEFAULT NULL,
                visibility TEXT DEFAULT 'pub',
                stability_level TEXT DEFAULT 'stable',
                item_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, container_path, item_name, item_kind)
            )
        """)

        # Create generic_constraints table for type bounds and lifetime parameters
        await db.execute("""
            CREATE TABLE IF NOT EXISTS generic_constraints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                item_path TEXT NOT NULL,
                param_name TEXT NOT NULL,
                param_kind TEXT NOT NULL CHECK(param_kind IN ('type', 'lifetime', 'const')),
                bounds TEXT DEFAULT NULL,
                default_value TEXT DEFAULT NULL,
                variance TEXT DEFAULT NULL,
                position INTEGER DEFAULT 0,
                item_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, item_path, param_name)
            )
        """)

        # Create optimized indexes for trait_implementations
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trait_impl_by_trait
            ON trait_implementations(crate_id, trait_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trait_impl_by_type
            ON trait_implementations(crate_id, impl_type_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trait_impl_blanket
            ON trait_implementations(crate_id, is_blanket)
            WHERE is_blanket = 1
        """)

        # Create optimized indexes for method_signatures
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_method_by_parent
            ON method_signatures(crate_id, parent_type_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_method_by_name
            ON method_signatures(crate_id, method_name)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_method_trait_source
            ON method_signatures(crate_id, trait_source)
            WHERE trait_source IS NOT NULL
        """)

        # Create optimized indexes for associated_items
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_assoc_by_container
            ON associated_items(crate_id, container_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_assoc_by_kind
            ON associated_items(crate_id, item_kind)
        """)

        # Create optimized indexes for generic_constraints
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_generic_by_item
            ON generic_constraints(crate_id, item_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_generic_by_kind
            ON generic_constraints(crate_id, param_kind)
        """)

        await db.commit()
        logger.info(
            "Successfully added trait/type system navigation tables with indexes"
        )


async def downgrade(db_path: Path) -> None:
    """Remove trait/type system navigation tables.

    Note: This will permanently delete all trait/type navigation data.
    """
    async with aiosqlite.connect(db_path) as db:
        logger.info("Removing trait/type system navigation tables")

        # Drop indexes first
        indexes_to_drop = [
            "idx_trait_impl_by_trait",
            "idx_trait_impl_by_type",
            "idx_trait_impl_blanket",
            "idx_method_by_parent",
            "idx_method_by_name",
            "idx_method_trait_source",
            "idx_assoc_by_container",
            "idx_assoc_by_kind",
            "idx_generic_by_item",
            "idx_generic_by_kind",
        ]

        for index_name in indexes_to_drop:
            await db.execute(f"DROP INDEX IF EXISTS {index_name}")

        # Drop tables
        await db.execute("DROP TABLE IF EXISTS generic_constraints")
        await db.execute("DROP TABLE IF EXISTS associated_items")
        await db.execute("DROP TABLE IF EXISTS method_signatures")
        await db.execute("DROP TABLE IF EXISTS trait_implementations")

        await db.commit()
        logger.info("Successfully removed trait/type system navigation tables")


async def is_applied(db_path: Path) -> bool:
    """Check if this migration has been applied."""
    async with aiosqlite.connect(db_path) as db:
        # Check if trait_implementations table exists
        cursor = await db.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='table' AND name='trait_implementations'
        """)
        return (await cursor.fetchone())[0] > 0
