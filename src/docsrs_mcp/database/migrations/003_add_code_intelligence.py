"""Migration to add code intelligence columns for Phase 5.

This migration extends the database schema to support comprehensive code intelligence
including safety information, error types, and feature requirements.
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
    """Add code intelligence columns to embeddings table."""
    async with aiosqlite.connect(db_path) as db:
        logger.info("Adding code intelligence columns for Phase 5")

        # Add safety information column
        await db.execute("""
            ALTER TABLE embeddings 
            ADD COLUMN safety_info TEXT DEFAULT NULL
        """)
        logger.info("Added safety_info column to embeddings table")

        # Add error types column for Result<T, E> patterns
        await db.execute("""
            ALTER TABLE embeddings 
            ADD COLUMN error_types TEXT DEFAULT NULL
        """)
        logger.info("Added error_types column to embeddings table")

        # Add feature requirements column for cfg attributes
        await db.execute("""
            ALTER TABLE embeddings 
            ADD COLUMN feature_requirements TEXT DEFAULT NULL
        """)
        logger.info("Added feature_requirements column to embeddings table")

        # Add is_safe flag for quick filtering
        await db.execute("""
            ALTER TABLE embeddings 
            ADD COLUMN is_safe BOOLEAN DEFAULT 1
        """)
        logger.info("Added is_safe column to embeddings table")

        # Create performance indexes for filtered queries
        # Partial index for unsafe items (much smaller than full index)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_unsafe_items 
            ON embeddings(item_type, crate_id) 
            WHERE is_safe = 0
        """)
        logger.info("Created partial index for unsafe items")

        # Partial index for items with error types
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_error_items 
            ON embeddings(item_type, crate_id) 
            WHERE error_types IS NOT NULL
        """)
        logger.info("Created partial index for items with error types")

        # Partial index for items with feature requirements
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_items 
            ON embeddings(crate_id) 
            WHERE feature_requirements IS NOT NULL
        """)
        logger.info("Created partial index for items with feature requirements")

        # Commit all changes
        await db.commit()
        logger.info("Successfully added code intelligence columns and indexes")


async def downgrade(db_path: Path) -> None:
    """Remove code intelligence columns from embeddings table."""
    async with aiosqlite.connect(db_path) as db:
        logger.info("Removing code intelligence columns for Phase 5 rollback")

        # Drop indexes first
        await db.execute("DROP INDEX IF EXISTS idx_unsafe_items")
        await db.execute("DROP INDEX IF EXISTS idx_error_items")
        await db.execute("DROP INDEX IF EXISTS idx_feature_items")
        logger.info("Dropped code intelligence indexes")

        # SQLite doesn't support DROP COLUMN directly, need to recreate table
        # First, create a temporary table with the original schema
        await db.execute("""
            CREATE TABLE embeddings_temp AS 
            SELECT 
                id, crate_id, item_path, item_type, embedding, signature,
                description, module_path, visibility, deprecated, generic_params,
                trait_bounds, return_type, parameters, examples, attributes,
                source_file, source_line, documentation_url, created_at
            FROM embeddings
        """)

        # Drop the original table
        await db.execute("DROP TABLE embeddings")

        # Rename temp table to original name
        await db.execute("ALTER TABLE embeddings_temp RENAME TO embeddings")

        # Recreate original indexes
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_crate_id 
            ON embeddings(crate_id)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_item_type 
            ON embeddings(item_type)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_module_path 
            ON embeddings(module_path)
        """)

        await db.commit()
        logger.info("Successfully removed code intelligence columns")
