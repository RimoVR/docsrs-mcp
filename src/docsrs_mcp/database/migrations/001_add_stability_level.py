"""Migration to add stability_level column to embeddings table.

This migration adds support for filtering search results by API stability level
(stable, unstable, experimental).
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
    """Add stability_level column to embeddings table."""
    async with aiosqlite.connect(db_path) as db:
        # Check if column already exists
        cursor = await db.execute(
            "SELECT COUNT(*) FROM pragma_table_info('embeddings') WHERE name='stability_level'"
        )
        column_exists = (await cursor.fetchone())[0] > 0

        if not column_exists:
            logger.info("Adding stability_level column to embeddings table")

            # Add stability_level column with default value 'stable'
            await db.execute("""
                ALTER TABLE embeddings 
                ADD COLUMN stability_level TEXT DEFAULT 'stable'
            """)

            # Create partial index for stable items (most common case)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_stability 
                ON embeddings(stability_level) 
                WHERE stability_level = 'stable'
            """)

            # Create composite index for efficient filtering
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_stability_type 
                ON embeddings(stability_level, item_type)
            """)

            # Update existing records based on deprecated flag
            # Deprecated items are considered unstable
            await db.execute("""
                UPDATE embeddings 
                SET stability_level = 'unstable' 
                WHERE deprecated = 1
            """)

            await db.commit()
            logger.info("Successfully added stability_level column and indices")
        else:
            logger.info("stability_level column already exists, skipping migration")


async def downgrade(db_path: Path) -> None:
    """Remove stability_level column from embeddings table.

    Note: SQLite doesn't support DROP COLUMN directly, so this creates
    a new table without the column and migrates data.
    """
    async with aiosqlite.connect(db_path) as db:
        logger.info("Removing stability_level column from embeddings table")

        # Create temporary table without stability_level
        await db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_path TEXT NOT NULL,
                header TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                item_type TEXT,
                signature TEXT,
                parent_id TEXT,
                examples TEXT,
                visibility TEXT DEFAULT 'public',
                deprecated BOOLEAN DEFAULT 0,
                generic_params TEXT DEFAULT NULL,
                trait_bounds TEXT DEFAULT NULL,
                UNIQUE(item_path)
            )
        """)

        # Copy data to new table
        await db.execute("""
            INSERT INTO embeddings_new 
            SELECT id, item_path, header, content, embedding, item_type, 
                   signature, parent_id, examples, visibility, deprecated,
                   generic_params, trait_bounds
            FROM embeddings
        """)

        # Drop indices
        await db.execute("DROP INDEX IF EXISTS idx_embeddings_stability")
        await db.execute("DROP INDEX IF EXISTS idx_embeddings_stability_type")

        # Drop old table and rename new one
        await db.execute("DROP TABLE embeddings")
        await db.execute("ALTER TABLE embeddings_new RENAME TO embeddings")

        await db.commit()
        logger.info("Successfully removed stability_level column")


async def check_migration_status(db_path: Path) -> bool:
    """Check if this migration has been applied."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM pragma_table_info('embeddings') WHERE name='stability_level'"
        )
        return (await cursor.fetchone())[0] > 0
