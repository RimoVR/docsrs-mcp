"""Database schema initialization and migrations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..config import DB_TIMEOUT, EMBEDDING_DIM
from .connection import execute_with_retry, load_sqlite_vec_extension

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def init_database(db_path: Path) -> None:
    """Initialize database with required tables and extensions."""
    import aiosqlite

    from .migrations import run_migrations

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Enable WAL mode for better concurrency
        await db.execute("PRAGMA journal_mode=WAL")

        # Load sqlite-vec extension
        await load_sqlite_vec_extension(db)

        # Create tables
        await db.execute("""
            CREATE TABLE IF NOT EXISTS crate_metadata (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                repository TEXT,
                documentation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                crate_id INTEGER,
                parent_id INTEGER,
                depth INTEGER DEFAULT 0,
                item_count INTEGER DEFAULT 0,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id),
                FOREIGN KEY (parent_id) REFERENCES modules(id)
            )
        """)

        # Create reexports table for auto-discovered re-export mappings and cross-references
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reexports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                alias_path TEXT NOT NULL,
                actual_path TEXT NOT NULL,
                is_glob BOOLEAN DEFAULT 0,
                link_text TEXT,
                link_type TEXT DEFAULT 'reexport',
                target_item_id TEXT,
                confidence_score REAL DEFAULT 1.0,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, alias_path, actual_path, link_type)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
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

        # Create vector index if it doesn't exist
        await db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # Note: Triggers cannot be used with vec0 virtual tables because the sqlite-vec
        # extension is not available in the trigger execution context. We use manual
        # synchronization in _store_batch instead.

        # Create indexes for module hierarchy
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_parent
            ON modules(parent_id)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_depth
            ON modules(depth)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_crate
            ON modules(crate_id, parent_id)
        """)

        # Create index for reexports lookup performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_lookup
            ON reexports(crate_id, alias_path)
        """)

        # Create composite indexes for cross-reference lookups
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_crossref_forward
            ON reexports(crate_id, alias_path, link_text)
            WHERE link_type = 'crossref'
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_crossref_reverse
            ON reexports(crate_id, target_item_id)
            WHERE link_type = 'crossref'
        """)

        # Create composite index for filtering performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_filter_composite
            ON embeddings(item_type, item_path)
        """)

        # Create compound index for common filter combinations
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_filter_combo
            ON embeddings(item_type, visibility, deprecated)
        """)

        # Create partial indexes for common filter patterns (performance optimization)
        # Partial index for non-deprecated items (most searches exclude deprecated)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_non_deprecated
            ON embeddings(item_type, item_path)
            WHERE deprecated = 0
        """)

        # Partial index for public items with specific types
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_public_functions
            ON embeddings(item_path, content)
            WHERE visibility = 'public' AND item_type = 'function'
        """)

        # Partial index for items with examples
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_has_examples
            ON embeddings(item_type, item_path)
            WHERE examples IS NOT NULL
        """)

        # Partial index for crate-specific searches (common pattern)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_crate_prefix
            ON embeddings(item_path, item_type)
            WHERE item_path GLOB '*::*'
        """)

        # Create example embeddings table for dedicated example search
        await db.execute("""
            CREATE TABLE IF NOT EXISTS example_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                item_path TEXT NOT NULL,
                crate_name TEXT NOT NULL,
                version TEXT NOT NULL,
                example_hash TEXT NOT NULL,
                example_text TEXT NOT NULL,
                language TEXT,
                context TEXT,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(crate_name, version, example_hash)
            )
        """)

        # Create virtual table for example vector search - always recreate for fresh database
        await db.execute("DROP TABLE IF EXISTS vec_example_embeddings")
        await db.execute(f"""
            CREATE VIRTUAL TABLE vec_example_embeddings USING vec0(
                example_embedding float[{EMBEDDING_DIM}]
            )
        """)

        # Create indexes for example embeddings performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_example_crate_version 
            ON example_embeddings(crate_name, version)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_example_hash 
            ON example_embeddings(example_hash)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_example_item_path
            ON example_embeddings(item_path)
        """)

        # Run ANALYZE to update query planner statistics
        await db.execute("ANALYZE")

        await db.commit()

        # Create ingestion status tracking table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_status (
                crate_id INTEGER PRIMARY KEY REFERENCES crate_metadata(id),
                status TEXT NOT NULL CHECK(status IN ('started', 'downloading', 'processing', 'completed', 'failed')),
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                error_message TEXT,
                items_processed INTEGER DEFAULT 0,
                total_items INTEGER,
                checkpoint_data TEXT,  -- JSON for extensibility
                ingestion_tier TEXT CHECK(ingestion_tier IN ('rustdoc_json', 'source_extraction', 'description_only', 'rust_lang_stdlib', NULL))
            )
        """)

        # Create partial index for fast incomplete detection (O(1) queries)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_incomplete_ingestions 
            ON ingestion_status(updated_at) 
            WHERE status != 'completed'
        """)

        await db.commit()

    # Run database migrations after initial schema creation
    await run_migrations(db_path)
    logger.info("Database initialization and migrations complete")


async def migrate_database_duplicates(db_path: Path) -> None:
    """Clean duplicate embeddings and add UNIQUE constraint if needed."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if UNIQUE constraint already exists
        cursor = await db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='embeddings' AND sql LIKE '%UNIQUE%'
        """)

        result = await cursor.fetchone()
        if result:
            # Already has UNIQUE constraint
            return

        logger.info("Migrating database to add UNIQUE constraint on embeddings")

        # First, identify duplicates
        duplicate_counts = await db.execute("""
            SELECT item_path, COUNT(*) as count 
            FROM embeddings 
            GROUP BY item_path 
            HAVING COUNT(*) > 1
        """)

        duplicates = []
        async for row in duplicate_counts:
            duplicates.append(row[0])

        if duplicates:
            logger.warning(
                f"Found {len(duplicates)} duplicate item_paths, cleaning up..."
            )

            # Keep only the most recent (highest ID) for each duplicate
            for item_path in duplicates:
                # Get the highest ID for this item_path
                max_id_result = await db.execute(
                    "SELECT MAX(id) FROM embeddings WHERE item_path = ?", (item_path,)
                )
                max_id = (await max_id_result.fetchone())[0]

                # Delete all except the most recent
                await execute_with_retry(
                    db,
                    "DELETE FROM embeddings WHERE item_path = ? AND id != ?",
                    (item_path, max_id),
                )

            logger.info(f"Cleaned up {len(duplicates)} duplicate embeddings")

        # Create new table with UNIQUE constraint
        await db.execute("""
            CREATE TABLE embeddings_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_path TEXT NOT NULL UNIQUE,
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
                trait_bounds TEXT DEFAULT NULL
            )
        """)

        # Copy data to new table
        await db.execute("""
            INSERT INTO embeddings_new 
            SELECT id, item_path, header, content, embedding, item_type, signature, 
                   parent_id, examples, visibility, deprecated, generic_params, trait_bounds
            FROM embeddings
        """)

        # Drop old table and rename new
        await db.execute("DROP TABLE embeddings")
        await db.execute("ALTER TABLE embeddings_new RENAME TO embeddings")

        # Rebuild the vec_embeddings virtual table to sync with cleaned embeddings
        await db.execute("DROP TABLE IF EXISTS vec_embeddings")

        # Load sqlite-vec extension
        await load_sqlite_vec_extension(db)

        await db.execute(f"""
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # Repopulate vec_embeddings from cleaned embeddings
        cursor = await db.execute("SELECT rowid, embedding FROM embeddings")
        vec_data = []
        async for row in cursor:
            rowid, embedding_blob = row
            vec_data.append((rowid, embedding_blob))

            # Process in batches of 100 for efficiency
            if len(vec_data) >= 100:
                await db.executemany(
                    "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    vec_data,
                )
                vec_data = []

        # Insert remaining data
        if vec_data:
            await db.executemany(
                "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)", vec_data
            )

        vec_count = await db.execute("SELECT COUNT(*) FROM vec_embeddings")
        vec_total = (await vec_count.fetchone())[0]
        logger.info(f"Rebuilt vec_embeddings with {vec_total} entries")

        await db.commit()
        logger.info(
            "Successfully migrated database with UNIQUE constraint and synced vector index"
        )


async def migrate_reexports_for_crossrefs(db_path: Path) -> None:
    """Migrate existing reexports table to support cross-references."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if migration is needed by checking for link_text column
        cursor = await db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='reexports'
        """)

        result = await cursor.fetchone()
        if result and "link_text" in result[0]:
            # Already has the columns for cross-reference support
            return

        logger.info("Migrating reexports table to support cross-references")

        # Create new table with additional columns
        await db.execute("""
            CREATE TABLE reexports_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                alias_path TEXT NOT NULL,
                actual_path TEXT NOT NULL,
                is_glob BOOLEAN DEFAULT 0,
                link_text TEXT,
                link_type TEXT DEFAULT 'reexport',
                target_item_id TEXT,
                confidence_score REAL DEFAULT 1.0,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, alias_path, actual_path, link_type)
            )
        """)

        # Copy existing data with default values for new columns
        await db.execute("""
            INSERT INTO reexports_new (id, crate_id, alias_path, actual_path, is_glob)
            SELECT id, crate_id, alias_path, actual_path, is_glob
            FROM reexports
        """)

        # Drop old table and rename new
        await db.execute("DROP TABLE reexports")
        await db.execute("ALTER TABLE reexports_new RENAME TO reexports")

        # Recreate indexes
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_lookup
            ON reexports(crate_id, alias_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_crossref_forward
            ON reexports(crate_id, alias_path, link_text)
            WHERE link_type = 'crossref'
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reexports_crossref_reverse
            ON reexports(crate_id, target_item_id)
            WHERE link_type = 'crossref'
        """)

        await db.commit()
        logger.info("Successfully migrated reexports table for cross-reference support")


async def migrate_add_generics_metadata(db_path: Path) -> None:
    """Add generic_params and trait_bounds columns for richer metadata."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if migration is needed by checking for generic_params column
        cursor = await db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='embeddings'
        """)

        result = await cursor.fetchone()
        if result and "generic_params" in result[0]:
            # Already has the columns
            return

        logger.info("Migrating database to add generic_params and trait_bounds columns")

        # Add new columns if they don't exist
        try:
            await db.execute(
                "ALTER TABLE embeddings ADD COLUMN generic_params TEXT DEFAULT NULL"
            )
            logger.info("Added generic_params column")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise

        try:
            await db.execute(
                "ALTER TABLE embeddings ADD COLUMN trait_bounds TEXT DEFAULT NULL"
            )
            logger.info("Added trait_bounds column")
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise

        await db.commit()
        logger.info("Successfully added generic_params and trait_bounds columns")


async def migrate_add_ingestion_tracking(db_path: Path) -> None:
    """Add ingestion_status table for tracking completion and recovery."""
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if migration is needed by checking for ingestion_status table
        cursor = await db.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ingestion_status'
        """)

        if await cursor.fetchone():
            # Table already exists
            return

        logger.info("Migrating database to add ingestion tracking functionality")

        # Create ingestion_status table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_status (
                crate_id INTEGER PRIMARY KEY REFERENCES crate_metadata(id),
                status TEXT NOT NULL CHECK(status IN ('started', 'downloading', 'processing', 'completed', 'failed')),
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                error_message TEXT,
                items_processed INTEGER DEFAULT 0,
                total_items INTEGER,
                checkpoint_data TEXT,  -- JSON for extensibility
                ingestion_tier TEXT CHECK(ingestion_tier IN ('rustdoc_json', 'source_extraction', 'description_only', 'rust_lang_stdlib', NULL))
            )
        """)

        # Create partial index for fast incomplete detection
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_incomplete_ingestions 
            ON ingestion_status(updated_at) 
            WHERE status != 'completed'
        """)

        await db.commit()
        logger.info(
            "Successfully added ingestion_status table with tracking capabilities"
        )


__all__ = [
    "init_database",
    "migrate_database_duplicates",
    "migrate_reexports_for_crossrefs",
    "migrate_add_generics_metadata",
    "migrate_add_ingestion_tracking",
]
