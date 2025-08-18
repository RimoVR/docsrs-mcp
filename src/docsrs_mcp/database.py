"""Database operations with SQLite and sqlite-vec."""

import asyncio
import functools
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
import sqlite_vec
import structlog

from . import config as app_config
from .cache import get_search_cache
from .config import CACHE_DIR, DB_TIMEOUT, EMBEDDING_DIM

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)

# Prepared statement cache for common queries
_prepared_statements = {}


def performance_timer(operation_name: str):
    """Decorator to track performance of database operations.

    Logs execution time and provides detailed metrics for monitoring.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log performance metrics
                if elapsed_ms > 100:
                    logger.warning(
                        f"{operation_name} took {elapsed_ms:.2f}ms (slow operation)"
                    )
                else:
                    logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")

                # Add timing to result if it's a dict
                if isinstance(result, dict) and "metrics" not in result:
                    result["metrics"] = {"execution_time_ms": elapsed_ms}

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log performance metrics
                if elapsed_ms > 100:
                    logger.warning(
                        f"{operation_name} took {elapsed_ms:.2f}ms (slow operation)"
                    )
                else:
                    logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RetryableTransaction:
    """Decorator for retryable database transactions with exponential backoff and jitter.

    Implements exponential backoff with full jitter to prevent thundering herd problem
    when multiple processes encounter database locks simultaneously.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        jitter: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds (cap for exponential growth)
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def __call__(self, func: Callable) -> Callable:
        """Make the class callable as a decorator."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiosqlite.OperationalError, aiosqlite.DatabaseError) as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Check if error is retryable
                    if any(
                        msg in error_msg
                        for msg in ["locked", "busy", "database is locked"]
                    ):
                        if attempt < self.max_retries:
                            # Calculate delay with exponential backoff
                            delay = min(self.base_delay * (2**attempt), self.max_delay)

                            # Add jitter to prevent thundering herd
                            if self.jitter:
                                delay = random.uniform(0, delay)

                            logger.warning(
                                f"Database locked on attempt {attempt + 1}/{self.max_retries + 1}, "
                                f"retrying in {delay:.3f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                            continue

                    # Non-retryable error, raise immediately
                    logger.error(f"Non-retryable database error: {e}")
                    raise
                except Exception as e:
                    # Non-database error, don't retry
                    logger.error(f"Unexpected error in transaction: {e}")
                    raise

            # All retries exhausted
            logger.error(
                f"Transaction failed after {self.max_retries + 1} attempts: {last_exception}"
            )
            raise last_exception

        return wrapper


async def execute_with_retry(
    conn: aiosqlite.Connection,
    query: str,
    params: tuple | None = None,
    use_immediate: bool = True,
) -> aiosqlite.Cursor:
    """Execute a query with retry logic and BEGIN IMMEDIATE for write operations.

    Args:
        conn: Database connection
        query: SQL query to execute
        params: Query parameters
        use_immediate: Whether to use BEGIN IMMEDIATE for write operations

    Returns:
        Query cursor
    """
    # Get configured values from environment
    max_retries = int(os.environ.get("TRANSACTION_MAX_RETRIES", "3"))
    busy_timeout = int(os.environ.get("TRANSACTION_BUSY_TIMEOUT", "5000"))

    # Set busy timeout on connection
    await conn.execute(f"PRAGMA busy_timeout = {busy_timeout}")

    # Determine if this is a write operation
    is_write = any(
        query.strip().upper().startswith(cmd)
        for cmd in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
    )

    @RetryableTransaction(max_retries=max_retries)
    async def _execute():
        if is_write and use_immediate:
            # Use BEGIN IMMEDIATE to avoid lock upgrades
            await conn.execute("BEGIN IMMEDIATE")
            try:
                if params:
                    cursor = await conn.execute(query, params)
                else:
                    cursor = await conn.execute(query)
                await conn.commit()
                return cursor
            except Exception:
                await conn.rollback()
                raise
        # Read operation or explicit transaction management
        elif params:
            return await conn.execute(query, params)
        else:
            return await conn.execute(query)

    return await _execute()


async def get_db_path(crate_name: str, version: str) -> Path:
    """Get the database path for a specific crate and version."""
    # Sanitize names for filesystem
    safe_crate = crate_name.replace("/", "_").replace("\\", "_")
    safe_version = version.replace("/", "_").replace("\\", "_")

    db_dir = CACHE_DIR / safe_crate
    db_dir.mkdir(parents=True, exist_ok=True)

    return db_dir / f"{safe_version}.db"


async def migrate_database_duplicates(db_path: Path) -> None:
    """Clean duplicate embeddings and add UNIQUE constraint if needed."""
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if UNIQUE constraint already exists
        cursor = await db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='embeddings' AND sql LIKE '%UNIQUE%'
        """)
        if await cursor.fetchone():
            # Already has UNIQUE constraint
            return

        logger.info("Migrating database to add UNIQUE constraint on embeddings")

        # Step 1: Clean existing duplicates (keep the one with highest rowid)
        await db.execute("""
            DELETE FROM embeddings 
            WHERE rowid NOT IN (
                SELECT MAX(rowid) 
                FROM embeddings 
                GROUP BY item_path
            )
        """)

        duplicates_removed = db.total_changes
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate embeddings")

        # Step 2: We'll rebuild vec_embeddings after migration

        # Step 3: Create new table with UNIQUE constraint and AUTOINCREMENT
        await db.execute("""
            CREATE TABLE embeddings_new (
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

        # Step 4: Copy data to new table
        await db.execute("""
            INSERT INTO embeddings_new 
            SELECT * FROM embeddings
        """)

        # Step 5: Drop old table and rename new one
        await db.execute("DROP TABLE embeddings")
        await db.execute("ALTER TABLE embeddings_new RENAME TO embeddings")

        # Step 6: Recreate indexes
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_filter_composite
            ON embeddings(item_type, item_path)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_filter_combo
            ON embeddings(item_type, visibility, deprecated)
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_non_deprecated
            ON embeddings(item_type, item_path)
            WHERE deprecated = 0
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_public_functions
            ON embeddings(item_path, content)
            WHERE visibility = 'public' AND item_type = 'function'
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_has_examples
            ON embeddings(item_type, item_path)
            WHERE examples IS NOT NULL
        """)

        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_crate_prefix
            ON embeddings(item_path, item_type)
            WHERE item_path GLOB '*::*'
        """)

        # Step 7: Rebuild vec_embeddings table to sync with cleaned embeddings
        logger.info("Rebuilding vec_embeddings table to sync with cleaned data")

        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Drop and recreate vec_embeddings table
        await db.execute("DROP TABLE IF EXISTS vec_embeddings")
        await db.execute(f"""
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # Re-populate vec_embeddings with cleaned data
        cursor = await db.execute("""
            SELECT id, embedding FROM embeddings ORDER BY id
        """)

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
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Check if migration is needed by checking for link_text column
        cursor = await db.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='reexports'
        """)

        result = await cursor.fetchone()
        if result and "link_text" in result[0]:
            # Check if we need to fix the unique constraint
            if "UNIQUE(crate_id, alias_path, actual_path, link_type)" not in result[0]:
                logger.info("Fixing unique constraint for cross-references")

                # Create new table with correct constraint
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

                # Copy data
                await db.execute("""
                    INSERT INTO reexports_new 
                    SELECT * FROM reexports
                """)

                # Drop old table and rename new one
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
                logger.info("Successfully fixed unique constraint")
            return

        logger.info("Migrating reexports table to support cross-references")

        # Add new columns if they don't exist
        try:
            await db.execute("ALTER TABLE reexports ADD COLUMN link_text TEXT")
        except Exception:
            pass  # Column might already exist

        try:
            await db.execute(
                "ALTER TABLE reexports ADD COLUMN link_type TEXT DEFAULT 'reexport'"
            )
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE reexports ADD COLUMN target_item_id TEXT")
        except Exception:
            pass

        try:
            await db.execute(
                "ALTER TABLE reexports ADD COLUMN confidence_score REAL DEFAULT 1.0"
            )
        except Exception:
            pass

        # Create new indexes for cross-reference lookups
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
        except Exception:
            pass  # Column might already exist

        try:
            await db.execute(
                "ALTER TABLE embeddings ADD COLUMN trait_bounds TEXT DEFAULT NULL"
            )
            logger.info("Added trait_bounds column")
        except Exception:
            pass  # Column might already exist

        await db.commit()
        logger.info("Successfully added generic_params and trait_bounds columns")


async def migrate_add_ingestion_tracking(db_path: Path) -> None:
    """Add ingestion_status table for tracking completion and recovery."""
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
                checkpoint_data TEXT  -- JSON for extensibility
            )
        """)

        # Create partial index for fast incomplete detection (O(1) queries)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_incomplete_ingestions 
            ON ingestion_status(updated_at) 
            WHERE status != 'completed'
        """)

        await db.commit()
        logger.info(
            "Successfully added ingestion_status table with tracking capabilities"
        )


async def init_database(db_path: Path) -> None:
    """Initialize database with required tables and extensions."""
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Enable WAL mode for better concurrency
        await db.execute("PRAGMA journal_mode=WAL")

        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

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
                checkpoint_data TEXT  -- JSON for extensibility
            )
        """)

        # Create partial index for fast incomplete detection (O(1) queries)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_incomplete_ingestions 
            ON ingestion_status(updated_at) 
            WHERE status != 'completed'
        """)


async def store_crate_metadata(
    db_path: Path,
    name: str,
    version: str,
    description: str,
    repository: str | None = None,
    documentation: str | None = None,
) -> int:
    """Store crate metadata and return the crate ID."""
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


async def get_discovered_reexports(
    db_path: Path,
    crate_name: str,
    version: str | None = None,
    include_crossrefs: bool = False,
) -> dict[str, str]:
    """Get auto-discovered re-exports and optionally cross-references for a crate.

    Args:
        db_path: Path to database
        crate_name: Name of the crate
        version: Optional version (for logging/future use)
        include_crossrefs: Whether to include cross-references

    Returns:
        Dictionary mapping alias_path to actual_path
    """
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
    """Get cross-references for a specific item.

    Args:
        db_path: Path to database
        item_path: Path of the item to get cross-references for
        direction: "from" (outgoing), "to" (incoming), or "both"

    Returns:
        Dictionary with 'from' and/or 'to' lists of cross-reference dicts
    """
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
                    target_path, link_text, confidence = row
                    from_refs.append(
                        {
                            "target_path": target_path,
                            "link_text": link_text,
                            "confidence": confidence,
                        }
                    )

                if from_refs:
                    result["from"] = from_refs

            # Get incoming cross-references (other items link to this)
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
                    source_path, link_text, confidence = row
                    to_refs.append(
                        {
                            "source_path": source_path,
                            "link_text": link_text,
                            "confidence": confidence,
                        }
                    )

                if to_refs:
                    result["to"] = to_refs

    except Exception as e:
        logger.warning(f"Error loading cross-references for {item_path}: {e}")

    return result


async def store_modules(db_path: Path, crate_id: int, modules: dict) -> None:
    """Store module hierarchy for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the parent crate
        modules: Dict of module_id -> module info
    """
    if not modules:
        return

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


async def get_module_tree(db_path: Path, crate_id: int | None = None) -> list[dict]:
    """Get the module hierarchy tree for a crate.

    Args:
        db_path: Path to database
        crate_id: Optional crate ID to filter modules

    Returns:
        List of module dictionaries with hierarchy info
    """
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
    """Get a specific module by its path.

    Args:
        db_path: Path to database
        module_path: Module path (e.g., "tokio::runtime")

    Returns:
        Module dict or None if not found
    """
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


def _apply_mmr_diversification(
    ranked_results: list[tuple[float, str, str, str, str]],
    embeddings: list[np.ndarray],
    k: int,
    lambda_param: float,
) -> list[tuple[float, str, str, str]]:
    """
    Apply Maximum Marginal Relevance (MMR) for result diversification.

    MMR balances relevance and diversity using the formula:
    MMR = λ * relevance + (1-λ) * diversity

    Enhanced with semantic similarity between embeddings for better diversity.

    Args:
        ranked_results: List of (score, path, header, content, item_type) tuples
        embeddings: List of numpy arrays containing embeddings for each result
        k: Number of results to select
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of diversified results as (score, path, header, content) tuples
    """
    if not ranked_results:
        return []

    # Start with the highest scoring result
    selected = []
    selected_indices = []
    candidates = list(ranked_results)
    candidate_embeddings = list(embeddings)

    # Select first item (highest relevance)
    first_item = candidates.pop(0)
    first_embedding = candidate_embeddings.pop(0)
    selected.append(first_item[:4])  # Exclude item_type from output
    selected_indices.append(0)

    # Track selected item types and embeddings for diversity calculation
    selected_types = [first_item[4]]
    selected_paths = [first_item[1]]
    selected_embeddings = [first_embedding]

    # Iteratively select remaining items
    while len(selected) < k and candidates:
        best_mmr_score = -1
        best_idx = -1

        for idx, (candidate, cand_embedding) in enumerate(
            zip(candidates, candidate_embeddings, strict=False)
        ):
            score, path, header, content, item_type = candidate

            # Calculate relevance (already computed score)
            relevance = score

            # Calculate diversity based on semantic similarity, item type, and path
            diversity = 1.0

            # Calculate semantic diversity (average dissimilarity to selected items)
            semantic_similarities = []
            for sel_embedding in selected_embeddings:
                # Cosine similarity between embeddings
                cos_sim = np.dot(cand_embedding, sel_embedding) / (
                    np.linalg.norm(cand_embedding) * np.linalg.norm(sel_embedding)
                )
                semantic_similarities.append(cos_sim)

            # Average semantic similarity to selected items
            avg_semantic_sim = np.mean(semantic_similarities)
            semantic_diversity = 1.0 - avg_semantic_sim

            # Penalize same item types (reduced weight with semantic similarity)
            type_penalty = selected_types.count(item_type) * 0.1  # Reduced from 0.2
            type_diversity = max(0.4, 1.0 - min(0.6, type_penalty))

            # Calculate path diversity with weighted depth consideration
            path_diversity = 1.0
            for sel_path in selected_paths:
                # Check if paths share the same module prefix
                sel_parts = sel_path.split("::")
                cand_parts = path.split("::")

                # Compare module prefixes with depth weighting
                if len(sel_parts) > 1 and len(cand_parts) > 1:
                    common_prefix_len = 0
                    for sp, cp in zip(sel_parts[:-1], cand_parts[:-1], strict=False):
                        if sp == cp:
                            common_prefix_len += 1
                        else:
                            break

                    # Weight by depth - deeper common paths are less diverse
                    if common_prefix_len > 0:
                        depth_penalty = common_prefix_len * 0.15
                        path_diversity = min(path_diversity, 1.0 - depth_penalty)

            # Combine diversity factors with configurable weights
            module_weight = getattr(app_config, "MODULE_DIVERSITY_WEIGHT", 0.15)
            diversity = (
                0.5 * semantic_diversity  # Semantic similarity weight
                + 0.35 * type_diversity  # Type diversity weight
                + module_weight * path_diversity  # Module diversity weight
            )

            # Ensure diversity stays in [0, 1]
            diversity = max(0.0, min(1.0, diversity))

            # Calculate MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_item = candidates.pop(best_idx)
            selected_embedding = candidate_embeddings.pop(best_idx)
            selected.append(selected_item[:4])  # Exclude item_type from output
            selected_types.append(selected_item[4])
            selected_paths.append(selected_item[1])
            selected_embeddings.append(selected_embedding)

    return selected


async def search_embeddings(
    db_path: Path,
    query_embedding: list[float],
    k: int = 5,
    type_filter: str | None = None,
    crate_filter: str | None = None,
    module_path: str | None = None,
    has_examples: bool | None = None,
    min_doc_length: int | None = None,
    visibility: str | None = None,
    deprecated: bool | None = None,
) -> list[tuple[float, str, str, str]]:
    """Search for similar embeddings using k-NN with enhanced ranking and caching.

    Implements progressive filtering with selectivity analysis for optimal performance.
    """
    if not db_path.exists():
        return []

    start_time = time.time()
    filter_times = {}
    debug_info = {"filters_applied": [], "results_at_stage": {}, "timing": {}}

    # Log search initiation with parameters
    logger.debug(
        "search_embeddings_start",
        k=k,
        type_filter=type_filter,
        crate_filter=crate_filter,
        module_path=module_path,
        has_examples=has_examples,
        min_doc_length=min_doc_length,
        visibility=visibility,
        deprecated=deprecated,
    )

    # Check cache first
    cache = get_search_cache()
    cached_results = cache.get(
        query_embedding,
        k,
        type_filter,
        crate_filter,
        module_path,
        has_examples,
        min_doc_length,
        visibility,
        deprecated,
    )
    if cached_results is not None:
        logger.debug(
            f"Cache hit for search with k={k}, latency: {(time.time() - start_time) * 1000:.2f}ms"
        )
        return cached_results

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Analyze filter selectivity for progressive filtering
        should_prefilter = False
        prefilter_count = 0

        # Check if filters would significantly reduce the dataset
        if any(
            [
                type_filter,
                crate_filter,
                module_path,
                has_examples,
                deprecated is not None,
            ]
        ):
            filter_start = time.time()

            # Prepare patterns for filtering
            crate_pattern = f"{crate_filter}::%%" if crate_filter else None
            # Module pattern needs to account for crate name prefix
            # Get crate name from the db path (format: cache/{crate}/{version}.db)
            crate_name = db_path.parent.name if db_path.parent.name != "cache" else None

            # Fix: Allow both the module itself AND items within it
            module_patterns = []
            if module_path and crate_name:
                module_patterns = [
                    f"{crate_name}::{module_path}",  # The module itself
                    f"{crate_name}::{module_path}::%",  # Items in the module
                ]

            # Build module filter clause for selectivity analysis
            module_filter_clause = "1=1"
            if module_patterns:
                module_conditions = " OR ".join(
                    ["item_path LIKE ?" for _ in module_patterns]
                )
                module_filter_clause = f"({module_conditions})"

            query = f"""
                SELECT COUNT(*) FROM embeddings
                WHERE (:type_filter IS NULL OR item_type = :type_filter)
                    AND (:crate_pattern IS NULL OR item_path LIKE :crate_pattern)
                    AND {module_filter_clause}
                    AND (:deprecated IS NULL OR deprecated = :deprecated)
                    AND (:has_examples IS NULL OR (:has_examples = 0 OR examples IS NOT NULL))
            """

            params = (
                [
                    type_filter,
                    crate_pattern,
                ]
                + module_patterns
                + [
                    deprecated,
                    has_examples,
                ]
            )

            cursor = await db.execute(query, params)
            result = await cursor.fetchone()
            prefilter_count = result[0] if result else 0
            filter_times["selectivity_analysis"] = (time.time() - filter_start) * 1000

            # Enhanced selectivity analysis with cardinality estimation
            # Estimate selectivity for each filter type
            total_count_query = "SELECT COUNT(*) FROM embeddings"
            total_cursor = await db.execute(total_count_query)
            total_count = (await total_cursor.fetchone())[0]

            # Calculate selectivity ratio
            selectivity_ratio = (
                prefilter_count / total_count if total_count > 0 else 1.0
            )

            # Dynamic threshold based on filter types
            # More selective filters (lower ratio) can handle larger result sets
            if selectivity_ratio < 0.05:  # Very selective (< 5% of data)
                threshold = 20000
            elif selectivity_ratio < 0.15:  # Selective (< 15% of data)
                threshold = 10000
            elif selectivity_ratio < 0.30:  # Moderately selective (< 30% of data)
                threshold = 5000
            else:  # Not very selective
                threshold = 2000

            # Use progressive filtering based on dynamic threshold
            should_prefilter = prefilter_count < threshold and prefilter_count > 0

            # Log filter selectivity analysis
            logger.debug(
                "filter_selectivity_analysis",
                prefilter_count=prefilter_count,
                should_prefilter=should_prefilter,
                elapsed_ms=filter_times["selectivity_analysis"],
                filters={
                    "type_filter": type_filter,
                    "crate_filter": crate_filter,
                    "module_path": module_path,
                    "has_examples": has_examples,
                    "deprecated": deprecated,
                },
            )

            debug_info["results_at_stage"]["after_filters"] = prefilter_count

        # Dynamic fetch_k adjustment based on filter selectivity
        # More selective filters need less over-fetching
        if "selectivity_ratio" in locals():
            if selectivity_ratio < 0.1:  # Very selective
                over_fetch = 5
            elif selectivity_ratio < 0.3:  # Moderately selective
                over_fetch = 10
            else:  # Less selective
                over_fetch = 15
        else:
            over_fetch = 10  # Default over-fetch

        # Adjust for MMR diversification if enabled
        if app_config.RANKING_DIVERSITY_WEIGHT > 0:
            over_fetch = int(
                over_fetch * 1.5
            )  # Need more candidates for diversification

        # Defensive bounds check to ensure k doesn't exceed safe limits
        # Cap k at 20 even if validation passed higher (prevents edge cases)
        safe_k = min(k, 20)
        fetch_k = min(safe_k + over_fetch, 50)  # Cap at 50 for performance

        # Prepare patterns for LIKE queries
        crate_pattern = f"{crate_filter}::%%" if crate_filter else None
        # Module pattern needs to account for crate name prefix
        crate_name = db_path.parent.name if db_path.parent.name != "cache" else None

        # Fix: Allow both the module itself AND items within it
        module_patterns = []
        if module_path and crate_name:
            module_patterns = [
                f"{crate_name}::{module_path}",  # The module itself
                f"{crate_name}::{module_path}::%",  # Items in the module
            ]

        # Build module filter clause for main query
        module_filter_clause = "1=1"
        module_params = []
        if module_patterns:
            module_conditions = " OR ".join(
                ["e.item_path LIKE ?" for _ in module_patterns]
            )
            module_filter_clause = f"({module_conditions})"
            module_params = module_patterns

        # Perform vector search with additional metadata for ranking and filtering
        query = f"""
            SELECT
                v.distance,
                e.item_path,
                e.header,
                e.content,
                e.item_type,
                LENGTH(e.content) as doc_length,
                e.examples,
                e.visibility,
                e.deprecated,
                e.embedding
            FROM vec_embeddings v
            JOIN embeddings e ON v.rowid = e.id
            WHERE v.embedding MATCH ? AND k = ?
                AND (? IS NULL OR e.item_type = ?)
                AND (? IS NULL OR e.item_path LIKE ?)
                AND {module_filter_clause}
                AND (? IS NULL OR e.visibility = ?)
                AND (? IS NULL OR e.deprecated = ?)
                AND (? IS NULL OR (? = 0 OR e.examples IS NOT NULL))
                AND (? IS NULL OR LENGTH(e.content) >= ?)
            ORDER BY v.distance
            """

        params = (
            [
                bytes(sqlite_vec.serialize_float32(query_embedding)),
                fetch_k,
                type_filter,
                type_filter,
                crate_pattern,
                crate_pattern,
            ]
            + module_params
            + [
                visibility,
                visibility,
                deprecated,
                deprecated,
                has_examples,
                has_examples,
                min_doc_length,
                min_doc_length,
            ]
        )

        cursor = await db.execute(query, params)

        results = await cursor.fetchall()

        # Apply enhanced ranking algorithm
        ranked_results = []
        embeddings = []  # Store embeddings for MMR
        for row in results:
            (
                distance,
                item_path,
                header,
                content,
                item_type,
                doc_length,
                examples,
                visibility,
                deprecated,
                embedding_blob,
            ) = row

            # Deserialize embedding for MMR
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings.append(embedding)

            # Base vector similarity score
            base_score = 1.0 - distance

            # Type-specific weight
            type_weight = (
                app_config.TYPE_WEIGHTS.get(item_type, 1.0) if item_type else 1.0
            )

            # Documentation quality score (normalize to 0-1)
            doc_quality = min(1.0, (doc_length or 0) / 1000)

            # Examples presence boost
            has_examples = 1.2 if examples else 1.0

            # Compute composite score
            final_score = (
                app_config.RANKING_VECTOR_WEIGHT * base_score
                + app_config.RANKING_TYPE_WEIGHT * (base_score * type_weight)
                + app_config.RANKING_QUALITY_WEIGHT * doc_quality
                + app_config.RANKING_EXAMPLES_WEIGHT * has_examples
            )

            # Ensure score stays in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))

            # Validate score
            if not 0.0 <= final_score <= 1.0:
                logger.warning(
                    f"Score out of range: {final_score} for item {item_path}"
                )
                final_score = max(0.0, min(1.0, final_score))

            ranked_results.append((final_score, item_path, header, content, item_type))

        # Sort by final score and keep embeddings aligned
        # Create paired list for sorting
        paired_results = list(zip(ranked_results, embeddings, strict=False))
        paired_results.sort(key=lambda x: x[0][0], reverse=True)

        # Unpack sorted results
        ranked_results = [r for r, _ in paired_results]
        embeddings = [e for _, e in paired_results]

        # Apply MMR diversification if we have enough results
        if len(ranked_results) > k and app_config.RANKING_DIVERSITY_WEIGHT > 0:
            top_results = _apply_mmr_diversification(
                ranked_results,
                embeddings,
                k,
                app_config.RANKING_DIVERSITY_LAMBDA,
            )
        else:
            # No diversification, just take top k
            top_results = [
                (score, path, header, content)
                for score, path, header, content, _ in ranked_results[:k]
            ]

        # Performance monitoring
        elapsed_ms = (time.time() - start_time) * 1000
        debug_info["timing"]["total_ms"] = elapsed_ms
        debug_info["timing"]["filter_times"] = filter_times
        debug_info["results_at_stage"]["final"] = len(top_results)

        # Calculate score distribution
        if top_results:
            scores = [score for score, _, _, _ in top_results]
            debug_info["score_distribution"] = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            }

        # Log comprehensive search metrics
        if elapsed_ms > 100:
            logger.warning(
                "slow_search_query",
                elapsed_ms=elapsed_ms,
                k=k,
                results_count=len(top_results),
                debug_info=debug_info,
            )
        else:
            logger.debug(
                "search_completed",
                elapsed_ms=elapsed_ms,
                k=k,
                results_count=len(top_results),
                cache_hit=False,
                filters_applied=bool(
                    type_filter
                    or crate_filter
                    or module_path
                    or has_examples
                    or deprecated is not None
                ),
                score_distribution=debug_info.get("score_distribution"),
                filter_impact=debug_info.get("results_at_stage"),
            )

        # Store in cache before returning
        cache.set(
            query_embedding,
            k,
            top_results,
            type_filter,
            crate_filter,
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
        )

        return top_results


async def get_see_also_suggestions(
    db_path: Path,
    query_embedding: list[float],
    original_item_paths: set[str],
    k: int = 8,
    similarity_threshold: float = 0.7,
    max_suggestions: int = 5,
) -> list[str]:
    """Find related items for see-also suggestions using vector similarity.

    Args:
        db_path: Path to the database
        query_embedding: Embedding vector to find similar items for
        original_item_paths: Set of item paths to exclude from suggestions
        k: Number of candidates to fetch (over-fetch for filtering)
        similarity_threshold: Minimum similarity score (1.0 - distance) for suggestions
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of item paths for see-also suggestions
    """
    if not db_path.exists():
        return []

    # Defensive bounds check for k parameter
    safe_k = min(k, 20)

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            await db.execute("PRAGMA journal_mode = WAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA synchronous = NORMAL")
            await db.execute("PRAGMA cache_size = -64000")

            # Load sqlite-vec extension
            await db.enable_load_extension(True)
            await db.load_extension(sqlite_vec.loadable_path())
            await db.enable_load_extension(False)

            # Perform vector search with explicit k parameter (required by sqlite-vec)
            cursor = await db.execute(
                """
                SELECT
                    v.distance,
                    e.item_path,
                    e.item_type
                FROM vec_embeddings v
                JOIN embeddings e ON v.rowid = e.id
                WHERE v.embedding MATCH :embedding AND k = :k
                ORDER BY v.distance
                """,
                {
                    "embedding": bytes(sqlite_vec.serialize_float32(query_embedding)),
                    "k": safe_k,
                },
            )

            results = await cursor.fetchall()

            # Process results: filter by threshold and exclude original items
            suggestions = []
            for distance, item_path, item_type in results:
                # Calculate similarity score (1.0 - distance)
                similarity_score = 1.0 - distance

                # Apply similarity threshold
                if similarity_score < similarity_threshold:
                    continue

                # Exclude original items
                if item_path in original_item_paths:
                    continue

                # Add to suggestions
                suggestions.append(item_path)

                # Stop if we have enough suggestions
                if len(suggestions) >= max_suggestions:
                    break

            return suggestions

    except Exception as e:
        logger.warning(f"Failed to get see-also suggestions: {e}")
        # Fail gracefully - return empty suggestions
        return []


async def search_example_embeddings(
    db_path: Path,
    query_embedding: list[float],
    k: int = 5,
    crate_filter: str | None = None,
    language_filter: str | None = None,
) -> list[dict]:
    """Search for similar code examples using dedicated example embeddings.

    Returns examples with their code, language, context, and similarity scores.
    """
    if not db_path.exists():
        return []

    start_time = time.time()

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Build query with optional filters
        query_parts = []
        params = []

        # Base query
        query = """
            SELECT 
                e.item_path,
                e.example_text,
                e.language,
                e.context,
                vec_distance_L2(v.example_embedding, ?) as distance,
                e.example_hash
            FROM example_embeddings e
            JOIN vec_example_embeddings v ON e.id = v.rowid
            WHERE 1=1
        """

        params.append(bytes(sqlite_vec.serialize_float32(query_embedding)))

        # Add filters
        if crate_filter:
            query += " AND e.crate_name = ?"
            params.append(crate_filter)

        if language_filter:
            query += " AND e.language = ?"
            params.append(language_filter)

        # Order by distance and limit
        query += " ORDER BY distance LIMIT ?"
        params.append(k)

        cursor = await db.execute(query, params)
        results = []

        async for row in cursor:
            item_path, example_text, language, context, distance, example_hash = row

            # Convert distance to similarity score
            score = 1.0 / (1.0 + distance)

            results.append(
                {
                    "item_path": item_path,
                    "code": example_text,
                    "language": language,
                    "context": context,
                    "score": score,
                    "hash": example_hash,
                }
            )

        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 100:
            logger.warning(f"Slow example search: {elapsed_ms:.2f}ms for k={k}")
        else:
            logger.debug(f"Example search completed in {elapsed_ms:.2f}ms")

        return results


# ============================================================
# Version Diff Database Functions
# ============================================================


async def get_all_items_for_version(
    db_path: Path,
    include_content: bool = False,
    include_examples: bool = False,
) -> dict[str, dict]:
    """Get all items from a specific version's database for comparison.

    Args:
        db_path: Path to the version's database
        include_content: Whether to include full documentation content
        include_examples: Whether to include code examples

    Returns:
        Dictionary mapping item_path to item data
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    items = {}

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Build query based on what to include
        columns = [
            "item_path",
            "header",
            "item_type",
            "signature",
            "visibility",
            "deprecated",
        ]
        if include_content:
            columns.append("content")
        if include_examples:
            columns.append("examples")

        # Add generic parameters and trait bounds for better comparison
        columns.extend(["generic_params", "trait_bounds"])

        query = f"SELECT {', '.join(columns)} FROM embeddings"

        cursor = await db.execute(query)
        async for row in cursor:
            # Build item dict from row
            item_data = {
                "item_path": row[0],
                "header": row[1],
                "item_type": row[2],
                "signature": row[3],
                "visibility": row[4],
                "deprecated": bool(row[5]),
                "generic_params": row[-2],  # second to last
                "trait_bounds": row[-1],  # last
            }

            # Add optional fields
            idx = 6
            if include_content:
                item_data["content"] = row[idx]
                idx += 1
            if include_examples:
                item_data["examples"] = row[idx]

            items[row[0]] = item_data  # Use item_path as key

    return items


async def get_item_signatures_batch(
    db_path: Path,
    item_paths: list[str],
) -> dict[str, dict]:
    """Get signatures for a batch of items efficiently.

    Args:
        db_path: Path to the database
        item_paths: List of item paths to retrieve

    Returns:
        Dictionary mapping item_path to signature data
    """
    if not db_path.exists():
        return {}

    if not item_paths:
        return {}

    signatures = {}

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Use parameterized query with IN clause
        placeholders = ",".join(["?"] * len(item_paths))
        query = f"""
            SELECT 
                item_path, 
                signature, 
                item_type, 
                visibility,
                deprecated,
                generic_params,
                trait_bounds
            FROM embeddings 
            WHERE item_path IN ({placeholders})
        """

        cursor = await db.execute(query, item_paths)
        async for row in cursor:
            signatures[row[0]] = {
                "signature": row[1],
                "item_type": row[2],
                "visibility": row[3],
                "deprecated": bool(row[4]),
                "generic_params": row[5],
                "trait_bounds": row[6],
            }

    return signatures


async def get_module_items(
    db_path: Path,
    module_path: str,
) -> list[dict]:
    """Get all items within a specific module.

    Args:
        db_path: Path to the database
        module_path: Module path to query (e.g., 'tokio::sync')

    Returns:
        List of items in the module with their metadata
    """
    if not db_path.exists():
        return []

    items = []

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Match items that start with the module path
        query = """
            SELECT 
                item_path,
                item_type,
                signature,
                visibility,
                deprecated
            FROM embeddings
            WHERE item_path LIKE ? || '::%'
            ORDER BY item_path
        """

        cursor = await db.execute(query, (module_path,))
        async for row in cursor:
            items.append(
                {
                    "item_path": row[0],
                    "item_type": row[1],
                    "signature": row[2],
                    "visibility": row[3],
                    "deprecated": bool(row[4]),
                }
            )

    return items


async def compute_item_hash(item_data: dict) -> str:
    """Compute a hash for an item to detect changes.

    Uses signature, visibility, deprecated status, and generics for comparison.
    """
    import hashlib

    # Create a stable string representation for hashing
    hash_parts = [
        str(item_data.get("signature", "")),
        str(item_data.get("visibility", "public")),
        str(item_data.get("deprecated", False)),
        str(item_data.get("generic_params", "")),
        str(item_data.get("trait_bounds", "")),
    ]

    hash_string = "|".join(hash_parts)
    return hashlib.sha256(hash_string.encode()).hexdigest()[:16]


async def get_all_item_paths(db_path: Path) -> list[str]:
    """Get all item paths from the database for fuzzy matching.

    Args:
        db_path: Path to the database file

    Returns:
        List of all unique item paths in the database
    """
    if not db_path.exists():
        return []

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            "SELECT DISTINCT item_path FROM embeddings WHERE item_path IS NOT NULL"
        )
        paths = [row[0] for row in await cursor.fetchall()]

        logger.debug(f"Retrieved {len(paths)} unique item paths from database")
        return paths


# ============================================================
# Ingestion Status Management Functions
# ============================================================

import json


@RetryableTransaction(max_retries=3)
async def set_ingestion_status(
    db_path: Path,
    crate_id: int,
    status: str,
    error_message: str | None = None,
    checkpoint_data: dict | None = None,
    items_processed: int | None = None,
    total_items: int | None = None,
) -> None:
    """Update ingestion status with retry logic.

    Args:
        db_path: Path to database
        crate_id: ID of the crate being ingested
        status: Status string ('started', 'downloading', 'processing', 'completed', 'failed')
        error_message: Optional error message for failed ingestions
        checkpoint_data: Optional checkpoint data as dict (will be JSON serialized)
        items_processed: Optional count of items processed
        total_items: Optional total count of items
    """
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Get current timestamp
        current_time = time.time()

        # Serialize checkpoint data if provided
        checkpoint_json = json.dumps(checkpoint_data) if checkpoint_data else None

        # Check if record exists
        cursor = await db.execute(
            "SELECT crate_id FROM ingestion_status WHERE crate_id = ?", (crate_id,)
        )
        exists = await cursor.fetchone()

        if exists:
            # Update existing record
            if status == "completed":
                await execute_with_retry(
                    db,
                    """
                    UPDATE ingestion_status 
                    SET status = ?, updated_at = ?, completed_at = ?, 
                        error_message = ?, checkpoint_data = ?,
                        items_processed = COALESCE(?, items_processed),
                        total_items = COALESCE(?, total_items)
                    WHERE crate_id = ?
                    """,
                    (
                        status,
                        current_time,
                        current_time,
                        error_message,
                        checkpoint_json,
                        items_processed,
                        total_items,
                        crate_id,
                    ),
                )
            else:
                await execute_with_retry(
                    db,
                    """
                    UPDATE ingestion_status 
                    SET status = ?, updated_at = ?, error_message = ?, 
                        checkpoint_data = ?,
                        items_processed = COALESCE(?, items_processed),
                        total_items = COALESCE(?, total_items)
                    WHERE crate_id = ?
                    """,
                    (
                        status,
                        current_time,
                        error_message,
                        checkpoint_json,
                        items_processed,
                        total_items,
                        crate_id,
                    ),
                )
        else:
            # Insert new record
            await execute_with_retry(
                db,
                """
                INSERT INTO ingestion_status 
                (crate_id, status, started_at, updated_at, completed_at, 
                 error_message, checkpoint_data, items_processed, total_items)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    crate_id,
                    status,
                    current_time,
                    current_time,
                    current_time if status == "completed" else None,
                    error_message,
                    checkpoint_json,
                    items_processed or 0,
                    total_items,
                ),
            )

        await db.commit()
        logger.debug(f"Updated ingestion status for crate_id={crate_id} to {status}")


async def get_ingestion_status(db_path: Path, crate_id: int) -> dict | None:
    """Get current ingestion status for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the crate

    Returns:
        Dict with status information or None if not found
    """
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            SELECT status, started_at, updated_at, completed_at, 
                   error_message, items_processed, total_items, checkpoint_data
            FROM ingestion_status 
            WHERE crate_id = ?
            """,
            (crate_id,),
        )
        row = await cursor.fetchone()

        if row:
            checkpoint_data = None
            if row[7]:  # checkpoint_data
                try:
                    checkpoint_data = json.loads(row[7])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse checkpoint data for crate_id={crate_id}"
                    )

            return {
                "status": row[0],
                "started_at": row[1],
                "updated_at": row[2],
                "completed_at": row[3],
                "error_message": row[4],
                "items_processed": row[5],
                "total_items": row[6],
                "checkpoint_data": checkpoint_data,
            }

        return None


async def find_incomplete_ingestions(
    cache_dir: Path,
    stale_threshold_seconds: int = 1800,  # 30 minutes default
) -> list[dict]:
    """Find all incomplete/stalled ingestions using partial index.

    Args:
        cache_dir: Cache directory containing crate databases
        stale_threshold_seconds: Seconds after which an ingestion is considered stalled

    Returns:
        List of dicts with incomplete ingestion information
    """
    incomplete = []
    current_time = time.time()
    stale_threshold = current_time - stale_threshold_seconds

    # Iterate through all crate directories
    for crate_dir in cache_dir.iterdir():
        if not crate_dir.is_dir():
            continue

        # Check each version database
        for db_file in crate_dir.glob("*.db"):
            try:
                async with aiosqlite.connect(db_file, timeout=DB_TIMEOUT) as db:
                    # Use partial index for efficient query
                    cursor = await db.execute(
                        """
                        SELECT 
                            cm.name, cm.version, cm.id,
                            s.status, s.started_at, s.updated_at, 
                            s.error_message, s.items_processed, s.total_items
                        FROM ingestion_status s
                        JOIN crate_metadata cm ON s.crate_id = cm.id
                        WHERE s.status != 'completed'
                          AND (s.status = 'failed' OR s.updated_at < ?)
                        """,
                        (stale_threshold,),
                    )

                    async for row in cursor:
                        incomplete.append(
                            {
                                "crate_name": row[0],
                                "version": row[1],
                                "crate_id": row[2],
                                "status": row[3],
                                "started_at": row[4],
                                "updated_at": row[5],
                                "error_message": row[6],
                                "items_processed": row[7],
                                "total_items": row[8],
                                "db_path": str(db_file),
                                "is_stalled": row[5] < stale_threshold
                                and row[3] != "failed",
                            }
                        )
            except Exception as e:
                logger.warning(f"Error checking ingestion status for {db_file}: {e}")
                continue

    logger.info(f"Found {len(incomplete)} incomplete ingestions")
    return incomplete


async def is_ingestion_complete(db_path: Path) -> bool:
    """Check if ingestion completed successfully.

    This replaces the simple db_path.exists() check with proper completion validation.

    Args:
        db_path: Path to database

    Returns:
        True if ingestion is complete, False otherwise
    """
    if not db_path.exists():
        return False

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # First check if tables exist
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name IN ('crate_metadata', 'ingestion_status')
                """
            )
            table_count = (await cursor.fetchone())[0]

            if table_count < 2:
                # Tables don't exist, ingestion not complete
                return False

            # Check ingestion status
            cursor = await db.execute(
                """
                SELECT s.status 
                FROM ingestion_status s
                JOIN crate_metadata cm ON s.crate_id = cm.id
                LIMIT 1
                """
            )
            row = await cursor.fetchone()

            if not row:
                # No ingestion status record, assume incomplete
                return False

            return row[0] == "completed"

    except Exception as e:
        logger.warning(f"Error checking ingestion completion for {db_path}: {e}")
        return False


async def detect_stalled_ingestions(
    cache_dir: Path, stale_threshold_seconds: int = 1800
) -> list[dict]:
    """Detect stalled ingestions that need recovery.

    Args:
        cache_dir: Cache directory containing crate databases
        stale_threshold_seconds: Seconds after which an ingestion is considered stalled

    Returns:
        List of stalled ingestions requiring recovery
    """
    stalled = []

    incomplete = await find_incomplete_ingestions(cache_dir, stale_threshold_seconds)

    for ingestion in incomplete:
        if ingestion["is_stalled"]:
            stalled.append(
                {
                    "crate_name": ingestion["crate_name"],
                    "version": ingestion["version"],
                    "db_path": ingestion["db_path"],
                    "status": ingestion["status"],
                    "stalled_since": time.time() - ingestion["updated_at"],
                    "items_processed": ingestion["items_processed"],
                    "total_items": ingestion["total_items"],
                }
            )

    if stalled:
        logger.warning(f"Found {len(stalled)} stalled ingestions requiring recovery")

    return stalled


async def reset_ingestion_status(db_path: Path, crate_id: int) -> None:
    """Reset ingestion status to trigger re-ingestion.

    Args:
        db_path: Path to database
        crate_id: ID of the crate to reset
    """
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        await execute_with_retry(
            db, "DELETE FROM ingestion_status WHERE crate_id = ?", (crate_id,)
        )
        await db.commit()
        logger.info(f"Reset ingestion status for crate_id={crate_id}")
