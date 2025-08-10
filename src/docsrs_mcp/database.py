"""Database operations with SQLite and sqlite-vec."""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiosqlite
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

        # Create reexports table for auto-discovered re-export mappings
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reexports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crate_id INTEGER NOT NULL,
                alias_path TEXT NOT NULL,
                actual_path TEXT NOT NULL,
                is_glob BOOLEAN DEFAULT 0,
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id) ON DELETE CASCADE,
                UNIQUE(crate_id, alias_path)
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
    """Store discovered re-export mappings for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the parent crate
        reexports: List of re-export dicts with alias_path, actual_path, is_glob
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
            )
            for reexport in reexports
        ]

        # Batch insert with IGNORE for duplicates
        await db.executemany(
            """
            INSERT OR IGNORE INTO reexports (crate_id, alias_path, actual_path, is_glob)
            VALUES (?, ?, ?, ?)
            """,
            reexport_data,
        )

        await db.commit()
        logger.info(f"Stored {len(reexports)} re-export mappings")


async def get_discovered_reexports(
    db_path: Path, crate_name: str, version: str | None = None
) -> dict[str, str]:
    """Get auto-discovered re-exports for a crate.

    Args:
        db_path: Path to database
        crate_name: Name of the crate
        version: Optional version (for logging/future use)

    Returns:
        Dictionary mapping alias_path to actual_path
    """
    reexport_map = {}

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Query re-exports for this crate
            cursor = await db.execute(
                """
                SELECT r.alias_path, r.actual_path, r.is_glob
                FROM reexports r
                JOIN crate_metadata c ON r.crate_id = c.id
                WHERE c.name = ?
                """,
                (crate_name,),
            )

            async for row in cursor:
                alias_path, actual_path, is_glob = row
                # Store mapping
                reexport_map[alias_path] = actual_path

                # Also store without crate prefix for convenience
                if alias_path.startswith(f"{crate_name}::"):
                    short_alias = alias_path[len(crate_name) + 2 :]
                    reexport_map[short_alias] = actual_path

            if reexport_map:
                logger.debug(
                    f"Loaded {len(reexport_map)} re-export mappings for {crate_name}"
                )
    except Exception as e:
        logger.warning(f"Error loading re-exports for {crate_name}: {e}")

    return reexport_map


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
    k: int,
    lambda_param: float,
) -> list[tuple[float, str, str, str]]:
    """
    Apply Maximum Marginal Relevance (MMR) for result diversification.

    MMR balances relevance and diversity using the formula:
    MMR = λ * relevance + (1-λ) * diversity

    Args:
        ranked_results: List of (score, path, header, content, item_type) tuples
        k: Number of results to select
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of diversified results as (score, path, header, content) tuples
    """
    if not ranked_results:
        return []

    # Start with the highest scoring result
    selected = []
    candidates = list(ranked_results)

    # Select first item (highest relevance)
    first_item = candidates.pop(0)
    selected.append(first_item[:4])  # Exclude item_type from output

    # Track selected item types for diversity calculation
    selected_types = [first_item[4]]
    selected_paths = [first_item[1]]

    # Iteratively select remaining items
    while len(selected) < k and candidates:
        best_mmr_score = -1
        best_idx = -1

        for idx, candidate in enumerate(candidates):
            score, path, header, content, item_type = candidate

            # Calculate relevance (already computed score)
            relevance = score

            # Calculate diversity based on item type and path similarity
            diversity = 1.0

            # Penalize same item types
            type_penalty = selected_types.count(item_type) * 0.2
            diversity -= min(0.6, type_penalty)

            # Penalize similar paths (same module)
            for sel_path in selected_paths:
                # Check if paths share the same module prefix
                sel_parts = sel_path.split("::")
                cand_parts = path.split("::")

                # Compare module prefixes
                if len(sel_parts) > 1 and len(cand_parts) > 1:
                    if sel_parts[:-1] == cand_parts[:-1]:
                        # Same module, reduce diversity
                        diversity -= 0.3
                        break

            # Ensure diversity stays in [0, 1]
            diversity = max(0.0, min(1.0, diversity))

            # Calculate MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_item = candidates.pop(best_idx)
            selected.append(selected_item[:4])  # Exclude item_type from output
            selected_types.append(selected_item[4])
            selected_paths.append(selected_item[1])

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

        fetch_k = min(k + over_fetch, 50)  # Cap at 50 for performance

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
                e.deprecated
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
            ) = row

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

        # Sort by final score
        ranked_results.sort(key=lambda x: x[0], reverse=True)

        # Apply MMR diversification if we have enough results
        if len(ranked_results) > k and app_config.RANKING_DIVERSITY_WEIGHT > 0:
            top_results = _apply_mmr_diversification(
                ranked_results,
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
                    "k": k,
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
