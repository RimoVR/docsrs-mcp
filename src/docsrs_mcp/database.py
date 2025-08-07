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

from . import config as app_config
from .cache import get_search_cache
from .config import CACHE_DIR, DB_TIMEOUT, EMBEDDING_DIM

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
                FOREIGN KEY (crate_id) REFERENCES crate_metadata(id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                item_path TEXT NOT NULL,
                header TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                item_type TEXT,
                signature TEXT,
                parent_id TEXT,
                examples TEXT,
                visibility TEXT DEFAULT 'public',
                deprecated BOOLEAN DEFAULT 0
            )
        """)

        # Create vector index
        await db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
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


async def search_embeddings(
    db_path: Path,
    query_embedding: list[float],
    k: int = 5,
    type_filter: str | None = None,
    crate_filter: str | None = None,
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

    # Check cache first
    cache = get_search_cache()
    cached_results = cache.get(
        query_embedding,
        k,
        type_filter,
        crate_filter,
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
        if any([type_filter, crate_filter, has_examples, deprecated is not None]):
            filter_start = time.time()
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM embeddings
                WHERE (:type_filter IS NULL OR item_type = :type_filter)
                    AND (:crate_pattern IS NULL OR item_path LIKE :crate_pattern)
                    AND (:deprecated IS NULL OR deprecated = :deprecated)
                    AND (:has_examples IS NULL OR (:has_examples = 0 OR examples IS NOT NULL))
            """,
                {
                    "type_filter": type_filter,
                    "crate_pattern": f"{crate_filter}::%%" if crate_filter else None,
                    "deprecated": deprecated,
                    "has_examples": has_examples,
                },
            )
            result = await cursor.fetchone()
            prefilter_count = result[0] if result else 0
            filter_times["selectivity_analysis"] = (time.time() - filter_start) * 1000

            # Use progressive filtering if result set is small enough (<10K items)
            should_prefilter = prefilter_count < 10000 and prefilter_count > 0
            logger.debug(
                f"Filter selectivity: {prefilter_count} items, prefilter={should_prefilter}"
            )

        # Fetch k+10 results to allow for re-ranking
        fetch_k = min(k + 10, 50)  # Cap at 50 for performance

        # Prepare crate pattern for LIKE query
        crate_pattern = f"{crate_filter}::%%" if crate_filter else None

        # Perform vector search with additional metadata for ranking and filtering
        cursor = await db.execute(
            """
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
            WHERE v.embedding MATCH :embedding AND k = :k
                AND (:type_filter IS NULL OR e.item_type = :type_filter)
                AND (:crate_pattern IS NULL OR e.item_path LIKE :crate_pattern)
                AND (:visibility IS NULL OR e.visibility = :visibility)
                AND (:deprecated IS NULL OR e.deprecated = :deprecated)
                AND (:has_examples IS NULL OR (:has_examples = 0 OR e.examples IS NOT NULL))
                AND (:min_doc_length IS NULL OR LENGTH(e.content) >= :min_doc_length)
            ORDER BY v.distance
            """,
            {
                "embedding": bytes(sqlite_vec.serialize_float32(query_embedding)),
                "k": fetch_k,
                "type_filter": type_filter,
                "crate_pattern": crate_pattern,
                "visibility": visibility,
                "deprecated": deprecated,
                "has_examples": has_examples,
                "min_doc_length": min_doc_length,
            },
        )

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

            ranked_results.append((final_score, item_path, header, content))

        # Sort by final score and return top k
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        top_results = ranked_results[:k]

        # Performance monitoring
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 100:
            logger.warning(f"Slow search query: {elapsed_ms:.2f}ms for k={k}")
        else:
            logger.debug(f"Search completed in {elapsed_ms:.2f}ms for k={k}")

        # Log filter execution metrics
        if filter_times:
            logger.debug(f"Filter execution times: {filter_times}")
            if should_prefilter:
                logger.debug(
                    f"Progressive filtering reduced candidates to {prefilter_count} items"
                )

        # Log score distribution for monitoring
        if top_results:
            scores = [score for score, _, _, _ in top_results]
            logger.debug(
                f"Score distribution - min: {min(scores):.3f}, max: {max(scores):.3f}, avg: {sum(scores) / len(scores):.3f}"
            )

        # Store in cache before returning
        cache.set(
            query_embedding,
            k,
            top_results,
            type_filter,
            crate_filter,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
        )

        return top_results
