"""Database operations with SQLite and sqlite-vec."""

import logging
import time
from pathlib import Path

import aiosqlite
import sqlite_vec

from . import config as app_config
from .cache import get_search_cache
from .config import CACHE_DIR, DB_TIMEOUT, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Prepared statement cache for common queries
_prepared_statements = {}


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
                examples TEXT
            )
        """)

        # Create vector index
        await db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
        """)

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
) -> list[tuple[float, str, str, str]]:
    """Search for similar embeddings using k-NN with enhanced ranking and caching."""
    if not db_path.exists():
        return []

    start_time = time.time()

    # Check cache first
    cache = get_search_cache()
    cached_results = cache.get(query_embedding, k, type_filter)
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

        # Fetch k+10 results to allow for re-ranking
        fetch_k = min(k + 10, 50)  # Cap at 50 for performance

        # Perform vector search with additional metadata for ranking
        cursor = await db.execute(
            """
            SELECT
                v.distance,
                e.item_path,
                e.header,
                e.content,
                e.item_type,
                LENGTH(e.content) as doc_length,
                e.examples
            FROM vec_embeddings v
            JOIN embeddings e ON v.rowid = e.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (bytes(sqlite_vec.serialize_float32(query_embedding)), fetch_k),
        )

        results = await cursor.fetchall()

        # Apply enhanced ranking algorithm
        ranked_results = []
        for row in results:
            distance, item_path, header, content, item_type, doc_length, examples = row

            # Base vector similarity score
            base_score = 1.0 - distance

            # Type-specific weight
            type_weight = app_config.TYPE_WEIGHTS.get(item_type, 1.0) if item_type else 1.0

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

        # Log score distribution for monitoring
        if top_results:
            scores = [score for score, _, _, _ in top_results]
            logger.debug(
                f"Score distribution - min: {min(scores):.3f}, max: {max(scores):.3f}, avg: {sum(scores) / len(scores):.3f}"
            )

        # Store in cache before returning
        cache.set(query_embedding, k, top_results, type_filter)

        return top_results
