"""Database operations with SQLite and sqlite-vec."""

import logging
from pathlib import Path

import aiosqlite
import sqlite_vec

from .config import CACHE_DIR, DB_TIMEOUT, EMBEDDING_DIM

logger = logging.getLogger(__name__)


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
    db_path: Path, query_embedding: list[float], k: int = 5
) -> list[tuple[float, str, str, str]]:
    """Search for similar embeddings using k-NN."""
    if not db_path.exists():
        return []

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Perform vector search using sqlite-vec MATCH operator
        # The distance column is automatically included
        # We need to add 'k = ?' constraint for older SQLite versions
        cursor = await db.execute(
            """
            SELECT
                v.distance,
                e.item_path,
                e.header,
                e.content
            FROM vec_embeddings v
            JOIN embeddings e ON v.rowid = e.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (bytes(sqlite_vec.serialize_float32(query_embedding)), k),
        )

        results = await cursor.fetchall()
        # Convert distance to similarity score (1 - normalized_distance)
        return [(1.0 - row[0], row[1], row[2], row[3]) for row in results]
