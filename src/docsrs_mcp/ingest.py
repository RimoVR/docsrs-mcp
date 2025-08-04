"""Ingestion pipeline for Rust crate documentation."""

import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
import aiosqlite
import sqlite_vec
from fastembed import TextEmbedding

from .config import (
    EMBEDDING_BATCH_SIZE,
    HTTP_TIMEOUT,
    MAX_DOWNLOAD_SIZE,
    MODEL_NAME,
)
from .database import get_db_path, init_database, store_crate_metadata

logger = logging.getLogger(__name__)


# Global embedding model instance
_embedding_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Get or create the embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _embedding_model = TextEmbedding(model_name=MODEL_NAME)
    return _embedding_model


async def fetch_crate_info(
    session: aiohttp.ClientSession, crate_name: str
) -> dict[str, Any]:
    """Fetch crate information from crates.io API."""
    url = f"https://crates.io/api/v1/crates/{crate_name}"
    async with session.get(
        url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Failed to fetch crate info: {resp.status}")
        data = await resp.json()
        return data["crate"]


async def resolve_version(
    crate_info: dict[str, Any], version: str | None = None
) -> str:
    """Resolve version string to actual version."""
    if version and version != "latest":
        return version

    # Get the latest stable version
    if "max_stable_version" in crate_info:
        return crate_info["max_stable_version"]
    elif "max_version" in crate_info:
        return crate_info["max_version"]
    elif "newest_version" in crate_info:
        return crate_info["newest_version"]

    raise Exception("No valid version found")


async def download_rustdoc(
    session: aiohttp.ClientSession, crate_name: str, version: str
) -> dict[str, Any]:
    """Download rustdoc JSON from docs.rs."""
    # Construct the rustdoc JSON URL
    url = f"https://docs.rs/{crate_name}/{version}/{crate_name.replace('-', '_')}.json"

    logger.info(f"Downloading rustdoc from: {url}")

    async with session.get(
        url,
        timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
        headers={"Accept": "application/json"},
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Failed to download rustdoc: {resp.status}")

        # Check size
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            raise Exception(f"File too large: {content_length} bytes")

        # Download and parse JSON
        text = await resp.text()
        return json.loads(text)


def extract_text_chunks(rustdoc: dict[str, Any]) -> list[dict[str, str]]:
    """Extract text chunks from rustdoc JSON for embedding."""
    chunks = []

    # For MVP, we'll just use the crate description as a single chunk
    # In a full implementation, this would parse the entire rustdoc structure
    
    # Since rustdoc JSON is complex and might not be available for all crates,
    # we'll return an empty list for now and rely on the crate metadata
    logger.info("Rustdoc parsing not yet implemented - using basic metadata only")
    
    return chunks


async def generate_embeddings(chunks: list[dict[str, str]]) -> list[list[float]]:
    """Generate embeddings for text chunks."""
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]

    embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = list(model.embed(batch))
        embeddings.extend(batch_embeddings)

    return embeddings


async def store_embeddings(
    db_path: Path, chunks: list[dict[str, str]], embeddings: list[list[float]]
) -> None:
    """Store chunks and their embeddings in the database."""
    if not chunks or not embeddings:
        return

    async with aiosqlite.connect(db_path) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Insert embeddings
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            # Insert into main table
            cursor = await db.execute(
                """
                INSERT INTO embeddings (item_path, header, content, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chunk["item_path"],
                    chunk["header"],
                    chunk["content"],
                    bytes(sqlite_vec.serialize_float32(embedding)),
                ),
            )

            # Insert into vector table
            await db.execute(
                "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, bytes(sqlite_vec.serialize_float32(embedding))),
            )

        await db.commit()


async def ingest_crate(crate_name: str, version: str | None = None) -> Path:
    """Ingest a crate's documentation and return the database path."""
    async with aiohttp.ClientSession() as session:
        # Fetch crate info
        crate_info = await fetch_crate_info(session, crate_name)

        # Resolve version
        resolved_version = await resolve_version(crate_info, version)

        # Get database path
        db_path = await get_db_path(crate_name, resolved_version)

        # Check if already ingested
        if db_path.exists():
            logger.info(f"Crate {crate_name}@{resolved_version} already ingested")
            # But check if it's properly initialized
            try:
                async with aiosqlite.connect(db_path) as db:
                    cursor = await db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='crate_metadata'"
                    )
                    if not await cursor.fetchone():
                        logger.warning("Database exists but not initialized, reinitializing...")
                        await init_database(db_path)
            except Exception as e:
                logger.error(f"Error checking database: {e}")
                await init_database(db_path)
            return db_path

        # Initialize database
        logger.info(f"Initializing database at {db_path}")
        await init_database(db_path)

        # Store crate metadata
        description = crate_info.get("description", "")
        repository = crate_info.get("repository")
        documentation = (
            crate_info.get("documentation") or f"https://docs.rs/{crate_name}"
        )

        await store_crate_metadata(
            db_path,
            crate_name,
            resolved_version,
            description,
            repository,
            documentation,
        )

        # For MVP, create a simple embedding from the crate description
        if description:
            chunks = [
                {
                    "item_path": "crate",
                    "header": f"{crate_name} - Crate Documentation",
                    "content": description,
                }
            ]
            
            try:
                # Generate embeddings
                embeddings = await generate_embeddings(chunks)
                
                # Store embeddings
                await store_embeddings(db_path, chunks, embeddings)
                
                logger.info(f"Successfully stored description embedding for {crate_name}@{resolved_version}")
            except Exception as e:
                logger.error(f"Failed to generate/store embeddings: {e}")
        
        # Note: Full rustdoc parsing would go here in a complete implementation
        logger.info(f"Successfully ingested {crate_name}@{resolved_version}")

        return db_path
