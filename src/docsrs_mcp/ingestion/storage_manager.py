"""Database storage operations and batch processing for embeddings.

This module handles:
- Batch embedding generation with streaming
- Database storage with transaction optimization
- Cleanup operations for re-ingestion
- Memory-efficient batch processing
- Code intelligence data storage (Phase 5)
"""

import json
import logging
import os
from pathlib import Path

import aiosqlite
import sqlite_vec

from ..config import DB_BATCH_SIZE
from ..database import execute_with_retry
from ..memory_utils import MemoryMonitor, get_adaptive_batch_size, trigger_gc_if_needed
from .embedding_manager import get_embedding_model

logger = logging.getLogger(__name__)


def generate_embeddings_streaming(chunks):
    """Generate embeddings for text chunks in streaming fashion.

    Args:
        chunks: Iterable of chunks (can be list or generator).

    Yields:
        tuple: (chunk, embedding) pairs.
    """
    model = get_embedding_model()

    # Buffer for batching
    chunk_buffer = []
    processed_count = 0
    batch_count = 0  # Track number of batches for process recycling

    # Get max text length from config
    max_text_length = int(os.getenv("FASTEMBED_MAX_TEXT_LENGTH", "100"))

    # Use adaptive batch size based on memory pressure
    batch_size = get_adaptive_batch_size()
    logger.info(f"Using adaptive batch size: {batch_size}")

    for chunk in chunks:
        # Handle both dict chunks and simple string chunks
        if isinstance(chunk, dict):
            header = chunk.get("header") or ""
            doc = chunk.get("doc") or ""
            text = header + " " + doc
        else:
            text = str(chunk)

        # Enforce max text length to prevent memory issues
        if len(text) > max_text_length:
            text = text[:max_text_length]

        chunk_buffer.append(chunk)

        # Process batch when buffer reaches batch size
        if len(chunk_buffer) >= batch_size:
            # Extract texts for embedding
            texts = []
            for c in chunk_buffer:
                if isinstance(c, dict):
                    header = c.get("header") or ""
                    doc = c.get("doc") or ""
                    t = header + " " + doc
                else:
                    t = str(c)
                # Enforce max length
                if len(t) > max_text_length:
                    t = t[:max_text_length]
                texts.append(t)

            # Generate embeddings for batch
            try:
                embeddings = list(model.embed(texts))

                # Yield (chunk, embedding) pairs
                for c, emb in zip(chunk_buffer, embeddings, strict=False):
                    yield (c, emb)
                    processed_count += 1

                # Log progress
                if processed_count % 100 == 0:
                    logger.debug(f"Generated {processed_count} embeddings")

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Skip this batch on error
                for c in chunk_buffer:
                    yield (c, None)  # Yield with None embedding on error

            # Clear buffer
            chunk_buffer = []
            batch_count += 1

            # Trigger GC periodically
            if batch_count % 10 == 0:
                trigger_gc_if_needed()

            # Check memory pressure and adjust batch size
            new_batch_size = get_adaptive_batch_size()
            if new_batch_size != batch_size:
                logger.info(f"Adjusting batch size: {batch_size} -> {new_batch_size}")
                batch_size = new_batch_size

    # Process remaining chunks in buffer
    if chunk_buffer:
        texts = []
        for c in chunk_buffer:
            if isinstance(c, dict):
                header = c.get("header") or ""
                doc = c.get("doc") or ""
                t = header + " " + doc
            else:
                t = str(c)
            if len(t) > max_text_length:
                t = t[:max_text_length]
            texts.append(t)

        try:
            embeddings = list(model.embed(texts))
            for c, emb in zip(chunk_buffer, embeddings, strict=False):
                yield (c, emb)
                processed_count += 1
        except Exception as e:
            logger.error(f"Error generating embeddings for final batch: {e}")
            for c in chunk_buffer:
                yield (c, None)

    logger.info(f"Completed embedding generation: {processed_count} items")


async def generate_embeddings(chunks: list[dict[str, str]]) -> list[list[float]]:
    """Generate embeddings for text chunks (backwards compatible).

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List[List[float]]: List of embedding vectors
    """
    embeddings = []
    for _chunk, embedding in generate_embeddings_streaming(chunks):
        embeddings.append(embedding)
    return embeddings


async def clean_existing_embeddings(db_path: Path, crate_name: str) -> None:
    """Clean up existing embeddings for a crate before re-ingestion.

    This prevents duplicates when re-ingesting a crate.

    Args:
        db_path: Path to the database
        crate_name: Name of the crate
    """
    async with aiosqlite.connect(db_path) as db:
        # Delete existing embeddings for this crate
        await execute_with_retry(
            db,
            "DELETE FROM embeddings WHERE item_path LIKE ?",
            (f"{crate_name}%",),
        )
        await db.commit()
        logger.info(f"Cleaned existing embeddings for {crate_name}")


async def _store_batch(
    db,
    chunk_buffer: list[dict],
    embedding_buffer: list[bytes],
    batch_num: int,
    total_items: int,
) -> None:
    """Store a batch of embeddings in the database.

    Args:
        db: Database connection
        chunk_buffer: List of chunks to store
        embedding_buffer: List of serialized embeddings
        batch_num: Batch number for logging
        total_items: Total items processed so far
    """
    if not chunk_buffer:
        return

    # Import validation function
    from ..validation import validate_item_path_with_fallback

    # Prepare batch data
    batch_data = []
    for chunk, embedding_bytes in zip(chunk_buffer, embedding_buffer, strict=False):
        # Validate and fix item_path
        item_path = chunk.get("item_path")
        item_id = chunk.get("item_id")
        item_type = chunk.get("item_type")
        
        # Ensure we have a valid item_path
        validated_path, used_fallback = validate_item_path_with_fallback(
            item_path, item_id, item_type
        )
        
        if used_fallback:
            logger.debug(f"Generated fallback path: {validated_path} for item_id: {item_id}")
        
        # Ensure we have a header (fallback to item_path or item_type)
        header = chunk.get("header")
        if not header:
            if item_path:
                # Use the last segment of the item_path as header
                header = item_path.split("::")[-1] if "::" in item_path else item_path
            elif item_type:
                header = f"{item_type} item"
            else:
                header = "Unknown item"
        
        # Convert intelligence data to JSON strings for storage (Phase 5)
        safety_info = chunk.get("safety_info")
        if safety_info and isinstance(safety_info, dict):
            safety_info_json = json.dumps(safety_info)
        else:
            safety_info_json = None

        error_types = chunk.get("error_types")
        if error_types and isinstance(error_types, list):
            error_types_json = json.dumps(error_types)
        else:
            error_types_json = None

        feature_requirements = chunk.get("feature_requirements")
        if feature_requirements and isinstance(feature_requirements, list):
            feature_requirements_json = json.dumps(feature_requirements)
        else:
            feature_requirements_json = None

        # Map to actual database columns
        # Ensure content is never None
        content = chunk.get("doc", "")
        if content is None:
            content = ""
        
        batch_data.append(
            (
                validated_path,  # Use validated path instead of raw item_path
                header,  # Use header with fallback
                content,  # content column (never None)
                embedding_bytes,
                chunk.get("item_type"),
                chunk.get("signature"),
                chunk.get("parent_id"),
                chunk.get("examples"),
                chunk.get("visibility", "public"),
                chunk.get("deprecated", False),
                chunk.get("generic_params"),
                chunk.get("trait_bounds"),
                safety_info_json,  # Phase 5: safety information
                error_types_json,  # Phase 5: error types
                feature_requirements_json,  # Phase 5: feature requirements
                chunk.get("is_safe", True),  # Phase 5: safety flag
            )
        )

    # Use REPLACE to handle duplicates efficiently
    # executemany for batch inserts
    await db.executemany(
        """
        INSERT OR REPLACE INTO embeddings
        (item_path, header, content, embedding, item_type, signature, 
         parent_id, examples, visibility, deprecated, generic_params, trait_bounds,
         safety_info, error_types, feature_requirements, is_safe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch_data,
    )

    await db.commit()

    logger.debug(
        f"Batch {batch_num}: Stored {len(chunk_buffer)} embeddings "
        f"(total: {total_items + len(chunk_buffer)})"
    )


async def store_embeddings_streaming(db_path: Path, chunk_embedding_pairs) -> None:
    """Store chunks and their embeddings in the database using streaming batch processing.

    Args:
        db_path: Path to the database file.
        chunk_embedding_pairs: Iterator of (chunk, embedding) tuples.
    """
    total_items = 0
    batch_num = 0

    async with aiosqlite.connect(db_path) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Buffer for batching
        chunk_buffer = []
        embedding_buffer = []

        with MemoryMonitor("store_embeddings"):
            for chunk, embedding in chunk_embedding_pairs:
                chunk_buffer.append(chunk)
                # Pre-serialize the embedding
                embedding_buffer.append(bytes(sqlite_vec.serialize_float32(embedding)))

                # Process batch when buffer reaches DB_BATCH_SIZE
                if len(chunk_buffer) >= DB_BATCH_SIZE:
                    await _store_batch(
                        db, chunk_buffer, embedding_buffer, batch_num, total_items
                    )

                    total_items += len(chunk_buffer)
                    batch_num += 1

                    # Clear buffers and trigger GC
                    chunk_buffer = []
                    embedding_buffer = []
                    trigger_gc_if_needed()

            # Process remaining items in buffer
            if chunk_buffer:
                await _store_batch(
                    db, chunk_buffer, embedding_buffer, batch_num, total_items
                )
                total_items += len(chunk_buffer)

        logger.info(f"Successfully stored {total_items} embeddings")


async def store_embeddings(
    db_path: Path, chunks: list[dict[str, str]], embeddings: list[list[float]]
) -> None:
    """Store chunks and their embeddings (backwards compatible).

    Args:
        db_path: Path to the database
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
    """
    # Convert to streaming format
    chunk_embedding_pairs = zip(chunks, embeddings, strict=False)
    await store_embeddings_streaming(db_path, chunk_embedding_pairs)
