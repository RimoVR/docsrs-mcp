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
from typing import Any

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

    # Sync to vec_embeddings virtual table (following pattern from code_examples.py)
    if chunk_buffer:  # Only sync if we actually stored data
        logger.debug(f"Batch {batch_num}: Syncing {len(chunk_buffer)} embeddings to vector index...")
        
        # Get the rowids of just inserted records to avoid resyncing everything
        # Find the minimum ID from this batch to only sync new records
        min_id = await db.execute("SELECT MIN(id) FROM embeddings WHERE rowid > (SELECT COALESCE(MAX(rowid), 0) FROM vec_embeddings)")
        min_id_result = await min_id.fetchone()
        min_id_value = min_id_result[0] if min_id_result and min_id_result[0] else 0
        
        # Populate the vector table from embeddings table for new records
        cursor = await db.execute("""
            SELECT id, embedding FROM embeddings 
            WHERE id >= ? AND id NOT IN (SELECT rowid FROM vec_embeddings)
        """, (min_id_value,))
        
        vec_data = []
        async for row in cursor:
            rowid, embedding_blob = row
            vec_data.append((rowid, embedding_blob))
            
            # Process in batches for efficiency (same pattern as code_examples.py)
            if len(vec_data) >= 100:
                await db.executemany(
                    "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    vec_data
                )
                vec_data = []
        
        # Insert remaining data
        if vec_data:
            await db.executemany(
                "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                vec_data
            )
        
        await db.commit()
        logger.debug(f"Batch {batch_num}: Vector index sync completed")

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


async def store_trait_implementations(
    db_path: Path, trait_implementations: list[dict[str, Any]], crate_id: int
) -> None:
    """Store trait implementations in the database.
    
    Args:
        db_path: Path to the database file
        trait_implementations: List of trait implementation dictionaries
        crate_id: Database crate ID for foreign key constraint
    """
    if not trait_implementations:
        logger.debug("No trait implementations to store")
        return
        
    async with aiosqlite.connect(db_path) as db:
        try:
            # Prepare batch insert for trait implementations
            trait_impl_data = []
            for impl in trait_implementations:
                trait_impl_data.append((
                    crate_id,
                    impl.get("trait_path", ""),
                    impl.get("impl_type_path", ""), 
                    impl.get("generic_params"),
                    impl.get("where_clauses"),
                    1 if impl.get("is_blanket", False) else 0,
                    1 if impl.get("is_negative", False) else 0,
                    impl.get("impl_signature"),
                    None,  # source_location - not available from rustdoc JSON
                    impl.get("stability_level", "stable"),
                    impl.get("item_id", "")
                ))
            
            # Batch insert with proper SQLite syntax
            await db.executemany("""
                INSERT OR IGNORE INTO trait_implementations (
                    crate_id, trait_path, impl_type_path, generic_params, 
                    where_clauses, is_blanket, is_negative, impl_signature,
                    source_location, stability_level, item_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, trait_impl_data)
            
            await db.commit()
            logger.info(f"Stored {len(trait_implementations)} trait implementations for crate {crate_id}")
            
        except Exception as e:
            logger.error(f"Error storing trait implementations: {e}")
            await db.rollback()
            raise


async def store_trait_definitions(
    db_path: Path, trait_definitions: list[dict[str, Any]], crate_id: int
) -> None:
    """Store trait definitions in the database.
    
    Note: This stores in trait_implementations table with special markers
    for trait definitions themselves.
    
    Args:
        db_path: Path to the database file
        trait_definitions: List of trait definition dictionaries
        crate_id: Database crate ID for foreign key constraint
    """
    if not trait_definitions:
        logger.debug("No trait definitions to store")
        return
        
    async with aiosqlite.connect(db_path) as db:
        try:
            # Store trait definitions as special entries
            # We could create a separate traits table, but for MVP we'll use
            # the existing structure with a marker
            trait_def_data = []
            for trait_def in trait_definitions:
                # Store trait definition as impl of itself
                trait_def_data.append((
                    crate_id,
                    trait_def.get("trait_path", ""),
                    f"_TRAIT_DEF_{trait_def.get('trait_path', '')}", # Special marker
                    trait_def.get("generic_params"),
                    json.dumps(trait_def.get("supertraits", [])) if trait_def.get("supertraits") else None,
                    0,  # not blanket
                    0,  # not negative
                    f"trait {trait_def.get('trait_path', '').split('::')[-1]}" if trait_def.get("trait_path") else "",
                    None,  # source_location
                    trait_def.get("stability_level", "stable"),
                    trait_def.get("item_id", "")
                ))
            
            await db.executemany("""
                INSERT OR IGNORE INTO trait_implementations (
                    crate_id, trait_path, impl_type_path, generic_params,
                    where_clauses, is_blanket, is_negative, impl_signature,
                    source_location, stability_level, item_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, trait_def_data)
            
            await db.commit()
            logger.info(f"Stored {len(trait_definitions)} trait definitions for crate {crate_id}")
            
        except Exception as e:
            logger.error(f"Error storing trait definitions: {e}")
            await db.rollback()
            raise


async def store_method_signatures(
    db_path: Path, method_signatures: list[dict[str, Any]], crate_id: int
) -> None:
    """Store method signatures in the database.
    
    Args:
        db_path: Path to the database file
        method_signatures: List of method signature dictionaries
        crate_id: Database crate ID for foreign key constraint
    """
    if not method_signatures:
        logger.debug("No method signatures to store")
        return
        
    async with aiosqlite.connect(db_path) as db:
        try:
            # Store method signatures
            method_data = []
            for method in method_signatures:
                method_data.append((
                    crate_id,
                    method.get("parent_type_path", ""),
                    method.get("method_name", ""),
                    method.get("signature", ""),  # full_signature
                    method.get("generic_params"),
                    method.get("where_clauses"),
                    method.get("return_type"),
                    1 if method.get("is_async", False) else 0,
                    1 if method.get("is_unsafe", False) else 0,
                    1 if method.get("is_const", False) else 0,
                    method.get("visibility", "pub"),
                    method.get("method_kind", "inherent"),
                    method.get("trait_source"),
                    method.get("receiver_type"),
                    method.get("stability_level", "stable"),
                    method.get("item_id", ""),
                ))
            
            await db.executemany(
                """
                INSERT OR IGNORE INTO method_signatures (
                    crate_id, parent_type_path, method_name, full_signature,
                    generic_params, where_clauses, return_type,
                    is_async, is_unsafe, is_const, visibility, method_kind,
                    trait_source, receiver_type, stability_level, item_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                method_data,
            )
            
            await db.commit()
            logger.info(f"Stored {len(method_signatures)} method signatures")
            
        except Exception as e:
            logger.error(f"Error storing method signatures: {e}")
            await db.rollback()
            raise


async def store_associated_items(
    db_path: Path, associated_items: list[dict[str, Any]], crate_id: int
) -> None:
    """Store associated items (types, consts, functions) in the database.

    Args:
        db_path: Path to the database file
        associated_items: List of associated item dictionaries
        crate_id: Database crate ID for foreign key constraint
    """
    if not associated_items:
        logger.debug("No associated items to store")
        return

    async with aiosqlite.connect(db_path) as db:
        try:
            assoc_data = []
            for it in associated_items:
                assoc_data.append(
                    (
                        crate_id,
                        it.get("container_path", ""),
                        it.get("item_name", ""),
                        it.get("item_kind", "type"),
                        it.get("item_signature", ""),
                        it.get("default_value"),
                        json.dumps(it.get("generic_params")) if it.get("generic_params") else None,
                        json.dumps(it.get("where_clauses")) if it.get("where_clauses") else None,
                        it.get("visibility", "pub"),
                        it.get("stability_level", "stable"),
                        it.get("item_id", ""),
                    )
                )

            await db.executemany(
                """
                INSERT OR IGNORE INTO associated_items (
                    crate_id, container_path, item_name, item_kind, item_signature,
                    default_value, generic_params, where_clauses, visibility, stability_level,
                    item_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                assoc_data,
            )

            await db.commit()
            logger.info(f"Stored {len(associated_items)} associated items")

        except Exception as e:
            logger.error(f"Error storing associated items: {e}")
            await db.rollback()
            raise


async def store_enhanced_items_streaming(
    db_path: Path, enhanced_items_stream, crate_id: int
) -> None:
    """Store enhanced rustdoc items with trait extraction support.
    
    This function handles both regular items and trait-specific data
    from the enhanced rustdoc parser.
    
    Args:
        db_path: Path to the database file
        enhanced_items_stream: Stream of enhanced items with trait data
        crate_id: Database crate ID for trait storage
    """
    from .signature_extractor import (
        extract_signature,
        extract_deprecated, 
        extract_visibility,
    )
    from .code_examples import extract_code_examples
    
    regular_items = []
    trait_implementations = []
    trait_definitions = []
    method_signatures = []
    associated_items = []
    modules_data = None
    
    # Collect items from stream
    async for item in enhanced_items_stream:
        if "_trait_impl" in item:
            # This is trait implementation data
            trait_implementations.append(item["_trait_impl"])
        elif "_trait_def" in item:
            # This is trait definition data
            trait_definitions.append(item["_trait_def"])
        elif "_method_signature" in item:
            # This is method signature data
            method_signatures.append(item["_method_signature"])
        elif "_associated_item" in item:
            # Trait/type associated items
            associated_items.append(item["_associated_item"])
        elif "_modules" in item:
            # Module hierarchy data
            modules_data = item["_modules"]
        else:
            # Regular item for embedding storage - enhance with metadata
            try:
                item["signature"] = extract_signature(item)
                item["deprecated"] = extract_deprecated(item)
                item["visibility"] = extract_visibility(item)
                item["examples"] = extract_code_examples(item.get("doc", ""))
                regular_items.append(item)
            except Exception as e:
                logger.warning(f"Error enhancing item metadata: {e}")
                regular_items.append(item)  # Store anyway
    
    # Store trait implementations and definitions first
    if trait_implementations:
        await store_trait_implementations(db_path, trait_implementations, crate_id)
        
    if trait_definitions:
        await store_trait_definitions(db_path, trait_definitions, crate_id)
    
    # Store method signatures
    if method_signatures:
        await store_method_signatures(db_path, method_signatures, crate_id)

    # Synthesize minimal method signatures from regular items to avoid empty results
    try:
        synthetic_methods = []
        for it in regular_items:
            if (it.get("item_type") or "").lower() == "method" and it.get("parent_id"):
                synthetic_methods.append(
                    {
                        "parent_type_path": it.get("parent_id"),
                        "method_name": it.get("name") or it.get("header") or "",
                        "item_id": it.get("item_id", ""),
                        "method_kind": "inherent",
                        "signature": it.get("signature", ""),
                        "generic_params": it.get("generic_params"),
                        "where_clauses": it.get("trait_bounds"),
                        "is_const": False,
                        "is_async": False,
                        "visibility": it.get("visibility", "public"),
                        "trait_source": None,
                        "receiver_type": None,
                        "stability_level": "stable",
                    }
                )
        if synthetic_methods:
            await store_method_signatures(db_path, synthetic_methods, crate_id)
            logger.info(f"Synthesized {len(synthetic_methods)} method signatures from items")
    except Exception as e:
        logger.warning(f"Failed to synthesize method signatures: {e}")
    
    # Store module hierarchy if present
    if modules_data:
        try:
            # Import and store modules
            from docsrs_mcp.database.storage import store_modules
            await store_modules(db_path, crate_id, modules_data)
            logger.info(f"Stored module hierarchy with {len(modules_data)} modules")
        except Exception as e:
            logger.warning(f"Error storing modules: {e}")
    
    # Generate embeddings and store regular items if any
    if regular_items:
        logger.info(f"Processing {len(regular_items)} regular items for embedding")
        chunk_embedding_pairs = generate_embeddings_streaming(regular_items)
        await store_embeddings_streaming(db_path, chunk_embedding_pairs)

    # Store associated items last (after trait defs are in place)
    if associated_items:
        await store_associated_items(db_path, associated_items, crate_id)
    
    logger.info(
        f"Enhanced storage complete: {len(regular_items)} items, "
        f"{len(trait_implementations)} trait impls, "
        f"{len(trait_definitions)} trait defs, "
        f"{len(method_signatures)} method signatures, "
        f"{len(associated_items)} associated items"
    )
