"""Ingestion pipeline for Rust crate documentation."""

import asyncio
import gzip
import io
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp
import aiosqlite
import ijson
import sqlite_vec
import zstandard
from fastembed import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    CACHE_DIR,
    CACHE_MAX_SIZE_BYTES,
    DB_BATCH_SIZE,
    DOWNLOAD_CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    HTTP_TIMEOUT,
    MAX_DECOMPRESSED_SIZE,
    MAX_DOWNLOAD_SIZE,
    MODEL_NAME,
)
from .database import get_db_path, init_database, store_crate_metadata

logger = logging.getLogger(__name__)


# Global embedding model instance
_embedding_model: TextEmbedding | None = None

# Global per-crate lock registry to prevent duplicate ingestion
_crate_locks: dict[str, asyncio.Lock] = {}


def get_embedding_model() -> TextEmbedding:
    """Get or create the embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _embedding_model = TextEmbedding(model_name=MODEL_NAME)
    return _embedding_model


async def get_crate_lock(crate_name: str, version: str) -> asyncio.Lock:
    """Get or create a lock for a specific crate@version."""
    key = f"{crate_name}@{version}"
    if key not in _crate_locks:
        _crate_locks[key] = asyncio.Lock()
    return _crate_locks[key]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def resolve_version(
    session: aiohttp.ClientSession, crate_name: str, version: str = "latest"
) -> tuple[str, str]:
    """Resolve the actual version from docs.rs, returning (version, rustdoc_url).

    Uses docs.rs redirect to resolve 'latest' or other version selectors.
    Returns the resolved version and the URL to the rustdoc JSON file.
    """
    # Construct the URL for version resolution
    base_url = f"https://docs.rs/crate/{crate_name}/{version}"

    # Make a HEAD request to get the redirect without downloading content
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with session.head(
        base_url, timeout=timeout, allow_redirects=True
    ) as response:
        if response.status != 200:
            raise Exception(
                f"Failed to resolve version for {crate_name}: HTTP {response.status}"
            )

        # Extract the final URL after redirects
        final_url = str(response.url)

        # Parse the version from the URL
        # Format: https://docs.rs/{crate_name}/{version}/{crate_name}/
        parts = final_url.strip("/").split("/")
        if len(parts) >= 5 and parts[3] == crate_name:
            resolved_version = parts[4]

            # Construct the rustdoc JSON URL
            # Convert crate name underscores for the JSON filename
            json_name = crate_name.replace("-", "_")
            rustdoc_url = (
                f"https://docs.rs/{crate_name}/{resolved_version}/{json_name}.json"
            )

            return resolved_version, rustdoc_url
        else:
            raise Exception(f"Unexpected URL format from docs.rs redirect: {final_url}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
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


async def resolve_version_from_crate_info(
    crate_info: dict[str, Any], version: str | None = None
) -> str:
    """Resolve version string to actual version from crate info."""
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
    session: aiohttp.ClientSession,
    crate_name: str,
    version: str,
    rustdoc_url: str | None = None,
) -> tuple[bytes, str]:
    """Download rustdoc JSON from docs.rs with compression support.

    Returns a tuple of (raw bytes, url) for further processing.
    Supports .json, .json.zst, and .json.gz formats.
    """
    # Use provided URL or construct default
    if rustdoc_url is None:
        json_name = crate_name.replace("-", "_")
        rustdoc_url = f"https://docs.rs/{crate_name}/{version}/{json_name}.json"

    # Try different compression formats in order of preference
    urls_to_try = []

    # If URL already has a compressed extension, use it as-is
    if rustdoc_url.endswith((".json.zst", ".json.gz")):
        urls_to_try.append(rustdoc_url)
    else:
        # Remove .json extension if present to get base URL
        base_url = (
            rustdoc_url.rstrip(".json")
            if rustdoc_url.endswith(".json")
            else rustdoc_url
        )
        # Try compressed formats first (preferred for size), then uncompressed
        urls_to_try.extend(
            [f"{base_url}.json.zst", f"{base_url}.json.gz", f"{base_url}.json"]
        )

    last_error = None
    for url in urls_to_try:
        try:
            logger.info(f"Attempting to download rustdoc from: {url}")

            # Use retry for transient network errors
            @retry(
                stop=stop_after_attempt(2),
                wait=wait_exponential(multiplier=1, min=1, max=5),
                reraise=True,
            )
            async def _download_with_retry(download_url):
                return await session.get(
                    download_url,
                    timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
                    headers={"Accept": "application/json, application/octet-stream"},
                )

            async with await _download_with_retry(url) as resp:
                if resp.status == 404:
                    # Try next format
                    continue
                elif resp.status != 200:
                    raise Exception(f"Failed to download rustdoc: HTTP {resp.status}")

                # Check compressed size
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                    raise Exception(
                        f"Compressed file too large: {content_length} bytes"
                    )

                # Stream download with size checking
                chunks = []
                total_size = 0

                async for chunk in resp.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                    chunks.append(chunk)
                    total_size += len(chunk)

                    if total_size > MAX_DOWNLOAD_SIZE:
                        raise Exception(
                            f"Download exceeded size limit: {total_size} bytes"
                        )

                content = b"".join(chunks)
                logger.info(f"Successfully downloaded {total_size} bytes from {url}")

                # Return the raw content, the URL is available in closure
                return (content, url)

        except Exception as e:
            last_error = e
            # Only log non-404 errors
            if "404" not in str(e):
                logger.warning(f"Failed to download from {url}: {e}")

    # All formats failed
    if last_error:
        raise last_error
    else:
        raise Exception(
            f"Rustdoc JSON not found for {crate_name}@{version} in any format"
        )


async def decompress_content(content: bytes, url: str) -> str:
    """Decompress content based on URL extension with size limits.

    Returns the decompressed JSON string.
    """
    if url.endswith(".json.zst"):
        # Zstandard decompression
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)

        # Stream decompress with size checking
        decompressed_chunks = []
        total_size = 0

        with dctx.stream_reader(io.BytesIO(content)) as reader:
            while True:
                chunk = reader.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                decompressed_chunks.append(chunk)
                total_size += len(chunk)

                if total_size > MAX_DECOMPRESSED_SIZE:
                    raise Exception(
                        f"Decompressed size exceeded limit: {total_size} bytes"
                    )

        decompressed = b"".join(decompressed_chunks)
        logger.info(
            f"Decompressed .zst file: {len(content)} -> {len(decompressed)} bytes"
        )
        return decompressed.decode("utf-8")

    elif url.endswith(".json.gz"):
        # Gzip decompression
        decompressed = gzip.decompress(content)

        if len(decompressed) > MAX_DECOMPRESSED_SIZE:
            raise Exception(
                f"Decompressed size exceeded limit: {len(decompressed)} bytes"
            )

        logger.info(
            f"Decompressed .gz file: {len(content)} -> {len(decompressed)} bytes"
        )
        return decompressed.decode("utf-8")

    else:
        # Uncompressed JSON
        if len(content) > MAX_DECOMPRESSED_SIZE:
            raise Exception(f"Uncompressed size exceeded limit: {len(content)} bytes")

        return content.decode("utf-8")


async def parse_rustdoc_items(json_content: str) -> list[dict[str, Any]]:
    """Parse rustdoc JSON using ijson for memory-efficient streaming.

    Extracts items (functions, structs, traits, modules) with their documentation.
    Returns a list of items with item_id, item_path, header, and docstring.
    """
    items = []

    # For rustdoc JSON format, we need to parse two main sections:
    # 1. "paths" - maps IDs to paths
    # 2. "index" - contains the actual items with their details

    # First pass: collect paths
    id_to_path = {}
    try:
        # Use items() for efficient iteration over dict items
        parser = ijson.kvitems(io.BytesIO(json_content.encode()), "paths")
        for item_id, path_info in parser:
            if isinstance(path_info, dict) and "path" in path_info:
                id_to_path[item_id] = "::".join(path_info["path"])
    except Exception as e:
        logger.warning(f"Error parsing paths section: {e}")

    # Second pass: collect items from index
    try:
        parser = ijson.kvitems(io.BytesIO(json_content.encode()), "index")
        for item_id, item_info in parser:
            if not isinstance(item_info, dict):
                continue

            # Extract relevant fields
            name = item_info.get("name", "")
            kind = item_info.get("kind", "")
            docs = item_info.get("docs", "")

            # Get the path from our mapping
            path = id_to_path.get(item_id, "")

            # Filter to relevant item types
            if isinstance(kind, str):
                kind_lower = kind.lower()
            elif isinstance(kind, dict) and len(kind) == 1:
                # Handle {"variant_name": data} format
                kind_lower = list(kind.keys())[0].lower()
            else:
                continue

            # Check if it's a type we want to index
            indexable_kinds = [
                "function",
                "struct",
                "trait",
                "mod",
                "module",
                "enum",
                "type",
                "typedef",
                "const",
                "static",
                "method",
            ]

            if any(k in kind_lower for k in indexable_kinds):
                # Create header based on item type
                if "function" in kind_lower or "method" in kind_lower:
                    header = f"fn {name}"
                elif "struct" in kind_lower:
                    header = f"struct {name}"
                elif "trait" in kind_lower:
                    header = f"trait {name}"
                elif "mod" in kind_lower:
                    header = f"mod {name}"
                elif "enum" in kind_lower:
                    header = f"enum {name}"
                elif "type" in kind_lower:
                    header = f"type {name}"
                elif "const" in kind_lower:
                    header = f"const {name}"
                elif "static" in kind_lower:
                    header = f"static {name}"
                else:
                    header = f"{kind_lower} {name}"

                # Ensure path includes the item name if not already
                if path and name and not path.endswith(name):
                    full_path = f"{path}::{name}"
                else:
                    full_path = path or name

                items.append(
                    {
                        "item_id": item_id,
                        "item_path": full_path,
                        "header": header,
                        "docstring": docs,
                        "kind": kind_lower,
                    }
                )

    except Exception as e:
        logger.warning(f"Error parsing index section: {e}")

    logger.info(f"Parsed {len(items)} items from rustdoc JSON")
    return items


def calculate_cache_size() -> int:
    """Calculate total size of cache directory using os.scandir for efficiency."""
    total_size = 0

    try:
        # Walk through cache directory
        for root, _dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith(".db"):
                    file_path = os.path.join(root, file)
                    try:
                        # Use os.stat for file size
                        stat_info = os.stat(file_path)
                        total_size += stat_info.st_size
                    except OSError as e:
                        logger.warning(f"Error getting size of {file_path}: {e}")

    except OSError as e:
        logger.error(f"Error calculating cache size: {e}")

    return total_size


async def evict_cache_if_needed() -> None:
    """Evict oldest cache files if total size exceeds limit."""
    current_size = calculate_cache_size()

    if current_size <= CACHE_MAX_SIZE_BYTES:
        logger.debug(f"Cache size {current_size} bytes within limit")
        return

    logger.info(
        f"Cache size {current_size} bytes exceeds limit {CACHE_MAX_SIZE_BYTES}, evicting..."
    )

    # Collect all cache files with their stats
    cache_files = []

    try:
        for root, _dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith(".db"):
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        cache_files.append(
                            {
                                "path": file_path,
                                "size": stat_info.st_size,
                                "mtime": stat_info.st_mtime,
                            }
                        )
                    except OSError:
                        pass

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x["mtime"])

        # Remove oldest files until under limit
        removed_size = 0
        for file_info in cache_files:
            if current_size - removed_size <= CACHE_MAX_SIZE_BYTES:
                break

            try:
                os.remove(file_info["path"])
                removed_size += file_info["size"]
                logger.info(
                    f"Evicted cache file: {file_info['path']} ({file_info['size']} bytes)"
                )
            except OSError as e:
                logger.warning(f"Error removing cache file {file_info['path']}: {e}")

        logger.info(f"Evicted {removed_size} bytes from cache")

    except Exception as e:
        logger.error(f"Error during cache eviction: {e}")


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
    """Store chunks and their embeddings in the database using batch processing."""
    if not chunks or not embeddings:
        return

    total_items = len(chunks)
    logger.info(f"Storing {total_items} embeddings in batches of {DB_BATCH_SIZE}")

    # Pre-serialize all vectors to reduce per-batch overhead
    serialized_embeddings = [
        bytes(sqlite_vec.serialize_float32(embedding)) for embedding in embeddings
    ]

    async with aiosqlite.connect(db_path) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Process in batches
        for batch_start in range(0, total_items, DB_BATCH_SIZE):
            batch_end = min(batch_start + DB_BATCH_SIZE, total_items)
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = serialized_embeddings[batch_start:batch_end]

            try:
                # Begin transaction for this batch
                await db.execute("BEGIN TRANSACTION")

                # Prepare batch data for embeddings table
                embeddings_data = [
                    (
                        chunk["item_path"],
                        chunk["header"],
                        chunk["content"],
                        embedding,
                    )
                    for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=False)
                ]

                # Batch insert into embeddings table
                await db.executemany(
                    """
                    INSERT INTO embeddings (item_path, header, content, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    embeddings_data,
                )

                # Get the rowids for the inserted records
                # SQLite's last_insert_rowid() gives us the last rowid
                cursor = await db.execute("SELECT last_insert_rowid()")
                last_rowid = (await cursor.fetchone())[0]
                first_rowid = last_rowid - len(batch_chunks) + 1

                # Prepare batch data for vec_embeddings table
                vec_data = [
                    (rowid, embedding)
                    for rowid, embedding in zip(
                        range(first_rowid, last_rowid + 1), batch_embeddings, strict=False
                    )
                ]

                # Batch insert into vec_embeddings table
                await db.executemany(
                    "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    vec_data,
                )

                # Commit this batch
                await db.commit()

                # Log progress for large datasets
                if total_items > DB_BATCH_SIZE:
                    progress = (batch_end / total_items) * 100
                    logger.info(
                        f"Batch {batch_start // DB_BATCH_SIZE + 1}: "
                        f"Processed {batch_end}/{total_items} items ({progress:.1f}%)"
                    )

            except Exception as e:
                # Rollback on error
                await db.execute("ROLLBACK")
                logger.error(f"Error in batch {batch_start}-{batch_end}: {e}")
                raise

        logger.info(f"Successfully stored {total_items} embeddings")


async def ingest_crate(crate_name: str, version: str | None = None) -> Path:
    """Ingest a crate's documentation and return the database path."""
    async with aiohttp.ClientSession() as session:
        # Fetch crate info first for basic metadata
        crate_info = await fetch_crate_info(session, crate_name)

        # Resolve version using crate info if not specified
        if not version or version == "latest":
            version = await resolve_version_from_crate_info(crate_info, version)

        # Acquire per-crate lock to prevent duplicate ingestion
        lock = await get_crate_lock(crate_name, version)

        async with lock:
            # Get database path
            db_path = await get_db_path(crate_name, version)

            # Check if already ingested
            if db_path.exists():
                logger.info(f"Crate {crate_name}@{version} already ingested")
                # But check if it's properly initialized
                try:
                    async with aiosqlite.connect(db_path) as db:
                        cursor = await db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='crate_metadata'"
                        )
                        if not await cursor.fetchone():
                            logger.warning(
                                "Database exists but not initialized, reinitializing..."
                            )
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
                version,
                description,
                repository,
                documentation,
            )

            # Try to download and process rustdoc JSON
            try:
                # Resolve version and get rustdoc URL from docs.rs
                logger.info(f"Resolving rustdoc URL for {crate_name}@{version}")
                resolved_version, rustdoc_url = await resolve_version(
                    session, crate_name, version
                )

                # Download rustdoc (with compression support)
                logger.info(f"Downloading rustdoc for {crate_name}@{resolved_version}")
                compressed_content, download_url = await download_rustdoc(
                    session, crate_name, resolved_version, rustdoc_url
                )

                # Decompress content
                logger.info("Decompressing rustdoc content")
                json_content = await decompress_content(
                    compressed_content, download_url
                )

                # Parse rustdoc items
                logger.info("Parsing rustdoc items")
                rustdoc_items = await parse_rustdoc_items(json_content)

                if rustdoc_items:
                    # Convert items to chunks for embedding
                    chunks = []
                    for item in rustdoc_items:
                        # Combine header and docstring for embedding
                        content = (
                            f"{item['header']}\n\n{item['docstring']}"
                            if item["docstring"]
                            else item["header"]
                        )

                        chunks.append(
                            {
                                "item_path": item["item_path"],
                                "header": item["header"],
                                "content": content,
                            }
                        )

                    # Generate embeddings in batches
                    logger.info(f"Generating embeddings for {len(chunks)} items")
                    embeddings = await generate_embeddings(chunks)

                    # Store embeddings
                    logger.info("Storing embeddings in database")
                    await store_embeddings(db_path, chunks, embeddings)

                    logger.info(
                        f"Successfully ingested {len(chunks)} items from rustdoc"
                    )
                else:
                    logger.warning("No items found in rustdoc JSON")
                    raise Exception("No items found in rustdoc JSON")

            except Exception as e:
                logger.warning(f"Failed to process rustdoc JSON: {e}")
                logger.info("Falling back to basic crate description embedding")

                # Fallback: create a simple embedding from the crate description
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

                        logger.info(
                            f"Successfully stored description embedding for {crate_name}@{version}"
                        )
                    except Exception as embed_error:
                        logger.error(
                            f"Failed to generate/store embeddings: {embed_error}"
                        )

            # Run cache eviction if needed
            await evict_cache_if_needed()

            logger.info(f"Successfully ingested {crate_name}@{version}")
            return db_path
