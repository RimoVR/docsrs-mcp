"""Ingestion pipeline for Rust crate documentation."""

import asyncio
import gzip
import io
import json
import logging
import os
import re
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
    PARSE_CHUNK_SIZE,
    RUST_VERSION_MANIFEST_URL,
    STDLIB_CRATES,
)
from .database import get_db_path, init_database, store_crate_metadata
from .memory_utils import (
    MemoryMonitor,
    get_adaptive_batch_size,
    trigger_gc_if_needed,
)

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


def is_stdlib_crate(crate_name: str) -> bool:
    """Check if a crate name is a Rust standard library crate."""
    return crate_name.lower() in STDLIB_CRATES


async def fetch_current_stable_version(session: aiohttp.ClientSession) -> str:
    """Fetch the current stable Rust version from the official channel."""
    try:
        async with session.get(
            RUST_VERSION_MANIFEST_URL,
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch Rust version: HTTP {resp.status}")

            # The version file contains the version string directly
            version_text = await resp.text()
            # Extract version number (e.g., "1.75.0 (2023-12-28)" -> "1.75.0")
            version_match = re.match(r"(\d+\.\d+\.\d+)", version_text.strip())
            if version_match:
                return version_match.group(1)
            else:
                raise Exception(f"Could not parse Rust version from: {version_text}")
    except Exception as e:
        logger.warning(f"Failed to fetch current stable version: {e}")
        # Fallback to a known stable version
        return "1.75.0"


async def resolve_stdlib_version(
    session: aiohttp.ClientSession, version: str | None = None
) -> str:
    """Resolve standard library version string to actual Rust version."""
    if not version or version in ["latest", "stable"]:
        return await fetch_current_stable_version(session)
    elif version in ["beta", "nightly"]:
        # For beta/nightly, we return the channel name
        # The actual version will be determined by the channel manifest
        return version
    else:
        # Assume it's a specific Rust version like "1.75.0"
        return version


def get_stdlib_url(crate_name: str, version: str) -> str:
    """Construct the URL for downloading standard library rustdoc JSON.

    Note: The standard library rustdoc JSON is distributed as part of
    the rust-docs-json component. For now, we'll try to use docs.rs
    which mirrors some stdlib crates.
    """
    # docs.rs mirrors std library crates with special version handling
    # For std library, the version is typically the Rust version
    json_name = crate_name.replace("-", "_")

    # Try multiple URL patterns that might work for stdlib
    # docs.rs sometimes uses 'latest' for stdlib crates
    if version in ["stable", "beta", "nightly"]:
        # For channel names, try 'latest' on docs.rs
        return f"https://docs.rs/{crate_name}/latest/{json_name}.json"
    else:
        # For specific versions, use the version directly
        return f"https://docs.rs/{crate_name}/{version}/{json_name}.json"


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


def normalize_item_type(kind: dict | str) -> str:
    """Normalize rustdoc kind to standard item_type."""
    if isinstance(kind, dict):
        kind_str = list(kind.keys())[0] if kind else "unknown"
    else:
        kind_str = str(kind).lower()

    # Map rustdoc kinds to standard types
    type_map = {
        "function": "function",
        "struct": "struct",
        "trait": "trait",
        "mod": "module",
        "module": "module",
        "method": "method",
        "enum": "enum",
        "type": "type",
        "typedef": "type",
        "const": "const",
        "static": "static",
    }

    # Find matching type
    for key, value in type_map.items():
        if key in kind_str:
            return value
    return kind_str


def format_signature(decl: dict) -> str:
    """Format a function/method declaration into a readable signature."""
    try:
        inputs = decl.get("inputs", [])
        output = decl.get("output")

        # Format parameters
        params = []
        for param in inputs:
            if isinstance(param, dict):
                param_name = param.get("name", "_")
                param_type = param.get("type", {})
                # Simple type extraction - can be enhanced
                if isinstance(param_type, dict):
                    type_str = param_type.get("name", "Unknown")
                else:
                    type_str = str(param_type)
                params.append(f"{param_name}: {type_str}")
            else:
                params.append(str(param))

        signature = f"({', '.join(params)})"

        # Add return type if present
        if output and output != "unit":
            if isinstance(output, dict):
                return_type = output.get("name", "")
            else:
                return_type = str(output)
            if return_type:
                signature += f" -> {return_type}"

        return signature
    except Exception:
        return ""


def extract_signature(item: dict) -> str | None:
    """Extract function/method signature."""
    try:
        inner = item.get("inner", {})

        # Check for function
        if isinstance(inner, dict) and "function" in inner:
            decl = inner["function"].get("decl", {})
            if decl:
                return format_signature(decl)

        # Check for method
        if isinstance(inner, dict) and "method" in inner:
            decl = inner["method"].get("decl", {})
            if decl:
                return format_signature(decl)

        # Check for other inner types that might have signatures
        for key in ["assoc_const", "assoc_type"]:
            if key in inner:
                # These might have type information
                type_info = inner[key].get("type", {})
                if isinstance(type_info, dict) and "name" in type_info:
                    return type_info["name"]
    except Exception as e:
        logger.warning(f"Could not extract signature: {e}")
    return None


def resolve_parent_id(item: dict, paths: dict) -> str | None:
    """Resolve the parent module/struct ID for an item."""
    try:
        # Check if item has a parent field
        if "parent" in item:
            parent = item["parent"]
            if parent and parent != "null":
                return parent

        # Try to infer from path if available
        if "path" in item:
            path_parts = item["path"]
            if isinstance(path_parts, list) and len(path_parts) > 1:
                # The parent would be all but the last part
                parent_path = "::".join(path_parts[:-1])
                # Try to find the parent ID from paths
                for pid, pinfo in paths.items():
                    if isinstance(pinfo, dict) and "path" in pinfo:
                        if "::".join(pinfo["path"]) == parent_path:
                            return pid
    except Exception as e:
        logger.warning(f"Could not resolve parent ID: {e}")
    return None


def extract_code_examples(docstring: str) -> list[str]:
    """Extract ```rust code blocks from documentation."""
    if not docstring:
        return []

    try:
        # Match ```rust blocks and also plain ``` blocks that are likely Rust
        pattern = r"```(?:rust)?\s*\n(.*?)```"
        examples = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

        # Clean up examples
        cleaned_examples = []
        for example in examples:
            # Remove leading/trailing whitespace but preserve internal formatting
            cleaned_example = example.strip()
            if cleaned_example:  # Only add non-empty examples
                cleaned_examples.append(cleaned_example)

        return cleaned_examples
    except Exception as e:
        logger.warning(f"Error extracting code examples: {e}")
        return []


def extract_visibility(item: dict) -> str:
    """Extract visibility level from rustdoc item."""
    try:
        inner = item.get("inner", {})

        # Check visibility field directly
        visibility = item.get("visibility")
        if visibility:
            if visibility == "public":
                return "public"
            elif visibility == "crate":
                return "crate"
            elif visibility == "private" or visibility == "restricted":
                return "private"

        # Check inner visibility
        if isinstance(inner, dict):
            # Check for pub keyword in various inner types
            for key in ["function", "struct", "trait", "enum", "mod", "module"]:
                if key in inner:
                    inner_item = inner[key]
                    if isinstance(inner_item, dict):
                        vis = inner_item.get("visibility")
                        if vis == "public":
                            return "public"
                        elif vis == "crate":
                            return "crate"
                        elif vis in ["private", "restricted"]:
                            return "private"

        # Default to public if not specified (common for public API items)
        return "public"
    except Exception as e:
        logger.warning(f"Error extracting visibility: {e}")
        return "public"


def extract_deprecated(item: dict) -> bool:
    """Extract deprecated status from rustdoc item attributes."""
    try:
        # Check attrs field for deprecated attribute
        attrs = item.get("attrs", [])
        if isinstance(attrs, list):
            for attr in attrs:
                if isinstance(attr, str):
                    if "deprecated" in attr.lower():
                        return True
                elif isinstance(attr, dict):
                    # Handle structured attributes
                    if attr.get("name") == "deprecated":
                        return True
                    # Check for deprecated in attribute content
                    content = attr.get("content", "")
                    if "deprecated" in str(content).lower():
                        return True

        # Check if item has deprecated field directly
        if item.get("deprecated", False):
            return True

        # Check inner for deprecated status
        inner = item.get("inner", {})
        if isinstance(inner, dict):
            if inner.get("deprecated", False):
                return True

        return False
    except Exception as e:
        logger.warning(f"Error extracting deprecated status: {e}")
        return False


async def parse_rustdoc_items_streaming(json_content: str):
    """Parse rustdoc JSON using ijson for memory-efficient streaming.

    Extracts items (functions, structs, traits, modules) with their documentation.
    Yields items progressively to avoid memory accumulation.

    Yields:
        dict: Item with item_id, item_path, header, docstring, and metadata.
    """

    # For rustdoc JSON format, we need to parse two main sections:
    # 1. "paths" - maps IDs to paths (must be collected first)
    # 2. "index" - contains the actual items (can be streamed)

    # First pass: collect paths (required for reference resolution)
    id_to_path = {}
    with MemoryMonitor("parse_rustdoc_paths"):
        try:
            parser = ijson.kvitems(io.BytesIO(json_content.encode()), "paths")
            for item_id, path_info in parser:
                if isinstance(path_info, dict) and "path" in path_info:
                    id_to_path[item_id] = "::".join(path_info["path"])
        except Exception as e:
            logger.warning(f"Error parsing paths section: {e}")

    # Second pass: stream items from index
    item_count = 0
    chunk_count = 0

    with MemoryMonitor("parse_rustdoc_index"):
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

                    # Extract additional metadata using helper functions
                    item_type = normalize_item_type(kind)
                    signature = extract_signature(item_info)
                    parent_id = resolve_parent_id(item_info, id_to_path)
                    examples = extract_code_examples(docs)
                    visibility = extract_visibility(item_info)
                    deprecated = extract_deprecated(item_info)

                    yield {
                        "item_id": item_id,
                        "item_path": full_path,
                        "header": header,
                        "docstring": docs,
                        "kind": kind_lower,
                        "item_type": item_type,
                        "signature": signature,
                        "parent_id": parent_id,
                        "examples": examples,
                        "visibility": visibility,
                        "deprecated": deprecated,
                    }

                    item_count += 1
                    chunk_count += 1

                    # Trigger GC periodically to control memory
                    if chunk_count >= PARSE_CHUNK_SIZE:
                        trigger_gc_if_needed()
                        chunk_count = 0
                        logger.debug(f"Parsed {item_count} items so far...")

        except Exception as e:
            logger.warning(f"Error parsing index section: {e}")

    logger.info(f"Streamed {item_count} items from rustdoc JSON")


async def parse_rustdoc_items(json_content: str) -> list[dict[str, Any]]:
    """Parse rustdoc JSON and return a list of items (backwards compatible)."""
    items = []
    async for item in parse_rustdoc_items_streaming(json_content):
        items.append(item)
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

    for chunk in chunks:
        chunk_buffer.append(chunk)

        # Calculate adaptive batch size based on memory
        batch_size = get_adaptive_batch_size(
            base_batch_size=EMBEDDING_BATCH_SIZE,
            min_size=16,
            max_size=128,  # Cap at reasonable size for embeddings
        )

        # Process batch when buffer reaches adaptive size
        if len(chunk_buffer) >= batch_size:
            texts = [c["content"] for c in chunk_buffer]
            batch_embeddings = list(model.embed(texts))

            # Yield chunk-embedding pairs
            for item, embedding in zip(chunk_buffer, batch_embeddings, strict=False):
                yield item, embedding
                processed_count += 1

            # Clear buffer and trigger GC if needed
            chunk_buffer = []
            if processed_count % 100 == 0:
                trigger_gc_if_needed()
                logger.debug(f"Generated {processed_count} embeddings...")

    # Process remaining chunks in buffer
    if chunk_buffer:
        texts = [c["content"] for c in chunk_buffer]
        batch_embeddings = list(model.embed(texts))

        for item, embedding in zip(chunk_buffer, batch_embeddings, strict=False):
            yield item, embedding
            processed_count += 1

    logger.info(f"Generated {processed_count} embeddings total")


async def generate_embeddings(chunks: list[dict[str, str]]) -> list[list[float]]:
    """Generate embeddings for text chunks (backwards compatible)."""
    embeddings = []
    for _chunk, embedding in generate_embeddings_streaming(chunks):
        embeddings.append(embedding)
    return embeddings


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
    """Store chunks and their embeddings (backwards compatible)."""
    # Convert to streaming format
    chunk_embedding_pairs = zip(chunks, embeddings, strict=False)
    await store_embeddings_streaming(db_path, chunk_embedding_pairs)


async def _store_batch(
    db: aiosqlite.Connection,
    chunks: list[dict],
    embeddings: list[bytes],
    batch_num: int,
    items_so_far: int,
) -> None:
    """Store a single batch of chunks and embeddings."""
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
                chunk.get("item_type"),
                chunk.get("signature"),
                chunk.get("parent_id"),
                json.dumps(chunk.get("examples", []))
                if chunk.get("examples")
                else None,
                chunk.get("visibility", "public"),
                chunk.get("deprecated", False),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=False)
        ]

        # Batch insert into embeddings table
        await db.executemany(
            """
            INSERT INTO embeddings (item_path, header, content, embedding, item_type, signature, parent_id, examples, visibility, deprecated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            embeddings_data,
        )

        # Get the rowids for the inserted records
        cursor = await db.execute("SELECT last_insert_rowid()")
        last_rowid = (await cursor.fetchone())[0]
        first_rowid = last_rowid - len(chunks) + 1

        # Prepare batch data for vec_embeddings table
        vec_data = [
            (rowid, embedding)
            for rowid, embedding in zip(
                range(first_rowid, last_rowid + 1),
                embeddings,
                strict=False,
            )
        ]

        # Batch insert into vec_embeddings table
        await db.executemany(
            "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
            vec_data,
        )

        # Commit this batch
        await db.commit()

        # Log progress
        total_processed = items_so_far + len(chunks)
        logger.info(
            f"Batch {batch_num + 1}: Processed {len(chunks)} items "
            f"(total: {total_processed})"
        )

    except Exception as e:
        # Rollback on error
        await db.execute("ROLLBACK")
        logger.error(f"Error in batch {batch_num}: {e}")
        raise


async def ingest_crate(crate_name: str, version: str | None = None) -> Path:
    """Ingest a crate's documentation and return the database path."""
    async with aiohttp.ClientSession() as session:
        # Check if this is a standard library crate
        if is_stdlib_crate(crate_name):
            # Handle stdlib crate specially
            version = await resolve_stdlib_version(session, version)
            crate_info = {
                "name": crate_name,
                "description": f"Rust standard library: {crate_name}",
                "repository": "https://github.com/rust-lang/rust",
                "documentation": f"https://doc.rust-lang.org/{crate_name}",
                "max_stable_version": version,
            }
        else:
            # Fetch crate info first for basic metadata (existing flow)
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
                # Resolve version and get rustdoc URL
                logger.info(f"Resolving rustdoc URL for {crate_name}@{version}")

                if is_stdlib_crate(crate_name):
                    # For stdlib, we already have the version resolved
                    resolved_version = version
                    rustdoc_url = get_stdlib_url(crate_name, version)
                else:
                    # Use existing docs.rs resolution for third-party crates
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

                # Parse rustdoc items (now returns an async generator)
                logger.info("Parsing rustdoc items in streaming mode")

                # Collect items from streaming parser
                items = []
                async for item in parse_rustdoc_items_streaming(json_content):
                    items.append(item)

                if not items:
                    logger.warning("No items found in rustdoc JSON")
                    raise Exception("No items found in rustdoc JSON")

                logger.info(f"Collected {len(items)} rustdoc items")

                # Transform items to chunks
                chunks = []
                for item in items:
                    # Combine header and docstring for embedding
                    content = (
                        f"{item['header']}\n\n{item['docstring']}"
                        if item["docstring"]
                        else item["header"]
                    )

                    chunk = {
                        "item_path": item["item_path"],
                        "header": item["header"],
                        "content": content,
                        "item_type": item.get("item_type"),
                        "signature": item.get("signature"),
                        "parent_id": item.get("parent_id"),
                        "examples": item.get("examples", []),
                        "visibility": item.get("visibility", "public"),
                        "deprecated": item.get("deprecated", False),
                    }
                    chunks.append(chunk)

                # Generate embeddings and store in streaming fashion
                logger.info(
                    f"Generating embeddings for {len(chunks)} items in streaming mode"
                )
                chunk_embedding_pairs = generate_embeddings_streaming(chunks)

                # Store embeddings in streaming fashion
                await store_embeddings_streaming(db_path, chunk_embedding_pairs)

                logger.info("Successfully completed streaming ingestion")

            except Exception as e:
                if is_stdlib_crate(crate_name):
                    logger.error(
                        f"Failed to download {crate_name} rustdoc JSON: {e}\n"
                        f"Standard library documentation may not be available on docs.rs.\n"
                        f"To generate it locally, try:\n"
                        f"  rustup component add --toolchain nightly rust-docs-json\n"
                        f"  Then use the generated JSON files from your Rust installation."
                    )
                else:
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
