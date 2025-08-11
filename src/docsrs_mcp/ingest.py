"""Ingestion pipeline for Rust crate documentation."""

import asyncio
import gc
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import aiohttp
import aiosqlite
import ijson
import sqlite_vec
import zstandard
from fastembed import TextEmbedding
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
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
    PRIORITY_CACHE_EVICTION_ENABLED,
    RUST_VERSION_MANIFEST_URL,
    STDLIB_CRATES,
)
from .database import (
    execute_with_retry,
    get_db_path,
    init_database,
    store_crate_metadata,
    store_modules,
    store_reexports,
)
from .memory_utils import (
    MemoryMonitor,
    get_adaptive_batch_size,
    trigger_gc_if_needed,
)

logger = logging.getLogger(__name__)


# Global embedding model instance
_embedding_model: TextEmbedding | None = None

# Global warmup status for health endpoint
_embeddings_warmed: bool = False

# Global per-crate lock registry to prevent duplicate ingestion
_crate_locks: dict[str, asyncio.Lock] = {}


def get_embedding_model() -> TextEmbedding:
    """Get or create the embedding model instance with optimized ONNX settings."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        # Note: FastEmbed doesn't directly expose ONNX session options
        # but we can set environment variables that ONNX Runtime respects
        import os

        # Disable CPU memory arena for smaller models to reduce memory usage
        os.environ["ORT_DISABLE_CPU_ARENA_ALLOCATOR"] = "1"
        # Disable memory pattern optimization to reduce memory overhead
        os.environ["ORT_DISABLE_MEMORY_PATTERN"] = "1"

        _embedding_model = TextEmbedding(model_name=MODEL_NAME)
    return _embedding_model


def cleanup_embedding_model() -> None:
    """Clean up the embedding model to free memory."""
    global _embedding_model
    if _embedding_model is not None:
        logger.debug("Cleaning up embedding model to free memory")
        try:
            # Attempt to clean up the model
            del _embedding_model
            _embedding_model = None

            # Force garbage collection to reclaim memory
            import gc

            gc.collect()

            # Trigger memory cleanup if needed
            from .memory_utils import trigger_gc_if_needed

            trigger_gc_if_needed()
        except Exception as e:
            logger.warning(f"Error during embedding model cleanup: {e}")


async def warmup_embedding_model() -> None:
    """Warm up the embedding model to eliminate cold-start latency."""
    import asyncio

    from . import config

    if not config.EMBEDDINGS_WARMUP_ENABLED:
        return

    try:
        # Create background task to not block startup
        warmup_task = asyncio.create_task(_perform_warmup())
        # Fire-and-forget pattern from popular_crates.py
        logger.info("Starting embedding model warmup in background")
    except Exception as e:
        logger.warning(f"Failed to start embedding warmup: {e}")
        # Non-critical - continue without warmup


async def _perform_warmup() -> None:
    """Perform actual warmup in background."""
    global _embeddings_warmed
    try:
        # Trigger model loading via existing function
        model = get_embedding_model()

        # Perform 3-5 representative embeddings (best practice)
        warmup_texts = [
            "async runtime spawn tasks",  # Short text
            "The Rust programming language provides memory safety without garbage collection through its ownership system and borrow checker",  # Medium text
            " ".join(["token"] * 100),  # Long text for edge case
        ]

        for text in warmup_texts:
            await asyncio.to_thread(model.embed, [text])

        # Set warmup status to true on success
        _embeddings_warmed = True
        logger.info("Embedding model warmup completed successfully")
    except Exception as e:
        logger.warning(f"Embedding warmup failed: {e}")
        # Non-critical failure - service continues


def get_warmup_status() -> bool:
    """Get the current warmup status for health endpoint."""
    return _embeddings_warmed


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

    Note: Standard library rustdoc JSON is generally NOT available on docs.rs.
    This function returns a URL pattern that will be tried, but is expected
    to fail for most stdlib crates. The ingestion will fall back to source
    extraction when this fails.

    To use actual stdlib rustdoc JSON, users need to generate it locally:
      rustup component add --toolchain nightly rust-docs-json
    """
    # Standard library crates are not available as rustdoc JSON on docs.rs
    # We still return a URL to try, but expect it to fail and trigger fallback
    json_name = crate_name.replace("-", "_")

    # Try the standard docs.rs pattern, but this will likely 404
    # The download_rustdoc function will handle the failure gracefully
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
    # Use the new rustdoc JSON API pattern introduced in May 2025
    # Direct JSON URL without needing to resolve redirects first
    if version == "latest":
        # For latest, use the direct JSON endpoint
        rustdoc_url = f"https://docs.rs/crate/{crate_name}/latest/json"

        # Make a HEAD request to get the actual version from redirect
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with session.head(
            rustdoc_url, timeout=timeout, allow_redirects=True
        ) as response:
            if response.status != 200:
                raise Exception(
                    f"Failed to resolve version for {crate_name}: HTTP {response.status}"
                )

            # Extract version from redirected URL
            # Format: https://static.docs.rs/{crate_name}/{version}/json
            final_url = str(response.url)
            parts = final_url.strip("/").split("/")

            # Try to extract version from the URL
            if "static.docs.rs" in final_url and len(parts) >= 5:
                resolved_version = parts[4]
            else:
                # Fallback: use "latest" if we can't parse
                resolved_version = "latest"

            return resolved_version, rustdoc_url
    else:
        # For specific versions, use the direct pattern
        rustdoc_url = f"https://docs.rs/crate/{crate_name}/{version}/json"
        return version, rustdoc_url


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
    if rustdoc_url.endswith((".json.zst", ".json.gz", ".gz")):
        urls_to_try.append(rustdoc_url)
    # For the new JSON API pattern (/crate/{name}/{version}/json)
    # We can append .gz for gzip compression
    elif rustdoc_url.endswith("/json"):
        # Try zstd (default), then gzip
        urls_to_try.extend([rustdoc_url, f"{rustdoc_url}.gz"])
    else:
        # Fallback for old-style URLs
        base_url = (
            rustdoc_url.rstrip(".json")
            if rustdoc_url.endswith(".json")
            else rustdoc_url
        )
        urls_to_try.extend(
            [f"{base_url}.json.zst", f"{base_url}.json.gz", f"{base_url}.json"]
        )

    last_error = None
    all_404 = True  # Track if all attempts were 404s
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
                    all_404 = False  # This wasn't a 404
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
            # Track if this was NOT a 404
            if "404" not in str(e):
                all_404 = False
                logger.warning(f"Failed to download from {url}: {e}")

    # All formats failed
    # Create a specific exception type to distinguish version unavailability
    class RustdocVersionNotFoundError(Exception):
        """Raised when a specific version doesn't have rustdoc JSON available."""

    if all_404:
        # All attempts were 404s - this version doesn't have rustdoc JSON
        raise RustdocVersionNotFoundError(
            f"Rustdoc JSON not found for {crate_name}@{version} - version may not have rustdoc JSON available"
        )
    elif last_error:
        # Some other error occurred (network, server error, etc.)
        raise last_error
    else:
        # Shouldn't reach here, but handle it
        raise Exception(f"Failed to download rustdoc JSON for {crate_name}@{version}")


async def decompress_content(content: bytes, url: str) -> str:
    """Decompress content based on URL extension with size limits.

    Returns the decompressed JSON string.
    """
    # Check if content is zstd compressed (default for /json endpoint)
    # zstd magic bytes: 0x28, 0xb5, 0x2f, 0xfd
    is_zstd = content[:4] == b"\x28\xb5\x2f\xfd" if len(content) >= 4 else False

    if url.endswith(".json.zst") or (url.endswith("/json") and is_zstd):
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

    elif url.endswith(".json.gz") or url.endswith("/json.gz"):
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
        "impl": "trait_impl",
    }

    # Find matching type
    for key, value in type_map.items():
        if key in kind_str:
            return value
    return kind_str


def format_signature(decl: dict, generics: dict | None = None) -> str:
    """Format a function/method declaration into a readable signature with generics."""
    try:
        # Extract generics if present
        generic_str = ""
        if generics and isinstance(generics, dict):
            params = generics.get("params", [])
            if params:
                generic_parts = []
                for param in params:
                    if isinstance(param, dict):
                        name = param.get("name", "")
                        kind = param.get("kind", {})

                        # Handle different generic parameter kinds
                        if isinstance(kind, dict):
                            if "lifetime" in kind:
                                generic_parts.append(f"'{name}")
                            elif "const" in kind:
                                const_type = (
                                    kind["const"].get("type")
                                    if isinstance(kind["const"], dict)
                                    else None
                                )
                                if const_type:
                                    type_name = extract_type_name(const_type)
                                    generic_parts.append(f"const {name}: {type_name}")
                                else:
                                    generic_parts.append(f"const {name}")
                            elif "type" in kind:
                                # Check for bounds on the type parameter
                                type_info = kind["type"]
                                if (
                                    isinstance(type_info, dict)
                                    and "bounds" in type_info
                                ):
                                    bounds = []
                                    for bound in type_info["bounds"]:
                                        if isinstance(bound, dict):
                                            bound_str = extract_type_name(
                                                bound.get("trait")
                                            )
                                            if bound_str:
                                                bounds.append(bound_str)
                                    if bounds:
                                        generic_parts.append(
                                            f"{name}: {' + '.join(bounds)}"
                                        )
                                    else:
                                        generic_parts.append(name)
                                else:
                                    generic_parts.append(name)
                            else:
                                generic_parts.append(name)
                        else:
                            generic_parts.append(name)

                if generic_parts:
                    generic_str = f"<{', '.join(generic_parts)}>"

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

        signature = f"{generic_str}({', '.join(params)})"

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
    """Extract function/method signature with generics."""
    try:
        inner = item.get("inner", {})

        # Check for function
        if isinstance(inner, dict) and "function" in inner:
            func_data = inner["function"]
            decl = func_data.get("decl", {})
            generics = func_data.get("generics")
            if decl:
                return format_signature(decl, generics)

        # Check for method
        if isinstance(inner, dict) and "method" in inner:
            method_data = inner["method"]
            decl = method_data.get("decl", {})
            generics = method_data.get("generics")
            if decl:
                return format_signature(decl, generics)

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


def build_module_hierarchy(paths: dict) -> dict:
    """Build complete module hierarchy from paths dictionary.

    Args:
        paths: Dictionary from rustdoc JSON paths section (id -> path_info dict)

    Returns:
        dict: Module ID -> module info with parent relationships
    """
    modules = {}
    total_entries = 0
    module_count = 0

    try:
        for id_str, path_info in paths.items():
            total_entries += 1
            if not isinstance(path_info, dict):
                continue

            # Check if it's a module
            kind = path_info.get("kind")
            if isinstance(kind, str) and kind.lower() in ["module", "mod"]:
                module_count += 1
                path_parts = path_info.get("path", [])

                # Module name is the last part of the path
                name = path_parts[-1] if path_parts else ""

                # The path already includes the module name
                full_path_parts = path_parts

                full_path = "::".join(full_path_parts)

                # Determine parent
                parent_id = None
                depth = len(full_path_parts)

                if depth > 1:
                    # Find parent module by path
                    parent_path_parts = full_path_parts[:-1]
                    parent_path = "::".join(parent_path_parts)

                    # Search for parent module ID
                    for pid, pinfo in paths.items():
                        if isinstance(pinfo, dict):
                            pkind = pinfo.get("kind", "")
                            if isinstance(pkind, str) and pkind.lower() in [
                                "module",
                                "mod",
                            ]:
                                pname = pinfo.get("name", "")
                                ppath = pinfo.get("path", [])
                                if pname and pname not in ppath:
                                    p_full = "::".join(ppath + [pname])
                                else:
                                    p_full = "::".join(ppath)
                                if p_full == parent_path:
                                    parent_id = pid
                                    break

                modules[id_str] = {
                    "name": name,
                    "path": full_path,
                    "parent_id": parent_id,
                    "depth": depth - 1,  # 0-indexed depth (crate root = 0)
                    "item_count": 0,  # Will be updated during index pass
                }

    except Exception as e:
        logger.warning(f"Error building module hierarchy: {e}")

    logger.info(
        f"Processed {total_entries} path entries, found {module_count} modules, built {len(modules)} module records"
    )
    return modules


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


def extract_type_name(type_info: dict | str | None) -> str:
    """Extract readable type name from rustdoc Type object.

    Args:
        type_info: Type info from rustdoc JSON (can be dict, string, or None)

    Returns:
        str: Extracted type name or "Unknown" if unable to extract
    """
    if not type_info:
        return "Unknown"

    if isinstance(type_info, str):
        return type_info

    if isinstance(type_info, dict):
        # Handle resolved_path type
        if "resolved_path" in type_info:
            resolved = type_info["resolved_path"]
            if isinstance(resolved, dict):
                name = resolved.get("name", "")
                if name:
                    return name

        # Handle path type
        if "path" in type_info:
            path = type_info["path"]
            if isinstance(path, dict):
                name = path.get("name", "")
                if name:
                    return name

        # Handle generic type
        if "generic" in type_info:
            return type_info["generic"]

        # Handle primitive type
        if "primitive" in type_info:
            return type_info["primitive"]

        # Try to get name directly
        if "name" in type_info:
            return type_info["name"]

    return "Unknown"


def extract_code_examples(docstring: str) -> str | None:
    """Extract code blocks from documentation with language detection.

    Returns JSON string with structure:
    [{"code": str, "language": str, "detected": bool}]
    """
    if not docstring:
        return None

    try:
        # Match code blocks with optional language tags
        # Captures: 1) optional language tag, 2) code content
        pattern = r"```(\w*)\s*\n(.*?)```"
        matches = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

        if not matches:
            return None

        examples = []
        for lang_tag, code in matches:
            code = code.strip()
            if not code:  # Skip empty examples
                continue

            # Determine language
            detected = False
            if lang_tag:
                # Explicit language tag provided
                language = lang_tag.lower()
            else:
                # Try to detect language using pygments
                try:
                    lexer = guess_lexer(code)
                    # Use confidence threshold - pygments returns confidence score
                    if hasattr(lexer, "analyse_text"):
                        confidence = lexer.analyse_text(code)
                        if confidence and confidence > 0.3:  # 30% confidence threshold
                            language = (
                                lexer.aliases[0]
                                if lexer.aliases
                                else lexer.name.lower()
                            )
                            detected = True
                        else:
                            language = "rust"  # Default for Rust crates
                            detected = True
                    else:
                        language = (
                            lexer.aliases[0] if lexer.aliases else lexer.name.lower()
                        )
                        detected = True
                except (ClassNotFound, Exception):
                    # Default to rust for Rust documentation
                    language = "rust"
                    detected = True

            examples.append({"code": code, "language": language, "detected": detected})

        return json.dumps(examples) if examples else None

    except Exception as e:
        logger.warning(f"Error extracting code examples: {e}")
        return None


def normalize_code(code: str) -> str:
    """Normalize code for consistent hashing and deduplication.

    Removes comments and normalizes whitespace to detect duplicate examples.
    """
    lines = code.strip().split("\n")
    normalized_lines = []

    for line in lines:
        # Skip comments and empty lines for hashing
        stripped = line.strip()
        # Skip various comment types
        if stripped and not any(
            [
                stripped.startswith("#"),  # Python comments
                stripped.startswith("//"),  # Rust/C++ comments
                stripped.startswith("/*"),  # Block comments
                stripped.startswith("*"),  # Continuation of block comments
                stripped.startswith("--"),  # SQL/Lua comments
            ]
        ):
            # Normalize whitespace but preserve structure
            normalized_lines.append(" ".join(stripped.split()))

    return "\n".join(normalized_lines)


def calculate_example_hash(example_text: str, language: str) -> str:
    """Generate hash for deduplication of code examples.

    Includes language in hash to avoid cross-language collisions.
    """
    # Normalize the code
    normalized = normalize_code(example_text)
    # Include language in hash to avoid cross-language collisions
    content = f"{language}:{normalized}"
    # Return first 16 chars of SHA256 hash for reasonable uniqueness
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def batch_examples(
    examples: list[dict], batch_size: int
) -> AsyncGenerator[list[dict], None]:
    """Yield batches of examples for processing.

    Memory-efficient batching for embedding generation.
    """
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        yield batch
        # Allow other async operations between batches
        await asyncio.sleep(0)


def format_example_for_embedding(example: dict) -> str:
    """Format a code example for embedding generation.

    Combines code with context for better semantic search.
    """
    # Include language as context
    language = example.get("language", "unknown")
    code = example.get("example_text", example.get("code", ""))
    context = example.get("context", "")

    # Combine elements for embedding
    parts = []
    if language and language != "unknown":
        parts.append(f"Language: {language}")
    if context:
        parts.append(f"Context: {context[:200]}")  # Limit context length
    parts.append(code)

    return "\n\n".join(parts)


async def generate_example_embeddings(
    db_path: Path, crate_name: str, version: str
) -> None:
    """Generate embeddings for code examples with deduplication.

    Extracts examples from existing embeddings table and generates
    dedicated embeddings for semantic search.
    """
    logger.info(f"Generating example embeddings for {crate_name}@{version}")

    async with aiosqlite.connect(db_path) as db:
        # Enable sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Extract examples from embeddings table
        cursor = await db.execute("""
            SELECT item_id, item_path, examples, content
            FROM embeddings
            WHERE examples IS NOT NULL AND examples != ''
        """)

        all_examples = []
        async for row in cursor:
            item_id, item_path, examples_json, content = row

            try:
                examples_data = json.loads(examples_json)

                # Handle string input - wrap in list to prevent character iteration
                if isinstance(examples_data, str):
                    examples_data = [examples_data]
                elif not examples_data:
                    logger.warning(
                        f"Empty examples_data for {crate_name}/{version} at {item_path}"
                    )
                    continue

                # Handle both old list format and new dict format
                if isinstance(examples_data, list) and all(
                    isinstance(e, str) for e in examples_data
                ):
                    examples_data = [
                        {"code": e, "language": "rust", "detected": False}
                        for e in examples_data
                    ]

                for example in examples_data:
                    if isinstance(example, str):
                        example = {
                            "code": example,
                            "language": "rust",
                            "detected": False,
                        }

                    code = example.get("code", "")
                    if not code:
                        continue

                    # Calculate hash for deduplication
                    language = example.get("language", "rust")
                    example_hash = calculate_example_hash(code, language)

                    all_examples.append(
                        {
                            "item_id": item_id,
                            "item_path": item_path,
                            "crate_name": crate_name,
                            "version": version,
                            "example_hash": example_hash,
                            "example_text": code,
                            "language": language,
                            "context": content[:500]
                            if content
                            else None,  # Store context
                        }
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse examples for {item_path}")
                continue

        if not all_examples:
            logger.info(f"No examples found for {crate_name}@{version}")
            return

        logger.info(f"Found {len(all_examples)} total examples")

        # Check for existing examples to avoid duplicates
        placeholders = ",".join(["?"] * len(all_examples))
        hashes = [ex["example_hash"] for ex in all_examples]

        cursor = await db.execute(
            f"""
            SELECT example_hash 
            FROM example_embeddings 
            WHERE crate_name = ? AND version = ? AND example_hash IN ({placeholders})
        """,
            [crate_name, version] + hashes,
        )

        existing_hashes = {row[0] for row in await cursor.fetchall()}

        # Filter out already processed examples
        new_examples = [
            ex for ex in all_examples if ex["example_hash"] not in existing_hashes
        ]

        if not new_examples:
            logger.info(f"All examples already embedded for {crate_name}@{version}")
            return

        logger.info(
            f"Processing {len(new_examples)} new examples (skipped {len(all_examples) - len(new_examples)} duplicates)"
        )

        # Process in batches
        embedding_model = get_embedding_model()
        batch_size = 16  # Conservative for CPU

        # Get max text length from config for memory leak mitigation
        import os

        max_text_length = int(os.getenv("FASTEMBED_MAX_TEXT_LENGTH", "100"))

        processed = 0
        async for batch in batch_examples(new_examples, batch_size):
            # Prepare texts for embedding with truncation to prevent memory leak
            texts = [format_example_for_embedding(ex)[:max_text_length] for ex in batch]

            # Generate embeddings using generator pattern
            embeddings = list(embedding_model.embed(texts))

            # Begin transaction for this batch
            await db.execute("BEGIN TRANSACTION")

            try:
                # Insert into example_embeddings table
                for example, embedding in zip(batch, embeddings, strict=False):
                    # Insert into main table
                    cursor = await db.execute(
                        """
                        INSERT INTO example_embeddings (
                            item_id, item_path, crate_name, version,
                            example_hash, example_text, language, context, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            example["item_id"],
                            example["item_path"],
                            example["crate_name"],
                            example["version"],
                            example["example_hash"],
                            example["example_text"],
                            example["language"],
                            example["context"],
                            bytes(sqlite_vec.serialize_float32(embedding)),
                        ),
                    )

                    rowid = cursor.lastrowid

                    # Insert into vector table
                    await db.execute(
                        """
                        INSERT INTO vec_example_embeddings(rowid, example_embedding)
                        VALUES (?, ?)
                    """,
                        (rowid, bytes(sqlite_vec.serialize_float32(embedding))),
                    )

                await db.commit()
                processed += len(batch)
                logger.debug(f"Processed {processed}/{len(new_examples)} examples")

            except Exception as e:
                await db.rollback()
                logger.error(f"Error processing batch: {e}")
                raise

            # Explicit memory cleanup between batches
            del embeddings
            gc.collect()

        logger.info(f"Successfully generated embeddings for {processed} examples")


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
            elif visibility in {"private", "restricted"}:
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


def extract_generic_params(item: dict) -> str | None:
    """Extract generic parameters from rustdoc item."""
    try:
        inner = item.get("inner", {})
        if not isinstance(inner, dict):
            return None

        # Look for generics in various item types
        generics = None

        # Direct generics field
        if "generics" in inner:
            generics = inner["generics"]
        # Function generics
        elif "function" in inner and isinstance(inner["function"], dict):
            generics = inner["function"].get("generics")
        # Struct generics
        elif "struct" in inner and isinstance(inner["struct"], dict):
            generics = inner["struct"].get("generics")
        # Trait generics
        elif "trait" in inner and isinstance(inner["trait"], dict):
            generics = inner["trait"].get("generics")
        # Enum generics
        elif "enum" in inner and isinstance(inner["enum"], dict):
            generics = inner["enum"].get("generics")
        # Type alias generics
        elif "typedef" in inner and isinstance(inner["typedef"], dict):
            generics = inner["typedef"].get("generics")
        # Impl block generics
        elif "impl" in inner and isinstance(inner["impl"], dict):
            generics = inner["impl"].get("generics")

        if not generics or not isinstance(generics, dict):
            return None

        # Extract generic parameters
        params = generics.get("params", [])
        if not params:
            return None

        result = []
        for param in params:
            if not isinstance(param, dict):
                continue

            param_info = {
                "name": param.get("name", ""),
                "kind": "type",  # default
            }

            # Determine parameter kind
            kind = param.get("kind")
            if isinstance(kind, dict):
                if "lifetime" in kind:
                    param_info["kind"] = "lifetime"
                elif "const" in kind:
                    param_info["kind"] = "const"
                    # Add const type if available
                    const_type = (
                        kind["const"].get("type")
                        if isinstance(kind["const"], dict)
                        else None
                    )
                    if const_type:
                        param_info["const_type"] = extract_type_name(const_type)
                elif "type" in kind:
                    param_info["kind"] = "type"
                    # Extract bounds if present
                    type_info = kind["type"]
                    if isinstance(type_info, dict) and "bounds" in type_info:
                        bounds = []
                        for bound in type_info["bounds"]:
                            if isinstance(bound, dict):
                                bound_str = extract_type_name(bound.get("trait"))
                                if bound_str:
                                    bounds.append(bound_str)
                        if bounds:
                            param_info["bounds"] = bounds
            elif isinstance(kind, str):
                param_info["kind"] = kind.lower()

            result.append(param_info)

        return json.dumps(result) if result else None

    except Exception as e:
        logger.warning(f"Failed to extract generic params: {e}")
        return None


def extract_trait_bounds(item: dict) -> str | None:
    """Extract trait bounds and where clauses from rustdoc item."""
    try:
        inner = item.get("inner", {})
        if not isinstance(inner, dict):
            return None

        # Look for generics which contain where predicates
        generics = None

        # Direct generics field
        if "generics" in inner:
            generics = inner["generics"]
        # Function generics
        elif "function" in inner and isinstance(inner["function"], dict):
            generics = inner["function"].get("generics")
        # Struct generics
        elif "struct" in inner and isinstance(inner["struct"], dict):
            generics = inner["struct"].get("generics")
        # Trait generics
        elif "trait" in inner and isinstance(inner["trait"], dict):
            generics = inner["trait"].get("generics")
        # Enum generics
        elif "enum" in inner and isinstance(inner["enum"], dict):
            generics = inner["enum"].get("generics")
        # Type alias generics
        elif "typedef" in inner and isinstance(inner["typedef"], dict):
            generics = inner["typedef"].get("generics")
        # Impl block generics and trait bounds
        elif "impl" in inner and isinstance(inner["impl"], dict):
            impl_data = inner["impl"]
            generics = impl_data.get("generics")

            # Also check for trait being implemented
            trait_info = impl_data.get("trait")
            if trait_info:
                trait_name = extract_type_name(trait_info)
                if trait_name:
                    # Include impl trait as a special bound
                    result = [{"type": "impl", "trait": trait_name}]
                    # Continue to check for where clauses
                    if generics and isinstance(generics, dict):
                        where_predicates = generics.get("where_predicates", [])
                        if where_predicates:
                            for pred in where_predicates:
                                if isinstance(pred, dict):
                                    result.append(extract_where_predicate(pred))
                    return json.dumps(result)

        if not generics or not isinstance(generics, dict):
            return None

        # Extract where predicates
        where_predicates = generics.get("where_predicates", [])
        if not where_predicates:
            return None

        result = []
        for predicate in where_predicates:
            if not isinstance(predicate, dict):
                continue

            pred_info = extract_where_predicate(predicate)
            if pred_info:
                result.append(pred_info)

        return json.dumps(result) if result else None

    except Exception as e:
        logger.warning(f"Failed to extract trait bounds: {e}")
        return None


def extract_where_predicate(predicate: dict) -> dict | None:
    """Extract a single where predicate into a structured format."""
    try:
        pred_type = predicate.get("type", "")

        if pred_type == "bound_predicate":
            # T: Display + Debug style
            type_name = extract_type_name(predicate.get("type"))
            bounds = []
            for bound in predicate.get("bounds", []):
                if isinstance(bound, dict):
                    trait_name = extract_type_name(bound.get("trait"))
                    if trait_name:
                        bounds.append(trait_name)

            if type_name and bounds:
                return {"type": "bound", "target": type_name, "bounds": bounds}

        elif pred_type == "region_predicate":
            # 'a: 'b style lifetime bounds
            lifetime = predicate.get("lifetime")
            bounds = predicate.get("bounds", [])
            if lifetime and bounds:
                return {"type": "lifetime", "lifetime": lifetime, "bounds": bounds}

        elif pred_type == "eq_predicate":
            # T = ConcreteType style
            lhs = extract_type_name(predicate.get("lhs"))
            rhs = extract_type_name(predicate.get("rhs"))
            if lhs and rhs:
                return {"type": "equality", "left": lhs, "right": rhs}

        return None

    except Exception:
        return None


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
    paths_data = {}  # Full path info for module hierarchy
    modules = {}
    with MemoryMonitor("parse_rustdoc_paths"):
        try:
            parser = ijson.kvitems(io.BytesIO(json_content.encode()), "paths")
            for item_id, path_info in parser:
                if isinstance(path_info, dict) and "path" in path_info:
                    id_to_path[item_id] = "::".join(path_info["path"])
                    paths_data[item_id] = path_info  # Store full info
        except Exception as e:
            logger.warning(f"Error parsing paths section: {e}")

    # Build module hierarchy from collected paths
    logger.info(f"Collected {len(paths_data)} paths entries")
    modules = build_module_hierarchy(paths_data)
    logger.info(f"Built hierarchy with {len(modules)} modules")

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
                docs = item_info.get("docs", "")
                inner = item_info.get("inner", {})

                # Get the path from our mapping
                path = id_to_path.get(item_id, "")

                # Extract kind from inner field
                if isinstance(inner, dict) and len(inner) == 1:
                    # The kind is the single key in the inner dict
                    kind_lower = list(inner.keys())[0].lower()
                else:
                    # Fallback: check if there's a direct kind field (older format?)
                    kind = item_info.get("kind", "")
                    if isinstance(kind, str):
                        kind_lower = kind.lower()
                    elif isinstance(kind, dict) and len(kind) == 1:
                        kind_lower = list(kind.keys())[0].lower()
                    else:
                        continue

                # Check if it's an import/re-export item
                if "use" in kind_lower or "import" in kind_lower:
                    # This is a re-export - extract mapping information
                    if isinstance(inner, dict):
                        use_info = inner.get("Use") or inner.get("use") or inner
                        if use_info:
                            # Extract source path and check for glob
                            source = use_info.get("source")
                            is_glob = use_info.get("is_glob", False)

                            # For re-exports, the name might be in use_info instead of item_info
                            use_name = use_info.get("name") or name

                            # Build the re-export mapping
                            if source and use_name:
                                # The alias path is the current path + name
                                alias_path = f"{path}::{use_name}" if path else use_name
                                # The actual path is the source
                                actual_path = source

                                # Yield a special re-export item
                                yield {
                                    "_reexport": {
                                        "alias_path": alias_path,
                                        "actual_path": actual_path,
                                        "is_glob": is_glob,
                                    }
                                }

                                # Log for debugging
                                logger.debug(
                                    f"Found re-export: {alias_path} -> {actual_path} (glob: {is_glob})"
                                )
                    continue  # Don't process as regular item

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
                    "impl",
                ]

                if any(k in kind_lower for k in indexable_kinds):
                    # Create header based on item type
                    if "function" in kind_lower or "method" in kind_lower:
                        header = f"fn {name}"
                    elif "struct" in kind_lower:
                        header = f"struct {name}"
                    elif "trait" in kind_lower:
                        header = f"trait {name}"
                    elif "impl" in kind_lower:
                        # Handle impl blocks - extract trait and type names
                        if inner:
                            trait_info = inner.get("trait")
                            type_info = inner.get("for")

                            if trait_info and type_info:
                                trait_name = extract_type_name(trait_info)
                                type_name = extract_type_name(type_info)
                                header = f"impl {trait_name} for {type_name}"
                                # For impl blocks, use a composite name
                                name = f"{trait_name}_for_{type_name}"
                            elif type_info:
                                # Inherent impl (no trait)
                                type_name = extract_type_name(type_info)
                                header = f"impl {type_name}"
                                name = f"impl_{type_name}"
                            else:
                                # Fallback
                                header = f"impl {name}"
                        else:
                            header = f"impl {name}"
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

                    # Validate and potentially generate fallback path
                    from .validation import validate_item_path_with_fallback

                    validated_path, used_fallback = validate_item_path_with_fallback(
                        full_path, item_id, kind_lower
                    )

                    if used_fallback:
                        logger.debug(
                            f"Generated fallback path for item: "
                            f"original_path='{path}', original_name='{name}', "
                            f"fallback='{validated_path}', kind='{kind_lower}'"
                        )

                    full_path = validated_path

                    # Extract additional metadata using helper functions
                    item_type = normalize_item_type(kind_lower)
                    signature = extract_signature(inner if inner else item_info)
                    parent_id = resolve_parent_id(item_info, id_to_path)

                    # Update item count for parent module
                    if parent_id and parent_id in modules:
                        modules[parent_id]["item_count"] += 1
                    examples = extract_code_examples(docs)
                    visibility = extract_visibility(item_info)
                    deprecated = extract_deprecated(item_info)
                    generic_params = extract_generic_params(item_info)
                    trait_bounds = extract_trait_bounds(item_info)

                    # Extract cross-references from the links field
                    links = item_info.get("links", {})
                    if links and isinstance(links, dict):
                        for link_text, target_id in links.items():
                            # Resolve target path using id_to_path mapping
                            if target_id in id_to_path:
                                target_path = id_to_path[target_id]
                                # Yield cross-reference marker
                                yield {
                                    "_crossref": {
                                        "source_path": full_path,
                                        "link_text": link_text,
                                        "target_path": target_path,
                                        "target_item_id": target_id,
                                        "confidence": 1.0,
                                    }
                                }
                                logger.debug(
                                    f"Found cross-reference: {full_path} -> {target_path} via '{link_text}'"
                                )

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
                        "generic_params": generic_params,
                        "trait_bounds": trait_bounds,
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

    # Return modules as final yield with special marker
    yield {"_modules": modules}


async def parse_rustdoc_items(json_content: str) -> list[dict[str, Any]]:
    """Parse rustdoc JSON and return a list of items (backwards compatible)."""
    items = []
    async for item in parse_rustdoc_items_streaming(json_content):
        # Skip the special modules marker
        if "_modules" not in item:
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
    """Evict cache files with priority-aware logic if total size exceeds limit."""
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

                        # Extract crate name from path (cache/{crate}/{version}.db)
                        path_obj = Path(file_path)
                        crate_name = None
                        try:
                            # Get parent directory name as crate name
                            crate_name = path_obj.parent.name
                        except Exception:
                            # If extraction fails, treat as unknown crate
                            crate_name = None

                        cache_files.append(
                            {
                                "path": file_path,
                                "size": stat_info.st_size,
                                "mtime": stat_info.st_mtime,
                                "crate_name": crate_name,
                            }
                        )
                    except OSError:
                        pass

        # Priority-aware eviction logic
        if PRIORITY_CACHE_EVICTION_ENABLED:
            try:
                # Get popular crates data for priority scoring
                from .popular_crates import get_popular_manager

                manager = get_popular_manager()
                popular_crates = await manager.get_popular_crates_with_metadata()

                # Create lookup dictionary for O(1) access
                popular_dict = (
                    {c.name: c.downloads for c in popular_crates}
                    if popular_crates
                    else {}
                )

                # Calculate priority scores for each file
                for file_info in cache_files:
                    crate_name = file_info.get("crate_name")
                    downloads = popular_dict.get(crate_name, 0) if crate_name else 0

                    # Calculate priority score using log scale for downloads
                    # Popular crates get higher priority (lower eviction priority)
                    if downloads > 0:
                        # Use log scale to prevent extreme differences
                        priority = math.log10(downloads + 1)
                    else:
                        priority = 0

                    file_info["priority"] = priority
                    file_info["downloads"] = downloads

                # Hybrid sorting: primary by priority (ascending = evict low priority first),
                # secondary by mtime (oldest first within same priority tier)
                cache_files.sort(
                    key=lambda x: (
                        x.get("priority", 0),  # Lower priority first (evict first)
                        x["mtime"],  # Older files first within same priority
                    )
                )

                logger.debug("Using priority-aware cache eviction")

            except Exception as e:
                # Fallback to time-based eviction if priority scoring fails
                logger.warning(f"Priority eviction failed, using time-based: {e}")
                cache_files.sort(key=lambda x: x["mtime"])
        else:
            # Standard time-based eviction
            cache_files.sort(key=lambda x: x["mtime"])

        # Remove files until under limit
        removed_size = 0
        removed_count = 0
        for file_info in cache_files:
            if current_size - removed_size <= CACHE_MAX_SIZE_BYTES:
                break

            try:
                os.remove(file_info["path"])
                removed_size += file_info["size"]
                removed_count += 1

                # Enhanced logging with priority information
                if PRIORITY_CACHE_EVICTION_ENABLED and "priority" in file_info:
                    crate_info = file_info.get("crate_name", "unknown")
                    priority = file_info.get("priority", 0)
                    downloads = file_info.get("downloads", 0)
                    age_days = (time.time() - file_info["mtime"]) / 86400

                    logger.info(
                        f"Evicted: {crate_info} "
                        f"(priority: {priority:.2f}, downloads: {downloads}, "
                        f"age: {age_days:.1f} days, size: {file_info['size']} bytes)"
                    )
                else:
                    logger.info(
                        f"Evicted cache file: {file_info['path']} ({file_info['size']} bytes)"
                    )

            except OSError as e:
                logger.warning(f"Error removing cache file {file_info['path']}: {e}")

        logger.info(
            f"Evicted {removed_count} files totaling {removed_size} bytes from cache"
        )

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
    batch_count = 0  # Track number of batches for process recycling

    # Get max text length from config (will be added to config.py)
    import os

    max_text_length = int(os.getenv("FASTEMBED_MAX_TEXT_LENGTH", "100"))
    max_batches = int(os.getenv("FASTEMBED_MAX_BATCHES", "50"))

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
            # Truncate texts to prevent memory leak (GitHub issue #222)
            texts = [
                c["content"][:max_text_length]
                if len(c["content"]) > max_text_length
                else c["content"]
                for c in chunk_buffer
            ]
            batch_embeddings = list(model.embed(texts))

            # Yield chunk-embedding pairs
            for item, embedding in zip(chunk_buffer, batch_embeddings, strict=False):
                yield item, embedding
                processed_count += 1

            # Clear buffer and aggressive memory cleanup
            chunk_buffer = []
            batch_count += 1

            # Force garbage collection after each batch to prevent memory leak
            gc.collect()

            if processed_count % 100 == 0:
                trigger_gc_if_needed(force=True)  # Force GC periodically
                logger.debug(
                    f"Generated {processed_count} embeddings, {batch_count} batches..."
                )

            # Check if process recycling is needed for FastEmbed memory leak mitigation
            if batch_count >= max_batches:
                logger.warning(
                    f"Reached max batch count {max_batches} for FastEmbed. "
                    "Process recycling may be needed to prevent memory leak."
                )
                # Note: Actual process recycling would need to be handled at a higher level
                # For now, we log a warning and continue

    # Process remaining chunks in buffer
    if chunk_buffer:
        # Truncate texts for remaining batch too
        texts = [
            c["content"][:max_text_length]
            if len(c["content"]) > max_text_length
            else c["content"]
            for c in chunk_buffer
        ]
        batch_embeddings = list(model.embed(texts))

        for item, embedding in zip(chunk_buffer, batch_embeddings, strict=False):
            yield item, embedding
            processed_count += 1

        # Final cleanup
        gc.collect()

    logger.info(
        f"Generated {processed_count} embeddings total in {batch_count} batches"
    )


async def generate_embeddings(chunks: list[dict[str, str]]) -> list[list[float]]:
    """Generate embeddings for text chunks (backwards compatible)."""
    embeddings = []
    for _chunk, embedding in generate_embeddings_streaming(chunks):
        embeddings.append(embedding)
    return embeddings


async def cleanup_existing_embeddings(
    db_path: Path, crate_name: str, version: str
) -> None:
    """Clean up existing embeddings for a crate before re-ingestion.

    This prevents duplicates when re-ingesting a crate.
    """
    async with aiosqlite.connect(db_path) as db:
        # Delete existing embeddings for this crate
        # We identify crate embeddings by checking if item_path starts with crate name
        # or is the crate itself
        await db.execute(
            """
            DELETE FROM embeddings 
            WHERE item_path = ? OR item_path LIKE ? || '::%'
        """,
            (crate_name, crate_name),
        )

        deleted_count = db.total_changes
        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} existing embeddings for {crate_name}@{version}"
            )

        # Also clean up corresponding vec_embeddings entries
        await db.execute("""
            DELETE FROM vec_embeddings 
            WHERE rowid NOT IN (
                SELECT rowid FROM embeddings
            )
        """)

        await db.commit()


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
    """Store a single batch of chunks and embeddings with manual vec_embeddings sync.

    Uses explicit DELETE + INSERT pattern with manual synchronization to vec_embeddings.
    AUTOINCREMENT ensures no rowid reuse conflicts. Now with retry logic for resilience.
    """
    try:
        # Use execute_with_retry for transaction management
        await execute_with_retry(db, "BEGIN IMMEDIATE")

        # Defensive validation - filter out invalid item_paths and deduplicate
        valid_chunks = []
        valid_embeddings = []
        skipped_count = 0
        seen_paths = set()
        duplicate_count = 0

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            # Defensive validation - should never fail after parse-time validation
            if not chunk.get("item_path") or not chunk["item_path"].strip():
                logger.error(
                    f"Invalid item_path in batch: {chunk.get('item_path')!r}, "
                    f"header={chunk.get('header')}, skipping chunk"
                )
                skipped_count += 1
                continue

            # Deduplicate within batch - keep first occurrence
            item_path = chunk["item_path"]
            if item_path in seen_paths:
                duplicate_count += 1
                logger.debug(f"Skipping duplicate item_path in batch: {item_path}")
                continue

            seen_paths.add(item_path)
            valid_chunks.append(chunk)
            valid_embeddings.append(embedding)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} chunks with invalid item_paths")

        if duplicate_count > 0:
            logger.warning(f"Skipped {duplicate_count} duplicate item_paths in batch")

        if not valid_chunks:
            logger.warning("No valid chunks to store in this batch")
            await db.execute("ROLLBACK")
            return

        # Process in smaller sub-batches to avoid SQL parameter limit issues
        MAX_PARAMS = 500  # SQLite limit is 999, but be conservative
        item_paths = [chunk["item_path"] for chunk in valid_chunks]

        for i in range(0, len(item_paths), MAX_PARAMS):
            batch_paths = item_paths[i : i + MAX_PARAMS]
            batch_chunks = valid_chunks[i : i + MAX_PARAMS]
            batch_embeddings = valid_embeddings[i : i + MAX_PARAMS]

            placeholders = ",".join(["?"] * len(batch_paths))

            # Step 1: Get existing rowids for vec_embeddings cleanup
            cursor = await db.execute(
                f"""
                SELECT id, item_path 
                FROM embeddings 
                WHERE item_path IN ({placeholders})
                """,
                batch_paths,
            )
            existing_rows = await cursor.fetchall()

            # Step 2: Delete from vec_embeddings for existing entries
            if existing_rows:
                existing_ids = [row[0] for row in existing_rows]
                id_placeholders = ",".join(["?"] * len(existing_ids))
                await db.execute(
                    f"DELETE FROM vec_embeddings WHERE rowid IN ({id_placeholders})",
                    existing_ids,
                )

            # Step 3: DELETE existing embeddings entries
            await db.execute(
                f"DELETE FROM embeddings WHERE item_path IN ({placeholders})",
                batch_paths,
            )

            # Step 4: INSERT new entries (AUTOINCREMENT ensures new unique IDs)
            insert_data = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings, strict=False):
                insert_data.append(
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
                        chunk.get("generic_params"),  # Already JSON string or None
                        chunk.get("trait_bounds"),  # Already JSON string or None
                    )
                )

            await db.executemany(
                """
                INSERT INTO embeddings 
                (item_path, header, content, embedding, item_type, signature, parent_id, examples, visibility, deprecated, generic_params, trait_bounds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data,
            )

            # Step 5: Get the new rowids and insert into vec_embeddings
            cursor = await db.execute(
                f"""
                SELECT id, item_path 
                FROM embeddings 
                WHERE item_path IN ({placeholders})
                """,
                batch_paths,
            )
            new_rows = await cursor.fetchall()

            # Build vec_data with correct rowids
            vec_data = []
            path_to_embedding = {
                chunk["item_path"]: emb
                for chunk, emb in zip(batch_chunks, batch_embeddings, strict=False)
            }

            for row_id, item_path in new_rows:
                if item_path in path_to_embedding:
                    vec_data.append((row_id, path_to_embedding[item_path]))

            # Insert into vec_embeddings with explicit rowids
            if vec_data:
                await db.executemany(
                    "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    vec_data,
                )

        # Commit this batch
        await db.commit()

        # Log progress
        total_processed = items_so_far + len(valid_chunks)
        logger.info(
            f"Batch {batch_num + 1}: Processed {len(valid_chunks)} items "
            f"(total: {total_processed})"
        )

    except Exception as e:
        # Rollback on error
        await db.execute("ROLLBACK")
        logger.error(f"Error in batch {batch_num}: {e}")
        raise


async def create_stdlib_fallback_documentation(
    db_path: Path,
    crate_id: int,
    crate_name: str,
    version: str,
    description: str,
) -> None:
    """Create fallback documentation for standard library crates.

    Since stdlib rustdoc JSON is not available on docs.rs, we create
    basic documentation entries for common stdlib items to enable
    at least partial functionality.
    """
    logger.info(f"Creating fallback documentation for {crate_name}")

    # Define common stdlib items for each crate
    stdlib_items = {
        "std": [
            # Core collection types
            ("std::vec::Vec", "Vec<T> - A contiguous growable array type", "struct"),
            (
                "std::collections::HashMap",
                "HashMap<K, V> - A hash map implementation",
                "struct",
            ),
            (
                "std::collections::BTreeMap",
                "BTreeMap<K, V> - An ordered map based on a B-Tree",
                "struct",
            ),
            (
                "std::collections::HashSet",
                "HashSet<T> - A hash set implementation",
                "struct",
            ),
            (
                "std::collections::BTreeSet",
                "BTreeSet<T> - An ordered set based on a B-Tree",
                "struct",
            ),
            (
                "std::collections::VecDeque",
                "VecDeque<T> - A double-ended queue",
                "struct",
            ),
            (
                "std::collections::LinkedList",
                "LinkedList<T> - A doubly-linked list",
                "struct",
            ),
            (
                "std::collections::BinaryHeap",
                "BinaryHeap<T> - A priority queue",
                "struct",
            ),
            # Core types
            ("std::option::Option", "Option<T> - Type for optional values", "enum"),
            ("std::result::Result", "Result<T, E> - Type for error handling", "enum"),
            (
                "std::string::String",
                "String - A growable UTF-8 encoded string",
                "struct",
            ),
            (
                "std::boxed::Box",
                "Box<T> - A pointer type for heap allocation",
                "struct",
            ),
            # Synchronization primitives
            (
                "std::sync::Arc",
                "Arc<T> - Atomically reference-counted pointer",
                "struct",
            ),
            ("std::sync::Mutex", "Mutex<T> - Mutual exclusion primitive", "struct"),
            ("std::sync::RwLock", "RwLock<T> - Reader-writer lock", "struct"),
            ("std::sync::Condvar", "Condvar - Condition variable", "struct"),
            (
                "std::sync::Barrier",
                "Barrier - Thread synchronization barrier",
                "struct",
            ),
            ("std::sync::Once", "Once - One-time initialization", "struct"),
            (
                "std::sync::mpsc",
                "mpsc - Multi-producer, single-consumer channels",
                "module",
            ),
            ("std::sync::atomic", "atomic - Atomic types", "module"),
            # I/O types
            ("std::io::Read", "Read - The Read trait for reading bytes", "trait"),
            ("std::io::Write", "Write - The Write trait for writing bytes", "trait"),
            ("std::io::BufRead", "BufRead - Buffered reading trait", "trait"),
            ("std::io::BufReader", "BufReader<R> - Buffered reader", "struct"),
            ("std::io::BufWriter", "BufWriter<W> - Buffered writer", "struct"),
            ("std::io::Error", "Error - I/O error type", "struct"),
            ("std::io::ErrorKind", "ErrorKind - I/O error kinds", "enum"),
            ("std::io::Stdin", "Stdin - Handle to standard input", "struct"),
            ("std::io::Stdout", "Stdout - Handle to standard output", "struct"),
            ("std::io::Stderr", "Stderr - Handle to standard error", "struct"),
            # File system
            ("std::fs::File", "File - File handle", "struct"),
            (
                "std::fs::OpenOptions",
                "OpenOptions - Options for opening files",
                "struct",
            ),
            ("std::fs::Metadata", "Metadata - File metadata", "struct"),
            ("std::fs::DirEntry", "DirEntry - Directory entry", "struct"),
            ("std::fs::DirBuilder", "DirBuilder - Directory builder", "struct"),
            # Path types
            ("std::path::Path", "Path - A slice of a path", "struct"),
            ("std::path::PathBuf", "PathBuf - An owned, mutable path", "struct"),
            # Process management
            ("std::process::Command", "Command - Process builder", "struct"),
            ("std::process::Child", "Child - Handle to a child process", "struct"),
            ("std::process::Output", "Output - Process output", "struct"),
            ("std::process::Stdio", "Stdio - Process I/O configuration", "struct"),
            # Threading
            ("std::thread::JoinHandle", "JoinHandle<T> - Thread join handle", "struct"),
            ("std::thread::ThreadId", "ThreadId - Thread identifier", "struct"),
            ("std::thread::Builder", "Builder - Thread builder", "struct"),
            # Error handling
            ("std::error::Error", "Error - Error trait", "trait"),
            ("std::panic", "panic - Panic support", "module"),
            # Common traits
            (
                "std::fmt::Display",
                "Display - Format trait for user-facing output",
                "trait",
            ),
            ("std::fmt::Debug", "Debug - Format trait for debugging", "trait"),
            # Modules
            ("std::vec", "vec - Vector module", "module"),
            ("std::collections", "collections - Collection types", "module"),
            ("std::io", "io - I/O traits, helpers, and type definitions", "module"),
            ("std::fs", "fs - Filesystem manipulation operations", "module"),
            ("std::thread", "thread - Native threads", "module"),
            ("std::sync", "sync - Synchronization primitives", "module"),
            ("std::process", "process - Process management", "module"),
            ("std::env", "env - Environment inspection and manipulation", "module"),
            ("std::path", "path - Cross-platform path manipulation", "module"),
            ("std::net", "net - Networking primitives", "module"),
            ("std::time", "time - Time-related functionality", "module"),
            ("std::fmt", "fmt - Formatting and printing", "module"),
        ],
        "core": [
            # Core types
            ("core::option::Option", "Option<T> - Type for optional values", "enum"),
            ("core::result::Result", "Result<T, E> - Type for error handling", "enum"),
            ("core::option::Option::Some", "Some(T) - Contains a value", "variant"),
            ("core::option::Option::None", "None - No value", "variant"),
            ("core::result::Result::Ok", "Ok(T) - Success value", "variant"),
            ("core::result::Result::Err", "Err(E) - Error value", "variant"),
            # Iterator traits
            (
                "core::iter::Iterator",
                "Iterator - Interface for iterating over collections",
                "trait",
            ),
            (
                "core::iter::IntoIterator",
                "IntoIterator - Conversion into an Iterator",
                "trait",
            ),
            (
                "core::iter::FromIterator",
                "FromIterator - Build from an iterator",
                "trait",
            ),
            (
                "core::iter::DoubleEndedIterator",
                "DoubleEndedIterator - Iterator with known end",
                "trait",
            ),
            (
                "core::iter::ExactSizeIterator",
                "ExactSizeIterator - Iterator with exact length",
                "trait",
            ),
            # Core traits
            ("core::clone::Clone", "Clone - Trait for duplicating values", "trait"),
            (
                "core::cmp::PartialEq",
                "PartialEq - Partial equality comparison",
                "trait",
            ),
            ("core::cmp::Eq", "Eq - Equality comparison", "trait"),
            ("core::cmp::PartialOrd", "PartialOrd - Partial ordering", "trait"),
            ("core::cmp::Ord", "Ord - Total ordering", "trait"),
            ("core::cmp::Ordering", "Ordering - Result of comparison", "enum"),
            ("core::default::Default", "Default - Default values", "trait"),
            ("core::hash::Hash", "Hash - Hashable types", "trait"),
            ("core::hash::Hasher", "Hasher - Hash state", "trait"),
            # Marker traits
            (
                "core::marker::Copy",
                "Copy - Types whose values can be duplicated by copying bits",
                "trait",
            ),
            (
                "core::marker::Send",
                "Send - Types that can be transferred across thread boundaries",
                "trait",
            ),
            (
                "core::marker::Sync",
                "Sync - Types for which references can be shared between threads",
                "trait",
            ),
            (
                "core::marker::Sized",
                "Sized - Types with known size at compile time",
                "trait",
            ),
            (
                "core::marker::Unpin",
                "Unpin - Types that can be moved after pinning",
                "trait",
            ),
            (
                "core::marker::PhantomData",
                "PhantomData<T> - Zero-sized type marker",
                "struct",
            ),
            # Conversion traits
            ("core::convert::From", "From - Simple value conversion", "trait"),
            ("core::convert::Into", "Into - Value conversion", "trait"),
            ("core::convert::TryFrom", "TryFrom - Fallible conversion", "trait"),
            ("core::convert::TryInto", "TryInto - Fallible conversion", "trait"),
            ("core::convert::AsRef", "AsRef - Cheap reference conversion", "trait"),
            (
                "core::convert::AsMut",
                "AsMut - Cheap mutable reference conversion",
                "trait",
            ),
            # Ops traits
            ("core::ops::Deref", "Deref - Dereference operator", "trait"),
            ("core::ops::DerefMut", "DerefMut - Mutable dereference", "trait"),
            ("core::ops::Drop", "Drop - Destructor", "trait"),
            ("core::ops::Fn", "Fn - Function trait", "trait"),
            ("core::ops::FnMut", "FnMut - Mutable function trait", "trait"),
            ("core::ops::FnOnce", "FnOnce - One-time function trait", "trait"),
            ("core::ops::Range", "Range<T> - Half-open range", "struct"),
            (
                "core::ops::RangeInclusive",
                "RangeInclusive<T> - Inclusive range",
                "struct",
            ),
            # Memory and pointer types
            (
                "core::mem::MaybeUninit",
                "MaybeUninit<T> - Uninitialized memory",
                "struct",
            ),
            (
                "core::mem::ManuallyDrop",
                "ManuallyDrop<T> - Wrapper to inhibit drop",
                "struct",
            ),
            ("core::ptr::NonNull", "NonNull<T> - Non-null raw pointer", "struct"),
            # Cell types
            ("core::cell::Cell", "Cell<T> - Mutable memory location", "struct"),
            (
                "core::cell::RefCell",
                "RefCell<T> - Mutable memory with dynamic borrowing",
                "struct",
            ),
            (
                "core::cell::UnsafeCell",
                "UnsafeCell<T> - Core primitive for interior mutability",
                "struct",
            ),
            # Format traits
            ("core::fmt::Display", "Display - Format for user-facing output", "trait"),
            ("core::fmt::Debug", "Debug - Format for debugging", "trait"),
            ("core::fmt::Binary", "Binary - Binary formatting", "trait"),
            ("core::fmt::Octal", "Octal - Octal formatting", "trait"),
            ("core::fmt::LowerHex", "LowerHex - Lowercase hex formatting", "trait"),
            ("core::fmt::UpperHex", "UpperHex - Uppercase hex formatting", "trait"),
            # Modules
            ("core::mem", "mem - Memory manipulation", "module"),
            ("core::ptr", "ptr - Raw pointer manipulation", "module"),
            ("core::slice", "slice - Slice primitive", "module"),
            ("core::str", "str - String slice primitive", "module"),
            ("core::option", "option - Optional values", "module"),
            ("core::result", "result - Error handling with Result", "module"),
            ("core::iter", "iter - Iteration traits", "module"),
            ("core::ops", "ops - Operator traits", "module"),
            ("core::cmp", "cmp - Comparison traits", "module"),
            ("core::convert", "convert - Conversion traits", "module"),
            ("core::marker", "marker - Marker traits", "module"),
            ("core::cell", "cell - Shareable mutable containers", "module"),
            ("core::fmt", "fmt - Formatting traits", "module"),
        ],
        "alloc": [
            # Core collection types
            ("alloc::vec::Vec", "Vec<T> - A contiguous growable array type", "struct"),
            (
                "alloc::string::String",
                "String - A growable UTF-8 encoded string",
                "struct",
            ),
            (
                "alloc::collections::BTreeMap",
                "BTreeMap<K, V> - An ordered map based on a B-Tree",
                "struct",
            ),
            (
                "alloc::collections::BTreeSet",
                "BTreeSet<T> - An ordered set based on a B-Tree",
                "struct",
            ),
            (
                "alloc::collections::BinaryHeap",
                "BinaryHeap<T> - A priority queue implemented with a binary heap",
                "struct",
            ),
            (
                "alloc::collections::LinkedList",
                "LinkedList<T> - A doubly-linked list",
                "struct",
            ),
            (
                "alloc::collections::VecDeque",
                "VecDeque<T> - A double-ended queue",
                "struct",
            ),
            # Smart pointers
            (
                "alloc::boxed::Box",
                "Box<T> - A pointer type for heap allocation",
                "struct",
            ),
            (
                "alloc::rc::Rc",
                "Rc<T> - A single-threaded reference-counting pointer",
                "struct",
            ),
            ("alloc::rc::Weak", "Weak<T> - A weak reference to an Rc", "struct"),
            (
                "alloc::sync::Arc",
                "Arc<T> - Atomically reference-counted pointer",
                "struct",
            ),
            ("alloc::sync::Weak", "Weak<T> - A weak reference to an Arc", "struct"),
            # String types
            ("alloc::str::FromStr", "FromStr - Parse from string slices", "trait"),
            ("alloc::string::ToString", "ToString - Convert to String", "trait"),
            (
                "alloc::string::FromUtf8Error",
                "FromUtf8Error - Error for invalid UTF-8",
                "struct",
            ),
            (
                "alloc::string::FromUtf16Error",
                "FromUtf16Error - Error for invalid UTF-16",
                "struct",
            ),
            # Allocation traits
            (
                "alloc::alloc::GlobalAlloc",
                "GlobalAlloc - Memory allocator trait",
                "trait",
            ),
            ("alloc::alloc::Layout", "Layout - Memory layout", "struct"),
            (
                "alloc::alloc::LayoutError",
                "LayoutError - Layout computation error",
                "struct",
            ),
            ("alloc::alloc::AllocError", "AllocError - Allocation failure", "struct"),
            # Format types
            ("alloc::fmt::format", "format - Format macro internals", "function"),
            ("alloc::format", "format! - String formatting macro", "macro"),
            # Vec-specific types
            ("alloc::vec::Drain", "Drain - Draining iterator for Vec", "struct"),
            ("alloc::vec::IntoIter", "IntoIter - Consuming iterator for Vec", "struct"),
            # Slice types
            (
                "alloc::slice::from_raw_parts",
                "from_raw_parts - Create slice from raw parts",
                "function",
            ),
            (
                "alloc::slice::from_raw_parts_mut",
                "from_raw_parts_mut - Create mutable slice from raw parts",
                "function",
            ),
            # Cow (Clone on Write)
            ("alloc::borrow::Cow", "Cow - Clone-on-write smart pointer", "enum"),
            (
                "alloc::borrow::ToOwned",
                "ToOwned - Create owned data from borrowed",
                "trait",
            ),
            # Collection traits
            (
                "alloc::collections::TryReserveError",
                "TryReserveError - Error for failed reserve",
                "struct",
            ),
            # Modules
            ("alloc::vec", "vec - Vector module", "module"),
            ("alloc::collections", "collections - Collection types", "module"),
            ("alloc::boxed", "boxed - Box pointer type", "module"),
            ("alloc::rc", "rc - Single-threaded reference counting", "module"),
            ("alloc::sync", "sync - Thread-safe reference counting", "module"),
            ("alloc::string", "string - UTF-8 string types", "module"),
            ("alloc::borrow", "borrow - Borrowed and owned values", "module"),
            ("alloc::fmt", "fmt - Formatting machinery", "module"),
            ("alloc::slice", "slice - Slice utilities", "module"),
            ("alloc::str", "str - String utilities", "module"),
            ("alloc::alloc", "alloc - Memory allocation APIs", "module"),
        ],
        "proc_macro": [
            (
                "proc_macro::TokenStream",
                "TokenStream - The main type for procedural macros",
                "struct",
            ),
            (
                "proc_macro::TokenTree",
                "TokenTree - A single token or delimited token tree",
                "enum",
            ),
            ("proc_macro::Group", "Group - A delimited token stream", "struct"),
            ("proc_macro::Ident", "Ident - An identifier", "struct"),
            ("proc_macro::Punct", "Punct - A punctuation character", "struct"),
            (
                "proc_macro::Literal",
                "Literal - A literal string, byte string, character, etc",
                "struct",
            ),
        ],
        "test": [
            ("test::TestFn", "TestFn - Function type for tests", "type"),
            ("test::TestDesc", "TestDesc - Test description", "struct"),
            ("test::TestResult", "TestResult - Result of running a test", "enum"),
        ],
    }

    # Get items for this crate, or use a default set
    items = stdlib_items.get(
        crate_name,
        [
            (
                f"{crate_name}",
                f"The {crate_name} crate - part of Rust standard library",
                "module",
            ),
        ],
    )

    # Create chunks for embedding
    chunks = []
    for item_path, header, item_type in items:
        content = f"{header}\n\nThis is part of the Rust standard library."
        chunks.append(
            {
                "item_path": item_path,
                "header": header,
                "content": content,
                "item_type": item_type,
                "signature": None,
                "parent_id": None,
                "examples": [],
                "visibility": "public",
                "deprecated": False,
                "generic_params": None,
                "trait_bounds": None,
            }
        )

    # Also add the crate-level documentation
    chunks.append(
        {
            "item_path": "crate",
            "header": f"{crate_name} - Rust Standard Library",
            "content": description
            or f"The {crate_name} crate is part of the Rust standard library.",
            "item_type": "module",
            "signature": None,
            "parent_id": None,
            "examples": [],
            "visibility": "public",
            "deprecated": False,
            "generic_params": None,
            "trait_bounds": None,
        }
    )

    try:
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} stdlib fallback items")
        from .memory_utils import MemoryMonitor

        with MemoryMonitor(f"stdlib_fallback_{crate_name}_{version}"):
            chunk_embedding_pairs = generate_embeddings_streaming(chunks)
            await store_embeddings_streaming(db_path, chunk_embedding_pairs)

        logger.info(f"Successfully stored fallback documentation for {crate_name}")

        # Store basic module hierarchy
        modules = {}
        for item_path, _header, item_type in items:
            if item_type == "module":
                # Extract module name from path
                parts = item_path.split("::")
                if len(parts) > 1:
                    module_name = parts[-1]
                    module_id = f"module_{item_path.replace('::', '_')}"
                    modules[module_id] = {
                        "name": module_name,
                        "path": item_path,
                        "parent_id": None,
                        "depth": len(parts) - 1,
                        "item_count": 0,
                    }

        if modules:
            await store_modules(db_path, crate_id, modules)
            logger.info(f"Stored {len(modules)} module entries for {crate_name}")

    except Exception as e:
        logger.error(f"Failed to create stdlib fallback documentation: {e}")
        # Don't re-raise - at least we have the crate metadata


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
                        else:
                            # Run migrations if needed
                            from .database import (
                                migrate_add_generics_metadata,
                                migrate_database_duplicates,
                                migrate_reexports_for_crossrefs,
                            )

                            await migrate_database_duplicates(db_path)
                            await migrate_reexports_for_crossrefs(db_path)
                            await migrate_add_generics_metadata(db_path)
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

            crate_id = await store_crate_metadata(
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

                # Try to download with the specific version first
                try:
                    compressed_content, download_url = await download_rustdoc(
                        session, crate_name, resolved_version, rustdoc_url
                    )
                except Exception:
                    # Only fall back to latest if this specific version doesn't have rustdoc JSON
                    # Check for the specific exception type that indicates version unavailability
                    # is_version_not_found = (
                    #     e.__class__.__name__ == "RustdocVersionNotFoundError"
                    #     or "RustdocVersionNotFoundError" in str(type(e))
                    # )

                    # Don't fall back to latest automatically - let source extraction handle it
                    raise

                # Decompress content
                logger.info("Decompressing rustdoc content")
                json_content = await decompress_content(
                    compressed_content, download_url
                )

                # Parse rustdoc items (now returns an async generator)
                logger.info("Parsing rustdoc items in streaming mode")

                # Collect items from streaming parser
                items = []
                modules = {}
                reexports = []
                crossrefs = []
                async for item in parse_rustdoc_items_streaming(json_content):
                    # Check for special modules marker
                    if "_modules" in item:
                        modules = item["_modules"]
                    # Check for re-export items
                    elif "_reexport" in item:
                        reexports.append(item["_reexport"])
                    # Check for cross-reference items
                    elif "_crossref" in item:
                        crossrefs.append(item["_crossref"])
                    else:
                        items.append(item)

                if not items:
                    logger.warning("No items found in rustdoc JSON")
                    raise Exception("No items found in rustdoc JSON")

                logger.info(f"Collected {len(items)} rustdoc items")

                # Store module hierarchy
                if modules:
                    await store_modules(db_path, crate_id, modules)
                    logger.info(f"Stored {len(modules)} modules with hierarchy")

                # Store discovered re-exports
                if reexports:
                    await store_reexports(db_path, crate_id, reexports)
                    logger.info(f"Stored {len(reexports)} re-export mappings")

                # Store cross-references
                if crossrefs:
                    # Convert crossrefs to reexports format with link_type='crossref'
                    crossref_data = []
                    for crossref in crossrefs:
                        crossref_data.append(
                            {
                                "alias_path": crossref["source_path"],
                                "actual_path": crossref["target_path"],
                                "is_glob": False,
                                "link_text": crossref["link_text"],
                                "link_type": "crossref",
                                "target_item_id": crossref.get("target_item_id"),
                                "confidence_score": crossref.get("confidence", 1.0),
                            }
                        )
                    await store_reexports(db_path, crate_id, crossref_data)
                    logger.info(f"Stored {len(crossrefs)} cross-reference mappings")

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
                        "generic_params": item.get("generic_params"),
                        "trait_bounds": item.get("trait_bounds"),
                    }
                    chunks.append(chunk)

                # Generate embeddings and store in streaming fashion with memory monitoring
                logger.info(
                    f"Generating embeddings for {len(chunks)} items in streaming mode"
                )

                # Use MemoryMonitor to track embedding generation
                from .memory_utils import MemoryMonitor

                with MemoryMonitor(f"embeddings_{crate_name}_{version}"):
                    chunk_embedding_pairs = generate_embeddings_streaming(chunks)

                    # Store embeddings in streaming fashion
                    await store_embeddings_streaming(db_path, chunk_embedding_pairs)

                logger.info("Successfully completed streaming ingestion")

                # Generate example embeddings for dedicated search
                try:
                    await generate_example_embeddings(db_path, crate_name, version)
                except Exception as ex_err:
                    logger.warning(f"Failed to generate example embeddings: {ex_err}")
                    # Continue - main embeddings are already stored

            except Exception as e:
                if is_stdlib_crate(crate_name):
                    # Count items for accurate reporting
                    stdlib_counts = {
                        "std": 62,  # Updated count after expansion
                        "core": 68,  # Updated count after expansion
                        "alloc": 43,  # Updated count after expansion
                        "proc_macro": 6,
                        "test": 3,
                    }

                    tutorial_message = f"""
================================================================================
STDLIB DOCUMENTATION NOTICE: Limited Fallback Active
================================================================================

📚 Status: Using minimal fallback for {crate_name} crate
   - Currently providing {stdlib_counts.get(crate_name, 1)} common items
   - Full documentation requires local rust-docs-json component

🚀 Quick Setup for Full Documentation:
   1. Install nightly toolchain (required):
      rustup toolchain install nightly

   2. Add rust-docs-json component:
      rustup component add --toolchain nightly rust-docs-json

   3. Verify installation:
      ls ~/.rustup/toolchains/nightly-*/lib/rustlib/*/json/
      (Should show std.json, core.json, alloc.json)

📖 Alternative: Use online documentation
   - Visit https://doc.rust-lang.org/{crate_name}/ for complete stdlib docs
   - Note: JSON ingestion from local files not yet supported

⚠️ Limitations:
   - JSON format is unstable (changes ~2x/month)
   - Requires nightly toolchain
   - Local ingestion planned for future release

💡 Tip: The fallback includes common types like Vec, HashMap, Result, Option,
   Iterator, Arc, Mutex, File, Path, and more. For other items, please refer
   to online documentation.

🔍 Original error: {e}
================================================================================
"""
                    logger.warning(tutorial_message)

                    # Create comprehensive stdlib fallback documentation
                    await create_stdlib_fallback_documentation(
                        db_path, crate_id, crate_name, version, description
                    )
                    return db_path
                else:
                    logger.warning(f"Failed to process rustdoc JSON: {e}")

                # Try source extraction fallback for non-stdlib crates
                if not is_stdlib_crate(crate_name):
                    logger.info("Attempting source extraction fallback")
                    try:
                        # Import extractor module from parent directory
                        import sys
                        from pathlib import Path

                        # Add parent directory to path if needed
                        parent_dir = Path(__file__).parent.parent.parent
                        if str(parent_dir) not in sys.path:
                            sys.path.insert(0, str(parent_dir))

                        from extractors.source_extractor import CratesIoSourceExtractor

                        # Create extractor with existing session
                        async with CratesIoSourceExtractor(
                            session=session,
                            memory_monitor=None,  # We'll use existing memory monitoring
                            timeout=30,
                        ) as extractor:
                            # Extract documentation from source
                            extracted_items = await extractor.extract_from_source(
                                name=crate_name, version=version
                            )

                            if extracted_items:
                                logger.info(
                                    f"Extracted {len(extracted_items)} items from source"
                                )

                                # Transform to chunk format
                                chunks = []
                                for item in extracted_items:
                                    # Extract code examples from docstring
                                    examples = []
                                    docstring = item.get("docstring", "")
                                    if docstring:
                                        # Find code blocks in documentation
                                        code_blocks = re.findall(
                                            r"```(?:rust|rs)?\n(.*?)\n```",
                                            docstring,
                                            re.DOTALL,
                                        )
                                        for code in code_blocks:
                                            if code.strip():
                                                examples.append(
                                                    {
                                                        "description": "Code example from documentation",
                                                        "code": code.strip(),
                                                        "language": "rust",
                                                    }
                                                )

                                    chunk = {
                                        "item_path": item["item_path"],
                                        "header": item["header"],
                                        "content": docstring,
                                        "item_type": item.get("item_type"),
                                        "signature": item.get("signature"),
                                        "parent_id": None,  # We don't track parent relationships in fallback
                                        "examples": examples,
                                        "visibility": item.get("visibility", "public"),
                                        "deprecated": False,
                                        "generic_params": None,  # Not available in fallback
                                        "trait_bounds": None,  # Not available in fallback
                                    }
                                    chunks.append(chunk)

                                # Generate and store embeddings using existing infrastructure
                                from .memory_utils import MemoryMonitor

                                with MemoryMonitor(
                                    f"fallback_embeddings_{crate_name}_{version}"
                                ):
                                    chunk_embedding_pairs = (
                                        generate_embeddings_streaming(chunks)
                                    )
                                    await store_embeddings_streaming(
                                        db_path, chunk_embedding_pairs
                                    )

                                logger.info(
                                    f"Successfully stored {len(chunks)} fallback items for {crate_name}@{version}"
                                )

                                # Skip the basic description fallback since we have real data
                                return db_path

                    except Exception as extraction_error:
                        logger.warning(
                            f"Source extraction fallback failed: {extraction_error}"
                        )

                        # Second fallback: Try latest version's rustdoc JSON
                        if version != "latest" and not is_stdlib_crate(crate_name):
                            logger.info(
                                "Source extraction failed, falling back to latest version's rustdoc JSON"
                            )
                            try:
                                # Resolve latest version URL
                                latest_version, latest_url = await resolve_version(
                                    session, crate_name, "latest"
                                )
                                logger.info(
                                    f"Trying with latest version: {latest_version}"
                                )

                                # Try downloading with latest version
                                (
                                    compressed_content,
                                    download_url,
                                ) = await download_rustdoc(
                                    session, crate_name, latest_version, latest_url
                                )

                                # Decompress and process
                                logger.info("Decompressing rustdoc content")
                                json_content = await decompress_content(
                                    compressed_content, download_url
                                )

                                # Parse and store using existing logic
                                items = []
                                modules = {}
                                reexports = []
                                crossrefs = []
                                async for item in parse_rustdoc_items_streaming(
                                    json_content
                                ):
                                    if "_modules" in item:
                                        modules = item["_modules"]
                                    elif "_reexports" in item:
                                        reexports = item["_reexports"]
                                    elif "_crossrefs" in item:
                                        crossrefs = item["_crossrefs"]
                                    else:
                                        items.append(item)

                                logger.info(
                                    f"Collected {len(items)} rustdoc items from latest version"
                                )

                                # Store modules and process items
                                await store_modules(db_path, crate_id, modules)

                                all_reexports = reexports + crossrefs
                                if all_reexports:
                                    await store_reexports(
                                        db_path, crate_id, all_reexports
                                    )

                                # Generate chunks and embeddings
                                chunks = []
                                for item in items:
                                    chunk = {
                                        "item_path": item["item_path"],
                                        "header": item["header"],
                                        "content": item.get("docstring", ""),
                                        "item_type": item.get("item_type"),
                                        "signature": item.get("signature"),
                                        "parent_id": item.get("parent_id"),
                                        "examples": item.get("examples", []),
                                        "visibility": item.get("visibility", "public"),
                                        "deprecated": item.get("deprecated", False),
                                        "generic_params": item.get("generic_params"),
                                        "trait_bounds": item.get("trait_bounds"),
                                    }
                                    chunks.append(chunk)

                                from .memory_utils import MemoryMonitor

                                with MemoryMonitor(
                                    f"embeddings_{crate_name}_{version}"
                                ):
                                    chunk_embedding_pairs = (
                                        generate_embeddings_streaming(chunks)
                                    )
                                    await store_embeddings_streaming(
                                        db_path, chunk_embedding_pairs
                                    )

                                logger.info(
                                    f"Successfully ingested latest version as fallback for {crate_name}@{version}"
                                )

                                # Try to generate example embeddings
                                try:
                                    await generate_example_embeddings(
                                        db_path, crate_name, version
                                    )
                                except Exception as ex_err:
                                    logger.warning(
                                        f"Failed to generate example embeddings: {ex_err}"
                                    )

                                return db_path

                            except Exception as latest_error:
                                logger.warning(
                                    f"Latest version fallback also failed: {latest_error}"
                                )
                                # Continue to basic description fallback

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
                        # First, clean up any partial writes from the failed main ingestion
                        # This prevents UNIQUE constraint violations on the "crate" item_path
                        async with aiosqlite.connect(db_path) as db:
                            await db.execute(
                                "DELETE FROM embeddings WHERE item_path = ?", ("crate",)
                            )
                            await db.commit()
                            logger.debug(
                                "Cleaned up existing 'crate' entry before fallback"
                            )

                        # Generate embeddings
                        embeddings = await generate_embeddings(chunks)

                        # Store embeddings (will use DELETE + INSERT via _store_batch)
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

            # Clean up embedding model to free memory after each crate
            cleanup_embedding_model()

            logger.info(f"Successfully ingested {crate_name}@{version}")
            return db_path
