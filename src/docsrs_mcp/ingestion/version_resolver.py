"""Version resolution and rustdoc downloading for Rust crates.

This module handles:
- Version resolution from docs.rs and crates.io
- Standard library version resolution
- Rustdoc JSON downloading with compression support
- Content decompression with size limits
"""

import gzip
import io
import logging
import re
from typing import Any

import aiohttp
import zstandard
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import (
    DOWNLOAD_CHUNK_SIZE,
    HTTP_TIMEOUT,
    MAX_DECOMPRESSED_SIZE,
    MAX_DOWNLOAD_SIZE,
    RUST_VERSION_MANIFEST_URL,
)

logger = logging.getLogger(__name__)

# Standard library crates that are part of Rust itself
STDLIB_CRATES = {
    "std",
    "core",
    "alloc",
    "proc_macro",
    "test",
}

# Feature flag for compressed rustdoc support
COMPRESSED_RUSTDOC_SUPPORTED = True


class RustdocVersionNotFoundError(Exception):
    """Raised when a specific version doesn't have rustdoc JSON available."""


def is_stdlib_crate(crate_name: str) -> bool:
    """Check if a crate name is a Rust standard library crate.

    Args:
        crate_name: Name of the crate to check

    Returns:
        bool: True if the crate is a standard library crate
    """
    return crate_name.lower() in STDLIB_CRATES


async def fetch_current_stable_version(session: aiohttp.ClientSession) -> str:
    """Fetch the current stable Rust version from the official channel.

    Args:
        session: HTTP session for making requests

    Returns:
        str: Current stable Rust version (e.g., "1.75.0")
    """
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
    """Resolve standard library version string to actual Rust version.

    Args:
        session: HTTP session for making requests
        version: Version string (e.g., "latest", "stable", "1.75.0")

    Returns:
        str: Resolved Rust version
    """
    if not version or version in ["latest", "stable"]:
        return await fetch_current_stable_version(session)
    elif version in ["beta", "nightly"]:
        # For beta/nightly, we return the channel name
        # The actual version will be determined by the channel manifest
        return version
    else:
        # Assume it's a specific Rust version like "1.75.0"
        return version


def construct_stdlib_url(crate_name: str, version: str) -> str:
    """Construct the URL for downloading standard library rustdoc JSON.

    Note: Standard library rustdoc JSON is generally NOT available on docs.rs.
    This function returns a URL pattern that will be tried, but is expected
    to fail for most stdlib crates. The ingestion will fall back to source
    extraction when this fails.

    To use actual stdlib rustdoc JSON, users need to generate it locally:
      rustup component add --toolchain nightly rust-docs-json

    Args:
        crate_name: Name of the stdlib crate
        version: Rust version string

    Returns:
        str: URL to try for rustdoc JSON (likely to 404)
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

    Args:
        session: HTTP session for making requests
        crate_name: Name of the crate
        version: Version string to resolve (default: "latest")

    Returns:
        Tuple[str, str]: (resolved_version, rustdoc_url)
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
    """Fetch crate information from crates.io API.

    Args:
        session: HTTP session for making requests
        crate_name: Name of the crate

    Returns:
        dict: Crate information from crates.io
    """
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
    """Resolve version string to actual version from crate info.

    Args:
        crate_info: Crate information from crates.io
        version: Optional version string to resolve

    Returns:
        str: Resolved version string
    """
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

    Args:
        session: HTTP session for making requests
        crate_name: Name of the crate
        version: Version of the crate
        rustdoc_url: Optional custom URL to use

    Returns:
        Tuple[bytes, str]: (raw_content, url_used)

    Raises:
        RustdocVersionNotFoundError: If rustdoc JSON is not available for this version
    """
    # Use provided URL or construct default using correct docs.rs API pattern
    if rustdoc_url is None:
        # Use the correct docs.rs API pattern: /crate/{name}/{version}/json
        rustdoc_url = f"https://docs.rs/crate/{crate_name}/{version}/json"

    # Try different compression formats in order of preference
    urls_to_try = []

    # If URL already has a compressed extension, use it as-is
    if rustdoc_url.endswith((".json.zst", ".json.gz", ".gz")):
        urls_to_try.append(rustdoc_url)
    # For the new JSON API pattern (/crate/{name}/{version}/json)
    elif rustdoc_url.endswith("/json"):
        # Try zstd (default), then gzip, then uncompressed
        urls_to_try.extend([f"{rustdoc_url}.zst", f"{rustdoc_url}.gz", rustdoc_url])
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

                # Return the raw content and URL
                return (content, url)

        except Exception as e:
            last_error = e
            # Track if this was NOT a 404
            if "404" not in str(e):
                all_404 = False
                logger.warning(f"Failed to download from {url}: {e}")

    # All formats failed
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

    Args:
        content: Raw compressed bytes
        url: URL used to download (for format detection)

    Returns:
        str: Decompressed JSON string

    Raises:
        Exception: If decompressed size exceeds limits
    """
    logger.debug(f"Attempting to decompress {len(content)} bytes from {url}")
    
    # Check if content is zstd compressed (default for /json endpoint)
    # zstd magic bytes: 0x28, 0xb5, 0x2f, 0xfd
    is_zstd = content[:4] == b"\x28\xb5\x2f\xfd" if len(content) >= 4 else False
    
    logger.debug(f"Content analysis: magic bytes={content[:4].hex() if len(content) >= 4 else 'too short'}, is_zstd={is_zstd}")

    if url.endswith(".json.zst") or url.endswith("/json.zst") or (url.endswith("/json") and is_zstd):
        # Zstandard decompression
        try:
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
            
        except zstandard.ZstdError as e:
            logger.warning(f"Zstd decompression failed: {e}")
            # If zstd fails but URL suggests compression, the content might actually be uncompressed
            # Try treating as plain JSON
            if len(content) > MAX_DECOMPRESSED_SIZE:
                raise Exception(f"Content size exceeded limit: {len(content)} bytes")
            try:
                # Attempt to decode as UTF-8 and parse as JSON to verify
                text_content = content.decode("utf-8")
                import json
                json.loads(text_content[:1000])  # Quick sanity check
                logger.info(f"Content appears to be uncompressed JSON despite .zst extension")
                return text_content
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Re-raise the original zstd error
                raise Exception(f"zstd decompress error: {e}")

    elif url.endswith(".json.gz") or url.endswith("/json.gz"):
        # Gzip decompression
        try:
            decompressed = gzip.decompress(content)

            if len(decompressed) > MAX_DECOMPRESSED_SIZE:
                raise Exception(
                    f"Decompressed size exceeded limit: {len(decompressed)} bytes"
                )

            logger.info(
                f"Decompressed .gz file: {len(content)} -> {len(decompressed)} bytes"
            )
            return decompressed.decode("utf-8")
            
        except (gzip.BadGzipFile, OSError) as e:
            logger.warning(f"Gzip decompression failed: {e}")
            # Similar fallback as with zstd
            if len(content) > MAX_DECOMPRESSED_SIZE:
                raise Exception(f"Content size exceeded limit: {len(content)} bytes")
            try:
                text_content = content.decode("utf-8")
                import json
                json.loads(text_content[:1000])  # Quick sanity check
                logger.info(f"Content appears to be uncompressed JSON despite .gz extension")
                return text_content
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise Exception(f"gzip decompress error: {e}")

    else:
        # Uncompressed JSON
        if len(content) > MAX_DECOMPRESSED_SIZE:
            raise Exception(f"Uncompressed size exceeded limit: {len(content)} bytes")

        logger.info(f"Processing uncompressed content: {len(content)} bytes")
        return content.decode("utf-8")
