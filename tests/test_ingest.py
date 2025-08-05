"""Tests for the ingestion pipeline."""

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import zstandard

from docsrs_mcp.ingest import (
    calculate_cache_size,
    decompress_content,
    download_rustdoc,
    evict_cache_if_needed,
    get_crate_lock,
    parse_rustdoc_items,
    resolve_version,
)


@pytest.mark.asyncio
async def test_get_crate_lock():
    """Test per-crate lock mechanism."""
    # Test that same crate@version returns same lock
    lock1 = await get_crate_lock("serde", "1.0.0")
    lock2 = await get_crate_lock("serde", "1.0.0")
    assert lock1 is lock2

    # Test that different versions get different locks
    lock3 = await get_crate_lock("serde", "1.0.1")
    assert lock1 is not lock3

    # Test that different crates get different locks
    lock4 = await get_crate_lock("tokio", "1.0.0")
    assert lock1 is not lock4


@pytest.mark.asyncio
async def test_resolve_version():
    """Test version resolution via docs.rs redirects."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.url = "https://docs.rs/serde/1.0.193/serde/"

    mock_session = MagicMock()
    # Create a proper async context manager mock
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None
    # Mock head as a regular method that returns the context manager
    mock_session.head = MagicMock(return_value=mock_context)

    version, rustdoc_url = await resolve_version(mock_session, "serde", "latest")

    assert version == "1.0.193"
    assert rustdoc_url == "https://docs.rs/serde/1.0.193/serde.json"

    # Verify HEAD request was made correctly
    mock_session.head.assert_called_once()
    call_args = mock_session.head.call_args
    assert "https://docs.rs/crate/serde/latest" in str(call_args)


@pytest.mark.asyncio
async def test_resolve_version_error():
    """Test version resolution error handling."""
    mock_response = AsyncMock()
    mock_response.status = 404

    mock_session = MagicMock()
    # Create a proper async context manager mock
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None
    # Mock head as a regular method that returns the context manager
    mock_session.head = MagicMock(return_value=mock_context)

    with pytest.raises(Exception) as exc_info:
        await resolve_version(mock_session, "nonexistent", "latest")

    assert "Failed to resolve version" in str(exc_info.value)


@pytest.mark.asyncio
async def test_decompress_content_zst():
    """Test zstandard decompression."""
    # Create test data
    test_data = '{"test": "data"}'
    compressed = zstandard.compress(test_data.encode())

    # Test decompression
    result = await decompress_content(compressed, "test.json.zst")
    assert result == test_data


@pytest.mark.asyncio
async def test_decompress_content_gz():
    """Test gzip decompression."""
    # Create test data
    test_data = '{"test": "data"}'
    compressed = gzip.compress(test_data.encode())

    # Test decompression
    result = await decompress_content(compressed, "test.json.gz")
    assert result == test_data


@pytest.mark.asyncio
async def test_decompress_content_uncompressed():
    """Test handling of uncompressed content."""
    test_data = '{"test": "data"}'

    result = await decompress_content(test_data.encode(), "test.json")
    assert result == test_data


@pytest.mark.asyncio
async def test_decompress_content_size_limit():
    """Test decompression size limit enforcement."""
    # Create large data that exceeds limit when decompressed
    large_data = "x" * (101 * 1024 * 1024)  # 101MB
    compressed = zstandard.compress(large_data.encode())

    with pytest.raises(Exception) as exc_info:
        await decompress_content(compressed, "test.json.zst")

    assert "exceeded limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_parse_rustdoc_items():
    """Test rustdoc JSON parsing."""
    # Sample rustdoc JSON structure
    rustdoc_json = {
        "paths": {
            "0:1": {"path": ["serde", "Serialize"]},
            "0:2": {"path": ["serde", "Deserialize"]},
            "0:3": {"path": ["serde", "de"]},
        },
        "index": {
            "0:1": {
                "name": "Serialize",
                "kind": "trait",
                "docs": "A data structure that can be serialized.",
            },
            "0:2": {
                "name": "Deserialize",
                "kind": "trait",
                "docs": "A data structure that can be deserialized.",
            },
            "0:3": {
                "name": "de",
                "kind": "module",
                "docs": "Generic deserialization framework.",
            },
        },
    }

    json_content = json.dumps(rustdoc_json)
    items = await parse_rustdoc_items(json_content)

    assert len(items) == 3

    # Check first item
    assert items[0]["item_id"] == "0:1"
    assert items[0]["item_path"] == "serde::Serialize"
    assert items[0]["header"] == "trait Serialize"
    assert "serialized" in items[0]["docstring"]

    # Check module
    module_item = next(i for i in items if i["kind"] == "module")
    assert module_item["header"] == "mod de"


@pytest.mark.asyncio
async def test_parse_rustdoc_items_empty():
    """Test parsing empty rustdoc JSON."""
    rustdoc_json = {"paths": {}, "index": {}}

    json_content = json.dumps(rustdoc_json)
    items = await parse_rustdoc_items(json_content)

    assert items == []


@pytest.mark.asyncio
async def test_parse_rustdoc_items_malformed():
    """Test parsing malformed rustdoc JSON."""
    # Invalid JSON should not crash but return empty list
    json_content = '{"paths": {invalid json here'

    items = await parse_rustdoc_items(json_content)
    assert items == []  # Should handle gracefully


def test_calculate_cache_size(tmp_path):
    """Test cache size calculation."""
    # Create mock cache structure
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create some test files
    (cache_dir / "crate1").mkdir()
    db1 = cache_dir / "crate1" / "1.0.0.db"
    db1.write_bytes(b"x" * 1000)

    (cache_dir / "crate2").mkdir()
    db2 = cache_dir / "crate2" / "2.0.0.db"
    db2.write_bytes(b"x" * 2000)

    # Non-db file should be ignored
    (cache_dir / "other.txt").write_text("ignored")

    with patch("docsrs_mcp.ingest.CACHE_DIR", cache_dir):
        size = calculate_cache_size()

    assert size == 3000  # Only .db files counted


@pytest.mark.asyncio
async def test_evict_cache_if_needed(tmp_path):
    """Test cache eviction based on size limit."""
    # Create mock cache structure
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create test files with different mtimes
    import time

    (cache_dir / "old").mkdir()
    old_db = cache_dir / "old" / "1.0.0.db"
    old_db.write_bytes(b"x" * 1000)
    time.sleep(0.1)

    (cache_dir / "newer").mkdir()
    newer_db = cache_dir / "newer" / "1.0.0.db"
    newer_db.write_bytes(b"x" * 1000)
    time.sleep(0.1)

    (cache_dir / "newest").mkdir()
    newest_db = cache_dir / "newest" / "1.0.0.db"
    newest_db.write_bytes(b"x" * 1000)

    # Set cache limit to 2500 bytes (should evict oldest)
    with patch("docsrs_mcp.ingest.CACHE_DIR", cache_dir):
        with patch("docsrs_mcp.ingest.CACHE_MAX_SIZE_BYTES", 2500):
            await evict_cache_if_needed()

    # Oldest file should be removed
    assert not old_db.exists()
    assert newer_db.exists()
    assert newest_db.exists()


@pytest.mark.asyncio
async def test_download_rustdoc_with_compression():
    """Test downloading rustdoc with compression format detection."""
    # Mock compressed content
    test_json = '{"test": "data"}'
    compressed_content = zstandard.compress(test_json.encode())

    # Mock response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Length": str(len(compressed_content))}

    # Create async iterator for content chunks
    async def async_iter_chunked(chunk_size):
        yield compressed_content

    mock_resp.content.iter_chunked = async_iter_chunked

    # Mock session that returns 404 for .json but 200 for .json.zst
    mock_session = MagicMock()

    # Create different response objects for different URLs
    mock_404_resp = AsyncMock()
    mock_404_resp.status = 404

    # Track which URLs have been tried
    tried_urls = []

    async def mock_get(url, **kwargs):
        tried_urls.append(url)
        mock_context = AsyncMock()
        if url.endswith(".json.zst"):
            mock_context.__aenter__.return_value = mock_resp
        else:
            mock_context.__aenter__.return_value = mock_404_resp
        mock_context.__aexit__.return_value = None
        return mock_context

    mock_session.get = mock_get

    # Test download
    content, url = await download_rustdoc(mock_session, "test-crate", "1.0.0")

    assert content == compressed_content
    assert url.endswith(".json.zst")

    # Verify it tried the URLs in the correct order
    assert len(tried_urls) >= 1  # Should have tried at least one URL
    # Check that we found the .json.zst URL
    assert any(u.endswith(".json.zst") for u in tried_urls)

    # Debug output if test fails
    if not (content == compressed_content and url.endswith(".json.zst")):
        print(f"Tried URLs: {tried_urls}")
        print(f"Returned URL: {url}")
        print(f"Content length: {len(content) if content else 'None'}")


@pytest.mark.asyncio
async def test_download_rustdoc_size_limit():
    """Test download size limit enforcement."""
    # Mock response with large content
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100MB

    mock_session = AsyncMock()
    # Create proper async context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_resp
    mock_context.__aexit__.return_value = None
    mock_session.get.return_value = mock_context

    with pytest.raises(Exception) as exc_info:
        await download_rustdoc(mock_session, "test-crate", "1.0.0")

    assert "too large" in str(exc_info.value)


@pytest.mark.asyncio
async def test_download_rustdoc_not_found():
    """Test handling of missing rustdoc."""
    # Mock 404 responses for all formats
    mock_resp = AsyncMock()
    mock_resp.status = 404

    mock_session = AsyncMock()
    # Create proper async context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_resp
    mock_context.__aexit__.return_value = None
    mock_session.get.return_value = mock_context

    with pytest.raises(Exception) as exc_info:
        await download_rustdoc(mock_session, "test-crate", "1.0.0")

    assert "not found" in str(exc_info.value)
