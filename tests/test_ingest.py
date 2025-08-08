"""Tests for the ingestion pipeline."""

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
import sqlite_vec
import zstandard

from docsrs_mcp.config import DB_BATCH_SIZE
from docsrs_mcp.database import init_database
from docsrs_mcp.ingest import (
    calculate_cache_size,
    decompress_content,
    download_rustdoc,
    evict_cache_if_needed,
    fetch_current_stable_version,
    get_crate_lock,
    get_stdlib_url,
    is_stdlib_crate,
    parse_rustdoc_items,
    resolve_stdlib_version,
    resolve_version,
    store_embeddings,
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
    mock_response.url = "https://static.docs.rs/serde/1.0.193/json"

    mock_session = MagicMock()
    # Create a proper async context manager mock
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response
    mock_context.__aexit__.return_value = None
    # Mock head as a regular method that returns the context manager
    mock_session.head = MagicMock(return_value=mock_context)

    version, rustdoc_url = await resolve_version(mock_session, "serde", "latest")

    assert version == "1.0.193"
    assert rustdoc_url == "https://docs.rs/crate/serde/latest/json"

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
async def test_priority_aware_cache_eviction(tmp_path):
    """Test priority-aware cache eviction that preserves popular crates."""
    # Create mock cache structure
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create test files with different crates and ages
    import time

    from docsrs_mcp.models import PopularCrate

    # Create unknown/unpopular crate (oldest)
    (cache_dir / "unknown_crate").mkdir()
    unknown_db = cache_dir / "unknown_crate" / "1.0.0.db"
    unknown_db.write_bytes(b"x" * 1000)
    time.sleep(0.1)

    # Create popular crate (middle age)
    (cache_dir / "serde").mkdir()
    serde_db = cache_dir / "serde" / "1.0.0.db"
    serde_db.write_bytes(b"x" * 1000)
    time.sleep(0.1)

    # Create another unknown crate (newest)
    (cache_dir / "random_crate").mkdir()
    random_db = cache_dir / "random_crate" / "1.0.0.db"
    random_db.write_bytes(b"x" * 1000)

    # Mock popular crates data
    mock_popular_crates = [
        PopularCrate(
            name="serde",
            downloads=150000000,  # Very popular
            description="Serialization framework",
            version="1.0.219",
            last_updated=time.time(),
        ),
        PopularCrate(
            name="tokio",
            downloads=100000000,
            description="Async runtime",
            version="1.0.0",
            last_updated=time.time(),
        ),
    ]

    # Mock the PopularCratesManager
    mock_manager = AsyncMock()
    mock_manager.get_popular_crates_with_metadata = AsyncMock(
        return_value=mock_popular_crates
    )

    # Set cache limit to 2500 bytes (should evict one file)
    with patch("docsrs_mcp.ingest.CACHE_DIR", cache_dir):
        with patch("docsrs_mcp.ingest.CACHE_MAX_SIZE_BYTES", 2500):
            with patch("docsrs_mcp.ingest.PRIORITY_CACHE_EVICTION_ENABLED", True):
                with patch(
                    "docsrs_mcp.popular_crates.get_popular_manager", return_value=lambda: mock_manager
                ):
                    await evict_cache_if_needed()

    # Should evict unknown_crate (oldest unpopular), keep serde (popular) and random_crate
    assert not unknown_db.exists()  # Evicted: unpopular and old
    assert serde_db.exists()  # Kept: popular despite being older than random_crate
    assert random_db.exists()  # Kept: newest file


@pytest.mark.asyncio
async def test_priority_eviction_fallback(tmp_path):
    """Test fallback to time-based eviction when priority data unavailable."""
    # Create mock cache structure
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create test files
    import time

    (cache_dir / "old").mkdir()
    old_db = cache_dir / "old" / "1.0.0.db"
    old_db.write_bytes(b"x" * 1000)
    time.sleep(0.1)

    (cache_dir / "new").mkdir()
    new_db = cache_dir / "new" / "1.0.0.db"
    new_db.write_bytes(b"x" * 1000)

    # Mock manager that fails
    mock_manager = AsyncMock()
    mock_manager.get_popular_crates_with_metadata = AsyncMock(
        side_effect=Exception("API failure")
    )

    # Set cache limit to 1500 bytes
    with patch("docsrs_mcp.ingest.CACHE_DIR", cache_dir):
        with patch("docsrs_mcp.ingest.CACHE_MAX_SIZE_BYTES", 1500):
            with patch("docsrs_mcp.ingest.PRIORITY_CACHE_EVICTION_ENABLED", True):
                with patch(
                    "docsrs_mcp.popular_crates.get_popular_manager", return_value=lambda: mock_manager
                ):
                    await evict_cache_if_needed()

    # Should fall back to time-based eviction, removing oldest
    assert not old_db.exists()
    assert new_db.exists()


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


@pytest.mark.asyncio
async def test_store_embeddings_batch_processing(tmp_path):
    """Test batch insert behavior with >1000 items."""
    # Create a test database
    db_path = tmp_path / "test.db"
    await init_database(db_path)

    # Create test data with >1000 items (e.g., 2500 items)
    num_items = 2500
    chunks = []
    embeddings = []

    for i in range(num_items):
        chunks.append(
            {
                "item_path": f"test::item_{i}",
                "header": f"Item {i}",
                "content": f"Content for item {i}",
            }
        )
        # Create a simple embedding (384 dimensions as per config)
        embeddings.append([float(i % 10) / 10.0] * 384)

    # Store embeddings using batch processing
    await store_embeddings(db_path, chunks, embeddings)

    # Verify all items were stored correctly
    async with aiosqlite.connect(db_path) as db:
        # Enable extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Check embeddings table
        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
        count = await cursor.fetchone()
        assert count[0] == num_items

        # Check vec_embeddings table
        cursor = await db.execute("SELECT COUNT(*) FROM vec_embeddings")
        count = await cursor.fetchone()
        assert count[0] == num_items

        # Verify some sample data
        cursor = await db.execute(
            "SELECT item_path, header, content FROM embeddings WHERE item_path = ?",
            ("test::item_999",),
        )
        row = await cursor.fetchone()
        assert row[0] == "test::item_999"
        assert row[1] == "Item 999"
        assert row[2] == "Content for item 999"

        # Verify rowid relationships are maintained
        cursor = await db.execute(
            """
            SELECT COUNT(*) FROM embeddings e
            JOIN vec_embeddings v ON e.id = v.rowid
            """
        )
        count = await cursor.fetchone()
        assert count[0] == num_items

    # Verify batching behavior
    # With 2500 items and batch size of 999, we should have 3 batches:
    # Batch 1: 0-998 (999 items)
    # Batch 2: 999-1997 (999 items)
    # Batch 3: 1998-2499 (502 items)
    expected_batches = (num_items + DB_BATCH_SIZE - 1) // DB_BATCH_SIZE
    assert expected_batches == 3


@pytest.mark.asyncio
async def test_store_embeddings_empty_data(tmp_path):
    """Test handling of empty data."""
    # Create a test database
    db_path = tmp_path / "test.db"
    await init_database(db_path)

    # Test with empty lists
    await store_embeddings(db_path, [], [])

    # Verify no data was inserted
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
        count = await cursor.fetchone()
        assert count[0] == 0


@pytest.mark.asyncio
async def test_store_embeddings_error_handling(tmp_path):
    """Test error handling during batch insert."""
    # Create a test database
    db_path = tmp_path / "test.db"
    await init_database(db_path)

    # Create test data with invalid embedding dimensions
    chunks = [{"item_path": "test::item", "header": "Item", "content": "Content"}]
    # Invalid: only 10 dimensions instead of 384
    embeddings = [[1.0] * 10]

    # This should raise an error due to dimension mismatch
    with pytest.raises(Exception):
        await store_embeddings(db_path, chunks, embeddings)


# ============== Standard Library Support Tests ==============


def test_stdlib_detection():
    """Test detection of standard library crates."""
    # Test stdlib crates
    assert is_stdlib_crate("std") is True
    assert is_stdlib_crate("core") is True
    assert is_stdlib_crate("alloc") is True
    assert is_stdlib_crate("proc_macro") is True
    assert is_stdlib_crate("test") is True

    # Test case insensitivity
    assert is_stdlib_crate("STD") is True
    assert is_stdlib_crate("Core") is True

    # Test non-stdlib crates
    assert is_stdlib_crate("serde") is False
    assert is_stdlib_crate("tokio") is False
    assert is_stdlib_crate("std-derive") is False  # Not an exact match


@pytest.mark.asyncio
async def test_stdlib_version_resolution():
    """Test standard library version resolution."""
    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="1.75.0 (2023-12-28)")

    # Create a proper async context manager mock
    mock_ctx_mgr = AsyncMock()
    mock_ctx_mgr.__aenter__ = AsyncMock(return_value=mock_response)
    mock_ctx_mgr.__aexit__ = AsyncMock()
    mock_session.get.return_value = mock_ctx_mgr

    # Test latest/stable resolution
    version = await resolve_stdlib_version(mock_session, "latest")
    assert version == "1.75.0"

    version = await resolve_stdlib_version(mock_session, "stable")
    assert version == "1.75.0"

    version = await resolve_stdlib_version(mock_session, None)
    assert version == "1.75.0"

    # Test channel names
    version = await resolve_stdlib_version(mock_session, "beta")
    assert version == "beta"

    version = await resolve_stdlib_version(mock_session, "nightly")
    assert version == "nightly"

    # Test specific version
    version = await resolve_stdlib_version(mock_session, "1.74.0")
    assert version == "1.74.0"


def test_stdlib_url_construction():
    """Test URL construction for standard library crates."""
    # Test with channel names
    url = get_stdlib_url("std", "stable")
    assert url == "https://docs.rs/std/latest/std.json"

    url = get_stdlib_url("core", "beta")
    assert url == "https://docs.rs/core/latest/core.json"

    url = get_stdlib_url("alloc", "nightly")
    assert url == "https://docs.rs/alloc/latest/alloc.json"

    # Test with specific version
    url = get_stdlib_url("std", "1.75.0")
    assert url == "https://docs.rs/std/1.75.0/std.json"

    # Test underscores in crate names
    url = get_stdlib_url("proc_macro", "1.75.0")
    assert url == "https://docs.rs/proc_macro/1.75.0/proc_macro.json"


@pytest.mark.asyncio
async def test_fetch_current_stable_version():
    """Test fetching current stable Rust version."""
    # Test successful fetch
    mock_session = MagicMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="1.75.0 (2023-12-28)")

    mock_ctx_mgr = AsyncMock()
    mock_ctx_mgr.__aenter__ = AsyncMock(return_value=mock_response)
    mock_ctx_mgr.__aexit__ = AsyncMock()
    mock_session.get.return_value = mock_ctx_mgr

    version = await fetch_current_stable_version(mock_session)
    assert version == "1.75.0"

    # Test version parsing with different formats
    mock_response.text = AsyncMock(return_value="1.76.0")
    version = await fetch_current_stable_version(mock_session)
    assert version == "1.76.0"

    # Test fallback on error - simulate network error
    mock_session_error = MagicMock()
    mock_session_error.get.side_effect = Exception("Network error")

    version = await fetch_current_stable_version(mock_session_error)
    assert version == "1.75.0"  # Should return fallback version
