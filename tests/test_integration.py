"""Integration tests for end-to-end ingestion pipeline."""

import asyncio
import json
from unittest.mock import patch

import aiosqlite
import pytest
import zstandard

from docsrs_mcp.ingest import ingest_crate


@pytest.fixture
async def mock_crate_info():
    """Mock crate info response."""
    return {
        "name": "test-crate",
        "max_stable_version": "1.0.0",
        "description": "A test crate for integration testing",
        "repository": "https://github.com/test/test-crate",
        "documentation": "https://docs.rs/test-crate",
    }


@pytest.fixture
def sample_rustdoc_json():
    """Sample rustdoc JSON for testing."""
    return {
        "format_version": 32,
        "paths": {
            "0:1": {"path": ["test_crate"], "kind": "module"},
            "0:2": {"path": ["test_crate", "TestStruct"]},
            "0:3": {"path": ["test_crate", "test_function"]},
            "0:4": {"path": ["test_crate", "TestTrait"]},
        },
        "index": {
            "0:1": {
                "name": "test_crate",
                "kind": "module",
                "docs": "The test crate root module.",
            },
            "0:2": {
                "name": "TestStruct",
                "kind": "struct",
                "docs": "A test struct for demonstration.",
            },
            "0:3": {
                "name": "test_function",
                "kind": "function",
                "docs": "A test function that does something.",
            },
            "0:4": {
                "name": "TestTrait",
                "kind": "trait",
                "docs": "A test trait for implementing things.",
            },
        },
    }


@pytest.mark.asyncio
async def test_ingest_crate_uncompressed(
    mock_crate_info, sample_rustdoc_json, tmp_path
):
    """Test ingesting a crate with uncompressed rustdoc JSON."""
    # Mock the HTTP responses
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with patch("docsrs_mcp.ingest.resolve_version") as mock_resolve:
            mock_resolve.return_value = (
                "1.0.0",
                "https://docs.rs/test-crate/1.0.0/test_crate.json",
            )

            with patch("docsrs_mcp.ingest.download_rustdoc") as mock_download:
                json_content = json.dumps(sample_rustdoc_json)
                mock_download.return_value = (json_content.encode(), "test.json")

                with (
                    patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
                    patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
                ):
                    # Run ingestion
                    db_path = await ingest_crate("test-crate", "1.0.0")

                    # Verify database was created
                    assert db_path.exists()
                    assert db_path.name == "1.0.0.db"

                    # Check database contents
                    async with aiosqlite.connect(db_path) as db:
                        # Check crate metadata
                        cursor = await db.execute(
                            "SELECT name, version, description FROM crate_metadata"
                        )
                        row = await cursor.fetchone()
                        assert row[0] == "test-crate"
                        assert row[1] == "1.0.0"
                        assert row[2] == "A test crate for integration testing"

                        # Check embeddings were created
                        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
                        count = await cursor.fetchone()
                        assert count[0] == 4  # 4 items from rustdoc

                        # Check specific items
                        cursor = await db.execute(
                            "SELECT item_path, header FROM embeddings WHERE header LIKE '%TestStruct%'"
                        )
                        row = await cursor.fetchone()
                        assert row[0] == "test_crate::TestStruct"
                        assert row[1] == "struct TestStruct"


@pytest.mark.asyncio
async def test_ingest_crate_zst_compressed(
    mock_crate_info, sample_rustdoc_json, tmp_path
):
    """Test ingesting a crate with zstandard compressed rustdoc."""
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with patch("docsrs_mcp.ingest.resolve_version") as mock_resolve:
            mock_resolve.return_value = (
                "1.0.0",
                "https://docs.rs/test-crate/1.0.0/test_crate.json",
            )

            with patch("docsrs_mcp.ingest.download_rustdoc") as mock_download:
                # Compress the JSON
                json_content = json.dumps(sample_rustdoc_json)
                compressed = zstandard.compress(json_content.encode())
                mock_download.return_value = (compressed, "test.json.zst")

                with (
                    patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
                    patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
                ):
                    # Run ingestion
                    db_path = await ingest_crate("test-crate", "1.0.0")

                    # Verify it worked
                    assert db_path.exists()

                    async with aiosqlite.connect(db_path) as db:
                        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
                        count = await cursor.fetchone()
                        assert count[0] == 4  # All items processed


@pytest.mark.asyncio
async def test_ingest_crate_fallback_to_description(mock_crate_info, tmp_path):
    """Test fallback to description embedding when rustdoc fails."""
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with patch("docsrs_mcp.ingest.resolve_version") as mock_resolve:
            # Make resolve_version fail
            mock_resolve.side_effect = Exception("Rustdoc not available")

            with (
                patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
                patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
            ):
                # Ensure clean state - remove any existing database
                crate_dir = tmp_path / "test-crate"
                if crate_dir.exists():
                    import shutil

                    shutil.rmtree(crate_dir)

                # Run ingestion
                db_path = await ingest_crate("test-crate", "1.0.0")

                # Should still create database with description
                assert db_path.exists()

                async with aiosqlite.connect(db_path) as db:
                    # Check that description embedding was created
                    cursor = await db.execute(
                        "SELECT item_path, header, content FROM embeddings"
                    )
                    row = await cursor.fetchone()
                    # The item_path should be "crate" for the fallback case
                    assert row is not None, "No embeddings found in database"
                    assert row[0] == "crate", (
                        f"Expected item_path 'crate', got '{row[0]}'"
                    )
                    assert "test-crate" in row[1], (
                        f"Expected 'test-crate' in header, got '{row[1]}'"
                    )
                    assert row[2] == "A test crate for integration testing", (
                        f"Expected description content, got '{row[2]}'"
                    )


@pytest.mark.asyncio
async def test_ingest_crate_already_exists(mock_crate_info, tmp_path):
    """Test that re-ingesting returns existing database."""
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with (
            patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
            patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
        ):
            # First ingestion
            db_path1 = await ingest_crate("test-crate", "1.0.0")

            # Get modification time
            mtime1 = db_path1.stat().st_mtime

            # Second ingestion should return same path
            db_path2 = await ingest_crate("test-crate", "1.0.0")

            assert db_path1 == db_path2
            assert db_path2.stat().st_mtime == mtime1  # Not recreated


@pytest.mark.asyncio
async def test_ingest_crate_version_resolution(mock_crate_info, tmp_path):
    """Test version resolution from 'latest'."""
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with (
            patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
            patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
        ):
            # Request 'latest' version
            db_path = await ingest_crate("test-crate", "latest")

            # Should resolve to max_stable_version
            assert db_path.name == "1.0.0.db"


@pytest.mark.asyncio
async def test_ingest_crate_concurrent_lock(
    mock_crate_info, sample_rustdoc_json, tmp_path
):
    """Test that concurrent ingestion is prevented by locks."""
    ingestion_started = asyncio.Event()
    ingestion_can_continue = asyncio.Event()

    async def slow_fetch(*args, **kwargs):
        # Signal that first ingestion started
        ingestion_started.set()
        # Wait for signal to continue
        await ingestion_can_continue.wait()
        return mock_crate_info

    with patch("docsrs_mcp.ingest.fetch_crate_info", side_effect=slow_fetch):
        with patch("docsrs_mcp.ingest.resolve_version") as mock_resolve:
            mock_resolve.return_value = (
                "1.0.0",
                "https://docs.rs/test-crate/1.0.0/test_crate.json",
            )

            with patch("docsrs_mcp.ingest.download_rustdoc") as mock_download:
                json_content = json.dumps(sample_rustdoc_json)
                mock_download.return_value = (json_content.encode(), "test.json")

                with (
                    patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
                    patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
                ):
                    # Start first ingestion
                    task1 = asyncio.create_task(ingest_crate("test-crate", "1.0.0"))

                    # Wait for it to start
                    await ingestion_started.wait()

                    # Try second ingestion - should wait for lock
                    task2 = asyncio.create_task(ingest_crate("test-crate", "1.0.0"))

                    # Give it time to hit the lock
                    await asyncio.sleep(0.1)

                    # Second task should be blocked
                    assert not task2.done()

                    # Let first ingestion continue
                    ingestion_can_continue.set()

                    # Both should complete
                    db_path1 = await task1
                    db_path2 = await task2

                    assert db_path1 == db_path2


@pytest.mark.asyncio
async def test_ingest_crate_cache_eviction(mock_crate_info, tmp_path):
    """Test cache eviction during ingestion."""
    # Create existing cache files that exceed limit
    cache_dir = tmp_path

    # Create old cache file
    old_crate_dir = cache_dir / "old-crate"
    old_crate_dir.mkdir()
    old_db = old_crate_dir / "1.0.0.db"
    old_db.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with (
            patch("docsrs_mcp.ingest.CACHE_DIR", cache_dir),
            patch("docsrs_mcp.database.CACHE_DIR", cache_dir),
        ):
            with patch(
                "docsrs_mcp.ingest.CACHE_MAX_SIZE_BYTES", 1024 * 1024
            ):  # 1MB limit
                # Ingest new crate
                await ingest_crate("test-crate", "1.0.0")

                # Old cache should be evicted
                assert not old_db.exists()


@pytest.mark.asyncio
async def test_ingest_crate_malformed_json(mock_crate_info, tmp_path):
    """Test handling of malformed rustdoc JSON."""
    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
        mock_fetch.return_value = mock_crate_info

        with patch("docsrs_mcp.ingest.resolve_version") as mock_resolve:
            mock_resolve.return_value = (
                "1.0.0",
                "https://docs.rs/test-crate/1.0.0/test_crate.json",
            )

            with patch("docsrs_mcp.ingest.download_rustdoc") as mock_download:
                # Return malformed JSON
                mock_download.return_value = (b'{"invalid": json', "test.json")

                with (
                    patch("docsrs_mcp.ingest.CACHE_DIR", tmp_path),
                    patch("docsrs_mcp.database.CACHE_DIR", tmp_path),
                ):
                    # Should fall back to description
                    db_path = await ingest_crate("test-crate", "1.0.0")

                    assert db_path.exists()

                    async with aiosqlite.connect(db_path) as db:
                        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
                        count = await cursor.fetchone()
                        assert count[0] == 1  # Just the description
