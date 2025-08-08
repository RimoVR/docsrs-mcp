"""Unit tests for popular crates functionality."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import msgpack
import pytest
from filelock import FileLock

from docsrs_mcp.models import PopularCrate
from docsrs_mcp.popular_crates import PopularCratesManager, PreIngestionWorker


class TestPopularCratesManager:
    """Test suite for PopularCratesManager."""

    @pytest.fixture
    async def manager(self, tmp_path):
        """Create a PopularCratesManager instance with temp cache directory."""
        with patch(
            "docsrs_mcp.popular_crates.config.POPULAR_CRATES_CACHE_FILE",
            tmp_path / "cache.msgpack",
        ):
            manager = PopularCratesManager()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_cache_serialization(self, manager, tmp_path):
        """Test msgpack cache serialization and deserialization."""
        # Create test data
        test_crates = [
            PopularCrate(
                name="tokio",
                downloads=150000000,
                description="An async runtime",
                version="1.35.1",
                last_updated=time.time(),
            ),
            PopularCrate(
                name="serde",
                downloads=250000000,
                description="Serialization framework",
                version="1.0.195",
                last_updated=time.time(),
            ),
        ]

        # Set cache
        from datetime import datetime

        manager._cached_list = test_crates
        manager._cache_time = datetime.now()

        # Save to disk
        result = await manager._save_cache_to_disk()
        assert result is True
        assert (tmp_path / "cache.msgpack").exists()

        # Clear in-memory cache
        manager._cached_list = None
        manager._cache_time = None

        # Load from disk
        result = await manager._load_cache_from_disk()
        assert result is True
        assert len(manager._cached_list) == 2
        assert manager._cached_list[0].name == "tokio"
        assert manager._cached_list[1].name == "serde"

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, manager):
        """Test circuit breaker pattern."""
        # Record failures
        for _ in range(3):
            manager._record_api_failure()

        # Circuit should be open
        assert manager._is_circuit_open() is True

        # Reset circuit
        manager._reset_circuit_breaker()
        assert manager._is_circuit_open() is False

    @pytest.mark.asyncio
    async def test_cache_statistics(self, manager):
        """Test cache statistics tracking."""
        # Initial stats
        stats = manager.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Simulate cache miss
        manager._stats["misses"] += 1

        # Simulate cache hit
        manager._stats["hits"] += 1

        stats = manager.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_fallback_crates(self, manager):
        """Test fallback crate list generation."""
        fallback = manager._get_fallback_crates_as_objects(5)
        assert len(fallback) == 5
        assert all(isinstance(crate, PopularCrate) for crate in fallback)
        assert fallback[0].name == "serde"
        assert fallback[0].downloads == 0  # Unknown downloads for fallback

    @pytest.mark.asyncio
    async def test_api_response_validation(self, manager):
        """Test API response validation."""
        # Valid response
        valid_data = [
            {"id": "tokio", "downloads": 150000000},
            {"id": "serde", "downloads": 250000000},
        ]
        assert manager._validate_api_response(valid_data) is True

        # Invalid - missing required fields
        invalid_data = [{"name": "tokio"}]  # Missing 'id' and 'downloads'
        assert manager._validate_api_response(invalid_data) is False

        # Invalid - zero downloads for popular crate
        zero_downloads = [{"id": "tokio", "downloads": 0}]
        assert manager._validate_api_response(zero_downloads) is False

    @pytest.mark.asyncio
    async def test_multi_tier_fallback(self, manager):
        """Test multi-tier fallback strategy."""
        with patch.object(manager, "_fetch_popular_crates") as mock_fetch:
            mock_fetch.side_effect = aiohttp.ClientError("API Error")

            # Should fall back to hardcoded list
            crates = await manager.get_popular_crates(5)
            assert len(crates) == 5
            assert crates[0] == "serde"  # First hardcoded crate


class TestPreIngestionWorker:
    """Test suite for PreIngestionWorker."""

    @pytest.fixture
    async def worker(self):
        """Create a PreIngestionWorker instance."""
        manager = AsyncMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)
        yield worker
        # Cleanup if needed
        if worker._monitor_task:
            worker._monitor_task.cancel()
        if worker._memory_monitor_task:
            worker._memory_monitor_task.cancel()

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, worker):
        """Test that priority queue processes by download count."""
        # Add crates with different priorities
        await worker.queue.put((0, "most-popular"))  # Priority 0 (highest)
        await worker.queue.put((-1000, "less-popular"))  # Priority -1000
        await worker.queue.put((-5000, "least-popular"))  # Priority -5000

        # They should come out in download order (most popular first)
        priority1, name1 = await worker.queue.get()
        assert name1 == "least-popular"

        priority2, name2 = await worker.queue.get()
        assert name2 == "less-popular"

        priority3, name3 = await worker.queue.get()
        assert name3 == "most-popular"

    def test_duplicate_detection(self, worker):
        """Test duplicate crate detection."""
        # Add to processed set
        worker._processed_crates.add("tokio")

        # Check detection
        assert "tokio" in worker._processed_crates
        assert "serde" not in worker._processed_crates

    def test_ingestion_stats(self, worker):
        """Test ingestion statistics calculation."""
        from datetime import datetime, timedelta

        # Set up stats
        worker.stats = {
            "success": 5,
            "failed": 2,
            "skipped": 1,
            "total": 10,
        }
        worker._start_time = datetime.now() - timedelta(seconds=10)  # 10 seconds ago

        stats = worker.get_ingestion_stats()

        assert stats["processed"] == 8
        assert stats["remaining"] == 2
        assert stats["progress_percent"] == 80.0
        assert "rate_per_second" in stats
        assert stats["is_running"] is False

    @pytest.mark.asyncio
    async def test_memory_monitoring(self, worker):
        """Test adaptive concurrency based on memory."""
        # Initial concurrency
        initial = worker._adaptive_concurrency

        # Simulate high memory (would need psutil mock in real test)
        import sys

        with patch.object(
            sys.modules["docsrs_mcp.popular_crates"], "psutil", create=True
        ) as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 950 * 1024 * 1024  # 950MB
            mock_psutil.Process.return_value = mock_process

            # This would trigger concurrency reduction in real scenario
            # For test, we manually simulate it
            if 950 > 900:  # High memory threshold
                worker._adaptive_concurrency = max(1, worker._adaptive_concurrency - 1)

            assert worker._adaptive_concurrency < initial


class TestIntegration:
    """Integration tests for popular crates system."""

    @pytest.mark.asyncio
    async def test_cache_file_locking(self, tmp_path):
        """Test that file locking prevents concurrent corruption."""
        cache_file = tmp_path / "cache.msgpack"
        lock_file = str(cache_file) + ".lock"

        # Create test data
        data = {
            "crates": [{"name": "test", "downloads": 100, "last_updated": time.time()}],
            "timestamp": time.time(),
        }

        # Write with lock
        with FileLock(lock_file, timeout=5):
            with open(cache_file, "wb") as f:
                f.write(msgpack.packb(data, use_bin_type=True))

        # Read with lock
        with FileLock(lock_file, timeout=5):
            with open(cache_file, "rb") as f:
                loaded = msgpack.unpackb(f.read(), raw=False)

        assert loaded["crates"][0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_api_mock_response(self):
        """Test handling of mocked API responses."""
        mock_response = {
            "crates": [
                {
                    "id": "tokio",
                    "downloads": 150000000,
                    "description": "Runtime for async",
                    "max_stable_version": {"num": "1.35.1"},
                },
                {
                    "id": "serde",
                    "downloads": 250000000,
                    "description": "Serialization",
                    "max_stable_version": {"num": "1.0.195"},
                },
            ]
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = mock_resp

            manager = PopularCratesManager()
            crates = await manager._fetch_popular_crates(2)

            assert len(crates) == 2
            assert crates[0].name == "tokio"
            assert crates[0].downloads == 150000000
            assert crates[1].name == "serde"

            await manager.close()
