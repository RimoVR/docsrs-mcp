"""
Tests for background ingestion stall fixes.

This module tests the comprehensive fixes applied to resolve the background
ingestion stall issue, including worker lifecycle, scheduler management,
crate spec parsing, and semaphore safety.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docsrs_mcp.popular_crates import (
    IngestionScheduler,
    PopularCratesManager,
    PreIngestionWorker,
    WorkerState,
    parse_crate_spec,
    queue_for_ingestion,
)


class TestCrateSpecParsing:
    """Test crate specification parsing for bulk ingestion support."""

    def test_parse_crate_spec_with_version(self):
        """Test parsing crate@version format."""
        name, version = parse_crate_spec("serde@1.0.219")
        assert name == "serde"
        assert version == "1.0.219"

    def test_parse_crate_spec_without_version(self):
        """Test parsing crate name without version."""
        name, version = parse_crate_spec("tokio")
        assert name == "tokio"
        assert version == "latest"

    def test_parse_crate_spec_latest(self):
        """Test parsing crate@latest format."""
        name, version = parse_crate_spec("reqwest@latest")
        assert name == "reqwest"
        assert version == "latest"

    def test_parse_crate_spec_complex_version(self):
        """Test parsing complex version strings."""
        name, version = parse_crate_spec("serde_json@1.0.0-alpha.1")
        assert name == "serde_json"
        assert version == "1.0.0-alpha.1"

    def test_parse_crate_spec_multiple_at_signs(self):
        """Test parsing with multiple @ signs (should split on first only)."""
        name, version = parse_crate_spec("weird@crate@1.0.0")
        assert name == "weird"
        assert version == "crate@1.0.0"


@pytest.mark.asyncio
class TestWorkerLifecycle:
    """Test worker lifecycle management and anti-premature-exit fixes."""

    async def test_worker_does_not_exit_on_empty_queue(self):
        """Test that workers don't exit when queue is temporarily empty."""
        manager = MagicMock(spec=PopularCratesManager)
        manager.get_popular_crates_with_metadata = AsyncMock(return_value=[])
        manager.close = AsyncMock()

        worker = PreIngestionWorker(manager)
        worker._state = WorkerState.RUNNING

        # Mock the queue to return TimeoutError then a real item
        queue_calls = 0

        async def mock_queue_get():
            nonlocal queue_calls
            queue_calls += 1
            if queue_calls <= 3:  # First few calls timeout
                raise asyncio.TimeoutError()
            else:  # Then return an item
                return (0, "test_crate")

        # Patch queue and ingest method
        with patch.object(worker.queue, "get", side_effect=mock_queue_get), patch.object(
            worker, "_ingest_single_crate", new_callable=AsyncMock
        ):
            # Start worker
            worker_task = asyncio.create_task(worker._ingest_worker(0))

            # Let it timeout a few times
            await asyncio.sleep(0.1)

            # Worker should still be alive (not exited due to timeout)
            assert not worker_task.done()

            # Stop the worker cleanly
            worker._state = WorkerState.STOPPING
            await asyncio.sleep(0.1)

            # Now it should exit
            assert worker_task.done()

    async def test_worker_exits_on_stopping_state(self):
        """Test that workers properly exit when STOPPING state is set."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)
        worker._state = WorkerState.RUNNING

        # Start worker
        worker_task = asyncio.create_task(worker._ingest_worker(0))
        await asyncio.sleep(0.05)  # Let worker start

        # Set stopping state
        worker._state = WorkerState.STOPPING
        await asyncio.sleep(0.1)  # Give it time to exit

        assert worker_task.done()

    async def test_spawn_workers_creates_correct_number(self):
        """Test that _spawn_workers creates the right number of workers."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)
        worker._adaptive_concurrency = 3

        # Initially no workers
        assert len(worker._workers) == 0

        # Spawn workers
        worker._spawn_workers()

        # Should have 3 workers
        assert len(worker._workers) == 3

        # All should be running
        for w in worker._workers:
            assert not w.done()

        # Cleanup
        for w in worker._workers:
            w.cancel()
        await asyncio.gather(*worker._workers, return_exceptions=True)

    async def test_spawn_workers_cleans_finished_tasks(self):
        """Test that _spawn_workers cleans up finished worker tasks."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)
        worker._adaptive_concurrency = 2

        # Create a finished task
        finished_task = asyncio.create_task(asyncio.sleep(0))
        await finished_task
        worker._workers = [finished_task]

        # Spawn workers
        worker._spawn_workers()

        # Should have 2 new workers (finished one removed)
        assert len(worker._workers) == 2
        for w in worker._workers:
            assert not w.done()

        # Cleanup
        for w in worker._workers:
            w.cancel()
        await asyncio.gather(*worker._workers, return_exceptions=True)


@pytest.mark.asyncio
class TestSchedulerWorkerManagement:
    """Test scheduler worker management fixes."""

    async def test_scheduler_spawns_workers_when_none_running(self):
        """Test that scheduler spawns workers when none are running."""
        manager = MagicMock(spec=PopularCratesManager)
        manager.get_popular_crates_with_metadata = AsyncMock(
            return_value=[
                MagicMock(name="tokio", downloads=1000),
                MagicMock(name="serde", downloads=2000),
            ]
        )

        worker = PreIngestionWorker(manager)
        scheduler = IngestionScheduler(manager, worker)

        # No workers initially
        assert len(worker._workers) == 0

        # Mock _should_schedule to return True
        scheduler._should_schedule = AsyncMock(return_value=True)

        # Mock queue operations
        worker.queue.put = AsyncMock()
        worker.queue.empty = MagicMock(return_value=False)
        worker.queue.join = AsyncMock()

        # Mock _spawn_workers to track calls
        original_spawn = worker._spawn_workers
        worker._spawn_workers = MagicMock(side_effect=original_spawn)

        # Run scheduler
        await scheduler._schedule_ingestion()

        # Should have called _spawn_workers
        worker._spawn_workers.assert_called_once()

    async def test_scheduler_skips_spawn_when_workers_running(self):
        """Test that scheduler doesn't spawn workers when they're already running."""
        manager = MagicMock(spec=PopularCratesManager)
        manager.get_popular_crates_with_metadata = AsyncMock(
            return_value=[MagicMock(name="tokio", downloads=1000)]
        )

        worker = PreIngestionWorker(manager)
        scheduler = IngestionScheduler(manager, worker)

        # Create a running worker task
        running_task = asyncio.create_task(asyncio.sleep(10))  # Long-running
        worker._workers = [running_task]

        # Mock required methods
        scheduler._should_schedule = AsyncMock(return_value=True)
        worker.queue.put = AsyncMock()
        worker.queue.empty = MagicMock(return_value=False)
        worker.queue.join = AsyncMock()
        worker._spawn_workers = MagicMock()

        try:
            # Run scheduler
            await scheduler._schedule_ingestion()

            # Should NOT have called _spawn_workers
            worker._spawn_workers.assert_not_called()
        finally:
            # Cleanup
            running_task.cancel()
            try:
                await running_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
class TestSemaphoreSafety:
    """Test semaphore resizing safety fixes."""

    async def test_semaphore_update_under_lock(self):
        """Test that semaphore updates are protected by lock."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)

        # Simulate concurrent access
        original_semaphore = worker.semaphore
        lock_acquired = False

        async def mock_memory_monitor():
            """Simulate memory monitor updating semaphore."""
            nonlocal lock_acquired
            async with worker._semaphore_lock:
                lock_acquired = True
                worker.semaphore = asyncio.Semaphore(1)  # Update semaphore

        async def mock_worker():
            """Simulate worker acquiring semaphore."""
            # Should get snapshot under lock
            async with worker._semaphore_lock:
                semaphore = worker.semaphore
            async with semaphore:
                await asyncio.sleep(0.1)  # Simulate work

        # Run both concurrently
        monitor_task = asyncio.create_task(mock_memory_monitor())
        worker_task = asyncio.create_task(mock_worker())

        await asyncio.gather(monitor_task, worker_task)

        # Lock should have been acquired
        assert lock_acquired
        # Semaphore should have been updated
        assert worker.semaphore != original_semaphore


@pytest.mark.asyncio  
class TestBulkIngestionFlow:
    """Test bulk ingestion with crate@version parsing."""

    async def test_bulk_ingestion_parses_crate_specs(self):
        """Test that bulk ingestion properly parses crate@version specs."""
        # Mock the global worker creation
        with patch("docsrs_mcp.popular_crates._pre_ingestion_worker", None), patch(
            "docsrs_mcp.popular_crates._popular_crates_manager", None
        ):
            # Mock manager and worker
            mock_manager = MagicMock(spec=PopularCratesManager)
            mock_worker = MagicMock(spec=PreIngestionWorker)
            mock_worker.queue = MagicMock()
            mock_worker.queue.put = AsyncMock()
            mock_worker._task_refs = set()
            mock_worker.start = AsyncMock()

            with patch(
                "docsrs_mcp.popular_crates.PopularCratesManager",
                return_value=mock_manager,
            ), patch(
                "docsrs_mcp.popular_crates.PreIngestionWorker",
                return_value=mock_worker,
            ), patch("asyncio.create_task") as mock_create_task:
                # Setup create_task mock
                mock_task = MagicMock()
                mock_create_task.return_value = mock_task

                # Test crate specs with versions
                crate_specs = ["serde@1.0.219", "tokio@1.0.0", "reqwest@latest"]

                await queue_for_ingestion(crate_specs, concurrency=3)

                # Should have queued each crate spec
                assert mock_worker.queue.put.call_count == 3

                # Check that each call used the full crate spec (not just name)
                queued_items = [call[0][1] for call in mock_worker.queue.put.call_args_list]
                assert "serde@1.0.219" in queued_items
                assert "tokio@1.0.0" in queued_items
                assert "reqwest@latest" in queued_items

    async def test_ingest_single_crate_with_version_spec(self):
        """Test that _ingest_single_crate handles crate@version specs."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)

        # Mock the ingest_crate function
        with patch("docsrs_mcp.popular_crates.ingest_crate", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = True

            # Test with version spec
            await worker._ingest_single_crate("serde@1.0.219")

            # Should have called ingest_crate with parsed name and version
            mock_ingest.assert_called_once_with("serde", "1.0.219")

            # Should have successful stats
            assert worker.stats["success"] == 1
            assert worker.stats["failed"] == 0

    async def test_ingest_single_crate_without_version_spec(self):
        """Test that _ingest_single_crate handles crate names without versions."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)

        # Mock version fetching and ingestion
        with patch(
            "docsrs_mcp.popular_crates.fetch_current_stable_version",
            new_callable=AsyncMock,
        ) as mock_fetch_version, patch(
            "docsrs_mcp.popular_crates.ingest_crate", new_callable=AsyncMock
        ) as mock_ingest:
            mock_fetch_version.return_value = "1.0.0"
            mock_ingest.return_value = True

            # Test with just crate name
            await worker._ingest_single_crate("tokio")

            # Should have fetched version
            mock_fetch_version.assert_called_once_with("tokio")

            # Should have called ingest_crate with fetched version
            mock_ingest.assert_called_once_with("tokio", "1.0.0")

            # Should have successful stats
            assert worker.stats["success"] == 1


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests for common ingestion scenarios."""

    async def test_scheduler_with_worker_restart_scenario(self):
        """Test scheduler behavior when workers die and need restart."""
        manager = MagicMock(spec=PopularCratesManager)
        manager.get_popular_crates_with_metadata = AsyncMock(
            return_value=[MagicMock(name="serde", downloads=1000)]
        )

        worker = PreIngestionWorker(manager)
        scheduler = IngestionScheduler(manager, worker)

        # Mock required methods
        scheduler._should_schedule = AsyncMock(return_value=True)
        worker.queue.put = AsyncMock()
        worker.queue.empty = MagicMock(return_value=False)
        worker.queue.join = AsyncMock()

        # Simulate finished/dead workers
        dead_task1 = asyncio.create_task(asyncio.sleep(0))
        dead_task2 = asyncio.create_task(asyncio.sleep(0))
        await asyncio.gather(dead_task1, dead_task2)  # Ensure they're finished
        worker._workers = [dead_task1, dead_task2]

        # Track spawning
        spawn_called = False
        original_spawn = worker._spawn_workers

        def track_spawn():
            nonlocal spawn_called
            spawn_called = True
            return original_spawn()

        worker._spawn_workers = track_spawn

        # Run scheduler
        await scheduler._schedule_ingestion()

        # Should have detected dead workers and spawned new ones
        assert spawn_called

    async def test_mixed_crate_specs_in_queue(self):
        """Test handling mixed crate specs (with and without versions)."""
        manager = MagicMock(spec=PopularCratesManager)
        worker = PreIngestionWorker(manager)

        crate_specs = [
            "serde@1.0.219",  # With version
            "tokio",  # Without version
            "reqwest@latest",  # With 'latest'
        ]

        with patch(
            "docsrs_mcp.popular_crates.fetch_current_stable_version",
            new_callable=AsyncMock,
        ) as mock_fetch, patch(
            "docsrs_mcp.popular_crates.ingest_crate", new_callable=AsyncMock
        ) as mock_ingest:
            mock_fetch.return_value = "1.0.0"  # For tokio
            mock_ingest.return_value = True

            # Process each spec
            for spec in crate_specs:
                await worker._ingest_single_crate(spec)

            # Check calls
            expected_calls = [
                (("serde", "1.0.219"),),  # Direct version
                (("tokio", "1.0.0"),),  # Fetched version
                (("reqwest", "1.0.0"),),  # latest -> fetch
            ]

            assert mock_ingest.call_count == 3
            for i, expected in enumerate(expected_calls):
                actual = mock_ingest.call_args_list[i][0]
                assert actual == expected[0]


if __name__ == "__main__":
    pytest.main([__file__])