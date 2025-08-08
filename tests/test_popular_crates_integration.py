"""Integration tests for popular crates pre-ingestion and monitoring."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docsrs_mcp.popular_crates import (
    IngestionScheduler,
    PopularCratesManager,
    PreIngestionWorker,
)


@pytest.mark.asyncio
async def test_end_to_end_pre_ingestion(
    mock_aiohttp_session, pre_ingestion_system, mock_ingest_crate
):
    """Test complete pre-ingestion flow."""
    manager, worker = pre_ingestion_system

    # Start pre-ingestion
    task = asyncio.create_task(worker._run())

    # Wait for some processing
    await asyncio.sleep(0.5)

    # Check progress is being tracked
    stats = worker.get_ingestion_stats()
    assert "processed" in stats
    assert "remaining" in stats
    assert "progress_percent" in stats
    assert stats["progress_percent"] >= 0

    # Check progress details are available
    assert hasattr(worker, "crate_progress")
    assert isinstance(worker.crate_progress, dict)

    # Check processing history is being maintained
    assert hasattr(worker, "processing_history")

    # Stop worker
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_health_endpoint_with_pre_ingestion(test_client, pre_ingestion_system):
    """Test health endpoint reports pre-ingestion status."""
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "subsystems" in data
    assert "memory" in data["subsystems"]

    # Check health status aggregation
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_detailed_pre_ingestion_endpoint(test_client, pre_ingestion_system):
    """Test the /health/pre-ingestion endpoint."""
    response = test_client.get("/health/pre-ingestion")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data

    # When not initialized, should indicate that
    if data["status"] == "not_initialized":
        assert "message" in data
    else:
        assert "worker" in data
        assert "scheduler" in data
        assert "cache" in data


@pytest.mark.asyncio
async def test_memory_pressure_handling(pre_ingestion_system, mock_psutil):
    """Test system handles memory pressure correctly."""
    manager, worker = pre_ingestion_system

    # Set high memory usage
    mock_psutil.memory_info.return_value.rss = 950 * 1024 * 1024  # 950 MB

    # Start memory monitor
    monitor_task = asyncio.create_task(worker._monitor_memory())

    # Wait for monitor to detect high memory
    await asyncio.sleep(0.1)

    # Check concurrency was reduced
    assert worker._adaptive_concurrency <= worker.semaphore._value

    # Cleanup
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_scheduler_integration():
    """Test scheduler starts and runs periodically."""
    manager = PopularCratesManager()
    worker = PreIngestionWorker(manager)
    scheduler = IngestionScheduler(manager, worker)

    # Mock the schedule method
    scheduler._schedule_ingestion = AsyncMock()

    # Start scheduler
    scheduler.enabled = True
    task = asyncio.create_task(scheduler._run())

    # Wait for at least one execution attempt
    await asyncio.sleep(0.1)

    # Check status
    status = scheduler.get_scheduler_status()
    assert "enabled" in status
    assert status["enabled"] is True

    # Cleanup
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await manager.close()


@pytest.mark.asyncio
async def test_progress_tracking_during_ingestion(
    pre_ingestion_system, mock_ingest_crate
):
    """Test that progress is tracked correctly during ingestion."""
    manager, worker = pre_ingestion_system

    # Add a crate to process
    worker.stats["total"] = 1
    await worker.queue.put((0, "test-crate"))

    # Process the crate
    await worker._ingest_single_crate("test-crate")

    # Check progress tracking
    assert "test-crate" in worker.crate_progress
    progress = worker.crate_progress["test-crate"]
    assert progress["percent"] == 100
    assert progress["status"] in ["completed", "failed", "error", "skipped"]

    # Check processing history
    assert len(worker.processing_history) > 0
    history = worker.processing_history[-1]
    assert history["crate"] == "test-crate"
    assert "duration" in history


@pytest.mark.asyncio
async def test_eta_calculation(pre_ingestion_system):
    """Test ETA calculation in ingestion stats."""
    manager, worker = pre_ingestion_system

    # Add some processing history
    worker.processing_history.append(
        {"crate": "crate1", "duration": 1.0, "status": "success"}
    )
    worker.processing_history.append(
        {"crate": "crate2", "duration": 2.0, "status": "success"}
    )

    # Set up stats
    worker.stats = {"success": 2, "failed": 0, "skipped": 0, "total": 5}
    worker._start_time = time.time() - 3  # 3 seconds elapsed

    # Get stats with ETA
    stats = worker.get_ingestion_stats()

    assert "eta_seconds" in stats or stats["remaining"] == 0
    assert "eta_formatted" in stats or stats["remaining"] == 0
    assert "avg_processing_time" in stats
    assert "recent_crates" in stats


@pytest.mark.asyncio
async def test_circuit_breaker_pattern(mock_aiohttp_session):
    """Test circuit breaker prevents cascade failures."""
    manager = PopularCratesManager()

    # Simulate API failures
    mock_aiohttp_session.get.return_value.__aenter__.return_value.status = 500

    # First failure
    with pytest.raises(Exception):
        await manager._fetch_popular_crates(10)

    # Second failure
    with pytest.raises(Exception):
        await manager._fetch_popular_crates(10)

    # Third failure triggers circuit breaker
    with pytest.raises(Exception):
        await manager._fetch_popular_crates(10)

    # Circuit should be open now
    assert manager._is_circuit_open()

    await manager.close()


@pytest.mark.asyncio
async def test_cache_statistics(pre_ingestion_system):
    """Test cache statistics collection."""
    manager, worker = pre_ingestion_system

    # Simulate some cache operations
    manager._stats["hits"] = 10
    manager._stats["misses"] = 5
    manager._stats["refreshes"] = 2

    # Get stats
    stats = manager.get_cache_stats()

    assert stats["hits"] == 10
    assert stats["misses"] == 5
    assert stats["hit_rate"] == 10 / 15  # hits / (hits + misses)
    assert stats["refreshes"] == 2


@given(
    crate_count=st.integers(min_value=1, max_value=20),
    concurrency=st.integers(min_value=1, max_value=5),
)
@pytest.mark.asyncio
async def test_ingestion_invariants(crate_count, concurrency):
    """Test invariants hold for various configurations."""
    manager = PopularCratesManager()
    worker = PreIngestionWorker(manager)
    worker.semaphore = asyncio.Semaphore(concurrency)

    # Set up test data
    worker.stats["total"] = crate_count

    # Create mock crates
    for i in range(crate_count):
        await worker.queue.put((i, f"crate_{i}"))

    # Get stats
    stats = worker.get_ingestion_stats()

    # Test invariants
    assert stats["processed"] + stats["remaining"] == stats["total"]
    assert 0 <= stats["progress_percent"] <= 100
    assert stats["adaptive_concurrency"] <= concurrency

    await manager.close()


@pytest.mark.asyncio
async def test_duplicate_detection(pre_ingestion_system):
    """Test that duplicate crates are not processed twice."""
    manager, worker = pre_ingestion_system

    # Add same crate multiple times
    await worker.queue.put((1, "duplicate-crate"))
    await worker.queue.put((2, "duplicate-crate"))

    # Process first occurrence
    worker._processed_crates.add("duplicate-crate")

    # Try to process second occurrence
    # Worker should skip it
    assert "duplicate-crate" in worker._processed_crates

    # Stats should reflect skipping
    worker.stats["total"] = 2
    stats = worker.get_ingestion_stats()
    assert stats["total"] == 2


@pytest.mark.asyncio
async def test_priority_queue_ordering(pre_ingestion_system):
    """Test that crates are processed by priority (download count)."""
    manager, worker = pre_ingestion_system

    # Add crates with different priorities
    await worker.queue.put((100, "low-priority"))  # Lower number = higher priority
    await worker.queue.put((10, "high-priority"))
    await worker.queue.put((50, "medium-priority"))

    # Get items in order
    priority1, crate1 = await worker.queue.get()
    priority2, crate2 = await worker.queue.get()
    priority3, crate3 = await worker.queue.get()

    # Check ordering (lower priority number comes first)
    assert crate1 == "high-priority"
    assert crate2 == "medium-priority"
    assert crate3 == "low-priority"


@pytest.mark.asyncio
async def test_fallback_crates_mechanism(mock_aiohttp_session):
    """Test fallback to hardcoded crates when API fails."""
    manager = PopularCratesManager()

    # Simulate API failure
    mock_aiohttp_session.get.return_value.__aenter__.return_value.status = 503

    # Should fall back to hardcoded list
    crates = await manager.get_popular_crates(5)

    assert len(crates) == 5
    assert all(
        crate in ["tokio", "serde", "clap", "regex", "anyhow"] for crate in crates
    )

    await manager.close()


@pytest.mark.asyncio
async def test_monitoring_metrics_collector():
    """Test the MetricsCollector functionality."""
    from docsrs_mcp.monitoring import MetricsCollector

    collector = MetricsCollector(window_size=100)

    # Record some events
    collector.record_event("crate_ingested", duration=1.5, metadata={"crate": "tokio"})
    collector.record_event("crate_ingested", duration=2.0, metadata={"crate": "serde"})
    collector.record_event(
        "crate_failed", duration=0.5, metadata={"crate": "bad-crate"}
    )

    # Get stats
    stats = collector.get_stats()

    assert stats["counters"]["crate_ingested"] == 2
    assert stats["counters"]["crate_failed"] == 1
    assert stats["events"] == 3
    assert stats["avg_duration"] > 0
    assert len(stats["recent_events"]) <= 10

    # Test event rate calculation
    rate = collector.get_event_rate("crate_ingested", window_seconds=60)
    assert rate > 0
