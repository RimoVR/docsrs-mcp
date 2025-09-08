import asyncio
import time

import pytest

from docsrs_mcp.models import PopularCrate
from docsrs_mcp.popular_crates import IngestionScheduler, PreIngestionWorker


class DummyManager:
    def __init__(self):
        self.calls = 0

    async def get_popular_crates_with_metadata(self, count: int | None = None):
        # Return a small, deterministic set that changes per call to ensure
        # the scheduler has fresh work on subsequent runs.
        self.calls += 1
        now = time.time()
        return [
            PopularCrate(
                name=f"crate_a_{self.calls}",
                downloads=100,
                description=None,
                version="1.0.0",
                last_updated=now,
            ),
            PopularCrate(
                name=f"crate_b_{self.calls}",
                downloads=50,
                description=None,
                version="1.0.0",
                last_updated=now,
            ),
        ]

    async def close(self):
        # Match PopularCratesManager.close signature used by worker cleanup
        return None


@pytest.mark.asyncio
async def test_workers_persist_and_scheduler_processes(monkeypatch):
    calls = {"ingest": 0}

    async def fake_fetch_current_stable_version(crate_name: str) -> str:
        return "1.0.0"

    async def fake_ingest_crate(crate_name: str, version: str) -> bool:
        # Pretend ingestion takes a tiny amount of time
        await asyncio.sleep(0.01)
        calls["ingest"] += 1
        return True

    # Patch network/IO heavy functions
    monkeypatch.setattr(
        "docsrs_mcp.popular_crates.fetch_current_stable_version",
        fake_fetch_current_stable_version,
    )
    monkeypatch.setattr(
        "docsrs_mcp.popular_crates.ingest_crate",
        fake_ingest_crate,
    )

    manager = DummyManager()
    worker = PreIngestionWorker(manager)

    # Start the worker (this enqueues an initial batch and spawns workers)
    await worker.start()

    # Allow some processing time for initial run
    await asyncio.sleep(0.1)

    # Ensure some items were processed
    processed_initial = worker.stats["success"] + worker.stats["failed"] + worker.stats["skipped"]
    assert processed_initial >= 1

    # Create a scheduler and run a scheduled ingestion twice
    scheduler = IngestionScheduler(manager, worker)
    # Run internal scheduling method directly to keep test fast
    await scheduler._schedule_ingestion()

    # Allow time for processing
    await asyncio.sleep(0.1)
    processed_after_first_schedule = (
        worker.stats["success"] + worker.stats["failed"] + worker.stats["skipped"]
    )
    # Stats may reset between runs; instead validate ingest call count increased
    assert calls["ingest"] >= 3

    # Second schedule to confirm workers did not exit after first drain
    await scheduler._schedule_ingestion()
    await asyncio.sleep(0.1)

    processed_after_second_schedule = (
        worker.stats["success"] + worker.stats["failed"] + worker.stats["skipped"]
    )
    assert calls["ingest"] >= 5

    # Sanity check that our fake ingest was used
    assert calls["ingest"] >= 2
