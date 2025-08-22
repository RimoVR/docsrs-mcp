"""Service layer for ingestion-related operations."""

import asyncio
import logging
import os
from pathlib import Path

import aiohttp

from ..cargo import extract_crates_from_cargo, resolve_cargo_versions
from ..popular_crates import (
    _ingestion_scheduler,
    _pre_ingestion_worker,
    check_crate_exists,
    get_popular_crates_status,
    queue_for_ingestion,
    start_pre_ingestion,
)

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingestion-related operations."""

    async def start_pre_ingestion(
        self,
        count: int | None = None,
        concurrency: int | None = None,
        force: bool = False,
    ) -> dict:
        """Start background pre-ingestion of popular Rust crates.

        Args:
            count: Number of crates to pre-ingest
            concurrency: Number of parallel workers
            force: Force restart if already running

        Returns:
            Dictionary with operation status
        """
        # Check current state
        status = get_popular_crates_status()
        is_running = status.get("worker", {}).get("is_running", False) or status.get(
            "scheduler", {}
        ).get("is_running", False)

        # Handle force restart
        if is_running and not force:
            current_stats = status.get("worker", {}).get("stats", {})
            return {
                "status": "already_running",
                "message": (
                    f"Pre-ingestion already in progress. "
                    f"Processed {current_stats.get('processed', 0)}/{current_stats.get('total', 0)} crates. "
                    f"Use force=true to restart."
                ),
                "stats": current_stats,
            }

        # Apply configuration if provided
        if concurrency is not None:
            os.environ["PRE_INGEST_CONCURRENCY"] = str(concurrency)

        # Start pre-ingestion in background with force_start for MCP control
        asyncio.create_task(start_pre_ingestion(force_start=True))

        response_status = "restarted" if (is_running and force) else "started"

        return {
            "status": response_status,
            "message": (
                f"Pre-ingestion {response_status} successfully. "
                f"Processing {count or '100-500'} popular crates "
                f"with {concurrency or 3} concurrent workers. "
                f"Monitor progress via health endpoints."
            ),
            "stats": None,  # Stats not immediately available on start
        }

    async def control_pre_ingestion(self, action: str) -> dict:
        """Control the pre-ingestion worker.

        Args:
            action: Control action (pause/resume/stop)

        Returns:
            Dictionary with operation result
        """
        if not _pre_ingestion_worker:
            return {
                "status": "failed",
                "message": "Pre-ingestion worker not initialized. Start pre-ingestion first.",
                "worker_state": None,
            }

        success = False
        if action == "pause":
            success = await _pre_ingestion_worker.pause()
        elif action == "resume":
            success = await _pre_ingestion_worker.resume()
        elif action == "stop":
            success = await _pre_ingestion_worker.stop()

        # Get current stats
        current_stats = None
        if _ingestion_scheduler:
            current_stats = await _ingestion_scheduler.get_ingestion_stats()
        elif _pre_ingestion_worker:
            current_stats = _pre_ingestion_worker.get_ingestion_stats()

        return {
            "status": "success" if success else "no_change",
            "message": f"Worker {action} {'successful' if success else 'had no effect'}",
            "worker_state": str(_pre_ingestion_worker._state.value)
            if _pre_ingestion_worker
            else None,
            "current_stats": current_stats,
        }

    async def ingest_cargo_file(
        self,
        file_path: str,
        concurrency: int = 3,
        skip_existing: bool = True,
        resolve_versions: bool = False,
    ) -> dict:
        """Ingest crates from a Cargo.toml or Cargo.lock file.

        Args:
            file_path: Path to Cargo.toml or Cargo.lock
            concurrency: Number of parallel workers
            skip_existing: Skip already ingested crates
            resolve_versions: Resolve version specifications

        Returns:
            Dictionary with ingestion status
        """
        try:
            # Parse the Cargo file
            path = Path(file_path)
            crates = extract_crates_from_cargo(path)

            if not crates:
                return {
                    "status": "completed",
                    "message": f"No dependencies found in {path.name}",
                    "crates_found": 0,
                    "crates_queued": 0,
                    "crates_skipped": 0,
                }

            # Resolve version specifications if requested
            if resolve_versions:
                async with aiohttp.ClientSession() as session:
                    crates = await resolve_cargo_versions(crates, session, resolve=True)
                    logger.info(f"Resolved {len(crates)} crate versions")

            # Check which crates already exist
            crates_to_ingest = []
            crates_skipped = 0

            if skip_existing:
                for crate_spec in crates:
                    if not await check_crate_exists(crate_spec):
                        crates_to_ingest.append(crate_spec)
                    else:
                        crates_skipped += 1
            else:
                crates_to_ingest = crates

            # Queue for ingestion
            if crates_to_ingest:
                await queue_for_ingestion(crates_to_ingest, concurrency=concurrency)
                estimated_time = len(crates_to_ingest) * 2.0 / concurrency
            else:
                estimated_time = 0

            return {
                "status": "started" if crates_to_ingest else "completed",
                "message": f"Queued {len(crates_to_ingest)} crates from {path.name}",
                "crates_found": len(crates),
                "crates_queued": len(crates_to_ingest),
                "crates_skipped": crates_skipped,
                "estimated_time_seconds": estimated_time,
            }

        except Exception as e:
            logger.error(f"Failed to ingest Cargo file: {e}")
            return {
                "status": "failed",
                "message": str(e),
                "crates_found": 0,
                "crates_queued": 0,
                "crates_skipped": 0,
            }
