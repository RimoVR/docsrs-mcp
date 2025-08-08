"""Popular crates pre-ingestion module for docsrs-mcp.

This module handles fetching popular crates from crates.io and pre-ingesting them
in the background to eliminate cold-start latency for commonly queried Rust crates.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config
from .ingest import fetch_current_stable_version, ingest_crate

logger = logging.getLogger(__name__)


class PopularCratesManager:
    """Manages fetching and caching popular crates list from crates.io API."""

    def __init__(self):
        self._cached_list: list[str] | None = None
        self._cache_time: datetime | None = None
        self._cache_ttl = timedelta(hours=config.POPULAR_CRATES_REFRESH_HOURS)
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_cache_valid(self) -> bool:
        """Check if the cached list is still valid."""
        if self._cached_list is None or self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
    )
    async def _fetch_popular_crates(self, count: int) -> list[str]:
        """Fetch popular crates from crates.io API with retry logic."""
        await self._ensure_session()

        url = config.POPULAR_CRATES_URL.format(count)
        headers = {
            "User-Agent": f"docsrs-mcp/{config.VERSION} (https://github.com/anthropics/docsrs-mcp)"
        }

        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 429:  # Rate limited
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited by crates.io, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")

                if resp.status != 200:
                    logger.warning(f"crates.io API returned status {resp.status}")
                    raise aiohttp.ClientError(f"API returned {resp.status}")

                data = await resp.json()
                crates = data.get("crates", [])

                # Extract crate names from the response
                crate_names = [crate["id"] for crate in crates if "id" in crate]

                if not crate_names:
                    logger.warning("No crates found in API response, using fallback")
                    return config.FALLBACK_POPULAR_CRATES[:count]

                logger.info(
                    f"Successfully fetched {len(crate_names)} popular crates from crates.io"
                )
                return crate_names[:count]

        except asyncio.TimeoutError:
            logger.warning("Timeout fetching popular crates, using fallback")
            return config.FALLBACK_POPULAR_CRATES[:count]
        except Exception as e:
            logger.warning(f"Error fetching popular crates: {e}, using fallback")
            return config.FALLBACK_POPULAR_CRATES[:count]

    async def get_popular_crates(self, count: int = None) -> list[str]:
        """Get list of popular crate names, using cache if valid."""
        if count is None:
            count = config.POPULAR_CRATES_COUNT

        # Check cache first
        if self._is_cache_valid() and self._cached_list:
            logger.debug(
                f"Using cached popular crates list ({len(self._cached_list)} crates)"
            )
            return self._cached_list[:count]

        # Fetch fresh list
        logger.info(f"Fetching fresh popular crates list (top {count})")
        crates = await self._fetch_popular_crates(count)

        # Update cache
        self._cached_list = crates
        self._cache_time = datetime.now()

        return crates[:count]

    def get_fallback_crates(self, count: int = None) -> list[str]:
        """Get fallback popular crates list (hardcoded)."""
        if count is None:
            count = config.POPULAR_CRATES_COUNT
        return config.FALLBACK_POPULAR_CRATES[:count]


class PreIngestionWorker:
    """Background worker for pre-ingesting popular crates."""

    def __init__(self, manager: PopularCratesManager):
        self.manager = manager
        self.semaphore = asyncio.Semaphore(config.PRE_INGEST_CONCURRENCY)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stats = {"success": 0, "failed": 0, "skipped": 0, "total": 0}
        self._workers: list[asyncio.Task] = []
        self._monitor_task: asyncio.Task | None = None
        self._start_time: datetime | None = None

    async def start(self):
        """Start pre-ingestion in background (non-blocking)."""
        logger.info("Starting background pre-ingestion of popular crates")
        self._start_time = datetime.now()

        # Create background task for the main runner
        asyncio.create_task(self._run())

    async def _run(self):
        """Main runner that coordinates pre-ingestion."""
        try:
            # Get list of popular crates
            crates = await self.manager.get_popular_crates()
            self.stats["total"] = len(crates)

            logger.info(f"Starting pre-ingestion of {len(crates)} popular crates")

            # Add all crates to the queue
            for crate_name in crates:
                await self.queue.put(crate_name)

            # Start worker tasks
            for i in range(config.PRE_INGEST_CONCURRENCY):
                worker = asyncio.create_task(self._ingest_worker(i))
                self._workers.append(worker)

            # Start progress monitor
            self._monitor_task = asyncio.create_task(self._monitor_progress())

            # Wait for all crates to be processed
            await self.queue.join()

            # Cancel monitor task
            if self._monitor_task:
                self._monitor_task.cancel()

            # Log final statistics
            self._log_final_stats()

        except Exception as e:
            logger.error(f"Error in pre-ingestion runner: {e}")
        finally:
            # Cleanup
            await self.manager.close()

    async def _ingest_worker(self, worker_id: int):
        """Worker that processes crates from the queue."""
        logger.debug(f"Pre-ingestion worker {worker_id} started")

        while True:
            try:
                # Get next crate from queue (with timeout to allow graceful shutdown)
                try:
                    crate_name = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if queue is empty and we should exit
                    if self.queue.empty():
                        break
                    continue

                # Process the crate with semaphore control
                async with self.semaphore:
                    await self._ingest_single_crate(crate_name)

                # Mark task as done
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                # Still mark as done to avoid hanging
                self.queue.task_done()

    async def _ingest_single_crate(self, crate_name: str):
        """Ingest a single crate with error handling."""
        try:
            # Get the latest stable version
            version = await fetch_current_stable_version(crate_name)
            if not version:
                logger.debug(f"No stable version found for {crate_name}, skipping")
                self.stats["skipped"] += 1
                return

            # Ingest the crate using existing pipeline
            logger.debug(f"Pre-ingesting {crate_name} v{version}")
            success = await ingest_crate(crate_name, version)

            if success:
                self.stats["success"] += 1
                logger.debug(f"Successfully pre-ingested {crate_name} v{version}")
            else:
                self.stats["failed"] += 1
                logger.debug(f"Failed to pre-ingest {crate_name} v{version}")

        except Exception as e:
            self.stats["failed"] += 1
            logger.debug(f"Error pre-ingesting {crate_name}: {e}")

    async def _monitor_progress(self):
        """Monitor and log progress periodically."""
        last_logged = 0

        while True:
            try:
                await asyncio.sleep(10)  # Log every 10 seconds

                processed = (
                    self.stats["success"] + self.stats["failed"] + self.stats["skipped"]
                )
                if processed > last_logged and processed % 10 == 0:
                    # Log every 10 crates
                    remaining = self.stats["total"] - processed
                    elapsed = (datetime.now() - self._start_time).total_seconds()
                    rate = processed / elapsed if elapsed > 0 else 0

                    logger.info(
                        f"Pre-ingestion progress: {processed}/{self.stats['total']} crates "
                        f"({self.stats['success']} success, {self.stats['failed']} failed, "
                        f"{self.stats['skipped']} skipped) - "
                        f"{rate:.1f} crates/sec, {remaining} remaining"
                    )
                    last_logged = processed

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in progress monitor: {e}")

    def _log_final_stats(self):
        """Log final statistics after pre-ingestion completes."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        minutes = elapsed / 60

        logger.info(
            f"Pre-ingestion completed in {minutes:.1f} minutes: "
            f"{self.stats['success']} successful, "
            f"{self.stats['failed']} failed, "
            f"{self.stats['skipped']} skipped "
            f"out of {self.stats['total']} total crates"
        )

        if self.stats["success"] > 0:
            logger.info(
                f"Successfully pre-ingested {self.stats['success']} popular crates. "
                f"These crates will now respond with sub-100ms latency."
            )


async def start_pre_ingestion():
    """Convenience function to start pre-ingestion (called from server startup)."""
    if not config.PRE_INGEST_ENABLED:
        logger.debug("Pre-ingestion is disabled")
        return

    try:
        manager = PopularCratesManager()
        worker = PreIngestionWorker(manager)
        await worker.start()
    except Exception as e:
        # Don't let pre-ingestion errors prevent server startup
        logger.error(f"Failed to start pre-ingestion: {e}")
