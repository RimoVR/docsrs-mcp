"""Popular crates pre-ingestion module for docsrs-mcp.

This module handles fetching popular crates from crates.io and pre-ingesting them
in the background to eliminate cold-start latency for commonly queried Rust crates.
"""

import asyncio
import logging
import random
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import msgpack
from filelock import FileLock
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import config
from .ingest import ingest_crate
from .models import PopularCrate

logger = logging.getLogger(__name__)


async def fetch_current_stable_version(crate_name: str) -> str | None:
    """Fetch the current stable version of a crate from crates.io API."""
    try:
        url = f"https://crates.io/api/v1/crates/{crate_name}"
        headers = {
            "User-Agent": f"docsrs-mcp/{config.VERSION} (https://github.com/anthropics/docsrs-mcp)"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.debug(
                        f"Failed to fetch crate info for {crate_name}: HTTP {resp.status}"
                    )
                    return None

                data = await resp.json()
                crate_data = data.get("crate", {})

                # Try different version fields
                version = (
                    crate_data.get("max_stable_version")
                    or crate_data.get("max_version")
                    or crate_data.get("newest_version")
                )

                return version
    except Exception as e:
        logger.debug(f"Error fetching version for {crate_name}: {e}")
        return None


class PopularCratesManager:
    """Manages fetching and caching popular crates list from crates.io API."""

    def __init__(self):
        self._cached_list: list[PopularCrate] | None = None
        self._cache_time: datetime | None = None
        self._cache_ttl = timedelta(hours=config.POPULAR_CRATES_REFRESH_HOURS)
        self._session: aiohttp.ClientSession | None = None
        self._cache_file = config.POPULAR_CRATES_CACHE_FILE
        self._cache_lock = FileLock(str(self._cache_file) + ".lock")

        # Cache statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "refreshes": 0,
            "api_failures": 0,
            "last_refresh": None,
        }

        # Circuit breaker state
        self._circuit_breaker = {
            "failures": 0,
            "last_failure": None,
            "cooldown_until": None,
            "max_failures": 3,
            "cooldown_duration": 300,  # 5 minutes
        }

        # Load persistent cache on initialization
        asyncio.create_task(self._load_cache_from_disk())

    async def _ensure_session(self):
        """Ensure aiohttp session is created with connection pooling."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=config.HTTP_TIMEOUT)
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool limit
                limit_per_host=10,  # Per-host connection limit
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_cache_valid(self) -> bool:
        """Check if the cached list is still valid."""
        if self._cached_list is None or self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def _should_refresh(self) -> bool:
        """Check if cache should be refreshed (at 75% of TTL)."""
        if self._cached_list is None or self._cache_time is None:
            return True
        elapsed = datetime.now() - self._cache_time
        threshold = self._cache_ttl * config.POPULAR_CRATES_REFRESH_THRESHOLD
        return elapsed >= threshold

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (in cooldown)."""
        if self._circuit_breaker["cooldown_until"] is None:
            return False
        return time.time() < self._circuit_breaker["cooldown_until"]

    def _record_api_failure(self):
        """Record an API failure for circuit breaker."""
        self._circuit_breaker["failures"] += 1
        self._circuit_breaker["last_failure"] = time.time()
        self._stats["api_failures"] += 1

        if self._circuit_breaker["failures"] >= self._circuit_breaker["max_failures"]:
            self._circuit_breaker["cooldown_until"] = (
                time.time() + self._circuit_breaker["cooldown_duration"]
            )
            logger.warning(
                f"Circuit breaker opened after {self._circuit_breaker['failures']} failures, "
                f"cooldown for {self._circuit_breaker['cooldown_duration']}s"
            )

    def _reset_circuit_breaker(self):
        """Reset circuit breaker after successful API call."""
        self._circuit_breaker["failures"] = 0
        self._circuit_breaker["cooldown_until"] = None

    async def _load_cache_from_disk(self) -> bool:
        """Load cached data from disk using msgpack."""
        try:
            if not self._cache_file.exists():
                logger.debug("No cache file found, starting fresh")
                return False

            with self._cache_lock.acquire(timeout=5):
                with open(self._cache_file, "rb") as f:
                    data = msgpack.unpackb(f.read(), raw=False)

                # Convert to PopularCrate objects
                self._cached_list = [
                    PopularCrate(**crate_data) for crate_data in data["crates"]
                ]
                self._cache_time = datetime.fromtimestamp(data["timestamp"])

                # Restore statistics if available
                if "stats" in data:
                    self._stats.update(data["stats"])

                logger.info(
                    f"Loaded {len(self._cached_list)} popular crates from cache "
                    f"(age: {datetime.now() - self._cache_time})"
                )
                return True

        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            return False

    async def _save_cache_to_disk(self) -> bool:
        """Save cached data to disk using msgpack with atomic write."""
        if not self._cached_list:
            return False

        try:
            # Prepare data for serialization
            data = {
                "crates": [crate.model_dump() for crate in self._cached_list],
                "timestamp": self._cache_time.timestamp(),
                "stats": self._stats,
                "version": config.VERSION,
            }

            # Atomic write: temp file + rename
            with self._cache_lock.acquire(timeout=5):
                # Write to temp file first
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self._cache_file.parent,
                    delete=False,
                ) as tmp_file:
                    tmp_file.write(msgpack.packb(data, use_bin_type=True))
                    tmp_path = Path(tmp_file.name)

                # Atomic rename
                tmp_path.replace(self._cache_file)

            logger.debug(f"Saved {len(self._cached_list)} crates to cache file")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
    )
    async def _fetch_popular_crates(self, count: int) -> list[PopularCrate]:
        """Fetch popular crates from crates.io API with enhanced metadata."""
        # Check circuit breaker first
        if self._is_circuit_open():
            logger.debug("Circuit breaker is open, skipping API call")
            raise aiohttp.ClientError("Circuit breaker open")

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
                    self._record_api_failure()
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")

                if resp.status != 200:
                    logger.warning(f"crates.io API returned status {resp.status}")
                    self._record_api_failure()
                    raise aiohttp.ClientError(f"API returned {resp.status}")

                data = await resp.json()
                crates_data = data.get("crates", [])

                if not crates_data:
                    logger.warning("No crates found in API response")
                    self._record_api_failure()
                    return self._get_fallback_crates_as_objects(count)

                # Validate response structure
                if not self._validate_api_response(crates_data):
                    logger.warning("API response validation failed")
                    self._record_api_failure()
                    return self._get_fallback_crates_as_objects(count)

                # Extract enhanced metadata
                popular_crates = []
                current_time = time.time()

                for crate_data in crates_data[:count]:
                    try:
                        popular_crate = PopularCrate(
                            name=crate_data["id"],
                            downloads=crate_data.get("downloads", 0),
                            description=crate_data.get("description"),
                            version=crate_data.get("max_stable_version", {}).get("num"),
                            last_updated=current_time,
                        )
                        popular_crates.append(popular_crate)
                    except Exception as e:
                        logger.debug(
                            f"Failed to parse crate {crate_data.get('id')}: {e}"
                        )
                        continue

                if not popular_crates:
                    logger.warning("Failed to parse any crates from API response")
                    self._record_api_failure()
                    return self._get_fallback_crates_as_objects(count)

                # Success - reset circuit breaker
                self._reset_circuit_breaker()
                logger.info(
                    f"Successfully fetched {len(popular_crates)} popular crates with metadata"
                )
                return popular_crates

        except asyncio.TimeoutError:
            logger.warning("Timeout fetching popular crates")
            self._record_api_failure()
            raise
        except Exception as e:
            logger.warning(f"Error fetching popular crates: {e}")
            self._record_api_failure()
            raise

    def _validate_api_response(self, crates_data: list[dict]) -> bool:
        """Validate API response for anomalies."""
        if not crates_data:
            return False

        # Check for required fields in first few crates
        required_fields = ["id", "downloads"]
        for crate in crates_data[:5]:
            if not all(field in crate for field in required_fields):
                return False

            # Check for anomalies (e.g., zero downloads for popular crates)
            if crate.get("downloads", 0) == 0:
                logger.warning(
                    f"Anomaly detected: popular crate {crate.get('id')} has 0 downloads"
                )
                return False

        return True

    def _get_fallback_crates_as_objects(self, count: int) -> list[PopularCrate]:
        """Convert fallback crate names to PopularCrate objects."""
        current_time = time.time()
        return [
            PopularCrate(
                name=name,
                downloads=0,  # Unknown downloads for fallback crates
                description=None,
                version=None,
                last_updated=current_time,
            )
            for name in config.FALLBACK_POPULAR_CRATES[:count]
        ]

    async def get_popular_crates(self, count: int = None) -> list[str]:
        """Get list of popular crate names with multi-tier fallback strategy."""
        if count is None:
            count = config.POPULAR_CRATES_COUNT

        # Check if we should trigger background refresh (at 75% TTL)
        if self._should_refresh() and self._cached_list:
            # Trigger background refresh but serve stale data (stale-while-revalidate)
            asyncio.create_task(self._background_refresh(count))
            logger.debug("Triggered background refresh, serving cached data")

        # Multi-tier fallback strategy
        # Tier 1: Valid in-memory cache
        if self._is_cache_valid() and self._cached_list:
            self._stats["hits"] += 1
            logger.debug(
                f"Cache hit: serving {len(self._cached_list)} crates from memory"
            )
            return [crate.name for crate in self._cached_list[:count]]

        self._stats["misses"] += 1

        # Tier 2: Try to fetch fresh data from API
        try:
            logger.info(f"Cache miss: fetching fresh list (top {count})")
            crates = await self._fetch_popular_crates(count)

            # Update cache
            self._cached_list = crates
            self._cache_time = datetime.now()
            self._stats["refreshes"] += 1
            self._stats["last_refresh"] = self._cache_time.isoformat()

            # Save to disk asynchronously
            asyncio.create_task(self._save_cache_to_disk())

            return [crate.name for crate in crates[:count]]

        except Exception as e:
            logger.warning(f"Failed to fetch from API: {e}")

            # Tier 3: Fall back to disk cache (even if expired)
            if self._cached_list:
                logger.info("Using expired cache from memory")
                return [crate.name for crate in self._cached_list[:count]]

            # Tier 4: Try to load from disk
            if await self._load_cache_from_disk():
                logger.info("Loaded cache from disk as fallback")
                return [crate.name for crate in self._cached_list[:count]]

            # Tier 5: Ultimate fallback to hardcoded list
            logger.warning("Using hardcoded fallback list")
            fallback = self._get_fallback_crates_as_objects(count)
            return [crate.name for crate in fallback]

    async def _background_refresh(self, count: int):
        """Background task to refresh cache without blocking."""
        try:
            logger.debug("Starting background cache refresh")
            crates = await self._fetch_popular_crates(count)

            # Update cache
            self._cached_list = crates
            self._cache_time = datetime.now()
            self._stats["refreshes"] += 1
            self._stats["last_refresh"] = self._cache_time.isoformat()

            # Save to disk
            await self._save_cache_to_disk()

            logger.info("Background cache refresh completed successfully")

        except Exception as e:
            logger.warning(f"Background refresh failed: {e}")

    def get_fallback_crates(self, count: int = None) -> list[str]:
        """Get fallback popular crates list (hardcoded)."""
        if count is None:
            count = config.POPULAR_CRATES_COUNT
        return config.FALLBACK_POPULAR_CRATES[:count]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        stats = self._stats.copy()

        # Add current cache state
        stats["cache_valid"] = self._is_cache_valid()
        stats["cache_age_hours"] = None
        if self._cache_time:
            age = datetime.now() - self._cache_time
            stats["cache_age_hours"] = age.total_seconds() / 3600

        stats["cached_crates_count"] = (
            len(self._cached_list) if self._cached_list else 0
        )
        stats["circuit_breaker_open"] = self._is_circuit_open()

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (
            (stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        return stats

    async def get_popular_crates_with_metadata(
        self, count: int = None
    ) -> list[PopularCrate]:
        """Get popular crates with full metadata (for pre-ingestion priority)."""
        if count is None:
            count = config.POPULAR_CRATES_COUNT

        # Ensure cache is populated
        await self.get_popular_crates(count)

        if self._cached_list:
            return self._cached_list[:count]

        # Fallback
        return self._get_fallback_crates_as_objects(count)


class PreIngestionWorker:
    """Background worker for pre-ingesting popular crates with priority processing."""

    def __init__(self, manager: PopularCratesManager):
        self.manager = manager
        self.semaphore = asyncio.Semaphore(config.PRE_INGEST_CONCURRENCY)
        # Use PriorityQueue for processing most downloaded crates first
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.stats = {"success": 0, "failed": 0, "skipped": 0, "total": 0}
        self._workers: list[asyncio.Task] = []
        self._monitor_task: asyncio.Task | None = None
        self._start_time: datetime | None = None
        self._processed_crates: set[str] = set()  # For duplicate detection
        self._memory_monitor_task: asyncio.Task | None = None
        self._adaptive_concurrency = config.PRE_INGEST_CONCURRENCY
        # Progress tracking enhancements
        self.crate_progress = {}  # {crate_name: {"status": str, "started": float, "percent": int}}
        self.processing_history = deque(
            maxlen=100
        )  # Rolling window of completed crates
        self.current_crate = None  # Currently processing crate name

    async def start(self):
        """Start pre-ingestion in background (non-blocking)."""
        logger.info("Starting background pre-ingestion of popular crates")
        self._start_time = datetime.now()

        # Create background task for the main runner
        asyncio.create_task(self._run())

    async def _run(self):
        """Main runner that coordinates pre-ingestion with priority processing."""
        try:
            # Get list of popular crates with metadata for priority processing
            crates_with_metadata = await self.manager.get_popular_crates_with_metadata()
            self.stats["total"] = len(crates_with_metadata)

            logger.info(
                f"Starting pre-ingestion of {len(crates_with_metadata)} popular crates"
            )

            # Add all crates to priority queue (negative downloads for max-heap behavior)
            for crate in crates_with_metadata:
                # Skip if already processed (duplicate detection)
                if crate.name in self._processed_crates:
                    logger.debug(f"Skipping duplicate crate: {crate.name}")
                    self.stats["skipped"] += 1
                    continue

                # Priority based on downloads (negative for max-heap)
                priority = -crate.downloads if crate.downloads > 0 else 0
                await self.queue.put((priority, crate.name))

            # Start worker tasks with adaptive concurrency
            for i in range(self._adaptive_concurrency):
                worker = asyncio.create_task(self._ingest_worker(i))
                self._workers.append(worker)

            # Start progress monitor
            self._monitor_task = asyncio.create_task(self._monitor_progress())

            # Start memory monitor
            self._memory_monitor_task = asyncio.create_task(self._monitor_memory())

            # Wait for all crates to be processed
            await self.queue.join()

            # Cancel monitor tasks
            if self._monitor_task:
                self._monitor_task.cancel()
            if self._memory_monitor_task:
                self._memory_monitor_task.cancel()

            # Log final statistics
            self._log_final_stats()

        except Exception as e:
            logger.error(f"Error in pre-ingestion runner: {e}")
        finally:
            # Cleanup
            await self.manager.close()

    async def _ingest_worker(self, worker_id: int):
        """Worker that processes crates from priority queue."""
        logger.debug(f"Pre-ingestion worker {worker_id} started")

        while True:
            try:
                # Get next crate from priority queue (with timeout for graceful shutdown)
                try:
                    priority, crate_name = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check if queue is empty and we should exit
                    if self.queue.empty():
                        break
                    continue

                # Skip if already processed (double-check for duplicates)
                if crate_name in self._processed_crates:
                    logger.debug(f"Worker {worker_id}: skipping duplicate {crate_name}")
                    self.stats["skipped"] += 1
                    self.queue.task_done()
                    continue

                # Mark as processed to prevent duplicates
                self._processed_crates.add(crate_name)

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
        """Ingest a single crate with error handling and progress tracking."""
        start_time = time.time()
        self.current_crate = crate_name

        # Initialize progress tracking for this crate
        self.crate_progress[crate_name] = {
            "status": "resolving",
            "started": start_time,
            "percent": 0,
        }

        try:
            # Get the latest stable version (0-10% progress)
            version = await fetch_current_stable_version(crate_name)
            if not version:
                logger.debug(f"No stable version found for {crate_name}, skipping")
                self.stats["skipped"] += 1
                self.crate_progress[crate_name]["status"] = "skipped"
                self.crate_progress[crate_name]["percent"] = 100
                return

            # Update progress: downloading (10-30%)
            self.crate_progress[crate_name]["status"] = "downloading"
            self.crate_progress[crate_name]["percent"] = 10

            # Ingest the crate using existing pipeline
            logger.debug(f"Pre-ingesting {crate_name} v{version}")

            # Note: We can't track internal progress of ingest_crate without modifying it
            # So we'll mark it as processing (30-90%)
            self.crate_progress[crate_name]["status"] = "processing"
            self.crate_progress[crate_name]["percent"] = 30

            success = await ingest_crate(crate_name, version)

            # Mark as complete (100%)
            self.crate_progress[crate_name]["percent"] = 100

            if success:
                self.stats["success"] += 1
                self.crate_progress[crate_name]["status"] = "completed"
                logger.debug(f"Successfully pre-ingested {crate_name} v{version}")

                # Add to processing history
                duration = time.time() - start_time
                self.processing_history.append(
                    {
                        "crate": crate_name,
                        "version": version,
                        "duration": duration,
                        "status": "success",
                    }
                )
            else:
                self.stats["failed"] += 1
                self.crate_progress[crate_name]["status"] = "failed"
                logger.debug(f"Failed to pre-ingest {crate_name} v{version}")

                # Add to processing history
                duration = time.time() - start_time
                self.processing_history.append(
                    {
                        "crate": crate_name,
                        "version": version,
                        "duration": duration,
                        "status": "failed",
                    }
                )

        except Exception as e:
            self.stats["failed"] += 1
            self.crate_progress[crate_name]["status"] = "error"
            self.crate_progress[crate_name]["percent"] = 100
            logger.debug(f"Error pre-ingesting {crate_name}: {e}")

            # Add to processing history
            duration = time.time() - start_time
            self.processing_history.append(
                {
                    "crate": crate_name,
                    "version": "unknown",
                    "duration": duration,
                    "status": "error",
                    "error": str(e),
                }
            )
        finally:
            # Clear current crate when done
            if self.current_crate == crate_name:
                self.current_crate = None

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

    async def _monitor_memory(self):
        """Monitor memory usage and adjust concurrency if needed."""
        try:
            import psutil
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
            return

        process = psutil.Process()

        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Get memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)

                # Check if approaching 1GB limit
                if memory_mb > 900:  # 900MB threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")

                    # Reduce concurrency if needed
                    if self._adaptive_concurrency > 1:
                        self._adaptive_concurrency = max(
                            1, self._adaptive_concurrency - 1
                        )
                        logger.info(
                            f"Reduced concurrency to {self._adaptive_concurrency}"
                        )

                        # Update semaphore
                        self.semaphore = asyncio.Semaphore(self._adaptive_concurrency)

                elif (
                    memory_mb < 600
                    and self._adaptive_concurrency < config.PRE_INGEST_CONCURRENCY
                ):
                    # Increase concurrency if memory allows
                    self._adaptive_concurrency = min(
                        config.PRE_INGEST_CONCURRENCY, self._adaptive_concurrency + 1
                    )
                    logger.debug(
                        f"Increased concurrency to {self._adaptive_concurrency}"
                    )
                    self.semaphore = asyncio.Semaphore(self._adaptive_concurrency)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                await asyncio.sleep(10)

    def get_ingestion_stats(self) -> dict[str, Any]:
        """Get pre-ingestion statistics for health endpoint with enhanced progress tracking."""
        stats = self.stats.copy()

        # Add progress information
        processed = stats["success"] + stats["failed"] + stats["skipped"]
        stats["processed"] = processed
        stats["remaining"] = stats["total"] - processed if stats["total"] > 0 else 0
        stats["progress_percent"] = (
            (processed / stats["total"] * 100) if stats["total"] > 0 else 0
        )

        # Add timing information
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            stats["elapsed_seconds"] = elapsed
            stats["rate_per_second"] = processed / elapsed if elapsed > 0 else 0

            # Enhanced ETA calculation based on rolling average
            if self.processing_history and stats["remaining"] > 0:
                # Calculate average time from recent completions
                successful_durations = [
                    h["duration"]
                    for h in self.processing_history
                    if h.get("status") == "success"
                ]
                if successful_durations:
                    avg_time = sum(successful_durations) / len(successful_durations)
                    # Account for concurrent processing
                    effective_rate = (
                        self._adaptive_concurrency / avg_time if avg_time > 0 else 0
                    )
                    eta_seconds = (
                        stats["remaining"] / effective_rate if effective_rate > 0 else 0
                    )
                    stats["eta_seconds"] = int(eta_seconds)
                    stats["eta_formatted"] = self._format_duration(int(eta_seconds))
                    stats["avg_processing_time"] = avg_time
            elif stats["remaining"] > 0 and stats["rate_per_second"] > 0:
                # Fallback to simple calculation if no history
                eta_seconds = stats["remaining"] / stats["rate_per_second"]
                stats["eta_seconds"] = int(eta_seconds)
                stats["eta_formatted"] = self._format_duration(int(eta_seconds))

        # Add current crate details
        stats["current_crate"] = self.current_crate
        if self.current_crate and self.current_crate in self.crate_progress:
            stats["crate_details"] = self.crate_progress[self.current_crate]
        else:
            stats["crate_details"] = {}

        # Add recent processing history summary
        if self.processing_history:
            recent = list(self.processing_history)[-5:]  # Last 5 crates
            stats["recent_crates"] = [
                {
                    "crate": h["crate"],
                    "status": h.get("status", "unknown"),
                    "duration": round(h.get("duration", 0), 2),
                }
                for h in recent
            ]

        stats["adaptive_concurrency"] = self._adaptive_concurrency
        stats["is_running"] = (
            self._monitor_task is not None and not self._monitor_task.done()
        )

        return stats

    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

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


class IngestionScheduler:
    """Background scheduler for periodic crate ingestion."""

    def __init__(self, manager: PopularCratesManager, worker: PreIngestionWorker):
        self.manager = manager
        self.worker = worker
        self.enabled = config.SCHEDULER_ENABLED
        self.interval_hours = config.SCHEDULER_INTERVAL_HOURS
        self.jitter_percent = config.SCHEDULER_JITTER_PERCENT
        self.background_tasks: set[asyncio.Task] = (
            set()
        )  # Strong references to prevent GC
        self._scheduler_task: asyncio.Task | None = None
        self._last_run: datetime | None = None
        self._next_run: datetime | None = None
        self._runs_completed = 0
        self._is_running = False
        self._start_time: datetime | None = None

    async def start(self):
        """Non-blocking scheduler startup."""
        if not self.enabled:
            logger.debug("Scheduler is disabled")
            return

        self._start_time = datetime.now()
        task = asyncio.create_task(self._run())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        self._scheduler_task = task
        logger.info(f"Background scheduler started (interval: {self.interval_hours}h)")

    async def _run(self):
        """Main scheduler loop with interval-based execution."""
        while True:
            try:
                self._next_run = datetime.now() + timedelta(hours=self.interval_hours)
                await self._schedule_ingestion()

                # Calculate next run time with jitter
                interval_seconds = await self._calculate_next_interval()
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                # Backoff on error
                await asyncio.sleep(300)  # 5 min backoff

    async def _calculate_next_interval(self) -> float:
        """Calculate next run interval with jitter to prevent thundering herd."""
        base_interval = self.interval_hours * 3600

        # Add jitter (±jitter_percent%)
        jitter_range = base_interval * (self.jitter_percent / 100.0)
        jitter = random.uniform(-jitter_range, jitter_range)

        return base_interval + jitter

    async def _schedule_ingestion(self):
        """Schedule a batch of popular crates for ingestion."""
        self._is_running = True
        self._last_run = datetime.now()

        try:
            # Check memory before scheduling
            if not await self._should_schedule():
                logger.warning(
                    "Skipping scheduled ingestion due to resource constraints"
                )
                return

            logger.info(f"Starting scheduled ingestion run #{self._runs_completed + 1}")

            # Get fresh list of popular crates
            crates = await self.manager.get_popular_crates_with_metadata()

            # Reset worker stats for this run
            self.worker.stats = {
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "total": len(crates),
            }
            self.worker._processed_crates.clear()

            # Add crates to worker's priority queue
            for crate in crates:
                if crate.name not in self.worker._processed_crates:
                    priority = -crate.downloads if crate.downloads > 0 else 0
                    await self.worker.queue.put((priority, crate.name))

            # Process the queue (reuse existing worker logic)
            await self.worker.queue.join()

            self._runs_completed += 1
            logger.info(
                f"Scheduled ingestion run #{self._runs_completed} completed. "
                f"Success: {self.worker.stats['success']}, "
                f"Failed: {self.worker.stats['failed']}, "
                f"Skipped: {self.worker.stats['skipped']}"
            )

        except Exception as e:
            logger.error(f"Error in scheduled ingestion: {e}")
        finally:
            self._is_running = False

    async def _should_schedule(self) -> bool:
        """Check if scheduling should proceed based on system resources."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            if memory.percent > 80:
                logger.warning(
                    f"High memory usage ({memory.percent}%), delaying schedule"
                )
                return False

        except ImportError:
            # psutil not available, proceed anyway
            pass

        return True

    async def _monitor_schedule(self):
        """Monitor and log scheduler activity periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Log every 5 minutes

                if self._scheduler_task and not self._scheduler_task.done():
                    status = self.get_scheduler_status()
                    logger.info(
                        f"Scheduler status: Runs={status['runs_completed']}, "
                        f"Next run in {status['next_run_minutes']:.1f} minutes"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler monitor: {e}")

    def get_scheduler_status(self) -> dict[str, Any]:
        """Get scheduler status for health endpoint."""
        status = {
            "enabled": self.enabled,
            "is_running": self._is_running,
            "runs_completed": self._runs_completed,
            "interval_hours": self.interval_hours,
            "jitter_percent": self.jitter_percent,
        }

        if self._start_time:
            status["uptime_seconds"] = (
                datetime.now() - self._start_time
            ).total_seconds()

        if self._last_run:
            status["last_run"] = self._last_run.isoformat()
            status["last_run_minutes_ago"] = (
                datetime.now() - self._last_run
            ).total_seconds() / 60

        if self._next_run:
            status["next_run"] = self._next_run.isoformat()
            status["next_run_minutes"] = max(
                0, (self._next_run - datetime.now()).total_seconds() / 60
            )

        # Include worker stats if available
        if self.worker:
            status["last_run_stats"] = {
                "success": self.worker.stats.get("success", 0),
                "failed": self.worker.stats.get("failed", 0),
                "skipped": self.worker.stats.get("skipped", 0),
            }

        return status

    async def stop(self):
        """Stop the scheduler gracefully."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("Scheduler stopped")


# Global references for monitoring
_popular_crates_manager: PopularCratesManager | None = None
_pre_ingestion_worker: PreIngestionWorker | None = None
_ingestion_scheduler: IngestionScheduler | None = None


async def start_pre_ingestion() -> tuple[
    PopularCratesManager | None, PreIngestionWorker | None
]:
    """Start pre-ingestion and scheduler, return manager and worker for monitoring."""
    global _popular_crates_manager, _pre_ingestion_worker, _ingestion_scheduler

    if not config.PRE_INGEST_ENABLED:
        logger.debug("Pre-ingestion is disabled")
        return None, None

    try:
        _popular_crates_manager = PopularCratesManager()
        _pre_ingestion_worker = PreIngestionWorker(_popular_crates_manager)

        # Start the worker
        await _pre_ingestion_worker.start()

        # Create and start the scheduler if enabled
        if config.SCHEDULER_ENABLED:
            _ingestion_scheduler = IngestionScheduler(
                _popular_crates_manager, _pre_ingestion_worker
            )
            await _ingestion_scheduler.start()

            # Also start the monitor task
            monitor_task = asyncio.create_task(_ingestion_scheduler._monitor_schedule())
            _ingestion_scheduler.background_tasks.add(monitor_task)
            monitor_task.add_done_callback(
                _ingestion_scheduler.background_tasks.discard
            )

        return _popular_crates_manager, _pre_ingestion_worker
    except Exception as e:
        # Don't let pre-ingestion errors prevent server startup
        logger.error(f"Failed to start pre-ingestion: {e}")
        return None, None


def get_popular_manager() -> PopularCratesManager:
    """Get or create the global PopularCratesManager instance."""
    global _popular_crates_manager

    if _popular_crates_manager is None:
        _popular_crates_manager = PopularCratesManager()

    return _popular_crates_manager


def get_popular_crates_status() -> dict[str, Any]:
    """Get combined status for health endpoint."""
    status = {
        "cache_stats": {},
        "ingestion_stats": {},
        "scheduler_stats": {},
    }

    if _popular_crates_manager:
        status["cache_stats"] = _popular_crates_manager.get_cache_stats()

    if _pre_ingestion_worker:
        status["ingestion_stats"] = _pre_ingestion_worker.get_ingestion_stats()

    if _ingestion_scheduler:
        status["scheduler_stats"] = _ingestion_scheduler.get_scheduler_status()

    return status
