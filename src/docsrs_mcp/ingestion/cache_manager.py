"""Cache management and eviction for ingested crate databases.

This module handles:
- Cache size calculation
- LRU eviction with priority weighting
- Memory pressure handling
- Priority-based eviction for popular crates
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from ..config import CACHE_DIR, CACHE_MAX_SIZE_BYTES, PRIORITY_CACHE_EVICTION_ENABLED

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
MAX_CACHE_SIZE = CACHE_MAX_SIZE_BYTES


def calculate_cache_size() -> int:
    """Calculate total size of cache directory using os.scandir for efficiency.

    Walks through the cache directory and sums up sizes of all .db files.

    Returns:
        int: Total size of cache in bytes
    """
    total_size = 0

    try:
        # Walk through cache directory
        for root, _dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith(".db"):
                    file_path = os.path.join(root, file)
                    try:
                        # Use os.stat for file size
                        stat_info = os.stat(file_path)
                        total_size += stat_info.st_size
                    except OSError as e:
                        logger.warning(f"Error getting size of {file_path}: {e}")

    except OSError as e:
        logger.error(f"Error calculating cache size: {e}")

    return total_size


def _extract_crate_name(file_path: str) -> str | None:
    """Extract crate name from cache file path.

    Cache files are structured as: cache/{crate}/{version}.db

    Args:
        file_path: Full path to cache file

    Returns:
        Optional[str]: Crate name if extractable, None otherwise
    """
    try:
        path_obj = Path(file_path)
        # Get parent directory name as crate name
        return path_obj.parent.name
    except Exception:
        # If extraction fails, treat as unknown crate
        return None


def _collect_cache_files() -> list[dict[str, Any]]:
    """Collect all cache files with their metadata.

    Returns:
        List[Dict]: List of file info dictionaries with path, size, mtime, crate_name
    """
    cache_files = []

    try:
        for root, _dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith(".db"):
                    file_path = os.path.join(root, file)
                    try:
                        stat_info = os.stat(file_path)
                        crate_name = _extract_crate_name(file_path)

                        cache_files.append(
                            {
                                "path": file_path,
                                "size": stat_info.st_size,
                                "mtime": stat_info.st_mtime,
                                "crate_name": crate_name,
                            }
                        )
                    except OSError:
                        pass
    except Exception as e:
        logger.error(f"Error collecting cache files: {e}")

    return cache_files


async def _apply_priority_scoring(cache_files: list[dict[str, Any]]) -> None:
    """Apply priority scoring to cache files based on crate popularity.

    Popular crates get higher priority (lower eviction priority).
    Uses log scale for download counts to prevent extreme differences.

    Args:
        cache_files: List of file info dictionaries to score (modified in-place)
    """
    try:
        # Get popular crates data for priority scoring
        from ..popular_crates import get_popular_manager

        manager = get_popular_manager()
        popular_crates = await manager.get_popular_crates_with_metadata()

        # Create lookup dictionary for O(1) access
        popular_dict = (
            {c.name: c.downloads for c in popular_crates} if popular_crates else {}
        )

        # Calculate priority scores for each file
        for file_info in cache_files:
            crate_name = file_info.get("crate_name")
            downloads = popular_dict.get(crate_name, 0) if crate_name else 0

            # Calculate priority score using log scale for downloads
            # Popular crates get higher priority (lower eviction priority)
            if downloads > 0:
                # Use log scale to prevent extreme differences
                priority = math.log10(downloads + 1)
            else:
                priority = 0

            file_info["priority"] = priority
            file_info["downloads"] = downloads

        # Hybrid sorting: primary by priority (ascending = evict low priority first),
        # secondary by mtime (oldest first within same priority tier)
        cache_files.sort(
            key=lambda x: (
                x.get("priority", 0),  # Lower priority first (evict first)
                x["mtime"],  # Older files first within same priority
            )
        )

        logger.debug("Using priority-aware cache eviction")

    except Exception as e:
        # Fallback to time-based eviction if priority scoring fails
        logger.warning(f"Priority eviction failed, using time-based: {e}")
        cache_files.sort(key=lambda x: x["mtime"])


async def evict_cache_if_needed() -> None:
    """Evict cache files with priority-aware logic if total size exceeds limit.

    Uses a hybrid eviction strategy:
    1. Priority-based: Popular crates are kept longer (if enabled)
    2. LRU: Within same priority tier, oldest files are evicted first

    The eviction continues until cache size is below the configured limit.
    """
    current_size = calculate_cache_size()

    if current_size <= CACHE_MAX_SIZE_BYTES:
        logger.debug(f"Cache size {current_size} bytes within limit")
        return

    logger.info(
        f"Cache size {current_size} bytes exceeds limit {CACHE_MAX_SIZE_BYTES}, evicting..."
    )

    # Collect all cache files with their stats
    cache_files = _collect_cache_files()

    # Apply priority scoring if enabled
    if PRIORITY_CACHE_EVICTION_ENABLED:
        await _apply_priority_scoring(cache_files)
    else:
        # Standard time-based eviction
        cache_files.sort(key=lambda x: x["mtime"])

    # Remove files until under limit
    removed_size = 0
    removed_count = 0

    for file_info in cache_files:
        if current_size - removed_size <= CACHE_MAX_SIZE_BYTES:
            break

        try:
            os.remove(file_info["path"])
            removed_size += file_info["size"]
            removed_count += 1

            # Enhanced logging with priority information
            if PRIORITY_CACHE_EVICTION_ENABLED and "priority" in file_info:
                crate_info = file_info.get("crate_name", "unknown")
                priority = file_info.get("priority", 0)
                downloads = file_info.get("downloads", 0)
                age_days = (time.time() - file_info["mtime"]) / 86400

                logger.info(
                    f"Evicted: {crate_info} "
                    f"(priority: {priority:.2f}, downloads: {downloads}, "
                    f"age: {age_days:.1f} days, size: {file_info['size']} bytes)"
                )
            else:
                logger.info(
                    f"Evicted cache file: {file_info['path']} ({file_info['size']} bytes)"
                )

        except OSError as e:
            logger.warning(f"Error removing cache file {file_info['path']}: {e}")

    logger.info(
        f"Evicted {removed_count} files totaling {removed_size} bytes from cache"
    )


def get_cache_metadata(file_path: str) -> dict[str, Any] | None:
    """Get metadata for a specific cache file.

    Args:
        file_path: Path to cache file

    Returns:
        Optional[Dict]: Metadata dict with size, mtime, crate_name or None if file doesn't exist
    """
    try:
        stat_info = os.stat(file_path)
        crate_name = _extract_crate_name(file_path)

        return {
            "path": file_path,
            "size": stat_info.st_size,
            "mtime": stat_info.st_mtime,
            "crate_name": crate_name,
            "age_days": (time.time() - stat_info.st_mtime) / 86400,
        }
    except OSError:
        return None


def update_access_time(file_path: str) -> None:
    """Update the access time of a cache file for LRU tracking.

    This updates both access time and modification time to current time,
    making the file appear "recently used" for LRU eviction.

    Args:
        file_path: Path to cache file to update
    """
    try:
        # Update both access and modification times
        current_time = time.time()
        os.utime(file_path, (current_time, current_time))
        logger.debug(f"Updated access time for {file_path}")
    except OSError as e:
        logger.warning(f"Failed to update access time for {file_path}: {e}")
