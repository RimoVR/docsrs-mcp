"""Ingestion status tracking and recovery operations."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ..config import DB_TIMEOUT
from .connection import RetryableTransaction, execute_with_retry

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


def compute_item_hash(item: dict[str, Any]) -> str:
    """Compute a stable hash for an item to track processing state.

    Args:
        item: Dictionary containing item data

    Returns:
        16-character hash string
    """
    # Create stable hash from key fields
    hash_parts = [
        item.get("name", ""),
        item.get("path", ""),
        item.get("kind", ""),
        str(item.get("inner", {})),
    ]

    hash_string = "|".join(hash_parts)
    return hashlib.sha256(hash_string.encode()).hexdigest()[:16]


@RetryableTransaction(max_retries=3)
async def set_ingestion_status(
    db_path: Path,
    crate_id: int,
    status: str,
    error_message: str | None = None,
    checkpoint_data: dict | None = None,
    items_processed: int | None = None,
    total_items: int | None = None,
    ingestion_tier: str | None = None,
) -> None:
    """Set the ingestion status for a crate with recovery support.

    Args:
        db_path: Path to database
        crate_id: ID of the crate
        status: Status string (started, downloading, processing, completed, failed)
        error_message: Optional error message
        checkpoint_data: Optional checkpoint data for recovery
        items_processed: Number of items processed
        total_items: Total number of items
        ingestion_tier: Ingestion tier (rustdoc_json, source_extraction, etc.)
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        current_time = time.time()
        checkpoint_json = json.dumps(checkpoint_data) if checkpoint_data else None

        # Check if record exists
        cursor = await db.execute(
            "SELECT COUNT(*) FROM ingestion_status WHERE crate_id = ?", (crate_id,)
        )
        exists = (await cursor.fetchone())[0] > 0

        if exists:
            # Update existing record
            if status == "completed":
                await execute_with_retry(
                    db,
                    """
                    UPDATE ingestion_status 
                    SET status = ?, updated_at = ?, completed_at = ?, 
                        error_message = ?, checkpoint_data = ?,
                        items_processed = COALESCE(?, items_processed),
                        total_items = COALESCE(?, total_items),
                        ingestion_tier = COALESCE(?, ingestion_tier)
                    WHERE crate_id = ?
                    """,
                    (
                        status,
                        current_time,
                        current_time,
                        error_message,
                        checkpoint_json,
                        items_processed,
                        total_items,
                        ingestion_tier,
                        crate_id,
                    ),
                )
            else:
                await execute_with_retry(
                    db,
                    """
                    UPDATE ingestion_status 
                    SET status = ?, updated_at = ?, error_message = ?, 
                        checkpoint_data = ?,
                        items_processed = COALESCE(?, items_processed),
                        total_items = COALESCE(?, total_items),
                        ingestion_tier = COALESCE(?, ingestion_tier)
                    WHERE crate_id = ?
                    """,
                    (
                        status,
                        current_time,
                        error_message,
                        checkpoint_json,
                        items_processed,
                        total_items,
                        ingestion_tier,
                        crate_id,
                    ),
                )
        else:
            # Insert new record
            await execute_with_retry(
                db,
                """
                INSERT INTO ingestion_status 
                (crate_id, status, started_at, updated_at, completed_at, 
                 error_message, checkpoint_data, items_processed, total_items, ingestion_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    crate_id,
                    status,
                    current_time,
                    current_time,
                    current_time if status == "completed" else None,
                    error_message,
                    checkpoint_json,
                    items_processed or 0,
                    total_items,
                    ingestion_tier,
                ),
            )

        await db.commit()
        logger.info(f"Set ingestion status for crate_id={crate_id}: {status}")


async def get_ingestion_status(db_path: Path, crate_id: int) -> dict | None:
    """Get the ingestion status for a crate.

    Args:
        db_path: Path to database
        crate_id: ID of the crate

    Returns:
        Status dictionary or None if not found
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            SELECT status, started_at, updated_at, completed_at, 
                   error_message, items_processed, total_items, 
                   checkpoint_data, ingestion_tier
            FROM ingestion_status 
            WHERE crate_id = ?
            """,
            (crate_id,),
        )
        row = await cursor.fetchone()

        if row:
            checkpoint_data = json.loads(row[7]) if row[7] else None
            return {
                "status": row[0],
                "started_at": row[1],
                "updated_at": row[2],
                "completed_at": row[3],
                "error_message": row[4],
                "items_processed": row[5],
                "total_items": row[6],
                "checkpoint_data": checkpoint_data,
                "ingestion_tier": row[8],
            }

        return None


async def find_incomplete_ingestions(db_path: Path) -> list[dict]:
    """Find all incomplete ingestions that may need recovery.

    Args:
        db_path: Path to database

    Returns:
        List of incomplete ingestion records
    """
    import aiosqlite

    incomplete = []
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Use partial index for O(1) query
        cursor = await db.execute(
            """
            SELECT c.name, c.version, i.crate_id, i.status, 
                   i.started_at, i.updated_at, i.error_message,
                   i.items_processed, i.total_items
            FROM ingestion_status i
            JOIN crate_metadata c ON i.crate_id = c.id
            WHERE i.status != 'completed'
            ORDER BY i.updated_at DESC
            """
        )

        async for row in cursor:
            incomplete.append(
                {
                    "crate_name": row[0],
                    "version": row[1],
                    "crate_id": row[2],
                    "status": row[3],
                    "started_at": row[4],
                    "updated_at": row[5],
                    "error_message": row[6],
                    "items_processed": row[7],
                    "total_items": row[8],
                }
            )

    if incomplete:
        logger.info(f"Found {len(incomplete)} incomplete ingestions")

    return incomplete


async def is_ingestion_complete(db_path: Path) -> bool:
    """Check if the ingestion for this database is complete.

    Args:
        db_path: Path to database

    Returns:
        True if ingestion is complete, False otherwise
    """
    import aiosqlite

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Check if ingestion_status table exists
            cursor = await db.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='ingestion_status'
                """
            )
            if not await cursor.fetchone():
                return False

            # Check if there's a completed ingestion
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM ingestion_status 
                WHERE status = 'completed'
                """
            )
            count = (await cursor.fetchone())[0]
            return count > 0

    except Exception as e:
        logger.debug(f"Error checking ingestion status: {e}")
        return False


async def detect_stalled_ingestions(
    db_path: Path, stale_threshold_seconds: int = 600
) -> list[dict]:
    """Detect ingestions that appear to be stalled.

    Args:
        db_path: Path to database
        stale_threshold_seconds: Time in seconds before considering stalled

    Returns:
        List of stalled ingestion records
    """
    import aiosqlite

    current_time = time.time()
    threshold_time = current_time - stale_threshold_seconds

    stalled = []
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        cursor = await db.execute(
            """
            SELECT c.name, c.version, i.crate_id, i.status, 
                   i.updated_at, i.items_processed, i.total_items
            FROM ingestion_status i
            JOIN crate_metadata c ON i.crate_id = c.id
            WHERE i.status IN ('started', 'downloading', 'processing')
              AND i.updated_at < ?
            """,
            (threshold_time,),
        )

        async for row in cursor:
            stalled.append(
                {
                    "crate_name": row[0],
                    "version": row[1],
                    "crate_id": row[2],
                    "status": row[3],
                    "last_update": row[4],
                    "items_processed": row[5],
                    "total_items": row[6],
                    "stale_duration_seconds": int(current_time - row[4]),
                }
            )

    if stalled:
        logger.warning(f"Found {len(stalled)} stalled ingestions requiring recovery")

    return stalled


async def reset_ingestion_status(db_path: Path, crate_id: int) -> None:
    """Reset ingestion status to trigger re-ingestion.

    Args:
        db_path: Path to database
        crate_id: ID of the crate to reset
    """
    import aiosqlite

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        await execute_with_retry(
            db, "DELETE FROM ingestion_status WHERE crate_id = ?", (crate_id,)
        )
        await db.commit()
        logger.info(f"Reset ingestion status for crate_id={crate_id}")


__all__ = [
    "set_ingestion_status",
    "get_ingestion_status",
    "find_incomplete_ingestions",
    "is_ingestion_complete",
    "detect_stalled_ingestions",
    "reset_ingestion_status",
    "compute_item_hash",
]
