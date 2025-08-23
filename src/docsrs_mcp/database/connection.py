"""Database connection management, retry logic, and performance utilities."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ..config import CACHE_DIR, DB_TIMEOUT

if TYPE_CHECKING:
    import aiosqlite

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)

# Prepared statement cache for common queries
prepared_statements = {}


def performance_timer(operation_name: str):
    """Decorator to track performance of database operations.

    Logs execution time and provides detailed metrics for monitoring.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log performance metrics
                if elapsed_ms > 100:
                    logger.warning(
                        f"{operation_name} took {elapsed_ms:.2f}ms (slow operation)"
                    )
                else:
                    logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")

                # Add timing to result if it's a dict
                if isinstance(result, dict) and "metrics" not in result:
                    result["metrics"] = {"execution_time_ms": elapsed_ms}

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log performance metrics
                if elapsed_ms > 100:
                    logger.warning(
                        f"{operation_name} took {elapsed_ms:.2f}ms (slow operation)"
                    )
                else:
                    logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")

                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RetryableTransaction:
    """Decorator for retryable database transactions with exponential backoff and jitter.

    Implements exponential backoff with full jitter to prevent thundering herd problem
    when multiple processes encounter database locks simultaneously.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        jitter: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds (cap for exponential growth)
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def __call__(self, func: Callable) -> Callable:
        """Make the class callable as a decorator."""
        import aiosqlite

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiosqlite.OperationalError, aiosqlite.DatabaseError) as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Check if error is retryable
                    if any(
                        msg in error_msg
                        for msg in ["locked", "busy", "database is locked"]
                    ):
                        if attempt < self.max_retries:
                            # Calculate delay with exponential backoff
                            delay = min(self.base_delay * (2**attempt), self.max_delay)

                            # Add jitter to prevent thundering herd
                            if self.jitter:
                                delay = random.uniform(0, delay)

                            logger.warning(
                                f"Database locked on attempt {attempt + 1}/{self.max_retries + 1}, "
                                f"retrying in {delay:.3f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                            continue

                    # Non-retryable error, raise immediately
                    logger.error(f"Non-retryable database error: {e}")
                    raise
                except Exception as e:
                    # Non-database error, don't retry
                    logger.error(f"Unexpected error in transaction: {e}")
                    raise

            # All retries exhausted
            logger.error(
                f"Transaction failed after {self.max_retries + 1} attempts: {last_exception}"
            )
            raise last_exception

        return wrapper


async def execute_with_retry(
    conn: aiosqlite.Connection,
    query: str,
    params: tuple | None = None,
    use_immediate: bool = True,
) -> aiosqlite.Cursor:
    """Execute a query with retry logic and BEGIN IMMEDIATE for write operations.

    Args:
        conn: Database connection
        query: SQL query to execute
        params: Query parameters
        use_immediate: Whether to use BEGIN IMMEDIATE for write operations

    Returns:
        Query cursor
    """
    # Get configured values from environment
    max_retries = int(os.environ.get("TRANSACTION_MAX_RETRIES", "3"))
    busy_timeout = int(os.environ.get("TRANSACTION_BUSY_TIMEOUT", "5000"))

    # Set busy timeout on connection
    await conn.execute(f"PRAGMA busy_timeout = {busy_timeout}")

    # Determine if this is a write operation
    is_write = any(
        query.strip().upper().startswith(cmd)
        for cmd in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
    )

    @RetryableTransaction(max_retries=max_retries)
    async def _execute():
        if is_write and use_immediate:
            # Use BEGIN IMMEDIATE to avoid lock upgrades
            await conn.execute("BEGIN IMMEDIATE")
            try:
                if params:
                    cursor = await conn.execute(query, params)
                else:
                    cursor = await conn.execute(query)
                await conn.commit()
                return cursor
            except Exception:
                await conn.rollback()
                raise
        # Read operation or explicit transaction management
        elif params:
            return await conn.execute(query, params)
        else:
            return await conn.execute(query)

    return await _execute()


async def get_db_path(crate_name: str, version: str) -> Path:
    """Get the database path for a specific crate and version."""
    # Sanitize names for filesystem
    safe_crate = crate_name.replace("/", "_").replace("\\", "_")
    safe_version = version.replace("/", "_").replace("\\", "_")

    db_dir = CACHE_DIR / safe_crate
    db_dir.mkdir(parents=True, exist_ok=True)

    return db_dir / f"{safe_version}.db"


async def load_sqlite_vec_extension(db: aiosqlite.Connection) -> None:
    """Load the sqlite-vec extension for vector operations.

    Args:
        db: Database connection
    """
    import sqlite_vec

    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)


__all__ = [
    "performance_timer",
    "RetryableTransaction",
    "execute_with_retry",
    "get_db_path",
    "load_sqlite_vec_extension",
    "prepared_statements",
    "DB_TIMEOUT",
]
