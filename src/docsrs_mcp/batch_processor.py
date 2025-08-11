"""Batch processing utilities for memory-aware and resilient batch operations."""

import gc
import logging
import os
from collections.abc import Callable, Generator, Iterable
from typing import TypeVar

from .memory_utils import MemoryMonitor, get_adaptive_batch_size

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class MemoryRecycleRequired(Exception):
    """Raised when process recycling is needed due to memory pressure."""


class BatchProcessor:
    """
    Core batch processing abstraction with memory-aware sizing and cleanup.

    This class encapsulates batch processing logic with automatic memory management,
    adaptive batch sizing, and proper cleanup between batches.
    """

    def __init__(
        self,
        operation_type: str,
        memory_monitor: MemoryMonitor | None = None,
        max_batch_size: int = 512,
        min_batch_size: int = 16,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            operation_type: Type of operation (e.g., "embedding", "database", "validation")
            memory_monitor: Optional MemoryMonitor instance for tracking memory usage
            max_batch_size: Maximum items per batch
            min_batch_size: Minimum items per batch (used under memory pressure)
        """
        self.operation_type = operation_type
        self.memory_monitor = memory_monitor
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.processed_batches = 0
        self.total_items_processed = 0

    def process_batch(
        self,
        items: Iterable[T],
        processor_func: Callable[[list[T]], R],
        batch_size_override: int | None = None,
    ) -> Generator[R, None, None]:
        """
        Process items in batches with automatic memory management.

        Args:
            items: Iterable of items to process
            processor_func: Function to process a batch of items
            batch_size_override: Optional override for batch size

        Yields:
            Results from processing each batch

        Raises:
            MemoryRecycleRequired: When process recycling is needed
        """
        batch = []
        batch_size = batch_size_override or self._calculate_batch_size()

        for item in items:
            batch.append(item)

            if len(batch) >= batch_size:
                # Process the batch
                if self.memory_monitor:
                    with self.memory_monitor:
                        result = processor_func(batch)
                else:
                    result = processor_func(batch)

                yield result

                # Update counters
                self.total_items_processed += len(batch)
                self.processed_batches += 1

                # Cleanup after batch
                self._cleanup_batch()

                # Check if process recycling is needed
                if self._should_recycle_process():
                    raise MemoryRecycleRequired(
                        f"Process recycling needed after {self.processed_batches} batches"
                    )

                # Reset for next batch
                batch = []
                batch_size = batch_size_override or self._calculate_batch_size()

        # Process remaining items
        if batch:
            if self.memory_monitor:
                with self.memory_monitor:
                    result = processor_func(batch)
            else:
                result = processor_func(batch)

            yield result
            self.total_items_processed += len(batch)
            self.processed_batches += 1
            self._cleanup_batch()

    def _calculate_batch_size(self) -> int:
        """
        Calculate optimal batch size based on operation type and memory pressure.

        Returns:
            Calculated batch size
        """
        # Get base adaptive batch size
        base_size = get_adaptive_batch_size(
            min_size=self.min_batch_size,
            max_size=self.max_batch_size,
        )

        # Apply operation-specific adjustments
        if self.operation_type == "embedding":
            # Embeddings are memory-intensive, use smaller batches
            return min(base_size, 32)
        elif self.operation_type == "database":
            # Database operations limited by SQLite parameter count
            return min(base_size, 999)
        elif self.operation_type == "validation":
            # Validation is lightweight, can use larger batches
            return base_size
        else:
            # Default to base adaptive size
            return base_size

    def _cleanup_batch(self) -> None:
        """Perform explicit memory cleanup after processing a batch."""
        # Force garbage collection to free memory
        gc.collect()

        # Log memory status periodically
        if self.processed_batches % 10 == 0:
            logger.debug(
                f"BatchProcessor[{self.operation_type}]: Processed {self.processed_batches} batches, "
                f"{self.total_items_processed} total items"
            )

    def _should_recycle_process(self) -> bool:
        """
        Determine if process recycling is needed based on batch count and memory.

        Returns:
            True if process should be recycled
        """
        # Check batch count threshold (configurable via environment)
        max_batches = int(os.getenv("BATCH_PROCESSOR_MAX_BATCHES", "50"))

        if self.operation_type == "embedding" and self.processed_batches >= max_batches:
            logger.info(
                f"BatchProcessor[{self.operation_type}]: Reached max batch count {max_batches}"
            )
            return True

        # Check memory pressure
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:  # High memory threshold
                logger.warning(
                    f"BatchProcessor[{self.operation_type}]: High memory pressure {memory_percent}%"
                )
                return True
        except ImportError:
            pass  # psutil not available

        return False

    def reset_counters(self) -> None:
        """Reset internal counters (useful after process recycling)."""
        self.processed_batches = 0
        self.total_items_processed = 0


class BatchSizeCalculator:
    """
    Calculator for operation-specific batch sizes with memory trend analysis.
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize the calculator with a moving average window.

        Args:
            window_size: Size of the window for trend analysis
        """
        self.window_size = window_size
        self.memory_history: list[float] = []

    def calculate_size(
        self,
        operation_type: str,
        current_memory_percent: float,
        min_size: int = 16,
        max_size: int = 512,
    ) -> int:
        """
        Calculate optimal batch size based on operation type and memory trends.

        Args:
            operation_type: Type of operation
            current_memory_percent: Current memory usage percentage
            min_size: Minimum batch size
            max_size: Maximum batch size

        Returns:
            Calculated batch size
        """
        # Update memory history
        self.memory_history.append(current_memory_percent)
        if len(self.memory_history) > self.window_size:
            self.memory_history.pop(0)

        # Calculate memory trend (positive = increasing, negative = decreasing)
        memory_trend = self._calculate_trend()

        # Base calculation from current memory
        if current_memory_percent >= 90:
            base_size = min_size
        elif current_memory_percent >= 80:
            # Linear scaling between min and max
            scale = (90 - current_memory_percent) / 10
            base_size = int(min_size + (max_size - min_size) * scale)
        else:
            base_size = max_size

        # Adjust based on trend
        if memory_trend > 5:  # Memory increasing rapidly
            base_size = int(base_size * 0.7)  # Reduce size by 30%
        elif memory_trend > 2:  # Memory increasing slowly
            base_size = int(base_size * 0.9)  # Reduce size by 10%
        elif memory_trend < -2:  # Memory decreasing
            base_size = int(base_size * 1.1)  # Increase size by 10%

        # Apply operation-specific limits
        if operation_type == "embedding":
            # Memory-intensive, cap at 32
            base_size = min(base_size, 32)
        elif operation_type == "database":
            # SQLite parameter limit
            base_size = min(base_size, 999)
        elif operation_type == "download":
            # Network I/O bound, can be larger
            base_size = min(base_size, max_size)

        # Ensure within bounds
        return max(min_size, min(base_size, max_size))

    def _calculate_trend(self) -> float:
        """
        Calculate memory usage trend using simple moving average.

        Returns:
            Trend value (positive for increasing, negative for decreasing)
        """
        if len(self.memory_history) < 2:
            return 0.0

        # Calculate average change over window
        changes = []
        for i in range(1, len(self.memory_history)):
            changes.append(self.memory_history[i] - self.memory_history[i - 1])

        return sum(changes) / len(changes) if changes else 0.0
