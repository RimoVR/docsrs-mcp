"""Memory monitoring and adaptive batch sizing utilities."""

import gc
import logging

import psutil

from .config import (
    EMBEDDING_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MEMORY_THRESHOLD_CRITICAL,
    MEMORY_THRESHOLD_HIGH,
    MIN_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def get_memory_percent() -> float:
    """Get current memory usage percentage.

    Returns:
        Current memory usage as a percentage (0-100).
    """
    return psutil.virtual_memory().percent


def get_available_memory_mb() -> float:
    """Get available memory in megabytes.

    Returns:
        Available memory in MB.
    """
    return psutil.virtual_memory().available / (1024 * 1024)


def memory_pressure_detected() -> bool:
    """Check if system is under memory pressure.

    Returns:
        True if memory usage exceeds high threshold.
    """
    memory_percent = get_memory_percent()
    is_high = memory_percent >= MEMORY_THRESHOLD_HIGH

    if is_high:
        logger.warning(f"Memory pressure detected: {memory_percent:.1f}% used")

    return is_high


def get_adaptive_batch_size(
    base_batch_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    operation_type: str | None = None,
) -> int:
    """Calculate adaptive batch size based on available memory.

    Args:
        base_batch_size: Base batch size to adapt from (default: EMBEDDING_BATCH_SIZE)
        min_size: Minimum batch size (default: MIN_BATCH_SIZE)
        max_size: Maximum batch size (default: MAX_BATCH_SIZE)
        operation_type: Type of operation ("embedding", "database", "validation", etc.)

    Returns:
        Adapted batch size based on memory pressure and operation type.
    """
    if base_batch_size is None:
        base_batch_size = EMBEDDING_BATCH_SIZE
    if min_size is None:
        min_size = MIN_BATCH_SIZE
    if max_size is None:
        max_size = MAX_BATCH_SIZE

    memory_percent = get_memory_percent()

    if memory_percent >= MEMORY_THRESHOLD_CRITICAL:
        # Critical memory pressure - use minimum batch size and trigger GC
        logger.warning(
            f"Critical memory pressure: {memory_percent:.1f}% - "
            f"using minimum batch size {min_size}"
        )
        gc.collect()
        return min_size
    elif memory_percent >= MEMORY_THRESHOLD_HIGH:
        # High memory pressure - reduce batch size proportionally
        reduction_factor = (100 - memory_percent) / (100 - MEMORY_THRESHOLD_HIGH)
        reduced_size = int(base_batch_size * reduction_factor)
        batch_size = max(min_size, min(reduced_size, base_batch_size))
        logger.info(
            f"High memory pressure: {memory_percent:.1f}% - "
            f"reduced batch size to {batch_size}"
        )
        return batch_size
    # Normal memory usage - can potentially increase batch size
    elif memory_percent < 50:
        # Plenty of memory available - increase batch size
        increase_factor = min(2.0, (100 - memory_percent) / 50)
        increased_size = int(base_batch_size * increase_factor)
        batch_size = min(max_size, increased_size)
        if batch_size > base_batch_size:
            logger.debug(
                f"Low memory usage: {memory_percent:.1f}% - "
                f"increased batch size to {batch_size}"
            )
        return batch_size
    else:
        # Normal range - use base batch size
        batch_size = base_batch_size

    # Apply operation-specific limits
    if operation_type:
        if operation_type == "embedding":
            # Memory-intensive embeddings, cap at 32
            batch_size = min(batch_size, 32)
        elif operation_type == "database":
            # SQLite parameter limit
            batch_size = min(batch_size, 999)
        elif operation_type == "validation":
            # Validation is lightweight, can use full size
            pass
        # Add more operation types as needed

    return batch_size


def trigger_gc_if_needed(force: bool = False) -> bool:
    """Trigger garbage collection if memory pressure is high.

    Args:
        force: Force garbage collection regardless of memory pressure.

    Returns:
        True if garbage collection was triggered.
    """
    if force or get_memory_percent() >= MEMORY_THRESHOLD_HIGH:
        collected = gc.collect()
        logger.debug(f"Garbage collection triggered, collected {collected} objects")
        return True
    return False


def log_memory_status(context: str = "") -> None:
    """Log current memory status for debugging.

    Args:
        context: Optional context string to include in log message.
    """
    vm = psutil.virtual_memory()
    process = psutil.Process()
    process_info = process.memory_info()

    context_str = f" [{context}]" if context else ""
    logger.debug(
        f"Memory status{context_str}: "
        f"System: {vm.percent:.1f}% used ({vm.used / (1024**3):.2f}GB / {vm.total / (1024**3):.2f}GB), "
        f"Process RSS: {process_info.rss / (1024**2):.1f}MB"
    )


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""

    # Class-level memory history for trend analysis
    _memory_history: list[float] = []
    _history_window_size: int = 10

    def __init__(self, operation_name: str, log_level: int = logging.DEBUG):
        """Initialize memory monitor.

        Args:
            operation_name: Name of the operation being monitored.
            log_level: Logging level for memory status messages.
        """
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_memory = 0
        self.start_process_memory = 0

    def __enter__(self):
        """Record initial memory state."""
        self.start_memory = psutil.virtual_memory().used
        process = psutil.Process()
        self.start_process_memory = process.memory_info().rss

        logger.log(
            self.log_level,
            f"Starting {self.operation_name} - "
            f"System memory: {self.start_memory / (1024**2):.1f}MB, "
            f"Process RSS: {self.start_process_memory / (1024**2):.1f}MB",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log memory usage delta."""
        end_memory = psutil.virtual_memory().used
        process = psutil.Process()
        end_process_memory = process.memory_info().rss

        system_delta = (end_memory - self.start_memory) / (1024**2)
        process_delta = (end_process_memory - self.start_process_memory) / (1024**2)

        logger.log(
            self.log_level,
            f"Completed {self.operation_name} - "
            f"System memory delta: {system_delta:+.1f}MB, "
            f"Process RSS delta: {process_delta:+.1f}MB",
        )

        # Update memory history for trend analysis
        current_memory_percent = psutil.virtual_memory().percent
        self._memory_history.append(current_memory_percent)
        if len(self._memory_history) > self._history_window_size:
            self._memory_history.pop(0)

        # Calculate memory trend
        trend = self._calculate_memory_trend()
        if trend > 5:
            logger.warning(
                f"Memory usage trending up rapidly (trend: {trend:.1f}%/operation)"
            )

        # Trigger GC if significant memory was used
        if process_delta > 100:  # More than 100MB increase
            trigger_gc_if_needed(force=True)

    @classmethod
    def _calculate_memory_trend(cls) -> float:
        """Calculate memory usage trend.

        Returns:
            Average memory change per operation (positive = increasing).
        """
        if len(cls._memory_history) < 2:
            return 0.0

        changes = []
        for i in range(1, len(cls._memory_history)):
            changes.append(cls._memory_history[i] - cls._memory_history[i - 1])

        return sum(changes) / len(changes) if changes else 0.0

    @classmethod
    def get_memory_trend(cls) -> float:
        """Get current memory trend for external use.

        Returns:
            Current memory trend value.
        """
        return cls._calculate_memory_trend()
