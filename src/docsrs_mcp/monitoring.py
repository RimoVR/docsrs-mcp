"""Lightweight monitoring and metrics collection module."""

import time
from collections import deque
from typing import Any


class MetricsCollector:
    """Lightweight metrics collection without external dependencies."""

    def __init__(self, window_size: int = 1000):
        """Initialize metrics collector with configurable window size.

        Args:
            window_size: Maximum number of events to keep in memory (default: 1000)
        """
        self.events = deque(maxlen=window_size)
        self.counters: dict[str, int] = {}
        self.window_size = window_size

    def record_event(
        self,
        event_type: str,
        duration: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an event with optional duration and metadata.

        Args:
            event_type: Type/category of the event
            duration: Optional duration in seconds
            metadata: Optional additional data about the event
        """
        self.events.append(
            {
                "timestamp": time.time(),
                "type": event_type,
                "duration": duration,
                "metadata": metadata or {},
            }
        )
        self.counters[event_type] = self.counters.get(event_type, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics from recorded events.

        Returns:
            Dictionary containing:
            - counters: Event counts by type
            - events: Total number of events in window
            - avg_duration: Average duration of timed events
            - p95_duration: 95th percentile duration
            - recent_events: Last 10 events (without full metadata)
        """
        if not self.events:
            return {
                "counters": self.counters,
                "events": 0,
                "avg_duration": 0,
                "p95_duration": 0,
                "recent_events": [],
            }

        recent_events = list(self.events)
        durations = [e["duration"] for e in recent_events if e.get("duration")]

        # Calculate duration statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        p95_duration = 0
        if durations:
            sorted_durations = sorted(durations)
            p95_index = int(len(sorted_durations) * 0.95)
            p95_duration = (
                sorted_durations[p95_index]
                if p95_index < len(sorted_durations)
                else sorted_durations[-1]
            )

        # Get recent events summary (without full metadata to save memory)
        recent_summary = [
            {
                "timestamp": e["timestamp"],
                "type": e["type"],
                "duration": e.get("duration"),
            }
            for e in list(recent_events)[-10:]
        ]

        return {
            "counters": self.counters.copy(),
            "events": len(recent_events),
            "avg_duration": round(avg_duration, 3) if avg_duration else 0,
            "p95_duration": round(p95_duration, 3) if p95_duration else 0,
            "recent_events": recent_summary,
        }

    def reset(self) -> None:
        """Reset all metrics and clear the event history."""
        self.events.clear()
        self.counters.clear()

    def get_event_rate(self, event_type: str, window_seconds: int = 60) -> float:
        """Calculate the rate of a specific event type over a time window.

        Args:
            event_type: Type of event to calculate rate for
            window_seconds: Time window in seconds (default: 60)

        Returns:
            Events per second for the specified type
        """
        if not self.events:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        count = sum(
            1
            for e in self.events
            if e["type"] == event_type and e["timestamp"] >= cutoff_time
        )

        # Calculate actual time window (might be less than requested if not enough data)
        oldest_in_window = min(
            (e["timestamp"] for e in self.events if e["timestamp"] >= cutoff_time),
            default=current_time,
        )
        actual_window = current_time - oldest_in_window

        return count / actual_window if actual_window > 0 else 0.0


# Global metrics collector instance (singleton pattern)
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance.

    Returns:
        The global MetricsCollector instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_metric(
    event_type: str,
    duration: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience function to record a metric using the global collector.

    Args:
        event_type: Type/category of the event
        duration: Optional duration in seconds
        metadata: Optional additional data about the event
    """
    collector = get_metrics_collector()
    collector.record_event(event_type, duration, metadata)


def get_global_stats() -> dict[str, Any]:
    """Get statistics from the global metrics collector.

    Returns:
        Aggregated statistics dictionary
    """
    collector = get_metrics_collector()
    return collector.get_stats()
