"""Tests for batch processing enhancements."""

import gc
import os
from unittest.mock import MagicMock, patch

import pytest

from docsrs_mcp.batch_processor import (
    BatchProcessor,
    BatchSizeCalculator,
    MemoryRecycleRequired,
)
from docsrs_mcp.memory_utils import MemoryMonitor


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor("embedding", max_batch_size=100, min_batch_size=10)
        assert processor.operation_type == "embedding"
        assert processor.max_batch_size == 100
        assert processor.min_batch_size == 10
        assert processor.processed_batches == 0
        assert processor.total_items_processed == 0

    def test_batch_processor_with_items(self):
        """Test processing items in batches."""
        processor = BatchProcessor("validation", max_batch_size=3, min_batch_size=1)
        items = list(range(10))
        
        def process_func(batch):
            return sum(batch)
        
        # Use batch_size_override to force specific batch size for testing
        results = list(processor.process_batch(items, process_func, batch_size_override=3))
        
        # Should have processed in batches of 3 or less
        assert len(results) == 4  # 3 + 3 + 3 + 1
        assert processor.total_items_processed == 10
        assert processor.processed_batches == 4

    @patch("docsrs_mcp.batch_processor.get_adaptive_batch_size")
    def test_adaptive_batch_sizing(self, mock_get_adaptive):
        """Test adaptive batch sizing."""
        mock_get_adaptive.return_value = 2
        processor = BatchProcessor("database")
        items = list(range(5))
        
        def process_func(batch):
            return len(batch)
        
        results = list(processor.process_batch(items, process_func))
        
        # Should use adaptive size of 2
        assert results == [2, 2, 1]
        assert mock_get_adaptive.called

    def test_operation_specific_batch_sizes(self):
        """Test operation-specific batch size limits."""
        # Embedding operations should be capped at 32
        processor = BatchProcessor("embedding", max_batch_size=100)
        size = processor._calculate_batch_size()
        assert size <= 32
        
        # Database operations should be capped at 999
        processor = BatchProcessor("database", max_batch_size=2000)
        size = processor._calculate_batch_size()
        assert size <= 999
        
        # Validation can use full size
        processor = BatchProcessor("validation", max_batch_size=500)
        with patch("docsrs_mcp.batch_processor.get_adaptive_batch_size", return_value=500):
            size = processor._calculate_batch_size()
            assert size == 500

    @patch.dict(os.environ, {"BATCH_PROCESSOR_MAX_BATCHES": "2"})
    def test_process_recycling_by_batch_count(self):
        """Test process recycling after max batches."""
        processor = BatchProcessor("embedding", max_batch_size=2)
        items = list(range(10))
        
        def process_func(batch):
            return len(batch)
        
        # Should raise MemoryRecycleRequired after 2 batches
        # Use batch_size_override to ensure consistent batch sizes for testing
        with pytest.raises(MemoryRecycleRequired) as exc_info:
            list(processor.process_batch(items, process_func, batch_size_override=2))
        
        assert "Process recycling needed after 2 batches" in str(exc_info.value)

    @patch("psutil.virtual_memory")
    def test_process_recycling_by_memory_pressure(self, mock_vm):
        """Test process recycling on high memory pressure."""
        mock_vm.return_value = MagicMock(percent=90)  # High memory
        
        processor = BatchProcessor("embedding", max_batch_size=2)
        processor.processed_batches = 1  # Already processed one batch
        
        # Should detect high memory and request recycling
        assert processor._should_recycle_process() is True

    def test_cleanup_after_batch(self):
        """Test cleanup is called after each batch."""
        processor = BatchProcessor("validation", max_batch_size=2)
        items = list(range(5))
        
        # Track gc.collect calls
        with patch("gc.collect") as mock_gc:
            # Use batch_size_override to force specific batch size
            list(processor.process_batch(items, lambda b: len(b), batch_size_override=2))
            # Should have called gc.collect after each batch
            assert mock_gc.call_count >= 3  # 3 batches for 5 items with size 2

    def test_memory_monitor_integration(self):
        """Test integration with MemoryMonitor."""
        monitor = MemoryMonitor("test_operation")
        processor = BatchProcessor("validation", memory_monitor=monitor)
        items = [1, 2, 3]
        
        def process_func(batch):
            return sum(batch)
        
        results = list(processor.process_batch(items, process_func))
        assert results == [6]  # All items in one batch
        assert processor.total_items_processed == 3


class TestBatchSizeCalculator:
    """Test BatchSizeCalculator class."""

    def test_calculator_initialization(self):
        """Test BatchSizeCalculator initialization."""
        calc = BatchSizeCalculator(window_size=5)
        assert calc.window_size == 5
        assert calc.memory_history == []

    def test_memory_trend_calculation(self):
        """Test memory trend calculation."""
        calc = BatchSizeCalculator()
        
        # Add increasing memory values
        calc.memory_history = [50, 55, 60, 65, 70]
        trend = calc._calculate_trend()
        assert trend > 0  # Positive trend (increasing)
        
        # Add decreasing memory values
        calc.memory_history = [70, 65, 60, 55, 50]
        trend = calc._calculate_trend()
        assert trend < 0  # Negative trend (decreasing)
        
        # Stable memory
        calc.memory_history = [50, 50, 50, 50]
        trend = calc._calculate_trend()
        assert trend == 0  # No trend

    def test_batch_size_calculation_with_memory_pressure(self):
        """Test batch size calculation under different memory conditions."""
        calc = BatchSizeCalculator()
        
        # Critical memory (>= 90%)
        size = calc.calculate_size("database", 95, min_size=10, max_size=100)
        assert size == 10  # Should use minimum
        
        # High memory (80-90%)
        size = calc.calculate_size("database", 85, min_size=10, max_size=100)
        assert 10 < size < 100  # Should be between min and max
        
        # Normal memory (< 80%)
        size = calc.calculate_size("validation", 50, min_size=10, max_size=100)
        assert size == 100  # Should use maximum

    def test_operation_specific_limits_in_calculator(self):
        """Test operation-specific limits in calculator."""
        calc = BatchSizeCalculator()
        
        # Embedding operations capped at 32
        size = calc.calculate_size("embedding", 50, min_size=10, max_size=100)
        assert size <= 32
        
        # Database operations capped at 999
        size = calc.calculate_size("database", 50, min_size=10, max_size=2000)
        assert size <= 999
        
        # Download operations can use full size
        size = calc.calculate_size("download", 50, min_size=10, max_size=500)
        assert size == 500

    def test_memory_trend_adjustment(self):
        """Test batch size adjustment based on memory trend."""
        calc = BatchSizeCalculator()
        
        # Simulate rapidly increasing memory
        calc.memory_history = [50, 55, 60, 65, 70, 75]
        size = calc.calculate_size("validation", 75, min_size=10, max_size=100)
        # Should reduce size due to increasing trend
        assert size < 100
        
        # Simulate decreasing memory
        calc.memory_history = [75, 70, 65, 60, 55, 50]
        size = calc.calculate_size("validation", 50, min_size=10, max_size=100)
        # Should allow larger size due to decreasing trend
        assert size == 100

    def test_window_size_limit(self):
        """Test that memory history respects window size."""
        calc = BatchSizeCalculator(window_size=3)
        
        for i in range(10):
            calc.calculate_size("validation", 50 + i)
        
        # Should only keep last 3 values
        assert len(calc.memory_history) == 3


class TestMemoryRecycleRequired:
    """Test MemoryRecycleRequired exception."""

    def test_exception_message(self):
        """Test exception carries proper message."""
        exc = MemoryRecycleRequired("Test message")
        assert str(exc) == "Test message"

    def test_exception_inheritance(self):
        """Test exception is properly inherited from Exception."""
        exc = MemoryRecycleRequired()
        assert isinstance(exc, Exception)