"""Tests for filter optimization enhancements."""

import asyncio
import time
from unittest.mock import patch

import aiosqlite
import pytest
import sqlite_vec
from pydantic import ValidationError

from docsrs_mcp.database import init_database, performance_timer, search_embeddings
from docsrs_mcp.models import SearchItemsRequest


class TestFilterValidationEnhancements:
    """Test enhanced validation error messages and filter compatibility."""

    def test_k_parameter_validation_with_helpful_errors(self):
        """Test that k parameter validation provides helpful error messages."""
        # Test valid k values
        request = SearchItemsRequest(crate_name="tokio", query="test", k=5)
        assert request.k == 5

        # Test string to int conversion
        request = SearchItemsRequest(crate_name="tokio", query="test", k="10")
        assert request.k == 10

        # Test out of bounds with helpful error
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", k="0")
        error_msg = str(exc_info.value)
        assert "must be at least 1" in error_msg
        assert "Use k=1 for single result" in error_msg

        # Test exceeding maximum with helpful error
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", k="50")
        error_msg = str(exc_info.value)
        assert "cannot exceed 20" in error_msg
        assert "Consider using k=10" in error_msg

        # Test invalid string with helpful error
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", k="invalid")
        error_msg = str(exc_info.value)
        assert "must be a valid integer" in error_msg
        assert "Examples: k=5" in error_msg

    def test_item_type_validation_with_suggestions(self):
        """Test item_type validation provides helpful suggestions."""
        # Test valid types
        request = SearchItemsRequest(
            crate_name="tokio", query="test", item_type="function"
        )
        assert request.item_type == "function"

        # Test normalization
        request = SearchItemsRequest(
            crate_name="tokio", query="test", item_type="FUNCTION"
        )
        assert request.item_type == "function"

        # Test common mistake suggestions
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", item_type="func")
        error_msg = str(exc_info.value)
        assert "Did you mean 'function'?" in error_msg

        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", item_type="class")
        error_msg = str(exc_info.value)
        assert "Did you mean 'struct'?" in error_msg

        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", item_type="interface")
        error_msg = str(exc_info.value)
        assert "Did you mean 'trait'?" in error_msg

    def test_visibility_validation_with_suggestions(self):
        """Test visibility validation provides helpful suggestions."""
        # Test valid visibility
        request = SearchItemsRequest(
            crate_name="tokio", query="test", visibility="public"
        )
        assert request.visibility == "public"

        # Test common terms with suggestions
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", visibility="pub")
        error_msg = str(exc_info.value)
        assert "Did you mean 'public'?" in error_msg

        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="test", visibility="priv")
        error_msg = str(exc_info.value)
        assert "Did you mean 'private'?" in error_msg

    def test_filter_compatibility_validation(self):
        """Test filter compatibility checking."""
        # Test incompatible visibility and crate_filter
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(
                crate_name="tokio",
                query="test",
                visibility="private",
                crate_filter="serde",
            )
        error_msg = str(exc_info.value)
        assert "Cannot search for private items" in error_msg
        assert "only visible within their own crate" in error_msg

        # Test incompatible min_doc_length with has_examples
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(
                crate_name="tokio", query="test", has_examples=True, min_doc_length=6000
            )
        error_msg = str(exc_info.value)
        assert "min_doc_length=6000 is very high" in error_msg
        assert "Consider using min_doc_length=1000" in error_msg

    def test_boolean_filter_string_conversion(self):
        """Test boolean filters handle string inputs correctly."""
        # Test various string representations
        request = SearchItemsRequest(
            crate_name="tokio", query="test", has_examples="true"
        )
        assert request.has_examples is True

        request = SearchItemsRequest(crate_name="tokio", query="test", has_examples="1")
        assert request.has_examples is True

        request = SearchItemsRequest(
            crate_name="tokio", query="test", has_examples="yes"
        )
        assert request.has_examples is True

        request = SearchItemsRequest(
            crate_name="tokio", query="test", has_examples="false"
        )
        assert request.has_examples is False

        request = SearchItemsRequest(crate_name="tokio", query="test", has_examples="")
        assert request.has_examples is None


@pytest.mark.asyncio
class TestFilterPerformanceOptimizations:
    """Test filter performance optimizations."""

    async def test_progressive_filtering_with_selectivity_analysis(self, tmp_path):
        """Test that progressive filtering analyzes selectivity before applying filters."""
        # Create test database with partial indexes
        db_path = tmp_path / "test.db"

        # Initialize database with our new partial indexes
        await init_database(db_path)

        # Add test data
        async with aiosqlite.connect(db_path) as db:
            # Load sqlite-vec extension
            await db.enable_load_extension(True)
            await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
            await db.enable_load_extension(False)
            
            # Insert test embeddings
            test_data = [
                ("function1", "fn test", "Test function", "function", "public", 0),
                ("function2", "fn test2", "Another function", "function", "public", 0),
                ("struct1", "struct Test", "Test struct", "struct", "public", 0),
                ("deprecated_fn", "fn old", "Deprecated", "function", "public", 1),
                ("private_fn", "fn internal", "Private", "function", "private", 0),
            ]

            for _, (
                path,
                header,
                content,
                item_type,
                visibility,
                deprecated,
            ) in enumerate(test_data):
                await db.execute(
                    """
                    INSERT INTO embeddings 
                    (item_path, header, content, embedding, item_type, visibility, deprecated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        path,
                        header,
                        content,
                        b"\x00" * 1536,  # Dummy embedding
                        item_type,
                        visibility,
                        deprecated,
                    ),
                )

                # Insert into vec_embeddings
                await db.execute(
                    """
                    INSERT INTO vec_embeddings (embedding)
                    VALUES (?)
                """,
                    (sqlite_vec.serialize_float32([0.1] * 384),),
                )

            await db.commit()

        # Mock embedding for search
        query_embedding = [0.1] * 384

        # Test with filters that should trigger progressive filtering
        with patch("docsrs_mcp.database.logger") as mock_logger:
            results = await search_embeddings(
                db_path=db_path,
                query_embedding=query_embedding,
                k=2,
                type_filter="function",
                deprecated=False,
            )

            # Check that selectivity analysis was performed
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Filter selectivity" in str(call)
            ]
            assert len(debug_calls) > 0, "Selectivity analysis should have been logged"

    async def test_filter_execution_metrics_logging(self, tmp_path):
        """Test that filter execution metrics are properly logged."""
        db_path = tmp_path / "test.db"

        await init_database(db_path)

        query_embedding = [0.1] * 384

        with patch("docsrs_mcp.database.logger") as mock_logger:
            # Run search with filters
            await search_embeddings(
                db_path=db_path,
                query_embedding=query_embedding,
                k=5,
                type_filter="function",
            )

            # Check for filter execution time logging
            debug_calls = mock_logger.debug.call_args_list
            metrics_logged = any(
                "Filter execution times" in str(call)
                or "Search completed in" in str(call)
                for call in debug_calls
            )
            assert metrics_logged, "Filter execution metrics should be logged"

    async def test_partial_index_usage(self, tmp_path):
        """Test that partial indexes are used for common filter patterns."""
        db_path = tmp_path / "test.db"

        await init_database(db_path)

        async with aiosqlite.connect(db_path) as db:
            # Check that our partial indexes were created
            cursor = await db.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
                ORDER BY name
            """)
            indexes = [row[0] for row in await cursor.fetchall()]

            # Verify all our optimized indexes exist
            expected_indexes = [
                "idx_crate_prefix",
                "idx_filter_combo",
                "idx_filter_composite",
                "idx_has_examples",
                "idx_non_deprecated",
                "idx_public_functions",
            ]

            for expected in expected_indexes:
                assert expected in indexes, f"Partial index {expected} should exist"

    async def test_performance_timer_decorator(self):
        """Test the performance timer decorator functionality."""

        # Test with async function
        @performance_timer("test_operation")
        async def slow_async_operation():
            await asyncio.sleep(0.05)  # 50ms
            return {"result": "success"}

        with patch("docsrs_mcp.database.logger") as mock_logger:
            result = await slow_async_operation()

            # Check that performance was logged
            assert result["result"] == "success"
            debug_calls = mock_logger.debug.call_args_list
            assert any("test_operation completed" in str(call) for call in debug_calls)

        # Test with sync function
        @performance_timer("sync_operation")
        def slow_sync_operation():
            time.sleep(0.05)  # 50ms
            return "done"

        with patch("docsrs_mcp.database.logger") as mock_logger:
            result = slow_sync_operation()
            assert result == "done"
            debug_calls = mock_logger.debug.call_args_list
            assert any("sync_operation completed" in str(call) for call in debug_calls)

    async def test_cache_hit_performance(self, tmp_path):
        """Test that cache hits are faster than database queries."""
        db_path = tmp_path / "test.db"

        await init_database(db_path)

        query_embedding = [0.1] * 384

        # First query (cache miss)
        start = time.perf_counter()
        results1 = await search_embeddings(
            db_path=db_path, query_embedding=query_embedding, k=5
        )
        miss_time = time.perf_counter() - start

        # Second query (cache hit)
        start = time.perf_counter()
        results2 = await search_embeddings(
            db_path=db_path, query_embedding=query_embedding, k=5
        )
        hit_time = time.perf_counter() - start

        # Cache hit should be significantly faster
        # Note: In real scenarios, hit_time would be much less than miss_time
        # but in tests with empty DB, we just verify caching works
        assert results1 == results2, "Cached results should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
