"""Tests for workflow enhancement service."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docsrs_mcp.services.workflow_service import DetailLevel, WorkflowService


@pytest.fixture
def workflow_service():
    """Create a workflow service instance."""
    return WorkflowService()


@pytest.fixture
def mock_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        yield tmp.name


class TestProgressiveDetailLevels:
    """Test progressive detail level documentation."""

    @pytest.mark.asyncio
    async def test_get_summary_level(self, workflow_service, mock_db_path):
        """Test getting summary level documentation."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                mock_cursor = AsyncMock()
                mock_cursor.fetchone.return_value = (
                    "test::item",  # item_path
                    "Test item summary",  # header
                    "Full documentation",  # content
                    "fn test() -> String",  # signature
                    "function",  # item_type
                    "example code",  # examples
                    "public",  # visibility
                    False,  # deprecated
                    "<T>",  # generic_params
                    "T: Display",  # trait_bounds
                )

                mock_db = AsyncMock()
                mock_db.execute.return_value = mock_cursor
                mock_connect.return_value.__aenter__.return_value = mock_db

                result = await workflow_service.get_documentation_with_detail_level(
                    "test_crate", "test::item", DetailLevel.SUMMARY
                )

                assert result["detail_level"] == DetailLevel.SUMMARY
                assert result["summary"] == "Test item summary"
                assert result["signature"] == "fn test() -> String"
                assert "documentation" not in result
                assert "examples" not in result

    @pytest.mark.asyncio
    async def test_get_detailed_level(self, workflow_service, mock_db_path):
        """Test getting detailed level documentation."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                mock_cursor = AsyncMock()
                mock_cursor.fetchone.return_value = (
                    "test::item",  # item_path
                    "Test item summary",  # header
                    "Full documentation content",  # content
                    "fn test() -> String",  # signature
                    "function",  # item_type
                    "example code",  # examples
                    "public",  # visibility
                    False,  # deprecated
                    "<T>",  # generic_params
                    "T: Display",  # trait_bounds
                )

                mock_db = AsyncMock()
                mock_db.execute.return_value = mock_cursor
                mock_connect.return_value.__aenter__.return_value = mock_db

                result = await workflow_service.get_documentation_with_detail_level(
                    "test_crate", "test::item", DetailLevel.DETAILED
                )

                assert result["detail_level"] == DetailLevel.DETAILED
                assert result["documentation"] == "Full documentation content"
                assert result["examples"] == "example code"
                assert "generic_params" not in result
                assert "trait_bounds" not in result

    @pytest.mark.asyncio
    async def test_get_expert_level(self, workflow_service, mock_db_path):
        """Test getting expert level documentation."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                # Mock main query cursor
                mock_cursor = AsyncMock()
                mock_cursor.fetchone.return_value = (
                    "test::item",  # item_path
                    "Test item summary",  # header
                    "Full documentation content",  # content
                    "fn test() -> String",  # signature
                    "function",  # item_type
                    "example code",  # examples
                    "public",  # visibility
                    False,  # deprecated
                    "<T>",  # generic_params
                    "T: Display",  # trait_bounds
                )

                # Mock related items cursor
                mock_related_cursor = AsyncMock()
                mock_related_cursor.__aiter__.return_value = [
                    ("test::related", "fn related()", "function"),
                ]

                mock_db = AsyncMock()
                mock_db.execute.side_effect = [mock_cursor, mock_related_cursor]
                mock_connect.return_value.__aenter__.return_value = mock_db

                result = await workflow_service.get_documentation_with_detail_level(
                    "test_crate", "test::item", DetailLevel.EXPERT
                )

                assert result["detail_level"] == DetailLevel.EXPERT
                assert result["generic_params"] == "<T>"
                assert result["trait_bounds"] == "T: Display"
                assert "related_items" in result
                assert len(result["related_items"]) == 1

    @pytest.mark.asyncio
    async def test_caching(self, workflow_service, mock_db_path):
        """Test that results are cached properly."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                mock_cursor = AsyncMock()
                mock_cursor.fetchone.return_value = (
                    "test::item",  # item_path
                    "Test item summary",  # header
                    "Full documentation",  # content
                    "fn test() -> String",  # signature
                    "function",  # item_type
                    None,  # examples
                    "public",  # visibility
                    False,  # deprecated
                    None,  # generic_params
                    None,  # trait_bounds
                )

                mock_db = AsyncMock()
                mock_db.execute.return_value = mock_cursor
                mock_connect.return_value.__aenter__.return_value = mock_db

                # First call
                result1 = await workflow_service.get_documentation_with_detail_level(
                    "test_crate", "test::item", DetailLevel.SUMMARY
                )

                # Second call (should use cache)
                result2 = await workflow_service.get_documentation_with_detail_level(
                    "test_crate", "test::item", DetailLevel.SUMMARY
                )

                # Database should only be called once
                assert mock_ingest.call_count == 1
                assert result1 == result2


class TestUsagePatternExtraction:
    """Test usage pattern extraction."""

    @pytest.mark.asyncio
    async def test_extract_patterns(self, workflow_service, mock_db_path):
        """Test extracting usage patterns from documentation."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                mock_cursor = AsyncMock()
                mock_cursor.__aiter__.return_value = [
                    (
                        "test::func",
                        "fn process(data: Result<String, Error>) -> Option<String>",
                        "let result = data.unwrap();",
                        "function",
                    ),
                    (
                        "test::struct",
                        "struct Handler<T: Display>",
                        "handler.process()",
                        "struct",
                    ),
                ]

                mock_db = AsyncMock()
                mock_db.execute.return_value = mock_cursor
                mock_connect.return_value.__aenter__.return_value = mock_db

                patterns = await workflow_service.extract_usage_patterns(
                    "test_crate", limit=5, min_frequency=1
                )

                assert len(patterns) > 0
                # Check for different pattern types
                pattern_types = {p["pattern_type"] for p in patterns}
                assert "method_call" in pattern_types or "generic_type" in pattern_types

    @pytest.mark.asyncio
    async def test_pattern_categorization(self, workflow_service):
        """Test pattern categorization logic."""
        # These are the actual patterns extracted by the regex
        assert workflow_service._categorize_pattern(".process()") == "method_call"
        assert workflow_service._categorize_pattern("data.unwrap()") == "method_call"
        assert workflow_service._categorize_pattern("Result<T, E>") == "error_handling"
        assert workflow_service._categorize_pattern("Option<T>") == "optional_value"
        assert workflow_service._categorize_pattern("Generic<T>") == "generic_type"
        assert (
            workflow_service._categorize_pattern("impl Trait for Type")
            == "trait_implementation"
        )
        assert workflow_service._categorize_pattern("something_else") == "other"


class TestLearningPathGeneration:
    """Test learning path generation."""

    @pytest.mark.asyncio
    async def test_migration_path(self, workflow_service):
        """Test generating migration path between versions."""
        with patch(
            "docsrs_mcp.services.workflow_service.get_diff_engine"
        ) as mock_get_engine:
            mock_engine = AsyncMock()
            mock_diff_result = MagicMock()
            mock_diff_result.summary.breaking_changes = 3
            mock_diff_result.summary.deprecated_items = 2
            mock_diff_result.summary.added_items = 5
            mock_diff_result.migration_hints = [
                MagicMock(
                    affected_path="test::old_func",
                    suggested_fix="Use test::new_func instead",
                    severity=MagicMock(value="high"),
                )
            ]

            mock_engine.compare_versions.return_value = mock_diff_result
            mock_get_engine.return_value = mock_engine

            result = await workflow_service.generate_learning_path(
                "test_crate", from_version="1.0.0", to_version="2.0.0"
            )

            assert result["type"] == "migration"
            assert result["from_version"] == "1.0.0"
            assert result["to_version"] == "2.0.0"
            assert len(result["steps"]) > 0
            assert "estimated_effort" in result

    @pytest.mark.asyncio
    async def test_onboarding_path(self, workflow_service, mock_db_path):
        """Test generating onboarding path for new users."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                mock_cursor = AsyncMock()
                mock_cursor.__aiter__.return_value = [
                    ("test::module", "module", "Core module"),
                    ("test::Struct", "struct", "Main struct"),
                    ("test::func", "function", "Helper function"),
                    ("test::Trait", "trait", "Core trait"),
                ]

                mock_db = AsyncMock()
                mock_db.execute.return_value = mock_cursor
                mock_connect.return_value.__aenter__.return_value = mock_db

                result = await workflow_service.generate_learning_path(
                    "test_crate", to_version="latest"
                )

                assert result["type"] == "onboarding"
                assert result["from_version"] == "new_user"
                assert len(result["steps"]) >= 3
                assert result["steps"][0]["title"] == "Core Concepts"

    @pytest.mark.asyncio
    async def test_learning_path_with_focus_areas(self, workflow_service, mock_db_path):
        """Test generating learning path with focus areas."""
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            mock_ingest.return_value = mock_db_path

            with patch("aiosqlite.connect") as mock_connect:
                # Mock module query
                mock_module_cursor = AsyncMock()
                mock_module_cursor.__aiter__.return_value = [
                    ("test::module", "module", "Core module"),
                ]

                # Mock focus area query
                mock_focus_cursor = AsyncMock()
                mock_focus_cursor.__aiter__.return_value = [
                    ("test::auth::login", "function", "Login function"),
                    ("test::auth::User", "struct", "User struct"),
                ]

                mock_db = AsyncMock()
                mock_db.execute.side_effect = [mock_module_cursor, mock_focus_cursor]
                mock_connect.return_value.__aenter__.return_value = mock_db

                result = await workflow_service.generate_learning_path(
                    "test_crate", focus_areas=["authentication"]
                )

                assert result["type"] == "onboarding"
                # Should have the standard steps plus focus area
                assert any("Focus" in step["title"] for step in result["steps"])


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_detail_cache(self, workflow_service, mock_db_path):
        """Test detail level caching."""
        cache_key = "test:item:summary:latest"
        cached_result = {"cached": True, "data": "test"}

        workflow_service._detail_cache[cache_key] = cached_result

        # Should return cached result without database call
        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            result = await workflow_service.get_documentation_with_detail_level(
                "test", "item", DetailLevel.SUMMARY, None
            )

            assert result == cached_result
            mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_pattern_cache(self, workflow_service):
        """Test pattern extraction caching."""
        cache_key = "test:latest:patterns"
        cached_patterns = [{"pattern": "test.method()", "frequency": 10}]

        workflow_service._pattern_cache[cache_key] = cached_patterns

        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            result = await workflow_service.extract_usage_patterns(
                "test", None, limit=5
            )

            assert result == cached_patterns
            mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_learning_cache(self, workflow_service):
        """Test learning path caching."""
        cache_key = "test:new:latest"
        cached_path = {"type": "onboarding", "steps": []}

        workflow_service._learning_cache[cache_key] = cached_path

        with patch("docsrs_mcp.services.workflow_service.ingest_crate") as mock_ingest:
            result = await workflow_service.generate_learning_path("test", None, None)

            assert result == cached_path
            mock_ingest.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
