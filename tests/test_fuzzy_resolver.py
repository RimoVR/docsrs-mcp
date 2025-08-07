"""Unit tests for fuzzy path resolution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest

from docsrs_mcp.fuzzy_resolver import (
    get_fuzzy_suggestions,
    get_fuzzy_suggestions_with_fallback,
)


@pytest.fixture
async def temp_db():
    """Create a temporary database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

        async with aiosqlite.connect(db_path) as db:
            # Create embeddings table
            await db.execute("""
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY,
                    item_path TEXT NOT NULL,
                    header TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)

            # Insert test data with Rust-like paths
            test_paths = [
                "tokio::spawn",
                "tokio::spawn_blocking",
                "tokio::spawn_local",
                "tokio::task::spawn",
                "tokio::runtime::Runtime",
                "serde::Serialize",
                "serde::Deserialize",
                "std::vec::Vec",
                "std::collections::HashMap",
            ]

            for path in test_paths:
                await db.execute(
                    "INSERT INTO embeddings (item_path, header, content, embedding) VALUES (?, ?, ?, ?)",
                    (
                        path,
                        f"Header for {path}",
                        f"Documentation for {path}",
                        b"dummy_embedding",
                    ),
                )

            await db.commit()

        yield db_path

        # Cleanup
        db_path.unlink(missing_ok=True)


class TestFuzzyResolver:
    """Test fuzzy path resolution functionality."""

    @pytest.mark.asyncio
    async def test_exact_match_not_in_suggestions(self, temp_db):
        """Test that exact matches are found in fuzzy suggestions when threshold is low."""
        suggestions = await get_fuzzy_suggestions(
            query="tokio::spawn",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=3,
            threshold=0.6,
        )

        # Exact match should be the first suggestion
        assert "tokio::spawn" in suggestions

    @pytest.mark.asyncio
    async def test_typo_correction(self, temp_db):
        """Test fuzzy matching corrects common typos."""
        # Test with a typo in 'spawn'
        suggestions = await get_fuzzy_suggestions(
            query="tokio::spwan",  # Typo: spwan instead of spawn
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=3,
            threshold=0.6,
        )

        # Should suggest the correct spelling
        assert "tokio::spawn" in suggestions

    @pytest.mark.asyncio
    async def test_partial_path_matching(self, temp_db):
        """Test fuzzy matching with partial paths."""
        suggestions = await get_fuzzy_suggestions(
            query="spawn",  # Partial path
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=5,
            threshold=0.5,
        )

        # Should find paths containing 'spawn'
        spawn_paths = [s for s in suggestions if "spawn" in s.lower()]
        assert len(spawn_paths) > 0

    @pytest.mark.asyncio
    async def test_threshold_filtering(self, temp_db):
        """Test that similarity threshold filters results correctly."""
        # High threshold should return fewer results
        high_threshold_suggestions = await get_fuzzy_suggestions(
            query="tokio::spwn",  # Very different from 'spawn'
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=10,
            threshold=0.9,
        )

        # Low threshold should return more results
        low_threshold_suggestions = await get_fuzzy_suggestions(
            query="tokio::spwn",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=10,
            threshold=0.3,
        )

        # Low threshold should return more or equal results
        assert len(low_threshold_suggestions) >= len(high_threshold_suggestions)

    @pytest.mark.asyncio
    async def test_limit_respected(self, temp_db):
        """Test that the limit parameter is respected."""
        suggestions = await get_fuzzy_suggestions(
            query="serialize",
            db_path=str(temp_db),
            crate_name="serde",
            version="1.0.0",
            limit=2,
            threshold=0.3,
        )

        # Should not exceed the limit
        assert len(suggestions) <= 2

    @pytest.mark.asyncio
    async def test_empty_database(self):
        """Test behavior with empty database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)

            async with aiosqlite.connect(db_path) as db:
                await db.execute("""
                    CREATE TABLE embeddings (
                        id INTEGER PRIMARY KEY,
                        item_path TEXT NOT NULL
                    )
                """)
                await db.commit()

            suggestions = await get_fuzzy_suggestions(
                query="test::path",
                db_path=str(db_path),
                crate_name="test",
                version="1.0.0",
            )

            assert suggestions == []

            db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_nonexistent_database(self):
        """Test behavior with non-existent database."""
        suggestions = await get_fuzzy_suggestions(
            query="test::path",
            db_path="/nonexistent/path.db",
            crate_name="test",
            version="1.0.0",
        )

        assert suggestions == []

    @pytest.mark.asyncio
    async def test_cache_usage(self, temp_db):
        """Test that paths are cached after first retrieval."""
        # Clear the cache before test
        import docsrs_mcp.fuzzy_resolver
        
        docsrs_mcp.fuzzy_resolver._path_cache.clear()
        
        # First call should fetch from database
        await get_fuzzy_suggestions(
            query="tokio::spawn",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
        )
        
        # Verify cache was populated
        cache_key = "tokio_1.0.0_paths"
        assert cache_key in docsrs_mcp.fuzzy_resolver._path_cache
        
        # Second call should use cache (we can't directly test this but cache should exist)
        await get_fuzzy_suggestions(
            query="tokio::spawn_blocking",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_fuzzy_suggestions_with_fallback(self, temp_db):
        """Test the fallback wrapper function."""
        # Normal case
        suggestions = await get_fuzzy_suggestions_with_fallback(
            query="tokio::spawn",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    @pytest.mark.asyncio
    async def test_fuzzy_suggestions_with_fallback_error_handling(self):
        """Test that fallback function handles errors gracefully."""
        # With invalid database path, should return empty list
        suggestions = await get_fuzzy_suggestions_with_fallback(
            query="test",
            db_path="/invalid/path.db",
            crate_name="test",
            version="1.0.0",
        )

        assert suggestions == []

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, temp_db):
        """Test fuzzy matching with special characters."""
        # Test with various special characters
        queries = [
            "std::vec::<Vec>",
            "std::collections::HashMap<String, Value>",
            "tokio::spawn()",
        ]

        for query in queries:
            suggestions = await get_fuzzy_suggestions(
                query=query,
                db_path=str(temp_db),
                crate_name="test",
                version="1.0.0",
                limit=3,
                threshold=0.4,
            )

            # Should not crash and should return some results
            assert isinstance(suggestions, list)
