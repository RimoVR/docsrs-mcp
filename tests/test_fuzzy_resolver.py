"""Unit tests for fuzzy path resolution."""

import tempfile
import time
from pathlib import Path

import aiosqlite
import pytest

import docsrs_mcp.fuzzy_resolver
from docsrs_mcp.fuzzy_resolver import (
    calculate_path_bonus,
    composite_path_score,
    get_adaptive_threshold,
    get_fuzzy_suggestions,
    get_fuzzy_suggestions_with_fallback,
    normalize_query,
    resolve_path_alias,
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
                "tokio::task::spawn",
                "tokio::spawn_blocking",
                "tokio::spawn_local",
                "tokio::runtime::Runtime",
                "tokio::task::JoinHandle",
                "serde::ser::Serialize",
                "serde::de::Deserialize",
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

        # Should find tokio::task::spawn as a suggestion
        assert "tokio::task::spawn" in suggestions

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
        assert "tokio::task::spawn" in suggestions

    @pytest.mark.asyncio
    async def test_partial_path_matching(self, temp_db):
        """Test fuzzy matching with partial paths."""
        suggestions = await get_fuzzy_suggestions(
            query="task::spawn",  # More specific partial path
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

    @pytest.mark.asyncio
    async def test_resolve_path_alias_serde(self):
        """Test alias resolution for serde paths."""
        # Test serde aliases
        assert (
            await resolve_path_alias("serde", "serde::Deserialize")
            == "serde::de::Deserialize"
        )
        assert (
            await resolve_path_alias("serde", "serde::Serialize")
            == "serde::ser::Serialize"
        )
        assert (
            await resolve_path_alias("serde", "serde::Serializer")
            == "serde::ser::Serializer"
        )
        assert (
            await resolve_path_alias("serde", "serde::Deserializer")
            == "serde::de::Deserializer"
        )

    @pytest.mark.asyncio
    async def test_resolve_path_alias_tokio(self):
        """Test alias resolution for tokio paths."""
        # Test tokio aliases
        assert await resolve_path_alias("tokio", "tokio::spawn") == "tokio::task::spawn"
        assert (
            await resolve_path_alias("tokio", "tokio::JoinHandle")
            == "tokio::task::JoinHandle"
        )
        assert (
            await resolve_path_alias("tokio", "tokio::select")
            == "tokio::macros::select"
        )

    @pytest.mark.asyncio
    async def test_resolve_path_alias_std(self):
        """Test alias resolution for std paths."""
        # Test std aliases
        assert (
            await resolve_path_alias("std", "std::HashMap")
            == "std::collections::HashMap"
        )
        assert (
            await resolve_path_alias("std", "std::HashSet")
            == "std::collections::HashSet"
        )
        assert (
            await resolve_path_alias("std", "std::BTreeMap")
            == "std::collections::BTreeMap"
        )
        assert (
            await resolve_path_alias("std", "std::BTreeSet")
            == "std::collections::BTreeSet"
        )
        assert (
            await resolve_path_alias("std", "std::VecDeque")
            == "std::collections::VecDeque"
        )
        assert await resolve_path_alias("std", "std::Vec") == "std::vec::Vec"
        assert await resolve_path_alias("std", "std::Result") == "std::result::Result"
        assert await resolve_path_alias("std", "std::Option") == "std::option::Option"

    @pytest.mark.asyncio
    async def test_resolve_path_alias_no_alias(self):
        """Test that non-aliased paths return unchanged."""
        # Non-existent alias should return original
        assert await resolve_path_alias("unknown", "foo::bar") == "foo::bar"
        assert (
            await resolve_path_alias("serde", "serde::json::Value")
            == "serde::json::Value"
        )
        assert (
            await resolve_path_alias("tokio", "tokio::net::TcpListener")
            == "tokio::net::TcpListener"
        )

    @pytest.mark.asyncio
    async def test_resolve_path_alias_crate_level(self):
        """Test that crate-level paths are not altered."""
        # Crate-level path should remain unchanged
        assert await resolve_path_alias("serde", "crate") == "crate"
        assert await resolve_path_alias("tokio", "crate") == "crate"
        assert await resolve_path_alias("std", "crate") == "crate"

    @pytest.mark.asyncio
    async def test_resolve_path_alias_edge_cases(self):
        """Test edge cases for alias resolution."""
        # Empty strings
        assert await resolve_path_alias("", "") == ""
        assert await resolve_path_alias("serde", "") == ""
        assert (
            await resolve_path_alias("", "serde::Deserialize")
            == "serde::de::Deserialize"
        )

        # Whitespace (should not be aliased)
        assert (
            await resolve_path_alias("serde", " serde::Deserialize")
            == " serde::Deserialize"
        )
        assert (
            await resolve_path_alias("serde", "serde::Deserialize ")
            == "serde::Deserialize "
        )


class TestEnhancedFuzzyMatching:
    """Test the enhanced fuzzy matching functionality."""

    def test_normalize_query(self):
        """Test Unicode normalization."""
        # Test basic normalization
        assert normalize_query("hello") is not None

        # Test Unicode normalization (é can be single char or e + combining accent)
        assert normalize_query("café") == normalize_query("café")

        # Test empty string
        assert normalize_query("") == ""

        # Test with spaces and special chars
        normalized = normalize_query("Vec<T>")
        assert normalized is not None  # Should handle brackets

    def test_composite_path_score(self):
        """Test composite scoring algorithm."""
        # Exact match should score very high
        score = composite_path_score("tokio::spawn", "tokio::spawn")
        assert score >= 0.95

        # Similar paths should score well
        score = composite_path_score("spawn", "tokio::task::spawn")
        assert 0.4 <= score <= 0.9  # Enhanced scoring may give higher scores

        # Order variations should still match
        score = composite_path_score("spawn task tokio", "tokio::task::spawn")
        assert score >= 0.5

        # Completely different should score low
        score = composite_path_score("serde::Serialize", "tokio::spawn")
        assert score < 0.3

        # Partial matches should work
        score = composite_path_score("HashMap", "std::collections::HashMap")
        assert score >= 0.5

    def test_calculate_path_bonus(self):
        """Test path component bonus calculation."""
        # Exact final component match
        bonus = calculate_path_bonus("spawn", "tokio::task::spawn")
        assert bonus > 0.1  # Should get path_component_bonus

        # Partial final component match
        bonus = calculate_path_bonus("Hash", "std::collections::HashMap")
        assert 0.05 <= bonus <= 0.1  # Should get partial_component_bonus

        # No match in final component
        bonus = calculate_path_bonus("Vec", "std::collections::HashMap")
        assert bonus == 0.0

        # Full path exact match
        bonus = calculate_path_bonus("tokio::spawn", "tokio::spawn")
        assert bonus > 0.1

        # Empty inputs
        bonus = calculate_path_bonus("", "tokio::spawn")
        assert bonus == 0.0

        bonus = calculate_path_bonus("spawn", "")
        assert bonus == 0.0

    def test_get_adaptive_threshold(self):
        """Test adaptive threshold calculation."""
        # Very short queries (<=5 chars)
        assert get_adaptive_threshold("Vec") == 0.55
        assert get_adaptive_threshold("spawn") == 0.55

        # Medium queries (6-10 chars)
        assert get_adaptive_threshold("HashMap") == 0.60
        assert get_adaptive_threshold("Serialize") == 0.60

        # Longer queries (11-20 chars)
        assert get_adaptive_threshold("tokio::task::spawn") == 0.63

        # Very long queries (>20 chars)
        assert get_adaptive_threshold("std::collections::HashMap::new") == 0.65

        # Edge cases
        assert get_adaptive_threshold("") == 0.55
        assert get_adaptive_threshold("   ") == 0.55  # Whitespace

    @pytest.mark.asyncio
    async def test_enhanced_fuzzy_matching_integration(self, temp_db):
        """Test that enhanced scoring improves fuzzy matching."""
        # Test with a short query that should benefit from adaptive threshold
        suggestions = await get_fuzzy_suggestions(
            query="spawn",
            db_path=str(temp_db),
            crate_name="tokio",
            version="1.0.0",
            limit=3,
            threshold=0.6,
        )

        # Should find spawn-related functions with enhanced scoring
        assert any("spawn" in s.lower() for s in suggestions)

        # Test partial path matching
        suggestions = await get_fuzzy_suggestions(
            query="HashMap",
            db_path=str(temp_db),
            crate_name="std",
            version="1.0.0",
            limit=3,
            threshold=0.6,
        )

        # Should find HashMap even without full path
        assert "std::collections::HashMap" in suggestions

    @pytest.mark.asyncio
    async def test_unicode_handling(self, temp_db):
        """Test Unicode normalization in fuzzy matching."""
        # Add a path with Unicode characters to the database
        async with aiosqlite.connect(temp_db) as db:
            await db.execute(
                "INSERT INTO embeddings (item_path, header, content, embedding) VALUES (?, ?, ?, ?)",
                (
                    "café::résumé",
                    "Unicode test",
                    "Unicode content",
                    b"dummy_embedding",
                ),
            )
            await db.commit()

        # Clear cache to ensure fresh data
        docsrs_mcp.fuzzy_resolver._path_cache.clear()

        # Test with different Unicode representations
        suggestions = await get_fuzzy_suggestions(
            query="cafe::resume",  # ASCII version
            db_path=str(temp_db),
            crate_name="test",
            version="1.0.0",
            limit=3,
            threshold=0.5,
        )

        # Should handle Unicode normalization
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, temp_db):
        """Test performance with many paths."""
        # Add many paths to test performance
        async with aiosqlite.connect(temp_db) as db:
            for i in range(1000):
                await db.execute(
                    "INSERT INTO embeddings (item_path, header, content, embedding) VALUES (?, ?, ?, ?)",
                    (
                        f"module_{i}::function_{i}",
                        f"Header {i}",
                        f"Content {i}",
                        b"dummy",
                    ),
                )
            await db.commit()

        # Clear cache
        docsrs_mcp.fuzzy_resolver._path_cache.clear()

        start = time.time()

        suggestions = await get_fuzzy_suggestions(
            query="module_500::function_500",
            db_path=str(temp_db),
            crate_name="test",
            version="1.0.0",
            limit=5,
            threshold=0.6,
        )

        elapsed = time.time() - start

        # Should complete within reasonable time (< 1 second)
        assert elapsed < 1.0
        # Should find the exact match
        assert "module_500::function_500" in suggestions

    def test_composite_scoring_weights(self):
        """Test that composite scoring uses configurable weights correctly."""
        # Test with known inputs where different algorithms excel

        # token_set_ratio excels with subsets
        query = "HashMap"
        candidate = "std::collections::HashMap"
        score = composite_path_score(query, candidate)
        assert score > 0.5  # Should match well due to subset

        # token_sort_ratio excels with reordered words
        query = "spawn task tokio"
        candidate = "tokio::task::spawn"
        score = composite_path_score(query, candidate)
        assert score > 0.4  # Should match despite order

        # partial_ratio excels with substrings
        query = "Deserialize"
        candidate = "serde::de::Deserialize"
        score = composite_path_score(query, candidate)
        assert score > 0.5  # Should match the substring

    def test_path_bonus_edge_cases(self):
        """Test edge cases in path bonus calculation."""
        # Single component paths
        bonus = calculate_path_bonus("spawn", "spawn")
        assert bonus > 0.1  # Exact match

        # Multiple :: separators
        bonus = calculate_path_bonus("spawn", "tokio::::task::::spawn")
        assert bonus > 0.1  # Should handle empty components

        # Case insensitive matching
        bonus = calculate_path_bonus("SPAWN", "tokio::task::spawn")
        assert bonus > 0.1  # Should match case-insensitively
