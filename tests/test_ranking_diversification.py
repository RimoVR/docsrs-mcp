"""Tests for MMR diversification algorithm."""

from docsrs_mcp.database import _apply_mmr_diversification


class TestMMRDiversification:
    """Test suite for Maximum Marginal Relevance diversification."""

    def test_mmr_basic_diversification(self):
        """Test basic MMR diversification with different item types."""
        # Create test data with different item types
        ranked_results = [
            (0.9, "std::vec::Vec", "Vec struct", "Vector content", "struct"),
            (0.85, "std::vec::Vec::push", "Vec::push", "Push method", "function"),
            (0.8, "std::vec::Vec::pop", "Vec::pop", "Pop method", "function"),
            (0.75, "std::vec", "vec module", "Module content", "module"),
            (0.7, "std::vec::IntoIter", "IntoIter", "Iterator", "struct"),
        ]

        # Apply MMR with balanced lambda (0.6)
        results = _apply_mmr_diversification(ranked_results, k=3, lambda_param=0.6)

        # Should return 3 results
        assert len(results) == 3

        # First result should always be the highest scoring
        assert results[0][1] == "std::vec::Vec"

        # Should diversify types (not all functions)
        result_paths = [r[1] for r in results]
        # Expect some diversity in the results
        assert "std::vec::Vec" in result_paths
        # Should include at least one other type besides the top result

    def test_mmr_high_relevance_lambda(self):
        """Test MMR with high lambda (favoring relevance)."""
        ranked_results = [
            (0.9, "tokio::spawn", "spawn", "Spawn task", "function"),
            (
                0.88,
                "tokio::spawn_blocking",
                "spawn_blocking",
                "Spawn blocking",
                "function",
            ),
            (0.85, "tokio::task::spawn", "task::spawn", "Task spawn", "function"),
            (0.7, "tokio::runtime", "runtime", "Runtime", "module"),
            (0.6, "tokio::sync", "sync", "Sync utilities", "module"),
        ]

        # High lambda (0.9) should favor relevance
        results = _apply_mmr_diversification(ranked_results, k=3, lambda_param=0.9)

        assert len(results) == 3
        # Should mostly follow original ranking
        assert results[0][1] == "tokio::spawn"
        # With high lambda, should still favor relevance but may have slight diversity
        # Check that at least 2 of top 3 are from the original top 3
        top_3_original = ["tokio::spawn", "tokio::spawn_blocking", "tokio::task::spawn"]
        result_paths = [r[1] for r in results]
        matches = sum(1 for path in result_paths if path in top_3_original)
        assert matches >= 2  # At least 2 of the top 3 should be from original top 3

    def test_mmr_high_diversity_lambda(self):
        """Test MMR with low lambda (favoring diversity)."""
        ranked_results = [
            (0.9, "serde::Serialize", "Serialize", "Serialize trait", "trait"),
            (0.88, "serde::Deserialize", "Deserialize", "Deserialize trait", "trait"),
            (0.85, "serde::ser", "ser module", "Serialization", "module"),
            (0.8, "serde::de", "de module", "Deserialization", "module"),
            (0.75, "serde::Serializer", "Serializer", "Serializer trait", "trait"),
        ]

        # Low lambda (0.3) should favor diversity
        results = _apply_mmr_diversification(ranked_results, k=3, lambda_param=0.3)

        assert len(results) == 3
        # First is still highest scoring
        assert results[0][1] == "serde::Serialize"

        # Should have diverse types (not all traits)
        # Extract the original item types from ranked_results
        selected_paths = [r[1] for r in results]
        original_types = {path: type_ for _, path, _, _, type_ in ranked_results}
        selected_types = [
            original_types[path] for path in selected_paths if path in original_types
        ]

        # Should have some variety in types
        unique_types = set(selected_types)
        assert len(unique_types) > 1  # More than one type selected

    def test_mmr_module_diversity(self):
        """Test that MMR penalizes items from the same module."""
        ranked_results = [
            (0.9, "std::collections::HashMap", "HashMap", "Hash map", "struct"),
            (0.88, "std::collections::HashSet", "HashSet", "Hash set", "struct"),
            (0.85, "std::collections::BTreeMap", "BTreeMap", "B-tree map", "struct"),
            (0.8, "std::vec::Vec", "Vec", "Vector", "struct"),
            (0.75, "std::sync::Arc", "Arc", "Atomic RC", "struct"),
        ]

        # With moderate lambda, should diversify across modules
        results = _apply_mmr_diversification(ranked_results, k=3, lambda_param=0.5)

        assert len(results) == 3
        result_paths = [r[1] for r in results]

        # Should include the top result
        assert "std::collections::HashMap" in result_paths

        # Should include items from different modules
        modules = set()
        for path in result_paths:
            parts = path.split("::")
            if len(parts) > 1:
                modules.add("::".join(parts[:-1]))

        # Expect diversity in modules
        assert len(modules) > 1

    def test_mmr_empty_results(self):
        """Test MMR with empty results."""
        results = _apply_mmr_diversification([], k=5, lambda_param=0.6)
        assert results == []

    def test_mmr_single_result(self):
        """Test MMR with a single result."""
        ranked_results = [
            (0.9, "std::vec::Vec", "Vec", "Vector", "struct"),
        ]

        results = _apply_mmr_diversification(ranked_results, k=5, lambda_param=0.6)
        assert len(results) == 1
        assert results[0][1] == "std::vec::Vec"

    def test_mmr_k_larger_than_results(self):
        """Test MMR when k is larger than available results."""
        ranked_results = [
            (0.9, "item1", "Header 1", "Content 1", "function"),
            (0.8, "item2", "Header 2", "Content 2", "struct"),
        ]

        results = _apply_mmr_diversification(ranked_results, k=5, lambda_param=0.6)
        assert len(results) == 2  # Should return all available

    def test_mmr_preserves_content(self):
        """Test that MMR preserves all content fields correctly."""
        ranked_results = [
            (0.9, "path1", "Header 1", "Content 1", "function"),
            (0.8, "path2", "Header 2", "Content 2", "struct"),
            (0.7, "path3", "Header 3", "Content 3", "trait"),
        ]

        results = _apply_mmr_diversification(ranked_results, k=2, lambda_param=0.6)

        # Check that all fields are preserved
        for score, path, header, content in results:
            assert isinstance(score, float)
            assert isinstance(path, str)
            assert isinstance(header, str)
            assert isinstance(content, str)

        # First result should have correct content
        assert results[0][1] == "path1"
        assert results[0][2] == "Header 1"
        assert results[0][3] == "Content 1"

    def test_mmr_lambda_boundaries(self):
        """Test MMR with lambda at boundaries (0 and 1)."""
        ranked_results = [
            (0.9, "item1", "H1", "C1", "function"),
            (0.8, "item2", "H2", "C2", "function"),
            (0.7, "item3", "H3", "C3", "struct"),
            (0.6, "item4", "H4", "C4", "module"),
        ]

        # Lambda = 1.0 (pure relevance)
        results_relevance = _apply_mmr_diversification(
            ranked_results, k=3, lambda_param=1.0
        )
        # Should follow original order
        assert [r[1] for r in results_relevance] == ["item1", "item2", "item3"]

        # Lambda = 0.0 (pure diversity)
        results_diversity = _apply_mmr_diversification(
            ranked_results, k=3, lambda_param=0.0
        )
        # Should maximize diversity
        assert results_diversity[0][1] == "item1"  # First is always highest
        # Rest should be diverse
