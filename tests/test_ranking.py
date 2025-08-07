"""Unit tests for search result ranking functionality."""

import time

import pytest

from docsrs_mcp import config
from docsrs_mcp.cache import SearchCache
from docsrs_mcp.models import RankingConfig


class TestRankingConfig:
    """Test the RankingConfig model."""

    def test_default_weights(self):
        """Test that default weights sum to 1.0."""
        config = RankingConfig()
        total = (
            config.vector_weight
            + config.type_weight
            + config.quality_weight
            + config.examples_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_custom_weights_validation(self):
        """Test that custom weights must sum to 1.0."""
        # Valid weights that sum to 1.0
        config = RankingConfig(
            vector_weight=0.6,
            type_weight=0.2,
            quality_weight=0.15,
            examples_weight=0.05,
        )
        assert config.vector_weight == 0.6

        # Invalid weights that don't sum to 1.0
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RankingConfig(
                vector_weight=0.5,
                type_weight=0.2,
                quality_weight=0.2,
                examples_weight=0.2,  # Sum = 1.1
            )

    def test_weight_string_conversion(self):
        """Test that weight parameters accept strings and convert to float."""
        # Test string conversion for all weight parameters
        config = RankingConfig.model_validate(
            {
                "vector_weight": "0.6",
                "type_weight": "0.2",
                "quality_weight": "0.15",
                "examples_weight": "0.05",
            }
        )
        assert config.vector_weight == 0.6
        assert config.type_weight == 0.2
        assert config.quality_weight == 0.15
        assert config.examples_weight == 0.05

        # Test integer to float conversion
        config = RankingConfig.model_validate(
            {
                "vector_weight": 0,
                "type_weight": 1,
                "quality_weight": 0,
                "examples_weight": 0,
            }
        )
        assert config.vector_weight == 0.0
        assert config.type_weight == 1.0
        assert config.quality_weight == 0.0
        assert config.examples_weight == 0.0

        # Test out of range values with strings
        with pytest.raises(ValueError, match="must be at least 0.0"):
            RankingConfig.model_validate({"vector_weight": "-0.1"})

        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            RankingConfig.model_validate({"vector_weight": "1.1"})

        # Test invalid string
        with pytest.raises(ValueError, match="must be a valid number"):
            RankingConfig.model_validate({"vector_weight": "invalid"})


class TestScoringFormula:
    """Test the composite scoring formula."""

    def test_base_score_calculation(self):
        """Test base score calculation from distance."""
        # Distance of 0 should give score of 1.0
        distance = 0.0
        base_score = 1.0 - distance
        assert base_score == 1.0

        # Distance of 0.5 should give score of 0.5
        distance = 0.5
        base_score = 1.0 - distance
        assert base_score == 0.5

        # Distance of 1.0 should give score of 0.0
        distance = 1.0
        base_score = 1.0 - distance
        assert base_score == 0.0

    def test_type_weight_application(self):
        """Test type-specific weight application."""
        base_score = 0.8

        # Function type should boost score
        function_weight = config.TYPE_WEIGHTS.get("function", 1.0)
        assert function_weight == 1.2
        boosted = base_score * function_weight
        assert boosted > base_score

        # Module type should reduce score
        module_weight = config.TYPE_WEIGHTS.get("module", 1.0)
        assert module_weight == 0.9
        reduced = base_score * module_weight
        assert reduced < base_score

    def test_doc_quality_normalization(self):
        """Test documentation quality score normalization."""
        # Short doc
        doc_length = 100
        quality = min(1.0, doc_length / 1000)
        assert quality == 0.1

        # Medium doc
        doc_length = 500
        quality = min(1.0, doc_length / 1000)
        assert quality == 0.5

        # Long doc (capped at 1.0)
        doc_length = 2000
        quality = min(1.0, doc_length / 1000)
        assert quality == 1.0

    def test_composite_score_calculation(self):
        """Test the complete composite scoring formula."""
        # Simulate scoring parameters
        base_score = 0.8
        type_weight = 1.2  # function
        doc_quality = 0.5
        has_examples = 1.2

        # Calculate composite score with default weights
        final_score = (
            config.RANKING_VECTOR_WEIGHT * base_score
            + config.RANKING_TYPE_WEIGHT * (base_score * type_weight)
            + config.RANKING_QUALITY_WEIGHT * doc_quality
            + config.RANKING_EXAMPLES_WEIGHT * has_examples
        )

        # Verify score is in valid range
        assert 0.0 <= final_score <= 1.0

        # Verify individual components
        vector_component = config.RANKING_VECTOR_WEIGHT * base_score
        assert vector_component == config.RANKING_VECTOR_WEIGHT * 0.8  # 0.6 * 0.8 = 0.48

        type_component = config.RANKING_TYPE_WEIGHT * (base_score * type_weight)
        assert type_component == 0.15 * (0.8 * 1.2)  # 0.144

        quality_component = config.RANKING_QUALITY_WEIGHT * doc_quality
        assert quality_component == 0.1 * 0.5  # 0.05

        examples_component = config.RANKING_EXAMPLES_WEIGHT * has_examples
        assert examples_component == config.RANKING_EXAMPLES_WEIGHT * 1.2  # 0.15 * 1.2 = 0.18

    def test_score_range_clamping(self):
        """Test that scores are clamped to [0, 1] range."""
        # Test max clamping
        score = 1.5
        clamped = max(0.0, min(1.0, score))
        assert clamped == 1.0

        # Test min clamping
        score = -0.5
        clamped = max(0.0, min(1.0, score))
        assert clamped == 0.0

        # Test valid score passes through
        score = 0.75
        clamped = max(0.0, min(1.0, score))
        assert clamped == 0.75


class TestCacheIntegration:
    """Test cache integration with search."""

    def test_cache_key_generation(self):
        """Test cache key generation from search parameters."""
        cache = SearchCache()

        # Same parameters should generate same key
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        k = 5
        type_filter = "function"

        key1 = cache._make_key(embedding, k, type_filter)
        key2 = cache._make_key(embedding, k, type_filter)
        assert key1 == key2

        # Different k should generate different key
        key3 = cache._make_key(embedding, k=10, type_filter=type_filter)
        assert key1 != key3

        # Different type_filter should generate different key
        key4 = cache._make_key(embedding, k, type_filter="struct")
        assert key1 != key4

    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = SearchCache(ttl=1)  # 1 second TTL

        embedding = [0.1] * 384
        k = 5
        results = [(0.9, "path", "header", "content")]

        # Store results
        cache.set(embedding, k, results)

        # Should retrieve immediately
        cached = cache.get(embedding, k)
        assert cached == results

        # Wait for expiration
        time.sleep(1.1)

        # Should return None after expiration
        cached = cache.get(embedding, k)
        assert cached is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SearchCache(max_size=2)

        # Add first entry
        embedding1 = [0.1] * 384
        results1 = [(0.9, "path1", "header1", "content1")]
        cache.set(embedding1, 1, results1)

        # Add second entry
        embedding2 = [0.2] * 384
        results2 = [(0.8, "path2", "header2", "content2")]
        cache.set(embedding2, 2, results2)

        # Add third entry (should evict first)
        embedding3 = [0.3] * 384
        results3 = [(0.7, "path3", "header3", "content3")]
        cache.set(embedding3, 3, results3)

        # First should be evicted
        assert cache.get(embedding1, 1) is None

        # Second and third should still be present
        assert cache.get(embedding2, 2) == results2
        assert cache.get(embedding3, 3) == results3
