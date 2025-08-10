"""Tests for enhanced query preprocessing with fuzzy normalization."""

import pytest

from docsrs_mcp.models import SearchItemsRequest


class TestQueryPreprocessing:
    """Test suite for query preprocessing and normalization."""

    def test_unicode_nfkc_normalization(self):
        """Test that Unicode NFKC normalization is applied correctly."""
        # Test compatibility characters
        request = SearchItemsRequest(
            crate_name="test",
            query="ﬀ test",  # Ligature ff
        )
        assert request.query == "ff test"

        # Test combining characters
        request = SearchItemsRequest(
            crate_name="test",
            query="café",  # e with combining acute accent
        )
        # NFKC should normalize to precomposed form
        assert "café" in request.query or "cafe" in request.query

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized correctly."""
        # Multiple spaces
        request = SearchItemsRequest(
            crate_name="test",
            query="async   runtime   test",
        )
        assert request.query == "async runtime test"

        # Tabs and newlines
        request = SearchItemsRequest(
            crate_name="test",
            query="async\truntime\ntest",
        )
        assert request.query == "async runtime test"

        # Leading/trailing whitespace
        request = SearchItemsRequest(
            crate_name="test",
            query="  async runtime  ",
        )
        assert request.query == "async runtime"

    def test_fuzzy_normalization_british_american(self):
        """Test fuzzy normalization for British/American English variations."""
        # Serialise -> serialize
        request = SearchItemsRequest(
            crate_name="test",
            query="serialise data",
        )
        assert request.query == "serialize data"

        # Deserialise -> deserialize
        request = SearchItemsRequest(
            crate_name="test",
            query="deserialise JSON",
        )
        assert request.query == "deserialize JSON"

        # Colour -> color
        request = SearchItemsRequest(
            crate_name="test",
            query="colour settings",
        )
        assert request.query == "color settings"

        # Behaviour -> behavior
        request = SearchItemsRequest(
            crate_name="test",
            query="behaviour patterns",
        )
        assert request.query == "behavior patterns"

        # Synchronise -> synchronize
        request = SearchItemsRequest(
            crate_name="test",
            query="synchronise threads",
        )
        assert request.query == "synchronize threads"

        # Initialise -> initialize
        request = SearchItemsRequest(
            crate_name="test",
            query="initialise system",
        )
        assert request.query == "initialize system"

    def test_fuzzy_normalization_case_preservation(self):
        """Test that fuzzy normalization preserves case patterns."""
        # All caps
        request = SearchItemsRequest(
            crate_name="test",
            query="SERIALISE DATA",
        )
        assert request.query == "SERIALIZE DATA"

        # Title case
        request = SearchItemsRequest(
            crate_name="test",
            query="Serialise Data",
        )
        assert request.query == "Serialize Data"

        # Mixed case in sentence
        request = SearchItemsRequest(
            crate_name="test",
            query="How to Serialise data",
        )
        assert request.query == "How to Serialize data"

    def test_fuzzy_normalization_whole_words_only(self):
        """Test that fuzzy normalization only affects whole words."""
        # Should not replace "serialise" within a word
        request = SearchItemsRequest(
            crate_name="test",
            query="serialiser module",
        )
        # "serialiser" should not be changed to "serializer"
        # (only whole word "serialise" is replaced)
        assert request.query == "serialiser module"

        # Multiple words, only exact matches replaced
        request = SearchItemsRequest(
            crate_name="test",
            query="serialise and deserialise",
        )
        assert request.query == "serialize and deserialize"

    def test_quick_check_optimization(self):
        """Test that already normalized text skips normalization."""
        # Already normalized text should pass through quickly
        request = SearchItemsRequest(
            crate_name="test",
            query="async runtime",
        )
        assert request.query == "async runtime"

        # Text that needs normalization
        request = SearchItemsRequest(
            crate_name="test",
            query="async  runtime",  # Double space
        )
        assert request.query == "async runtime"

    def test_empty_query_validation(self):
        """Test that empty queries are properly rejected."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchItemsRequest(
                crate_name="test",
                query=None,
            )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchItemsRequest(
                crate_name="test",
                query="",
            )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchItemsRequest(
                crate_name="test",
                query="   ",  # Only whitespace
            )

    def test_query_length_validation(self):
        """Test that overly long queries are rejected."""
        # Create a 501 character query
        long_query = "a" * 501
        with pytest.raises(ValueError, match="Query too long"):
            SearchItemsRequest(
                crate_name="test",
                query=long_query,
            )

        # 500 characters should be OK
        ok_query = "a" * 500
        request = SearchItemsRequest(
            crate_name="test",
            query=ok_query,
        )
        assert len(request.query) == 500

    def test_international_queries(self):
        """Test that international characters are handled correctly."""
        # Japanese
        request = SearchItemsRequest(
            crate_name="test",
            query="非同期 runtime",
        )
        assert "非同期" in request.query
        assert "runtime" in request.query

        # Russian
        request = SearchItemsRequest(
            crate_name="test",
            query="асинхронный runtime",
        )
        assert "асинхронный" in request.query
        assert "runtime" in request.query

        # Arabic (RTL)
        request = SearchItemsRequest(
            crate_name="test",
            query="متزامن async",
        )
        assert "متزامن" in request.query
        assert "async" in request.query

    def test_combined_normalizations(self):
        """Test that all normalizations work together correctly."""
        # Combine Unicode, whitespace, and fuzzy normalization
        request = SearchItemsRequest(
            crate_name="test",
            query="  ﬀ  serialise   colour  behaviour  ",
        )
        assert request.query == "ff serialize color behavior"

        # Complex case with multiple normalizations
        request = SearchItemsRequest(
            crate_name="test",
            query="Initialise\tthe\nsynchronise\r\nbehaviour",
        )
        assert request.query == "Initialize the synchronize behavior"
