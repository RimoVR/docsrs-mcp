"""
Test token efficiency for MCP tool descriptions.

This module ensures that MCP tool tutorials remain within token limits
to optimize LLM context usage.
"""

import pytest
from fastapi.testclient import TestClient

from docsrs_mcp.app import app
from docsrs_mcp.validation import count_tokens, validate_tutorial_tokens


class TestTokenEfficiency:
    """Test suite for token efficiency of MCP tool descriptions."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_count_tokens_basic(self):
        """Test basic token counting functionality."""
        # Test simple text
        text = "Hello world"
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 10  # Simple text should be few tokens

    def test_count_tokens_fallback(self):
        """Test fallback to character estimation on error."""
        # Test with invalid encoding to trigger fallback
        text = "Test text for fallback"
        tokens = count_tokens(text, encoding="invalid_encoding")
        # Should fall back to character estimation (len/4)
        assert tokens == len(text) // 4

    def test_validate_tutorial_tokens_under_limit(self):
        """Test validation passes for tutorials under token limit."""
        tutorial = "Short tutorial text that is well under the token limit."
        result = validate_tutorial_tokens(tutorial, max_tokens=200)
        assert result == tutorial

    def test_validate_tutorial_tokens_none(self):
        """Test validation handles None input."""
        result = validate_tutorial_tokens(None)
        assert result is None

    def test_validate_tutorial_tokens_empty(self):
        """Test validation handles empty string."""
        result = validate_tutorial_tokens("")
        assert result is None
        result = validate_tutorial_tokens("   ")
        assert result is None

    def test_mcp_manifest_token_efficiency(self):
        """Test that all MCP tool tutorials are within token limits."""
        response = self.client.get("/mcp/manifest")
        assert response.status_code == 200

        manifest = response.json()
        assert "tools" in manifest

        max_tokens = 200  # Target limit

        for tool in manifest["tools"]:
            tool_name = tool.get("name", "unknown")

            # Check if tutorial exists and is within limits
            if "tutorial" in tool and tool["tutorial"]:
                tutorial = tool["tutorial"]
                tokens = count_tokens(tutorial)

                # Assert tutorial is within token limit
                assert tokens <= max_tokens, (
                    f"Tool '{tool_name}' tutorial exceeds {max_tokens} tokens: "
                    f"{tokens} tokens, {len(tutorial)} chars"
                )

            # Check examples if present
            if "examples" in tool and tool["examples"]:
                for i, example in enumerate(tool["examples"]):
                    if example:
                        example_tokens = count_tokens(example)
                        # Examples should be even more concise
                        assert example_tokens <= 50, (
                            f"Tool '{tool_name}' example {i} is too long: "
                            f"{example_tokens} tokens"
                        )

    def test_specific_tool_tutorials(self):
        """Test token counts for specific MCP tools."""
        response = self.client.get("/mcp/manifest")
        manifest = response.json()

        # Expected tools and their approximate token limits
        expected_tools = {
            "get_crate_summary": 100,  # Should be concise
            "search_items": 100,  # Key search tool
            "get_item_doc": 100,  # Documentation fetcher
            "search_examples": 100,  # Example finder
            "get_module_tree": 100,  # Module structure
            "start_pre_ingestion": 150,  # More complex, allow slightly more
        }

        tools_by_name = {tool["name"]: tool for tool in manifest["tools"]}

        for tool_name, expected_max in expected_tools.items():
            assert tool_name in tools_by_name, (
                f"Tool '{tool_name}' not found in manifest"
            )

            tool = tools_by_name[tool_name]
            if "tutorial" in tool and tool["tutorial"]:
                tokens = count_tokens(tool["tutorial"])
                assert tokens <= expected_max, (
                    f"Tool '{tool_name}' exceeds expected {expected_max} tokens: "
                    f"{tokens} tokens"
                )

    def test_total_manifest_size(self):
        """Test that total manifest size is reasonable."""
        response = self.client.get("/mcp/manifest")
        manifest = response.json()

        # Calculate total tokens across all tutorials
        total_tokens = 0
        for tool in manifest["tools"]:
            if "tutorial" in tool and tool["tutorial"]:
                total_tokens += count_tokens(tool["tutorial"])

        # Total should be under 1000 tokens for all tutorials combined
        assert total_tokens <= 1000, (
            f"Total manifest tutorials exceed 1000 tokens: {total_tokens} tokens"
        )

    def test_optimized_startpreingestion(self):
        """Test that startPreIngestion tutorial was successfully optimized."""
        response = self.client.get("/mcp/manifest")
        manifest = response.json()

        tools_by_name = {tool["name"]: tool for tool in manifest["tools"]}

        assert "start_pre_ingestion" in tools_by_name
        tool = tools_by_name["start_pre_ingestion"]

        if "tutorial" in tool and tool["tutorial"]:
            tutorial = tool["tutorial"]
            tokens = count_tokens(tutorial)

            # Should be well under 200 after optimization
            assert tokens < 150, (
                f"startPreIngestion not optimized enough: {tokens} tokens"
            )

            # Check it still contains key information
            assert "cache" in tutorial.lower()
            assert "monitor" in tutorial.lower() or "health" in tutorial.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
