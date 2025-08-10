"""
Fallback extraction modules for DocsRS MCP server.

This package contains extractors for obtaining documentation when rustdoc JSON is unavailable.
"""

from .source_extractor import CratesIoSourceExtractor

__all__ = ["CratesIoSourceExtractor"]
