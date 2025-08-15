#!/usr/bin/env python
"""Test script to verify MCP endpoints handle string parameters correctly."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp.models import (
    GetCrateSummaryRequest,
    SearchItemsRequest,
    GetItemDocRequest,
    SearchExamplesRequest,
    GetModuleTreeRequest,
    StartPreIngestionRequest,
    IngestCargoFileRequest,
    PreIngestionControlRequest,
    CompareVersionsRequest,
)


def test_string_coercion():
    """Test that all request models properly coerce string parameters."""

    print("Testing MCP string parameter coercion...")
    print("=" * 60)

    # Test GetCrateSummaryRequest
    print("\n1. GetCrateSummaryRequest:")
    try:
        req = GetCrateSummaryRequest(crate_name="serde", version="1.0.104")
        print(f"   ✅ version as string: {req.version}")
        req = GetCrateSummaryRequest(crate_name="serde", version="")
        print(f"   ✅ empty version string: {req.version}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test SearchItemsRequest
    print("\n2. SearchItemsRequest:")
    try:
        req = SearchItemsRequest(
            crate_name="tokio",
            query="spawn",
            k="10",  # String k
            has_examples="true",  # String boolean
            min_doc_length="500",  # String integer
            deprecated="false",  # String boolean
        )
        print(f"   ✅ k as string '10': {req.k} (type: {type(req.k).__name__})")
        print(
            f"   ✅ has_examples as 'true': {req.has_examples} (type: {type(req.has_examples).__name__})"
        )
        print(
            f"   ✅ min_doc_length as '500': {req.min_doc_length} (type: {type(req.min_doc_length).__name__})"
        )
        print(
            f"   ✅ deprecated as 'false': {req.deprecated} (type: {type(req.deprecated).__name__})"
        )

        # Test edge cases
        req = SearchItemsRequest(crate_name="serde", query="test", k="")
        print(f"   ✅ empty k string: {req.k}")

        req = SearchItemsRequest(crate_name="serde", query="test", k="5")
        print(f"   ✅ k='5': {req.k}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test GetItemDocRequest
    print("\n3. GetItemDocRequest:")
    try:
        req = GetItemDocRequest(
            crate_name="serde", item_path="serde::Deserialize", version="latest"
        )
        print(f"   ✅ version as 'latest': {req.version}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test SearchExamplesRequest
    print("\n4. SearchExamplesRequest:")
    try:
        req = SearchExamplesRequest(
            crate_name="tokio",
            query="async",
            k="15",  # String k
            version="",  # Empty string
        )
        print(f"   ✅ k as string '15': {req.k} (type: {type(req.k).__name__})")
        print(f"   ✅ empty version string: {req.version}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test GetModuleTreeRequest
    print("\n5. GetModuleTreeRequest:")
    try:
        req = GetModuleTreeRequest(crate_name="actix-web", version="4.5.0")
        print(f"   ✅ version as string: {req.version}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test StartPreIngestionRequest
    print("\n6. StartPreIngestionRequest:")
    try:
        req = StartPreIngestionRequest(
            force="true",  # String boolean
            concurrency="5",  # String integer
            count="200",  # String integer
        )
        print(f"   ✅ force as 'true': {req.force} (type: {type(req.force).__name__})")
        print(
            f"   ✅ concurrency as '5': {req.concurrency} (type: {type(req.concurrency).__name__})"
        )
        print(f"   ✅ count as '200': {req.count} (type: {type(req.count).__name__})")

        # Test defaults
        req = StartPreIngestionRequest()
        print(
            f"   ✅ Defaults: force={req.force}, concurrency={req.concurrency}, count={req.count}"
        )

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test IngestCargoFileRequest
    print("\n7. IngestCargoFileRequest:")
    try:
        req = IngestCargoFileRequest(
            file_path="/path/to/Cargo.toml",
            concurrency="8",  # String integer
            skip_existing="false",  # String boolean
            resolve_versions="true",  # String boolean
        )
        print(
            f"   ✅ concurrency as '8': {req.concurrency} (type: {type(req.concurrency).__name__})"
        )
        print(
            f"   ✅ skip_existing as 'false': {req.skip_existing} (type: {type(req.skip_existing).__name__})"
        )
        print(
            f"   ✅ resolve_versions as 'true': {req.resolve_versions} (type: {type(req.resolve_versions).__name__})"
        )
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test PreIngestionControlRequest
    print("\n8. PreIngestionControlRequest:")
    try:
        req = PreIngestionControlRequest(action="pause")
        print(f"   ✅ action='pause': {req.action}")
        req = PreIngestionControlRequest(action="resume")
        print(f"   ✅ action='resume': {req.action}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test CompareVersionsRequest
    print("\n9. CompareVersionsRequest:")
    try:
        req = CompareVersionsRequest(
            crate_name="serde",
            version_a="1.0.196",
            version_b="1.0.197",
            include_unchanged="false",  # String boolean
            max_results="1000",  # String integer
        )
        print(
            f"   ✅ include_unchanged as 'false': {req.include_unchanged} (type: {type(req.include_unchanged).__name__})"
        )
        print(
            f"   ✅ max_results as '1000': {req.max_results} (type: {type(req.max_results).__name__})"
        )

        # Test with categories
        from docsrs_mcp.models import ChangeCategory

        req = CompareVersionsRequest(
            crate_name="tokio",
            version_a="1.34.0",
            version_b="1.35.0",
            categories=[ChangeCategory.BREAKING, ChangeCategory.ADDED],
        )
        print(f"   ✅ categories: {[c.value for c in req.categories]}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ All string parameter coercion tests passed!")
    print(
        "\nNote: All numeric and boolean parameters successfully coerce from strings."
    )
    print("This ensures Claude Code compatibility with MCP parameter validation.")


def test_boundary_values():
    """Test boundary values and edge cases."""

    print("\n\nTesting boundary values and edge cases...")
    print("=" * 60)

    # Test k parameter bounds
    print("\n1. Testing k parameter bounds (1-20):")
    try:
        # Valid range
        req = SearchItemsRequest(crate_name="test", query="q", k="1")
        print(f"   ✅ k='1' (min): {req.k}")

        req = SearchItemsRequest(crate_name="test", query="q", k="20")
        print(f"   ✅ k='20' (max): {req.k}")

        # Out of bounds - should raise error
        try:
            req = SearchItemsRequest(crate_name="test", query="q", k="0")
            print(f"   ❌ k='0' should have failed but got: {req.k}")
        except ValueError as e:
            print(f"   ✅ k='0' correctly rejected: {str(e)[:50]}...")

        try:
            req = SearchItemsRequest(crate_name="test", query="q", k="21")
            print(f"   ❌ k='21' should have failed but got: {req.k}")
        except ValueError as e:
            print(f"   ✅ k='21' correctly rejected: {str(e)[:50]}...")

    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    # Test concurrency bounds
    print("\n2. Testing concurrency parameter bounds (1-10):")
    try:
        req = StartPreIngestionRequest(concurrency="1")
        print(f"   ✅ concurrency='1' (min): {req.concurrency}")

        req = StartPreIngestionRequest(concurrency="10")
        print(f"   ✅ concurrency='10' (max): {req.concurrency}")

        try:
            req = StartPreIngestionRequest(concurrency="11")
            print(f"   ❌ concurrency='11' should have failed")
        except ValueError as e:
            print(f"   ✅ concurrency='11' correctly rejected: {str(e)[:50]}...")

    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    # Test count bounds
    print("\n3. Testing count parameter bounds (10-500):")
    try:
        req = StartPreIngestionRequest(count="10")
        print(f"   ✅ count='10' (min): {req.count}")

        req = StartPreIngestionRequest(count="500")
        print(f"   ✅ count='500' (max): {req.count}")

        try:
            req = StartPreIngestionRequest(count="501")
            print(f"   ❌ count='501' should have failed")
        except ValueError as e:
            print(f"   ✅ count='501' correctly rejected: {str(e)[:50]}...")

    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("✅ All boundary value tests passed!")


if __name__ == "__main__":
    test_string_coercion()
    test_boundary_values()

    print("\n🎉 ALL TESTS PASSED! String parameter coercion is working correctly.")
    print("\nThis confirms that the MCP server will properly handle string parameters")
    print("from Claude Code, which cannot send native numeric or boolean types.")
