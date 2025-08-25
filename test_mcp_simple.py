#!/usr/bin/env python3
"""
Simple MCP test to verify compare_versions works in MCP mode.
This tests by calling the MCP SDK server functions directly.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp import mcp_sdk_server


async def test_mcp_compare_versions():
    """Test compare_versions function directly from MCP SDK server."""
    print("🧪 Testing MCP SDK server compare_versions function...")
    
    tests = [
        {
            "name": "Test 1: No categories (should use defaults)",
            "args": {
                "crate_name": "once_cell",  # Use a smaller crate for faster testing
                "version_a": "1.17.0",
                "version_b": "1.18.0",
                "categories": None
            }
        },
        {
            "name": "Test 2: String categories", 
            "args": {
                "crate_name": "once_cell",
                "version_a": "1.17.0",
                "version_b": "1.18.0", 
                "categories": "breaking,added"
            }
        },
        {
            "name": "Test 3: Invalid categories (should fail)",
            "args": {
                "crate_name": "once_cell",
                "version_a": "1.17.0",
                "version_b": "1.18.0",
                "categories": "breaking,invalid_category"
            },
            "should_fail": True
        }
    ]
    
    all_passed = True
    
    for test in tests:
        print(f"\n  {test['name']}")
        try:
            result = await mcp_sdk_server.compare_versions(**test["args"])
            
            if test.get("should_fail"):
                print(f"    ❌ FAILED: Expected error but got success")
                all_passed = False
            else:
                # Check if result has expected structure
                if isinstance(result, dict) and "crate_name" in result:
                    print(f"    ✅ PASSED: Got valid result")
                else:
                    print(f"    ❌ FAILED: Invalid result structure")
                    all_passed = False
                    
        except Exception as e:
            if test.get("should_fail"):
                if "Invalid category" in str(e):
                    print(f"    ✅ PASSED: Correctly rejected invalid categories")
                else:
                    print(f"    ❌ FAILED: Wrong error type: {e}")
                    all_passed = False
            else:
                print(f"    ❌ FAILED: Unexpected error: {e}")
                all_passed = False
    
    return all_passed


async def main():
    """Run the MCP test."""
    print("🚀 Testing MCP SDK server compare_versions...")
    
    success = await test_mcp_compare_versions()
    
    if success:
        print("\n🎉 All MCP tests passed! The fix works in MCP mode.")
        return True
    else:
        print("\n❌ Some MCP tests failed.")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)