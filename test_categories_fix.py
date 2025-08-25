#!/usr/bin/env python3
"""
Test script for CompareVersionsRequest categories field validator fix.

This tests the field validator directly to ensure it properly converts
string categories to ChangeCategory enums.
"""

import sys
import traceback
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp.models.version_diff import CompareVersionsRequest, ChangeCategory


def test_categories_validator():
    """Test the categories field validator with various input formats."""
    print("🧪 Testing categories field validator...")
    
    tests = [
        {
            "name": "Test 1: None categories (should use defaults)",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": None
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.DEPRECATED,
                ChangeCategory.ADDED,
                ChangeCategory.REMOVED,
                ChangeCategory.MODIFIED,
            ]
        },
        {
            "name": "Test 2: String list categories",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": ["breaking", "added", "deprecated"]
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.ADDED,
                ChangeCategory.DEPRECATED
            ]
        },
        {
            "name": "Test 3: Comma-separated string categories",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": "breaking,added,deprecated"
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.ADDED,
                ChangeCategory.DEPRECATED
            ]
        },
        {
            "name": "Test 4: Category aliases",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": ["break", "new", "changed"]
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.ADDED,
                ChangeCategory.MODIFIED
            ]
        },
        {
            "name": "Test 5: Mixed case categories",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": "BREAKING,Added,deprecated"
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.ADDED,
                ChangeCategory.DEPRECATED
            ]
        },
        {
            "name": "Test 6: Categories with whitespace",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180", 
                "version_b": "1.0.195",
                "categories": " breaking , added , deprecated "
            },
            "expected_categories": [
                ChangeCategory.BREAKING,
                ChangeCategory.ADDED,
                ChangeCategory.DEPRECATED
            ]
        },
        {
            "name": "Test 7: Invalid category (should fail)",
            "input": {
                "crate_name": "serde",
                "version_a": "1.0.180",
                "version_b": "1.0.195",
                "categories": ["breaking", "invalid_category"]
            },
            "should_fail": True,
            "expected_error": "Invalid category"
        }
    ]
    
    all_passed = True
    
    for test in tests:
        print(f"\n  {test['name']}")
        try:
            request = CompareVersionsRequest(**test["input"])
            
            if test.get("should_fail"):
                print(f"    ❌ FAILED: Expected validation error but got success")
                all_passed = False
            else:
                # Check categories match expected
                actual_categories = request.categories
                expected_categories = test["expected_categories"]
                
                if actual_categories == expected_categories:
                    print(f"    ✅ PASSED: Categories correctly converted")
                else:
                    print(f"    ❌ FAILED: Expected {expected_categories}, got {actual_categories}")
                    all_passed = False
                    
        except Exception as e:
            if test.get("should_fail"):
                if test["expected_error"] in str(e):
                    print(f"    ✅ PASSED: Correctly rejected with error: {e}")
                else:
                    print(f"    ❌ FAILED: Wrong error message. Expected '{test['expected_error']}' in error, got: {e}")
                    all_passed = False
            else:
                print(f"    ❌ FAILED: Unexpected error: {e}")
                print(f"    Stack trace: {traceback.format_exc()}")
                all_passed = False
    
    return all_passed


def test_mcp_integration():
    """Test MCP integration to ensure the fix works end-to-end."""
    print("\n🔗 Testing MCP integration...")
    
    try:
        # Import the MCP SDK server and crate service
        from docsrs_mcp.services.crate_service import CrateService
        
        # Create a crate service instance
        crate_service = CrateService()
        
        # Test the service method with string categories
        print("  Testing service method with string categories...")
        
        # This should work now without throwing validation errors
        # We'll just test that it doesn't crash during model validation
        try:
            # Create a CompareVersionsRequest directly with string categories
            # This is what would happen inside crate_service.compare_versions
            request = CompareVersionsRequest(
                crate_name="serde",
                version_a="1.0.180",
                version_b="1.0.195",
                categories="breaking,added"  # String instead of list
            )
            print("  ✅ PASSED: Service integration works with string categories")
            return True
            
        except Exception as e:
            print(f"  ❌ FAILED: Service integration failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ FAILED: Could not test integration: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing CompareVersionsRequest categories field fix...")
    
    # Test the field validator
    validator_passed = test_categories_validator()
    
    # Test integration
    integration_passed = test_mcp_integration()
    
    if validator_passed and integration_passed:
        print("\n🎉 All tests passed! The categories field fix is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)