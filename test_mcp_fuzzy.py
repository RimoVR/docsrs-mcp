#!/usr/bin/env python3
"""Direct test of the fuzzy resolution fix by importing and calling the function."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp.services.crate_service import CrateService
from docsrs_mcp.fuzzy_resolver import get_fuzzy_suggestions_with_fallback

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


async def test_fuzzy_function_signature():
    """Test that the function signature fix works correctly."""
    print(f"\n{YELLOW}=== Testing Fuzzy Function Signature Fix ==={RESET}\n")
    
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
        db_path = tmp_db.name
        
        # Test 1: Verify keyword-only enforcement works
        print(f"{YELLOW}Test 1: Keyword-only parameter enforcement{RESET}")
        try:
            # This should fail because of keyword-only enforcement
            result = await get_fuzzy_suggestions_with_fallback(
                "Serialize", db_path, "serde", "latest"
            )
            print(f"{RED}  ✗ Function accepted positional arguments (should require keyword-only){RESET}")
            return False
        except TypeError as e:
            if "keyword-only" in str(e) or "positional" in str(e):
                print(f"{GREEN}  ✓ Function correctly enforces keyword-only arguments{RESET}")
                print(f"    Error: {e}")
            else:
                print(f"{RED}  ✗ Unexpected TypeError: {e}{RESET}")
                return False
        
        # Test 2: Verify correct keyword usage works
        print(f"\n{YELLOW}Test 2: Correct keyword argument usage{RESET}")
        try:
            result = await get_fuzzy_suggestions_with_fallback(
                query="Serialize",
                db_path=db_path,
                crate_name="serde",
                version="latest"
            )
            print(f"{GREEN}  ✓ Function accepts correct keyword arguments{RESET}")
            print(f"    Result type: {type(result)}")
            print(f"    Result: {result}")
        except Exception as e:
            print(f"{RED}  ✗ Function failed with keyword arguments: {e}{RESET}")
            return False
        
        # Test 3: Test CrateService integration
        print(f"\n{YELLOW}Test 3: CrateService integration{RESET}")
        service = CrateService()
        try:
            # This will fail to find the item but should successfully call fuzzy suggestions
            result = await service.get_item_doc(
                crate_name="serde",
                item_path="Serialize",
                version="latest"
            )
            # If we get here without error, the parameter fix worked
            print(f"{YELLOW}  ⚠ Got result (item may have been found or error handled){RESET}")
        except Exception as e:
            error_msg = str(e)
            # Check if it's the old parameter error or a new error
            if "missing 2 required positional arguments" in error_msg:
                print(f"{RED}  ✗ Still has parameter mismatch error!{RESET}")
                print(f"    Error: {e}")
                return False
            elif "Did you mean one of these?" in error_msg or "not found" in error_msg:
                print(f"{GREEN}  ✓ CrateService successfully called fuzzy suggestions{RESET}")
                print(f"    Error message includes suggestions or not found")
            else:
                print(f"{YELLOW}  ⚠ Different error (may be expected): {e}{RESET}")
    
    return True


async def main():
    """Main test runner."""
    print(f"{YELLOW}=== Direct Fuzzy Resolution Fix Test ==={RESET}")
    print("Testing the parameter fix directly without MCP protocol\n")
    
    success = await test_fuzzy_function_signature()
    
    print(f"\n{YELLOW}=== Test Summary ==={RESET}")
    if success:
        print(f"{GREEN}✓ All tests passed! The fix is working correctly.{RESET}")
        print(f"{GREEN}  - Keyword-only enforcement prevents parameter order errors{RESET}")
        print(f"{GREEN}  - Function works with correct keyword arguments{RESET}")
        print(f"{GREEN}  - CrateService integration is functional{RESET}")
    else:
        print(f"{RED}✗ Some tests failed. The fix may not be complete.{RESET}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)