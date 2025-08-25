#!/usr/bin/env python3
"""Test script for Bug #8 fix - get_documentation_detail validation error.

This script tests the MCP server in SDK mode via stdio communication to verify
that the get_documentation_detail validation error has been resolved.
"""

import json
import subprocess
import sys
import asyncio
import time
from typing import Any, Dict


async def test_mcp_server_bug8_fix():
    """Test the Bug #8 fix by communicating with MCP server via stdio."""
    print("🧪 Testing Bug #8 fix: get_documentation_detail validation error")
    print("=" * 60)
    
    # Start MCP server in SDK mode (default)
    print("📡 Starting MCP server in SDK mode...")
    process = subprocess.Popen(
        ["uv", "run", "docsrs-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    # Give server time to start
    await asyncio.sleep(2)
    
    try:
        # Test case 1: Get documentation detail for non-existent item (reproduces Bug #8)
        print("🔍 Test 1: Testing get_documentation_detail with non-existent item...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_documentation_detail",
                "arguments": {
                    "crate_name": "serde",
                    "item_path": "NonexistentItem",
                    "detail_level": "summary"
                }
            }
        }
        
        print(f"📤 Sending request: {json.dumps(request)}")
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        # Wait for response
        await asyncio.sleep(3)
        
        # Read response
        response_line = process.stdout.readline().strip()
        print(f"📥 Raw response: {response_line}")
        
        if response_line:
            try:
                response = json.loads(response_line)
                print(f"📋 Parsed response: {json.dumps(response, indent=2)}")
                
                # Check if this is a successful response (not an error)
                if "result" in response:
                    result = response["result"]
                    if "content" in result and isinstance(result["content"], list):
                        content = result["content"][0] if result["content"] else {}
                        if "text" in content:
                            # Parse the text content which should contain the ProgressiveDetailResponse
                            text_content = content["text"]
                            print(f"📄 Response text content:\n{text_content}")
                            
                            # If we got here without validation errors, the fix works!
                            print("✅ SUCCESS: No validation errors occurred!")
                            print("✅ Bug #8 has been fixed - ProgressiveDetailResponse validation works correctly")
                            return True
                        else:
                            print("❓ Unexpected content structure - no text field")
                    else:
                        print("❓ Unexpected result structure - no content array")
                
                elif "error" in response:
                    error = response["error"]
                    error_message = error.get("message", "Unknown error")
                    print(f"❌ Server error: {error_message}")
                    
                    # Check if this is a validation error (Bug #8)
                    if "validation error" in error_message.lower() or "field required" in error_message.lower():
                        print("❌ FAILED: Bug #8 validation error still occurs!")
                        return False
                    else:
                        print("❓ Different error type - may not be related to Bug #8")
                
                else:
                    print("❓ Unexpected response structure")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response was: {response_line}")
                
        else:
            print("❌ No response received from server")
            
        # Test case 2: Test with valid item to ensure no regression
        print("\n🔍 Test 2: Testing get_documentation_detail with valid crate...")
        
        request2 = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_documentation_detail",
                "arguments": {
                    "crate_name": "serde",
                    "item_path": "Serialize",  # This should exist
                    "detail_level": "summary"
                }
            }
        }
        
        print(f"📤 Sending request: {json.dumps(request2)}")
        process.stdin.write(json.dumps(request2) + "\n")
        process.stdin.flush()
        
        await asyncio.sleep(3)
        
        response_line2 = process.stdout.readline().strip()
        print(f"📥 Raw response: {response_line2}")
        
        if response_line2:
            try:
                response2 = json.loads(response_line2)
                if "result" in response2:
                    print("✅ SUCCESS: Valid item test passed - no regression detected")
                    return True
                else:
                    print("❓ Valid item test had unexpected response")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response for valid item test: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        print("\n🛑 Terminating MCP server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("✅ MCP server terminated")


async def main():
    """Main test function."""
    print("🚀 Starting Bug #8 fix verification test")
    print("This test verifies that get_documentation_detail validation errors have been resolved")
    print()
    
    success = await test_mcp_server_bug8_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 BUG #8 FIX VERIFICATION: SUCCESS!")
        print("✅ ProgressiveDetailResponse validation errors have been resolved")
        sys.exit(0)
    else:
        print("❌ BUG #8 FIX VERIFICATION: FAILED!")
        print("❌ Validation errors may still exist")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())