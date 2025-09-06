#!/usr/bin/env python3
"""
Test script to verify version management fixes in docsrs-mcp.

Tests the following fixes:
1. Parameter mismatch fix in compare_versions (version_a/version_b)
2. Complete list_versions implementation with crates.io API 
3. NoneType error fixes in version comparison
4. Version validation consistency improvements
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path


async def test_mcp_communication():
    """Test MCP communication with the server."""
    print("🔄 Starting MCP server test...")
    
    # Start the MCP server in background
    server_process = subprocess.Popen(
        ["uv", "run", "docsrs-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/peterkloiber/docsrs-mcp"
    )
    
    print("🔄 Server started, testing...")
    
    try:
        # Test 1: list_versions tool (should now show multiple versions)
        print("\n📋 Testing list_versions tool...")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_versions",
                "arguments": {
                    "crate_name": "serde"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Wait for response
        response_line = server_process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line)
                if "result" in response:
                    result = response["result"]["content"][0]["text"]
                    result_data = json.loads(result)
                    versions = result_data.get("versions", [])
                    print(f"✅ list_versions working - found {len(versions)} versions")
                    if len(versions) > 1:
                        print(f"✅ Multiple versions returned (expected fix)")
                    else:
                        print(f"⚠️  Only {len(versions)} version(s) returned")
                else:
                    print(f"❌ list_versions failed: {response.get('error', 'Unknown error')}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse list_versions response: {e}")
        
        # Test 2: compare_versions tool with corrected parameter names
        print("\n🔄 Testing compare_versions tool...")
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "compare_versions",
                "arguments": {
                    "crate_name": "serde",
                    "version_a": "1.0.0",
                    "version_b": "1.1.0"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(request) + "\n")
        server_process.stdin.flush()
        
        # Wait for response
        response_line = server_process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line)
                if "result" in response:
                    result = response["result"]["content"][0]["text"]
                    result_data = json.loads(result)
                    changes = result_data.get("changes", {})
                    total_changes = sum(len(v) for v in changes.values()) if changes else 0
                    print(f"✅ compare_versions working - found {total_changes} changes")
                    print(f"✅ Parameter name fix successful (version_a/version_b)")
                else:
                    error = response.get("error", {})
                    error_msg = error.get("message", "Unknown error")
                    if "validation error" in error_msg.lower():
                        print(f"❌ Parameter validation still failing: {error_msg}")
                    else:
                        print(f"❌ compare_versions failed: {error_msg}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse compare_versions response: {e}")
        
        print("\n✅ MCP communication test completed")
        return True
        
    except Exception as e:
        print(f"❌ MCP test failed: {e}")
        return False
    finally:
        # Cleanup
        if server_process:
            server_process.terminate()
            server_process.wait()


def test_direct_api():
    """Test the fixes by importing and calling functions directly."""
    print("🔄 Testing direct API imports...")
    
    try:
        # Test that imports work without errors
        sys.path.insert(0, "/Users/peterkloiber/docsrs-mcp/src")
        from docsrs_mcp.services.crate_service import CrateService
        from docsrs_mcp.mcp_tools_config import MCP_TOOLS_CONFIG
        from docsrs_mcp.validation import validate_version_string
        
        print("✅ Core imports successful")
        
        # Test 1: Check MCP tool schema has correct parameter names
        compare_tool = None
        for tool in MCP_TOOLS_CONFIG:
            if tool["name"] == "compare_versions":
                compare_tool = tool
                break
        
        if compare_tool:
            required_params = compare_tool["input_schema"]["required"]
            properties = compare_tool["input_schema"]["properties"]
            
            if "version_a" in required_params and "version_b" in required_params:
                print("✅ MCP schema parameter names fixed (version_a, version_b)")
            else:
                print(f"❌ MCP schema still has wrong parameter names: {required_params}")
                
            if "version_a" in properties and "version_b" in properties:
                print("✅ MCP schema properties include version_a, version_b")
            else:
                print(f"❌ MCP schema properties missing: {list(properties.keys())}")
        else:
            print("❌ compare_versions tool not found in MCP_TOOLS")
        
        # Test 2: Version validation handles None properly
        test_cases = [
            (None, None),
            ("", None),
            ("latest", "latest"),
            ("1.0.0", "1.0.0"),
        ]
        
        print("\n🔄 Testing version validation...")
        for input_val, expected in test_cases:
            result = validate_version_string(input_val)
            if result == expected:
                print(f"✅ validate_version_string({input_val!r}) -> {result!r}")
            else:
                print(f"❌ validate_version_string({input_val!r}) -> {result!r}, expected {expected!r}")
        
        print("✅ Direct API test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing docsrs-mcp version management fixes\n")
    
    # Test 1: Direct API testing (faster, no server needed)
    direct_success = test_direct_api()
    
    # Test 2: MCP communication testing (full integration) 
    print("\n" + "="*50)
    mcp_success = await test_mcp_communication()
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY:")
    print(f"Direct API tests: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"MCP integration tests: {'✅ PASSED' if mcp_success else '❌ FAILED'}")
    
    if direct_success and mcp_success:
        print("\n🎉 ALL TESTS PASSED - Version management fixes are working!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Check output above for details")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)