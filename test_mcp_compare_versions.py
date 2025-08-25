#!/usr/bin/env python3
"""
Test script for comparing crate versions via MCP stdio interface.

This script tests the compare_versions fix by communicating directly with the
MCP server via stdio, bypassing the native MCP tools that use the old version.
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict


class MCPClient:
    """Simple MCP client for stdio communication."""
    
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.request_id = 1
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline().decode().strip()
        if not response_line:
            raise Exception("No response from MCP server")
        
        try:
            response = json.loads(response_line)
            return response
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {response_line}") from e


async def test_compare_versions():
    """Test the compare_versions functionality with various parameter combinations."""
    print("🚀 Testing compare_versions MCP fix via stdio...")
    
    # Start the MCP server using uvx
    print("📡 Starting MCP server with uvx...")
    cmd = ["uvx", "--from", ".", "docsrs-mcp"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        cwd="/Users/peterkloiber/docsrs-mcp"
    )
    
    client = MCPClient(process)
    
    try:
        # Initialize the connection
        print("🔗 Initializing MCP connection...")
        init_response = await client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        print(f"✅ Initialize response: {init_response.get('result', {}).get('protocolVersion')}")
        
        # Send initialized notification
        await client.send_request("notifications/initialized", {})
        
        # Test 1: Compare versions without categories parameter (should use defaults)
        print("\n🧪 Test 1: Compare versions without categories (should use defaults)")
        try:
            result = await client.send_request("tools/call", {
                "name": "compare_versions",
                "arguments": {
                    "crate_name": "serde",
                    "version_a": "1.0.180",
                    "version_b": "1.0.195"
                }
            })
            
            if "error" in result:
                print(f"❌ Test 1 FAILED: {result['error']}")
                return False
            else:
                print(f"✅ Test 1 PASSED: Got result with {len(result.get('result', {}).get('content', []))} content items")
        except Exception as e:
            print(f"❌ Test 1 FAILED with exception: {e}")
            return False
        
        # Test 2: Compare versions with categories parameter (comma-separated string)
        print("\n🧪 Test 2: Compare versions with categories parameter")
        try:
            result = await client.send_request("tools/call", {
                "name": "compare_versions",
                "arguments": {
                    "crate_name": "serde",
                    "version_a": "1.0.180",
                    "version_b": "1.0.195",
                    "categories": "breaking,added,deprecated"
                }
            })
            
            if "error" in result:
                print(f"❌ Test 2 FAILED: {result['error']}")
                return False
            else:
                print(f"✅ Test 2 PASSED: Got result with specific categories")
        except Exception as e:
            print(f"❌ Test 2 FAILED with exception: {e}")
            return False
        
        # Test 3: Compare versions with invalid category (should fail gracefully)
        print("\n🧪 Test 3: Compare versions with invalid category")
        try:
            result = await client.send_request("tools/call", {
                "name": "compare_versions",
                "arguments": {
                    "crate_name": "serde", 
                    "version_a": "1.0.180",
                    "version_b": "1.0.195",
                    "categories": "breaking,invalid_category"
                }
            })
            
            if "error" in result and "Invalid category" in str(result["error"]):
                print(f"✅ Test 3 PASSED: Correctly rejected invalid category")
            else:
                print(f"❌ Test 3 FAILED: Should have rejected invalid category")
                return False
        except Exception as e:
            print(f"❌ Test 3 FAILED with exception: {e}")
            return False
        
        # Test 4: Compare versions with category aliases
        print("\n🧪 Test 4: Compare versions with category aliases")
        try:
            result = await client.send_request("tools/call", {
                "name": "compare_versions",
                "arguments": {
                    "crate_name": "serde",
                    "version_a": "1.0.180", 
                    "version_b": "1.0.195",
                    "categories": "break,new,changed"  # aliases for breaking,added,modified
                }
            })
            
            if "error" in result:
                print(f"❌ Test 4 FAILED: {result['error']}")
                return False
            else:
                print(f"✅ Test 4 PASSED: Correctly handled category aliases")
        except Exception as e:
            print(f"❌ Test 4 FAILED with exception: {e}")
            return False
        
        print("\n🎉 All tests passed! The compare_versions fix is working correctly.")
        return True
        
    finally:
        # Clean up
        print("\n🧹 Cleaning up MCP server process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    result = asyncio.run(test_compare_versions())
    sys.exit(0 if result else 1)