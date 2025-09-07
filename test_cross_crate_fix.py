#!/usr/bin/env python3
"""
Test script for cross-crate search fix via MCP stdio mode.
Tests the actual local implementation, not the old GitHub version.
"""

import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict

async def send_mcp_request(process: subprocess.Popen, request: Dict[str, Any]) -> Dict[str, Any]:
    """Send an MCP request via stdio and get the response."""
    # Send request
    request_line = json.dumps(request) + '\n'
    print(f"Sending: {request_line.strip()}")
    
    process.stdin.write(request_line.encode())
    process.stdin.flush()
    
    # Read response
    response_line = process.stdout.readline().decode().strip()
    print(f"Received: {response_line}")
    
    if not response_line:
        raise Exception("No response received from MCP server")
    
    return json.loads(response_line)

async def test_cross_crate_search():
    """Test cross-crate search functionality via MCP stdio."""
    
    # Start MCP server in SDK mode via stdio
    print("Starting MCP server in SDK mode...")
    process = subprocess.Popen(
        ["uv", "run", "docsrs-mcp", "--mcp-implementation", "sdk"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/Users/peterkloiber/docsrs-mcp"
    )
    
    try:
        # Initialize MCP connection
        print("\n1. Testing MCP initialization...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        init_response = await send_mcp_request(process, init_request)
        if "error" in init_response:
            raise Exception(f"Initialization failed: {init_response['error']}")
        
        print("✅ MCP server initialized successfully")
        
        # Test 1: List tools to verify schema
        print("\n2. Testing tool listing (schema verification)...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        tools_response = await send_mcp_request(process, list_tools_request)
        if "error" in tools_response:
            raise Exception(f"Tool listing failed: {tools_response['error']}")
        
        # Find search_items tool and check its schema
        search_tool = None
        for tool in tools_response.get("result", {}).get("tools", []):
            if tool["name"] == "search_items":
                search_tool = tool
                break
        
        if not search_tool:
            raise Exception("search_items tool not found in tool listing")
        
        print(f"Found search_items tool: {search_tool['name']}")
        print(f"Schema type: {search_tool['inputSchema'].get('type', 'unknown')}")
        
        # Check if schema uses oneOf pattern
        if "oneOf" in search_tool['inputSchema']:
            print("✅ Schema uses oneOf pattern - cross-crate search should be supported")
        else:
            print("❌ Schema doesn't use oneOf pattern")
            print(f"Required fields: {search_tool['inputSchema'].get('required', [])}")
        
        # Test 2: Cross-crate search with query only (this should work with new schema)
        print("\n3. Testing cross-crate search with query only...")
        cross_crate_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_items",
                "arguments": {
                    "query": "deserialize"
                }
            }
        }
        
        cross_crate_response = await send_mcp_request(process, cross_crate_request)
        
        if "error" in cross_crate_response:
            print(f"❌ Cross-crate search failed: {cross_crate_response['error']}")
            # Check if it's still the old schema validation error
            error_message = cross_crate_response["error"].get("message", "")
            if "crate_name" in error_message and "required" in error_message:
                print("❌ Still getting old schema validation error - fix not working")
            else:
                print(f"❌ Different error: {error_message}")
        else:
            result = cross_crate_response.get("result", {})
            if "results" in result:
                print(f"✅ Cross-crate search successful! Found {len(result['results'])} results")
                if result["results"]:
                    first_result = result["results"][0]
                    print(f"First result: {first_result.get('item_path', 'unknown')}")
            else:
                print(f"✅ Cross-crate search response: {result}")
        
        # Test 3: Cross-crate search with specific crates
        print("\n4. Testing cross-crate search with specific crates...")
        specific_crates_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "search_items",
                "arguments": {
                    "query": "async",
                    "crates": ["tokio", "serde"]
                }
            }
        }
        
        specific_response = await send_mcp_request(process, specific_crates_request)
        
        if "error" in specific_response:
            print(f"❌ Specific crates search failed: {specific_response['error']}")
        else:
            result = specific_response.get("result", {})
            if "results" in result:
                print(f"✅ Specific crates search successful! Found {len(result['results'])} results")
            else:
                print(f"✅ Specific crates search response: {result}")
        
        # Test 4: Single-crate search (backward compatibility)
        print("\n5. Testing single-crate search (backward compatibility)...")
        single_crate_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "search_items",
                "arguments": {
                    "crate_name": "serde",
                    "query": "deserialize"
                }
            }
        }
        
        single_response = await send_mcp_request(process, single_crate_request)
        
        if "error" in single_response:
            print(f"❌ Single-crate search failed: {single_response['error']}")
        else:
            result = single_response.get("result", {})
            if "results" in result:
                print(f"✅ Single-crate search successful! Found {len(result['results'])} results")
            else:
                print(f"✅ Single-crate search response: {result}")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        # Print stderr for debugging
        stderr_output = process.stderr.read().decode()
        if stderr_output:
            print(f"Server stderr: {stderr_output}")
    
    finally:
        # Clean up
        print("\nCleaning up...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

if __name__ == "__main__":
    print("Testing cross-crate search fix via MCP stdio mode")
    print("=" * 60)
    asyncio.run(test_cross_crate_search())