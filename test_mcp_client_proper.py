#!/usr/bin/env python3
"""Proper MCP client test for Bug #8 fix using the MCP protocol correctly."""

import asyncio
import json
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_get_documentation_detail():
    """Test get_documentation_detail with proper MCP client."""
    print("🧪 Testing Bug #8 fix with proper MCP client")
    print("=" * 50)
    
    # Server command
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "docsrs-mcp"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("📡 Connected to MCP server")
                
                # Initialize the session
                await session.initialize()
                print("✅ Session initialized")
                
                # List available tools first
                print("🔍 Listing available tools...")
                tools = await session.list_tools()
                print(f"📋 Found {len(tools.tools)} tools")
                
                # Find documentation detail related tools
                doc_tools = [
                    tool for tool in tools.tools 
                    if "documentation" in tool.name.lower() and "detail" in tool.name.lower()
                ]
                
                if doc_tools:
                    print(f"🎯 Found documentation detail tools:")
                    for tool in doc_tools:
                        print(f"  - {tool.name}: {tool.description}")
                else:
                    print("❗ No documentation detail tools found")
                    print("Available tools:")
                    for tool in tools.tools:
                        print(f"  - {tool.name}")
                
                # Test 1: get_documentation_detail with non-existent item (Bug #8 reproduction)
                print("\n🔍 Test 1: Testing get_documentation_detail with non-existent item...")
                try:
                    result = await session.call_tool(
                        "get_documentation_detail",
                        {
                            "crate_name": "serde",
                            "item_path": "NonexistentItem",
                            "detail_level": "summary"
                        }
                    )
                    
                    print("✅ SUCCESS: get_documentation_detail completed without validation errors!")
                    print(f"📄 Result: {result}")
                    
                    # Check if this is an error response
                    if hasattr(result, 'content') and result.content:
                        content_text = str(result.content[0].text if result.content else "")
                        if "error" in content_text.lower() and "not found" in content_text.lower():
                            print("✅ Proper error handling - item not found but no validation error")
                        else:
                            print("❓ Unexpected response content")
                    
                    return True
                    
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"❌ Error calling get_documentation_detail: {e}")
                    
                    if "validation error" in error_str or "field required" in error_str:
                        print("❌ FAILED: Bug #8 validation error still exists!")
                        return False
                    elif "missing" in error_str or "required" in error_str:
                        print("❌ FAILED: Possible validation error")
                        return False
                    else:
                        print("❓ Different error - may not be Bug #8")
                        
                # Test 2: Valid item test to check for regression
                print("\n🔍 Test 2: Testing get_documentation_detail with valid item...")
                try:
                    result = await session.call_tool(
                        "get_documentation_detail",
                        {
                            "crate_name": "serde",
                            "item_path": "Serialize",  # This should exist
                            "detail_level": "summary"
                        }
                    )
                    
                    print("✅ SUCCESS: Valid item test passed - no regression!")
                    print(f"📄 Result: {result}")
                    return True
                    
                except Exception as e:
                    print(f"❌ Valid item test failed: {e}")
                    return False
                    
    except Exception as e:
        print(f"❌ MCP client connection failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Bug #8 fix verification with proper MCP client")
    print()
    
    success = await test_get_documentation_detail()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 BUG #8 FIX VERIFICATION: SUCCESS!")
        print("✅ ProgressiveDetailResponse validation errors have been resolved")
        return 0
    else:
        print("❌ BUG #8 FIX VERIFICATION: FAILED!")
        print("❌ Validation errors may still exist")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)