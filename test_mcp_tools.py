#!/usr/bin/env python3
"""Test script to check what MCP tools are available."""

import json
import subprocess
import asyncio


async def test_mcp_tools_list():
    """Test what tools are available in the MCP server."""
    print("🔍 Checking available MCP tools...")
    
    # Start MCP server in SDK mode
    process = subprocess.Popen(
        ["uv", "run", "docsrs-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    await asyncio.sleep(2)  # Give server time to start
    
    try:
        # Request list of available tools
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        print(f"📤 Sending tools/list request: {json.dumps(request)}")
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        await asyncio.sleep(3)
        
        # Read response
        response_line = process.stdout.readline().strip()
        print(f"📥 Raw response: {response_line}")
        
        if response_line:
            try:
                response = json.loads(response_line)
                print(f"📋 Parsed response:")
                print(json.dumps(response, indent=2))
                
                if "result" in response and "tools" in response["result"]:
                    tools = response["result"]["tools"]
                    print(f"\n🛠️  Available tools ({len(tools)}):")
                    for tool in tools:
                        name = tool.get("name", "Unknown")
                        description = tool.get("description", "No description")
                        print(f"  - {name}: {description}")
                        
                        # Check if this is a documentation detail related tool
                        if "documentation" in name.lower() or "detail" in name.lower():
                            print(f"    🎯 This might be our target tool!")
                            if "inputSchema" in tool:
                                print(f"    📝 Input schema: {json.dumps(tool['inputSchema'], indent=6)}")
                    
                    # Look for get_documentation_detail specifically
                    doc_detail_tools = [t for t in tools if "documentation_detail" in t.get("name", "")]
                    if doc_detail_tools:
                        print(f"\n🎯 Found get_documentation_detail tool:")
                        for tool in doc_detail_tools:
                            print(json.dumps(tool, indent=2))
                    else:
                        print(f"\n❗ get_documentation_detail tool NOT found in tools list")
                        print("Available tool names:")
                        for tool in tools:
                            print(f"  - {tool.get('name', 'Unknown')}")
                        
                else:
                    print("❌ No tools found in response")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        print("\n🛑 Terminating server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    asyncio.run(test_mcp_tools_list())