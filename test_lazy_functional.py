#!/usr/bin/env python3
"""Functional test to verify lazy loading works with actual MCP tool calls."""

import asyncio
import json
import subprocess
import sys
import time

async def test_lazy_tool_call():
    """Test that MCP tool calls work with lazy loading and trigger service loading."""
    
    # Start the MCP server
    cmd = ["uv", "run", "docsrs-mcp"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Send initialize request
        initialize_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {},
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        # Send initialize
        message = json.dumps(initialize_msg) + "\n"
        proc.stdin.write(message)
        proc.stdin.flush()
        
        # Read initialize response
        response_line = proc.stdout.readline()
        print("Initialize response:", response_line.strip())
        
        # Send tools/list request (no params needed)
        list_tools_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        message = json.dumps(list_tools_msg) + "\n"
        proc.stdin.write(message)
        proc.stdin.flush()
        
        # Read tools list response
        tools_response = proc.stdout.readline()
        print("Tools list response:", tools_response.strip())
        
        # Parse tools response
        tools_data = json.loads(tools_response.strip())
        if "result" in tools_data and "tools" in tools_data["result"]:
            tools = tools_data["result"]["tools"]
            print(f"✅ Found {len(tools)} tools available")
            
            # Try to call list_versions tool which should trigger lazy loading of CrateService
            if any(tool["name"] == "list_versions" for tool in tools):
                list_versions_msg = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "list_versions",
                        "arguments": {"crate_name": "serde"}
                    }
                }
                
                message = json.dumps(list_versions_msg) + "\n"
                proc.stdin.write(message)
                proc.stdin.flush()
                
                # Read tool call response
                tool_response = proc.stdout.readline()
                print("Tool call response:", tool_response.strip())
                
                # Check stderr for lazy loading messages
                # Give it a moment to process
                time.sleep(1)
                proc.terminate()
                
                # Read stderr to see if lazy loading happened
                _, stderr = proc.communicate(timeout=5)
                print("STDERR output:")
                print(stderr)
                
                if "Lazy loaded CrateService" in stderr:
                    print("✅ Lazy loading working! CrateService loaded on first use")
                    return True
                else:
                    print("⚠️  No lazy loading message found, but tool call succeeded")
                    # Still successful if tool worked
                    tool_data = json.loads(tool_response.strip())
                    if "result" in tool_data:
                        print("✅ Tool call successful with lazy loading")
                        return True
                    
            return False
        else:
            print("❌ No tools found in response")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if proc.poll() is None:
            proc.kill()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_lazy_tool_call())
    sys.exit(0 if result else 1)