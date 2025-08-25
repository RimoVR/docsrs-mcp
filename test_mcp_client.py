#!/usr/bin/env python3
"""
Test MCP client for testing the local docsrs-mcp implementation.
This script tests the local directory version, not the installed version.
"""

import asyncio
import json
import subprocess
import sys
import os
from typing import Any, Dict

class MCPClient:
    """Simple MCP client for testing local server."""
    
    def __init__(self, server_command: list):
        self.server_command = server_command
        self.process = None
        self.request_id = 1
    
    async def start_server(self):
        """Start the MCP server process."""
        print(f"Starting MCP server with command: {' '.join(self.server_command)}")
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send initialization request
        await self.send_request("initialize", {
            "protocolVersion": "2024-10-07",
            "capabilities": {
                "roots": {
                    "listChanged": True
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        
        # Wait for initialization response
        response = await self.read_response()
        print(f"Initialization response: {response}")
        
        # Send initialized notification
        await self.send_notification("notifications/initialized")
        
    async def send_request(self, method: str, params: Dict[str, Any] = None):
        """Send a JSON-RPC request to the server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        self.request_id += 1
        
        message = json.dumps(request) + "\n"
        print(f"Sending request: {message.strip()}")
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
    async def send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send a JSON-RPC notification to the server."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        message = json.dumps(notification) + "\n"
        print(f"Sending notification: {message.strip()}")
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
    async def read_response(self):
        """Read a response from the server."""
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10)
            if line:
                response_text = line.decode().strip()
                print(f"Received: {response_text}")
                return json.loads(response_text)
            else:
                stderr_line = await self.process.stderr.readline()
                if stderr_line:
                    print(f"Server stderr: {stderr_line.decode().strip()}")
                return None
        except asyncio.TimeoutError:
            print("Timeout waiting for server response")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
            
    async def list_tools(self):
        """List available tools."""
        await self.send_request("tools/list")
        return await self.read_response()
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool."""
        await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return await self.read_response()
        
    async def stop_server(self):
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            await self.process.wait()

async def test_local_mcp_server():
    """Test the local MCP server implementation."""
    
    # Use uv run to execute the local version
    server_command = ["uv", "run", "docsrs-mcp"]
    
    client = MCPClient(server_command)
    
    try:
        # Start the server
        await client.start_server()
        
        # Test 1: List available tools
        print("\n=== Testing tools/list ===")
        tools_response = await client.list_tools()
        if tools_response and "result" in tools_response:
            tools = tools_response["result"]["tools"]
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description'][:80]}...")
        else:
            print("Failed to get tools list")
            return
            
        # Test 2: Call list_versions (simple tool)
        print("\n=== Testing list_versions ===")
        list_versions_response = await client.call_tool("list_versions", {
            "crate_name": "serde"
        })
        print(f"list_versions response: {json.dumps(list_versions_response, indent=2)}")
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        await client.stop_server()

if __name__ == "__main__":
    asyncio.run(test_local_mcp_server())
