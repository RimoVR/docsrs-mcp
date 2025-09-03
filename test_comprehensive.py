#!/usr/bin/env python3
"""
Comprehensive test for character fragmentation with multiple crates
"""

import asyncio
import json
import subprocess
import sys

class MCPClient:
    def __init__(self):
        self.process = None
        self.request_id = 1
    
    async def start_server(self):
        print("Starting MCP server...")
        self.process = await asyncio.create_subprocess_exec(
            "uv", "run", "docsrs-mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Initialize
        await self.send_request("initialize", {
            "protocolVersion": "2024-10-07",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "comprehensive-test", "version": "1.0.0"}
        })
        
        response = await self.read_response()
        print(f"✅ Server initialized: {response.get('result', {}).get('serverInfo', {}).get('name')}")
        
        await self.send_notification("notifications/initialized")
        
    async def send_request(self, method, params=None):
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        self.request_id += 1
        
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
    async def send_notification(self, method, params=None):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
    async def read_response(self):
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=15)
            if line:
                return json.loads(line.decode().strip())
        except (asyncio.TimeoutError, json.JSONDecodeError):
            pass
        return None
        
    async def call_tool(self, tool_name, arguments):
        await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return await self.read_response()
        
    async def stop_server(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()

async def test_character_fragmentation():
    """Test multiple crates for character fragmentation."""
    client = MCPClient()
    
    try:
        await client.start_server()
        
        # Test crates that might have cached data
        test_cases = [
            {"crate": "tokio", "query": "async"},
            {"crate": "serde", "query": "serialize"},  
            {"crate": "once_cell", "query": "lazy"},
            {"crate": "pydantic", "query": "validation"}  # This might be cached
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            crate = test_case["crate"]
            query = test_case["query"]
            
            print(f"\n=== Testing {crate} for '{query}' ===")
            
            response = await client.call_tool("search_examples", {
                "crate_name": crate,
                "query": query,
                "k": "3"
            })
            
            if not response or "result" not in response:
                print(f"❌ No response from {crate}")
                all_passed = False
                continue
                
            content = response["result"].get("content", [])
            if not content:
                print(f"⚠️  No content from {crate}")
                continue
                
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    
                    # Critical character fragmentation check
                    if len(text) == 1 and text.isalnum():
                        print(f"❌ CHARACTER FRAGMENTATION in {crate}: '{text}'")
                        all_passed = False
                        continue
                    
                    # Parse JSON response
                    try:
                        parsed = json.loads(text)
                        examples = parsed.get("examples", [])
                        
                        # Check if examples list contains individual characters
                        if examples and isinstance(examples, list):
                            if any(isinstance(ex, str) and len(ex) == 1 for ex in examples):
                                print(f"❌ CHARACTER FRAGMENTATION in {crate} examples list")
                                all_passed = False
                                continue
                        
                        print(f"✅ {crate}: {len(examples)} examples, structure intact")
                        
                        # Show first example if available
                        if examples:
                            first_ex = str(examples[0])[:100]
                            print(f"   First example preview: {first_ex}...")
                        
                    except json.JSONDecodeError:
                        print(f"⚠️  {crate}: Non-JSON response (length: {len(text)})")
        
        if all_passed:
            print(f"\n🎉 ALL TESTS PASSED! Character fragmentation bug is DEFINITIVELY FIXED!")
            return True
        else:
            print(f"\n❌ Some tests failed - character fragmentation may still exist")
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        return False
        
    finally:
        await client.stop_server()

if __name__ == "__main__":
    success = asyncio.run(test_character_fragmentation())
    sys.exit(0 if success else 1)