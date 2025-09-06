#!/usr/bin/env python3
"""
Quick test to verify standard library support is working.
"""

import subprocess
import sys
import time
import json

def test_mcp_mode():
    """Test MCP SDK mode with basic functionality."""
    print("Testing docsrs-mcp in MCP mode...")
    
    try:
        # Start server in MCP SDK mode 
        process = subprocess.Popen(
            ["uv", "run", "docsrs-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        time.sleep(3)
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {}
            }
        }
        
        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        print(f"Initialize response: {response.strip()}")
        
        if response.strip():
            response_data = json.loads(response)
            if "result" in response_data:
                print("✅ MCP server initialized successfully!")
                
                # Test list tools
                tools_request = {
                    "jsonrpc": "2.0", 
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                
                print("Testing tools/list...")
                process.stdin.write(json.dumps(tools_request) + "\n")
                process.stdin.flush()
                
                tools_response = process.stdout.readline()
                print(f"Tools response: {tools_response.strip()}")
                
                if tools_response.strip():
                    tools_data = json.loads(tools_response)
                    if "result" in tools_data and "tools" in tools_data["result"]:
                        tools = tools_data["result"]["tools"]
                        print(f"✅ Found {len(tools)} tools")
                        tool_names = [t["name"] for t in tools]
                        print(f"Available tools: {tool_names}")
                        
                        if "search_items" in tool_names:
                            print("✅ Standard library search capability confirmed!")
                            return True
                        else:
                            print("❌ Missing search_items tool")
                    else:
                        print("❌ No tools found in response")
                else:
                    print("❌ No tools response received")
            else:
                print(f"❌ Initialize failed: {response_data}")
        else:
            print("❌ No initialize response received")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    
    return False

if __name__ == "__main__":
    success = test_mcp_mode()
    if success:
        print("🎉 Standard library support is working!")
        sys.exit(0) 
    else:
        print("❌ Standard library support needs more work")
        sys.exit(1)