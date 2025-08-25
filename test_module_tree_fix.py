#!/usr/bin/env python3
"""Test script for module tree fix via MCP stdio communication."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_mcp_module_tree():
    """Test the module tree functionality via MCP stdio."""
    
    print("🚀 Starting MCP server test for module tree fix...")
    
    # Start the MCP server in stdio mode
    cmd = ["uv", "run", "docsrs-mcp"]
    print(f"Running command: {' '.join(cmd)}")
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent,
    )
    
    try:
        # Initialize the MCP connection
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "experimental": {},
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        print("📤 Sending initialize request...")
        proc.stdin.write((json.dumps(init_request) + "\n").encode())
        await proc.stdin.drain()
        
        # Read initialization response
        response_line = await proc.stdout.readline()
        init_response = json.loads(response_line.decode())
        print(f"📥 Initialize response: {init_response.get('result', {}).get('serverInfo', {})}")
        
        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        proc.stdin.write((json.dumps(initialized_notif) + "\n").encode())
        await proc.stdin.drain()
        
        # First, let's ingest a small crate
        print("\n📦 Ingesting httpx crate...")
        ingest_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "ingest_cargo_file",
                "arguments": {
                    "file_path": "./Cargo.toml"
                }
            },
            "id": 2
        }
        
        proc.stdin.write((json.dumps(ingest_request) + "\n").encode())
        await proc.stdin.drain()
        
        # Read ingest response
        response_line = await proc.stdout.readline()
        ingest_response = json.loads(response_line.decode())
        print(f"📥 Ingest response: {ingest_response.get('result', {})}")
        
        # Wait a bit for ingestion to complete
        await asyncio.sleep(5)
        
        # Now test get_module_tree
        print("\n🌳 Testing get_module_tree for httpx...")
        tree_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "get_module_tree",
                "arguments": {
                    "crate_name": "httpx"
                }
            },
            "id": 3
        }
        
        proc.stdin.write((json.dumps(tree_request) + "\n").encode())
        await proc.stdin.drain()
        
        # Read module tree response
        response_line = await proc.stdout.readline()
        tree_response = json.loads(response_line.decode())
        
        if "error" in tree_response:
            print(f"❌ Error received: {tree_response['error']}")
            return False
        
        result = tree_response.get("result", {})
        if "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                content_data = json.loads(content[0]["text"])
            else:
                content_data = content
        else:
            content_data = result
            
        print(f"📥 Module tree response structure:")
        print(f"   - Has crate_name: {'crate_name' in content_data}")
        print(f"   - Has version: {'version' in content_data}")
        print(f"   - Has tree: {'tree' in content_data}")
        
        if "tree" in content_data:
            tree = content_data["tree"]
            print(f"   - Tree type: {type(tree)}")
            
            # Check if tree is a proper dict/object (not a list)
            if isinstance(tree, dict):
                print(f"   ✅ Tree is a dictionary (ModuleTreeNode)")
                print(f"   - Tree has name: {'name' in tree}")
                print(f"   - Tree has path: {'path' in tree}")
                print(f"   - Tree has children: {'children' in tree}")
                
                if "children" in tree:
                    print(f"   - Children count: {len(tree['children'])}")
                    
                print("\n✅ Module tree fix verified successfully!")
                return True
            elif isinstance(tree, list):
                print(f"   ❌ Tree is still a list (flat structure) - FIX NOT WORKING")
                print(f"   - List length: {len(tree)}")
                return False
            else:
                print(f"   ❓ Unexpected tree type: {type(tree)}")
                return False
        else:
            print("❌ No tree field in response")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        proc.terminate()
        await proc.wait()


async def main():
    """Run the test."""
    success = await test_mcp_module_tree()
    
    if success:
        print("\n🎉 TEST PASSED: Module tree fix is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED: Module tree issue persists")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())