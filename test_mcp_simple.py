#!/usr/bin/env python3
import asyncio
import json
import subprocess
import sys

async def test_mcp():
    print("Testing MCP server std library support...")
    
    # Start server
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "docsrs-mcp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Initialize with proper MCP protocol
        init_msg = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }) + "\n"
        proc.stdin.write(init_msg.encode())
        await proc.stdin.drain()
        
        # Read init response
        line = await proc.stdout.readline()
        print("Init response:", line.decode().strip())
        
        # Send initialized notification
        init_notif = json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}
        }) + "\n"
        proc.stdin.write(init_notif.encode()) 
        await proc.stdin.drain()
        
        # Test std library search_items
        search_msg = json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "search_items", "arguments": {"crate_name": "std", "query": "HashMap", "k": "5"}}
        }) + "\n"
        proc.stdin.write(search_msg.encode())
        await proc.stdin.drain()
        
        # Read search response  
        line = await proc.stdout.readline()
        response = json.loads(line.decode().strip())
        
        if "result" in response:
            content = response["result"].get("content", [])
            if content and len(content) > 0:
                search_results = json.loads(content[0]["text"])
                results = search_results.get("results", [])
                print(f"✅ SUCCESS: search_items returned {len(results)} results!")
                if results:
                    for i, r in enumerate(results[:2]):
                        print(f"  {i+1}. {r.get('item_path', 'N/A')}: {r.get('snippet', '')[:100]}...")
                elif len(results) == 0:
                    print("⚠️  No results found for std::HashMap - may need ingestion")
                    # Test if std crate is ingested
                    print("Testing if std library is available...")
            else:
                print("❌ No content in search response")
        else:
            print(f"❌ FAILED: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        await proc.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp())
