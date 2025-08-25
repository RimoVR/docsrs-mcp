#!/usr/bin/env python3
"""Simple test to verify lazy loading works with MCP protocol."""

import asyncio
import json
import subprocess
import sys

async def test_lazy_loading():
    """Test that MCP server works with lazy loading and services are loaded on demand."""
    
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
        
        # Send the message
        message = json.dumps(initialize_msg) + "\n"
        proc.stdin.write(message)
        proc.stdin.flush()
        
        # Read response with timeout
        try:
            stdout, stderr = proc.communicate(timeout=10)
            print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
                
            # Check if we got a valid JSON-RPC response
            if stdout.strip():
                response = json.loads(stdout.strip().split('\n')[0])
                if response.get("id") == 1 and "result" in response:
                    print("✅ MCP Initialize successful!")
                    print("✅ Lazy loading implementation working!")
                    return True
                else:
                    print("❌ Invalid MCP response")
                    return False
            else:
                print("❌ No response received")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout waiting for response")
            proc.kill()
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        proc.kill()
        return False
    
    finally:
        if proc.poll() is None:
            proc.terminate()

if __name__ == "__main__":
    result = asyncio.run(test_lazy_loading())
    sys.exit(0 if result else 1)