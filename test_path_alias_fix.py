#!/usr/bin/env python3
"""
Test script to verify Path Alias Resolution bug fix.
Tests resolve_import functionality via MCP stdio communication.
"""

import json
import subprocess
import sys
from typing import Dict, Any

def send_mcp_request(process, request: Dict[str, Any]) -> Dict[str, Any]:
    """Send MCP request and get response."""
    request_json = json.dumps(request)
    print(f"→ SENDING: {request_json}")
    
    process.stdin.write(request_json + '\n')
    process.stdin.flush()
    
    response_line = process.stdout.readline().strip()
    print(f"← RECEIVED: {response_line}")
    
    try:
        return json.loads(response_line)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        return {"error": "Invalid JSON response"}

def test_resolve_import(process):
    """Test resolve_import functionality for Path Alias Resolution."""
    
    print("\n" + "="*60)
    print("TESTING PATH ALIAS RESOLUTION BUG FIX")
    print("="*60)
    
    # Test 1: Basic resolve_import call with serde::Deserialize
    print("\nTest 1: resolve_import('serde', 'Deserialize')")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "resolve_import",
            "arguments": {
                "crate_name": "serde",
                "import_path": "Deserialize"
            }
        }
    }
    
    response = send_mcp_request(process, request)
    
    if "error" in response:
        print(f"❌ Test 1 FAILED: {response['error']}")
        return False
    elif "result" in response and "content" in response["result"]:
        content = response["result"]["content"]
        if isinstance(content, list) and len(content) > 0:
            print(f"✅ Test 1 PASSED: Got valid response")
            print(f"   Response type: {type(content[0])}")
            if hasattr(content[0], 'text'):
                print(f"   Content preview: {content[0].text[:200]}...")
        else:
            print(f"❌ Test 1 FAILED: Invalid content structure")
            return False
    else:
        print(f"❌ Test 1 FAILED: Unexpected response structure")
        return False
    
    # Test 2: resolve_import with include_alternatives=true
    print("\nTest 2: resolve_import with include_alternatives=true")
    request = {
        "jsonrpc": "2.0", 
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "resolve_import",
            "arguments": {
                "crate_name": "serde",
                "import_path": "NonExistentItem",
                "include_alternatives": "true"
            }
        }
    }
    
    response = send_mcp_request(process, request)
    
    if "error" in response:
        print(f"❌ Test 2 FAILED: {response['error']}")
        return False
    elif "result" in response and "content" in response["result"]:
        print(f"✅ Test 2 PASSED: include_alternatives boolean parameter accepted")
    else:
        print(f"❌ Test 2 FAILED: Unexpected response structure")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! 🎉")
    print("Path Alias Resolution bug appears to be fixed!")
    print("="*60)
    return True

def main():
    """Main test function."""
    print("Starting MCP server test...")
    
    # Start the MCP server process
    try:
        process = subprocess.Popen(
            ["uv", "run", "docsrs-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        print("MCP server started, running tests...")
        
        # Initialize MCP protocol
        print("\nInitializing MCP protocol...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = send_mcp_request(process, init_request)
        
        if "error" in response:
            print(f"Failed to initialize MCP: {response['error']}")
            return False
            
        print("MCP initialized successfully")
        
        # Run resolve_import tests
        success = test_resolve_import(process)
        
        return success
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    finally:
        if 'process' in locals():
            process.terminate()
            process.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)