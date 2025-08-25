#!/bin/bash
# Test MCP mode with example processing

echo "=== Testing MCP Server Example Processing ==="
echo

# Clean cache
echo "1. Cleaning cache..."
rm -rf cache/*

# Start MCP server in background with our fixed code
echo "2. Starting MCP server with fixed code..."
uvx --from . docsrs-mcp > mcp_test.log 2>&1 &
MCP_PID=$!
echo "   Server PID: $MCP_PID"

# Wait for server to start
sleep 5

# Create a simple test that sends MCP commands via stdin
echo "3. Creating MCP test client..."
cat > test_mcp_client.py << 'EOF'
import json
import subprocess
import time

def send_mcp_command(cmd):
    """Send command to MCP server via uvx."""
    proc = subprocess.Popen(
        ["uvx", "--from", ".", "docsrs-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send initialize
    init_req = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "clientInfo": {"name": "test_client", "version": "1.0.0"},
            "protocolVersion": "0.1.0"
        },
        "id": 1
    }
    
    proc.stdin.write(f"Content-Length: {len(json.dumps(init_req))}\r\n\r\n")
    proc.stdin.write(json.dumps(init_req))
    proc.stdin.flush()
    
    # Send tool call
    tool_req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_examples",
            "arguments": {
                "crate_name": "serde",
                "query": "derive"
            }
        },
        "id": 2
    }
    
    proc.stdin.write(f"Content-Length: {len(json.dumps(tool_req))}\r\n\r\n")
    proc.stdin.write(json.dumps(tool_req))
    proc.stdin.flush()
    
    # Wait and get output
    time.sleep(2)
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    
    return stdout, stderr

stdout, stderr = send_mcp_command("search_examples")
print("STDOUT:", stdout[:500] if stdout else "None")
print("STDERR:", stderr[:500] if stderr else "None")
EOF

echo "4. Running MCP test..."
uv run python test_mcp_client.py

# Kill the MCP server
echo
echo "5. Cleaning up..."
kill $MCP_PID 2>/dev/null || true

echo
echo "=== Test Complete ==="
echo "Check mcp_test.log for server logs"