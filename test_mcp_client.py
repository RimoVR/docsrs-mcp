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
