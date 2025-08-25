#!/usr/bin/env python3
"""Test script for MCP example search functionality."""

import json
import sys

def send_mcp_request(request):
    """Send an MCP request to the server via stdin."""
    request_str = json.dumps(request)
    print(f"Content-Length: {len(request_str)}\r\n\r\n{request_str}", end="", flush=True)

def test_example_search():
    """Test the search_examples functionality."""
    
    # First, let's ingest a simple crate that should have examples
    ingest_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "ingest_cargo_file",
            "arguments": {
                "file_path": "./Cargo.toml",
                "skip_existing": "false"
            }
        },
        "id": 1
    }
    
    print("Testing MCP example search...")
    print(f"Request: {json.dumps(ingest_request, indent=2)}")
    
    # Now search for examples
    search_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_examples",
            "arguments": {
                "crate_name": "serde",
                "query": "derive",
                "k": "5"
            }
        },
        "id": 2
    }
    
    print(f"Search Request: {json.dumps(search_request, indent=2)}")

if __name__ == "__main__":
    test_example_search()