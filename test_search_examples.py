#\!/usr/bin/env python3
import json
import sys

# MCP protocol test for searchExamples
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "search_examples",
        "arguments": {
            "crate_name": "serde",
            "query": "deserialize json",
            "k": 3
        }
    }
}

print(json.dumps(request))
