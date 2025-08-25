#!/usr/bin/env python3
"""Test example storage through MCP protocol."""

import json
import sys
import time
import sqlite3
from pathlib import Path

# MCP protocol request to get std crate summary (triggers ingestion)
request = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "get_crate_summary",
        "arguments": {
            "crate_name": "std",
            "version": "latest"
        }
    },
    "id": 1
}

# Send to server via stdin
print(json.dumps(request), flush=True)

# Wait for ingestion to complete
time.sleep(15)

# Check database
print("\nChecking database for examples...", file=sys.stderr)
for db_path in Path("cache").rglob("*.db"):
    if "std" in str(db_path):
        print(f"Found database: {db_path}", file=sys.stderr)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check embeddings table
        cursor.execute("SELECT item_path, examples FROM embeddings WHERE examples IS NOT NULL")
        results = cursor.fetchall()
        print(f"\nItems with examples in embeddings: {len(results)}", file=sys.stderr)
        for path, examples in results[:3]:
            print(f"  {path}: {examples[:100] if examples else 'None'}", file=sys.stderr)
        
        # Check example_embeddings table  
        cursor.execute("SELECT COUNT(*) FROM example_embeddings")
        count = cursor.fetchone()[0]
        print(f"\nExample embeddings count: {count}", file=sys.stderr)
        
        if count > 0:
            cursor.execute("SELECT item_path, substr(example_text, 1, 80) FROM example_embeddings LIMIT 3")
            for path, text in cursor.fetchall():
                print(f"  {path}: {text}...", file=sys.stderr)
        
        conn.close()
        break
