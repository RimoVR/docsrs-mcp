#!/usr/bin/env python3
"""Test stdlib crate ingestion with examples."""

import subprocess
import sqlite3
import time
from pathlib import Path

# Clean cache
subprocess.run("rm -rf cache/*", shell=True)

# Start server
print("Starting server...")
server = subprocess.Popen(
    ["uvx", "--from", ".", "docsrs-mcp", "--mode", "rest"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(5)

try:
    # Ingest std crate directly
    print("\nIngesting std crate...")
    result = subprocess.run(
        """curl -X POST http://localhost:8000/mcp/tools/get_crate_summary \
           -H "Content-Type: application/json" \
           -d '{"crate_name": "std", "version": "latest"}'""",
        shell=True,
        capture_output=True,
        text=True
    )
    print(f"Response: {result.stdout[:200]}")
    
    time.sleep(10)
    
    # Check database
    print("\nChecking database...")
    std_db = None
    for db_path in Path("cache").rglob("*.db"):
        if "std" in str(db_path):
            std_db = db_path
            print(f"Found std database: {db_path}")
            break
    
    if std_db:
        conn = sqlite3.connect(std_db)
        cursor = conn.cursor()
        
        # Check embeddings
        cursor.execute("SELECT item_path, substr(examples, 1, 100) FROM embeddings WHERE examples IS NOT NULL")
        results = cursor.fetchall()
        print(f"\nEmbeddings with examples: {len(results)}")
        for path, ex in results[:3]:
            print(f"  {path}: {ex}")
        
        # Check example_embeddings
        cursor.execute("SELECT COUNT(*) FROM example_embeddings")
        count = cursor.fetchone()[0]
        print(f"\nExample embeddings count: {count}")
        
        if count > 0:
            cursor.execute("SELECT item_path, substr(example_text, 1, 50), language FROM example_embeddings LIMIT 3")
            for path, text, lang in cursor.fetchall():
                print(f"  {path} [{lang}]: {text}...")
                
            print("\n✅ SUCCESS: Examples stored in std crate!")
        
        conn.close()
    else:
        print("❌ No std database found")
        
finally:
    server.terminate()
    print("\nTest complete")
