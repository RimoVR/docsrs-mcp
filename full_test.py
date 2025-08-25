#!/usr/bin/env python3
"""Full test of example processing with local fixed code."""

import subprocess
import time
import sqlite3
import json
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def test_example_processing():
    """Test the complete example processing pipeline."""
    
    print("=" * 60)
    print("TESTING EXAMPLE PROCESSING WITH FIXED CODE")
    print("=" * 60)
    
    # 1. Clean cache
    print("\n1. Cleaning cache...")
    run_command("rm -rf cache/*")
    
    # 2. Start the server in background with our fixed code
    print("\n2. Starting MCP server with uvx (fixed code)...")
    server_proc = subprocess.Popen(
        ["uvx", "--from", ".", "docsrs-mcp", "--mode", "rest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(8)  # Wait for server to start
    
    try:
        # 3. Ingest std crate (uses fallback mode with our enhanced examples)
        print("\n3. Ingesting std crate (will use fallback mode)...")
        result = run_command("""
            curl -X POST http://localhost:8000/mcp/tools/ingest_cargo_file \
            -H "Content-Type: application/json" \
            -d '{"file_path": "./Cargo.toml", "skip_existing": "false"}'
        """)
        print(f"Ingestion response: {result[:200]}")
        
        # Wait for ingestion to complete
        print("   Waiting for ingestion to complete...")
        time.sleep(20)
        
        # 4. Check database for examples
        print("\n4. Checking database for stored examples...")
        
        # Find the std database
        std_db = None
        for db_path in Path("cache").rglob("*.db"):
            print(f"   Found database: {db_path}")
            if "std" in str(db_path) or "serde" in str(db_path):
                std_db = db_path
                break
        
        if std_db:
            conn = sqlite3.connect(std_db)
            cursor = conn.cursor()
            
            # Check embeddings table for examples
            cursor.execute("SELECT item_path, examples FROM embeddings WHERE examples IS NOT NULL LIMIT 5")
            results = cursor.fetchall()
            print(f"\n   Examples in embeddings table: {len(results)}")
            for path, examples in results:
                if examples:
                    print(f"     - {path}: {examples[:100]}...")
            
            # Check example_embeddings table
            cursor.execute("SELECT COUNT(*) FROM example_embeddings")
            count = cursor.fetchone()[0]
            print(f"\n   Total examples in example_embeddings table: {count}")
            
            if count > 0:
                cursor.execute("SELECT item_path, example_text, language FROM example_embeddings LIMIT 3")
                examples = cursor.fetchall()
                print("\n   Sample examples:")
                for path, text, lang in examples:
                    print(f"     - {path} [{lang}]: {text[:80]}...")
                    
                print("\n✅ SUCCESS: Examples are being stored correctly!")
            else:
                print("\n⚠️  WARNING: No examples in example_embeddings table")
            
            conn.close()
        else:
            print("   ❌ No database found")
        
        # 5. Test search
        print("\n5. Testing example search...")
        search_result = run_command("""
            curl -X POST http://localhost:8000/mcp/tools/search_examples \
            -H "Content-Type: application/json" \
            -d '{"crate_name": "serde", "query": "derive", "k": "5"}'
        """)
        
        if search_result:
            try:
                search_json = json.loads(search_result)
                example_count = len(search_json.get("examples", []))
                print(f"   Found {example_count} examples")
                
                if example_count > 0:
                    print("\n✅ EXAMPLE SEARCH WORKING!")
                    for i, ex in enumerate(search_json["examples"][:2], 1):
                        print(f"\n   Example {i}:")
                        print(f"     Language: {ex.get('language')}")
                        print(f"     Code: {ex.get('code', '')[:100]}...")
                else:
                    print("\n⚠️  No examples returned from search")
            except json.JSONDecodeError:
                print(f"   Failed to parse search response: {search_result[:200]}")
        
    finally:
        # Clean up
        print("\n6. Stopping server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_example_processing()