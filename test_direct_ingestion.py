#!/usr/bin/env python3
"""Test ingestion directly without MCP protocol."""

import asyncio
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate
import sqlite3

async def main():
    print("Testing std crate ingestion...")
    
    # Ingest std crate (will use fallback mode)
    db_path = await ingest_crate("std", "latest")
    print(f"Database created at: {db_path}")
    
    # Check for examples
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check embeddings table
    cursor.execute("SELECT item_path, examples FROM embeddings WHERE examples IS NOT NULL")
    results = cursor.fetchall()
    print(f"\nItems with examples in embeddings: {len(results)}")
    for path, examples in results[:5]:
        print(f"  {path}: {examples[:100] if examples else 'None'}...")
    
    # Check example_embeddings table
    cursor.execute("SELECT COUNT(*) FROM example_embeddings")
    count = cursor.fetchone()[0]
    print(f"\nExample embeddings count: {count}")
    
    if count > 0:
        cursor.execute("SELECT item_path, substr(example_text, 1, 80), language FROM example_embeddings LIMIT 5")
        for path, text, lang in cursor.fetchall():
            print(f"  {path} [{lang}]: {text}...")
    
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
