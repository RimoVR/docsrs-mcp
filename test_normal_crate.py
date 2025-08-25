#!/usr/bin/env python3
"""Test normal crate fallback ingestion."""

import asyncio
import sys
import sqlite3
sys.path.insert(0, 'src')

from pathlib import Path
from docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate

async def main():
    print("Testing serde crate ingestion (normal crate)...")
    
    # Ingest serde - a normal crate that will use fallback
    db_path = await ingest_crate("serde", "1.0.0")
    print(f"Database created at: {db_path}")
    
    # Check for content
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check embeddings table
    cursor.execute("SELECT item_path, header, substr(content, 1, 100), examples FROM embeddings")
    results = cursor.fetchall()
    print(f"\nItems in embeddings: {len(results)}")
    for path, header, content, examples in results:
        print(f"  Path: {path}")
        print(f"  Header: {header}")
        print(f"  Content: {content}...")
        print(f"  Has examples: {'Yes' if examples else 'No'}")
    
    # Check ingestion status
    cursor.execute("SELECT status, ingestion_tier FROM ingestion_status")
    status = cursor.fetchone()
    if status:
        print(f"\nIngestion status: {status[0]}")
        print(f"Ingestion tier: {status[1]}")
    
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
