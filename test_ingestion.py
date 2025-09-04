#!/usr/bin/env python3

import asyncio
import sqlite3
from pathlib import Path

async def test_ingestion():
    """Test ingestion of serde crate and check example extraction"""
    
    # Import after setting up Python path
    from src.docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate
    
    print("Starting ingestion of serde...")
    
    # Ingest the crate
    result = await ingest_crate("serde", "latest")
    
    print(f"Ingestion result: {result}")
    
    # Check database
    db_path = Path("cache/serde/latest.db")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check embeddings table
        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE examples IS NOT NULL")
        example_count = cursor.fetchone()[0]
        print(f"Embeddings with examples: {example_count}")
        
        # Check if any examples exist
        cursor.execute("SELECT item_path, examples FROM embeddings WHERE examples IS NOT NULL LIMIT 1")
        row = cursor.fetchone()
        if row:
            print(f"Sample item with examples: {row[0]}")
            print(f"Examples content (first 200 chars): {row[1][:200]}")
        
        # Check example_embeddings table
        cursor.execute("SELECT COUNT(*) FROM example_embeddings")
        example_embedding_count = cursor.fetchone()[0]
        print(f"Example embeddings count: {example_embedding_count}")
        
        conn.close()
    else:
        print(f"Database not found at {db_path}")

if __name__ == "__main__":
    asyncio.run(test_ingestion())