#!/usr/bin/env python3
"""Test the complete four-tier ingestion system."""

import asyncio
import sys
import sqlite3
sys.path.insert(0, 'src')

from pathlib import Path
from docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate

async def test_crate(crate_name, version):
    """Test ingestion of a specific crate and report tier used."""
    print(f"\n{'='*60}")
    print(f"Testing {crate_name}@{version}")
    print('='*60)
    
    try:
        db_path = await ingest_crate(crate_name, version)
        print(f"✓ Database created at: {db_path}")
        
        # Check ingestion details
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get ingestion tier
        cursor.execute("SELECT status, ingestion_tier FROM ingestion_status")
        status = cursor.fetchone()
        if status:
            print(f"✓ Ingestion status: {status[0]}")
            print(f"✓ Ingestion tier used: {status[1] or 'Unknown'}")
        
        # Count items
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        print(f"✓ Items in embeddings: {count}")
        
        # Check for examples
        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE examples IS NOT NULL")
        example_count = cursor.fetchone()[0]
        print(f"✓ Items with examples: {example_count}")
        
        # Check example embeddings
        cursor.execute("SELECT COUNT(*) FROM example_embeddings")
        example_emb_count = cursor.fetchone()[0]
        print(f"✓ Example embeddings: {example_emb_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

async def main():
    print("Testing Four-Tier Ingestion System")
    print("="*60)
    
    # Test 1: Old crate version that likely has no rustdoc JSON
    # This should trigger SOURCE_EXTRACTION (Tier 2)
    await test_crate("serde", "1.0.50")
    
    # Test 2: Latest version of a popular crate
    # This might have rustdoc JSON (Tier 1)
    await test_crate("tokio", "latest")
    
    # Test 3: Standard library crate
    # This should use stdlib fallback (special handling)
    await test_crate("std", "latest")
    
    # Test 4: Very old or obscure crate
    # This should cascade through all tiers
    await test_crate("lazy_static", "0.1.0")
    
    print("\n" + "="*60)
    print("Tier System Test Complete")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
