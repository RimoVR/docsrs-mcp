#!/usr/bin/env python3
"""Test fresh ingestion with improved trait extraction."""

import asyncio
import logging
import sqlite3
from pathlib import Path

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)

async def test_fresh_ingestion():
    """Test ingestion with a small crate to verify trait extraction fixes."""
    
    print("🧪 Testing fresh ingestion with trait extraction fixes...")
    
    # Use a smaller crate first for testing
    test_crate = "lazy_static"
    test_version = "1.4.0"
    
    try:
        from src.docsrs_mcp.ingest import ingest_crate
        
        print(f"Ingesting {test_crate} v{test_version}...")
        db_path = await ingest_crate(test_crate, test_version)
        
        print(f"✅ Ingestion complete! Database: {db_path}")
        
        # Check the results
        with sqlite3.connect(db_path) as conn:
            # Check trait implementations
            cursor = conn.execute("SELECT COUNT(*) FROM trait_implementations")
            trait_count = cursor.fetchone()[0]
            print(f"📊 Found {trait_count} trait implementations")
            
            if trait_count > 0:
                # Sample a few trait implementations
                cursor = conn.execute("""
                    SELECT trait_path, impl_type_path, item_id 
                    FROM trait_implementations 
                    LIMIT 5
                """)
                
                print("📋 Sample trait implementations:")
                for i, (trait_path, impl_type_path, item_id) in enumerate(cursor.fetchall(), 1):
                    print(f"  {i}. {trait_path} for {impl_type_path}")
                    print(f"     (item_id: {item_id})")
                
                # Check if we have FQN trait paths now
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM trait_implementations 
                    WHERE trait_path LIKE '%::%'
                """)
                fqn_count = cursor.fetchone()[0]
                print(f"🎯 Trait paths with FQN (containing '::'): {fqn_count}")
                
                # Check if type paths are readable (not JSON)
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM trait_implementations 
                    WHERE impl_type_path NOT LIKE '%{%'
                """)
                readable_types = cursor.fetchone()[0]
                print(f"📝 Readable type paths (not JSON): {readable_types}")
                
                return trait_count, fqn_count, readable_types
            else:
                print("⚠️ No trait implementations found")
                return 0, 0, 0
                
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    asyncio.run(test_fresh_ingestion())