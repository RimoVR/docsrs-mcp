#!/usr/bin/env python3
"""Test fresh serde ingestion with trait extraction fixes."""

import asyncio
import logging
import sqlite3
from pathlib import Path

# Configure logging to see trait extraction debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)

async def test_serde_fresh():
    """Test serde ingestion with the trait extraction fixes."""
    
    print("🧪 Testing fresh serde ingestion with trait extraction fixes...")
    
    # Remove existing serde cache to force fresh ingestion
    serde_dir = Path("./cache/serde")
    test_version = "1.0.210"
    test_db = serde_dir / f"{test_version}.db"
    
    if test_db.exists():
        test_db.unlink()
        print(f"🗑️ Removed existing database: {test_db}")
    
    try:
        from src.docsrs_mcp.ingest import ingest_crate
        
        print(f"🚀 Ingesting serde v{test_version} (this will take a minute)...")
        db_path = await ingest_crate("serde", test_version)
        
        print(f"✅ Fresh ingestion complete! Database: {db_path}")
        
        # Analyze the results
        with sqlite3.connect(db_path) as conn:
            # Check trait implementations  
            cursor = conn.execute("SELECT COUNT(*) FROM trait_implementations")
            trait_count = cursor.fetchone()[0]
            print(f"\n📊 Analysis of trait_implementations table:")
            print(f"   Total trait implementations: {trait_count}")
            
            if trait_count > 0:
                # Check for FQN trait paths (containing '::')
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM trait_implementations 
                    WHERE trait_path LIKE '%::%'
                """)
                fqn_count = cursor.fetchone()[0]
                
                # Check for readable type paths (not starting with '{')
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM trait_implementations 
                    WHERE impl_type_path NOT LIKE '{%'
                """)
                readable_types = cursor.fetchone()[0]
                
                print(f"   FQN trait paths (with '::'): {fqn_count}/{trait_count} ({fqn_count/trait_count*100:.1f}%)")
                print(f"   Readable type paths (not JSON): {readable_types}/{trait_count} ({readable_types/trait_count*100:.1f}%)")
                
                # Sample the new data
                print(f"\n📋 Sample of new trait implementations:")
                cursor = conn.execute("""
                    SELECT trait_path, impl_type_path, item_id 
                    FROM trait_implementations 
                    ORDER BY id DESC
                    LIMIT 8
                """)
                
                for i, (trait_path, impl_type_path, item_id) in enumerate(cursor.fetchall(), 1):
                    print(f"   {i}. {trait_path}")
                    print(f"      for {impl_type_path}")
                    print(f"      (item: {item_id})")
                
                # Quick fix assessment
                print(f"\n🎯 Fix Assessment:")
                if fqn_count > trait_count * 0.5:
                    print(f"   ✅ Trait path extraction: MUCH IMPROVED ({fqn_count}/{trait_count} FQN)")
                elif fqn_count > 0:
                    print(f"   🔄 Trait path extraction: PARTIALLY IMPROVED ({fqn_count}/{trait_count} FQN)")
                else:
                    print(f"   ❌ Trait path extraction: STILL NEEDS WORK (no FQN found)")
                
                if readable_types > trait_count * 0.8:
                    print(f"   ✅ Type path extraction: FIXED ({readable_types}/{trait_count} readable)")
                elif readable_types > trait_count * 0.3:
                    print(f"   🔄 Type path extraction: PARTIALLY FIXED ({readable_types}/{trait_count} readable)")
                else:
                    print(f"   ❌ Type path extraction: STILL BROKEN ({readable_types}/{trait_count} readable)")
                
                return True
            else:
                print("   ❌ No trait implementations found - ingestion may have failed")
                return False
                
    except Exception as e:
        print(f"❌ Error during fresh ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_serde_fresh())
    if success:
        print("\n🎉 Fresh ingestion test completed!")
    else:
        print("\n💥 Fresh ingestion test failed!")