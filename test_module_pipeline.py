#!/usr/bin/env python3
"""
Test script to verify the module storage pipeline works end-to-end after the import fix.

This script:
1. Ingests a simple crate
2. Checks if modules are stored in the database
3. Tests the get_module_tree functionality
"""

import asyncio
import sqlite3
from pathlib import Path
import sys
sys.path.append('src')

from docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate


async def test_module_pipeline():
    """Test the complete module storage pipeline."""
    
    print("=== Testing Module Storage Pipeline ===")
    
    # Test with a simple, well-known crate
    crate_name = "serde"
    version = "latest"
    
    print(f"1. Ingesting {crate_name} {version}...")
    
    try:
        # Ingest the crate
        result = await ingest_crate(crate_name, version)
        print(f"   Ingestion result: {result}")
        
        # Check the database for modules
        if isinstance(result, Path):
            db_path = result
        else:
            db_path = Path(result)
        
        if not db_path.exists():
            print(f"   ERROR: Database file not found at {db_path}")
            # Let's check what's in the cache directory
            cache_dir = Path("cache")
            if cache_dir.exists():
                print("   Files in cache directory:")
                for file in cache_dir.rglob("*.db"):
                    print(f"     - {file}")
            return False
        
        print(f"2. Checking database at {db_path}...")
        
        # Connect to database and check modules table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if modules table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='modules';")
        table_exists = cursor.fetchone()
        if not table_exists:
            print("   ERROR: modules table does not exist")
            conn.close()
            return False
        
        print("   modules table exists ✓")
        
        # Check if modules were actually stored
        cursor.execute("SELECT COUNT(*) FROM modules;")
        module_count = cursor.fetchone()[0]
        print(f"   Found {module_count} modules in database")
        
        if module_count == 0:
            print("   ERROR: No modules found in database")
            conn.close()
            return False
        
        # Show some sample module data
        cursor.execute("SELECT path, name, parent_id, item_count FROM modules LIMIT 5;")
        sample_modules = cursor.fetchall()
        print("   Sample modules:")
        for module in sample_modules:
            print(f"     - {module[0]} (name: {module[1]}, parent: {module[2]}, items: {module[3]})")
        
        conn.close()
        
        print("3. Testing MCP module tree functionality...")
        
        # Now test the MCP functionality
        from docsrs_mcp.services.crate_service import CrateService
        
        service = CrateService()
        tree_result = await service.get_module_tree(crate_name, version)
        
        print(f"   Module tree result: {tree_result}")
        
        # Check if we got a proper tree structure
        if isinstance(tree_result, dict) and 'tree' in tree_result and tree_result['tree']:
            tree = tree_result['tree']
            print(f"   Root module: {tree.name}")
            print(f"   Children count: {len(tree.children) if tree.children else 0}")
            print("   Module tree retrieval working ✓")
        else:
            print("   ERROR: Module tree is empty or malformed")
            return False
        
        print("\n=== Module Pipeline Test PASSED ===")
        return True
        
    except Exception as e:
        print(f"   ERROR during pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_module_pipeline())
    
    if success:
        print("\n✅ All tests passed! Module storage pipeline is working.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Module storage pipeline needs further investigation.")
        sys.exit(1)