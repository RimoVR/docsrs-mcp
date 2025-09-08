#!/usr/bin/env python3
"""Debug script to test trait path extraction."""

import asyncio
import logging
import sqlite3
import json
from pathlib import Path

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)

async def debug_trait_extraction():
    """Debug trait extraction and database content."""
    
    # Check current database content
    db_path = Path("./cache/serde/latest.db")
    if not db_path.exists():
        print("❌ No serde database found")
        return
    
    print("🔍 Analyzing current trait_implementations data...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT trait_path, impl_type_path, crate_id, item_id
            FROM trait_implementations 
            LIMIT 5
        """)
        
        rows = cursor.fetchall()
        print(f"Found {len(rows)} sample trait implementations:")
        
        for i, (trait_path, impl_type_path, crate_id, item_id) in enumerate(rows, 1):
            print(f"\n  {i}. Trait: '{trait_path}'")
            print(f"     Type:  '{impl_type_path}'")
            print(f"     Crate ID: {crate_id}")
            print(f"     Item ID:  {item_id}")
            
            # Check if impl_type_path is JSON
            try:
                if impl_type_path.startswith('{'):
                    parsed = json.loads(impl_type_path)
                    print(f"     Type (parsed): {json.dumps(parsed, indent=6)}")
            except:
                pass
    
    print(f"\n🧪 Testing path resolution logic...")
    
    # Test the path resolution logic directly
    from src.docsrs_mcp.ingestion.enhanced_trait_extractor import EnhancedTraitExtractor
    
    extractor = EnhancedTraitExtractor()
    
    # Sample trait_info structures that might come from rustdoc
    test_cases = [
        # Case 1: Direct name (what we're currently getting)
        {"name": "Deserialize"},
        
        # Case 2: Resolved path structure
        {"resolved_path": {"name": "Deserialize", "id": 123}},
        
        # Case 3: Full path structure
        {"path": {"name": "serde::de::Deserialize", "segments": ["serde", "de", "Deserialize"]}},
        
        # Case 4: With ID for paths cache lookup
        {"id": 456}
    ]
    
    # Mock paths cache with some sample data
    extractor.paths_cache = {
        123: {
            "name": "Deserialize",
            "path": ["serde", "de", "Deserialize"]
        },
        456: {
            "name": "Serialize", 
            "path": ["serde", "ser", "Serialize"]
        }
    }
    
    print("Testing trait path extraction:")
    for i, trait_info in enumerate(test_cases, 1):
        result = extractor._extract_trait_path(trait_info)
        print(f"  {i}. Input:  {trait_info}")
        print(f"     Output: '{result}'")
    
    # Test type path extraction with complex JSON from the database
    test_type_cases = [
        # Case 1: Arc<T> - from database
        {
            "resolved_path": {
                "path": "Arc", 
                "id": 102, 
                "args": {
                    "angle_bracketed": {
                        "args": [{"type": {"generic": "T"}}], 
                        "constraints": []
                    }
                }
            }
        },
        # Case 2: Tuple - from database
        {
            "tuple": [{"generic": "T"}]
        },
        # Case 3: Simple resolved path
        {
            "resolved_path": {
                "path": "String",
                "id": 200
            }
        }
    ]
    
    print(f"\nTesting type path extraction:")
    for i, test_type_info in enumerate(test_type_cases, 1):
        result = extractor._extract_type_path(test_type_info)
        print(f"  {i}. Input:  {test_type_info}")
        print(f"     Output: '{result}'")
    
    # Check what the database actually stores vs what we extract
    print(f"\n📊 Analysis:")
    print(f"  - Current DB trait paths are bare names (Deserialize, Serialize)")  
    print(f"  - Current DB type paths are JSON objects")
    print(f"  - Expected: FQN trait paths (serde::de::Deserialize)")
    print(f"  - Expected: Readable type paths (Arc<T>)")

if __name__ == "__main__":
    asyncio.run(debug_trait_extraction())