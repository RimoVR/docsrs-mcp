#!/usr/bin/env python3
"""Debug script to examine rustdoc JSON paths structure."""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

async def debug_paths_cache():
    """Debug paths cache structure and lookup."""
    
    print("🔍 Debugging paths cache structure...")
    
    try:
        from src.docsrs_mcp.ingestion.version_resolver import download_rustdoc
        import aiohttp
        
        # Download a small rustdoc JSON to examine structure  
        print("📥 Downloading rustdoc JSON to examine structure...")
        async with aiohttp.ClientSession() as session:
            json_content, url_used = await download_rustdoc(session, "serde", "latest")
        
        if not json_content:
            print("❌ Failed to download rustdoc JSON")
            return
            
        print(f"✅ Downloaded rustdoc JSON from {url_used}, decompressing...")
        from src.docsrs_mcp.ingestion.version_resolver import decompress_content
        json_str = await decompress_content(json_content, url_used)
        print(f"✅ Decompressed, parsing JSON structure...")
        data = json.loads(json_str)
        
        # Examine the paths structure
        paths = data.get("paths", {})
        print(f"📊 Paths structure: {len(paths)} entries")
        
        # Show a sample of paths entries
        print(f"\n📋 Sample paths entries:")
        count = 0
        for path_id, path_info in paths.items():
            if count >= 5:
                break
            print(f"  ID {path_id}: {path_info}")
            count += 1
        
        # Look for trait-related paths
        trait_paths = {}
        for path_id, path_info in paths.items():
            if isinstance(path_info, dict):
                path_segments = path_info.get("path", [])
                if isinstance(path_segments, list) and len(path_segments) > 0:
                    path_str = "::".join(path_segments)
                    if any(trait_name in path_str.lower() for trait_name in ["serialize", "deserialize"]):
                        trait_paths[path_id] = {"path_str": path_str, "info": path_info}
                        
        print(f"\n🎯 Found {len(trait_paths)} serialize/deserialize related paths:")
        for path_id, info in trait_paths.items():
            print(f"  ID {path_id}: {info['path_str']}")
            print(f"    Raw: {info['info']}")
        
        # Now examine a trait implementation from the index
        index = data.get("index", {})
        impl_items = {}
        
        for item_id, item_data in index.items():
            inner = item_data.get("inner", {})
            if isinstance(inner, dict) and "impl" in inner:
                impl_items[item_id] = item_data
                
        print(f"\n🔧 Found {len(impl_items)} impl items in index")
        
        # Examine one impl item in detail
        if impl_items:
            sample_id, sample_item = next(iter(impl_items.items()))
            print(f"\n📋 Sample impl item {sample_id}:")
            
            inner = sample_item["inner"]
            impl_data = inner["impl"]
            
            trait_info = impl_data.get("trait")
            for_type = impl_data.get("for")
            
            print(f"  Trait info: {trait_info}")
            print(f"  For type: {for_type}")
            
            # Test path resolution
            from src.docsrs_mcp.ingestion.enhanced_trait_extractor import EnhancedTraitExtractor
            
            extractor = EnhancedTraitExtractor()
            extractor.paths_cache = paths  # Set the actual paths from rustdoc
            
            # Try to extract trait path
            if trait_info:
                trait_path = extractor._extract_trait_path(trait_info)
                print(f"  🎯 Extracted trait path: '{trait_path}'")
                
                # Debug the lookup process
                print(f"  🔍 Debug trait path extraction:")
                trait_id = None
                if "id" in trait_info:
                    trait_id = trait_info["id"]
                    print(f"    Found trait ID: {trait_id}")
                elif "resolved_path" in trait_info and isinstance(trait_info["resolved_path"], dict):
                    trait_id = trait_info["resolved_path"].get("id")
                    print(f"    Found trait ID from resolved_path: {trait_id}")
                elif "path" in trait_info and isinstance(trait_info["path"], dict):
                    trait_id = trait_info["path"].get("id")
                    print(f"    Found trait ID from path: {trait_id}")
                
                if trait_id and trait_id in paths:
                    path_entry = paths[trait_id]
                    print(f"    ✅ Found in paths cache: {path_entry}")
                    
                    if isinstance(path_entry, dict) and "path" in path_entry:
                        path_segments = path_entry["path"]
                        if isinstance(path_segments, list) and path_segments:
                            fqn = "::".join(path_segments)
                            print(f"    🎯 FQN would be: {fqn}")
                        else:
                            print(f"    ❌ Path segments not valid: {path_segments}")
                    else:
                        print(f"    ❌ Path entry format unexpected: {path_entry}")
                else:
                    print(f"    ❌ Trait ID {trait_id} not found in paths cache")
            
            # Try to extract type path
            if for_type:
                type_path = extractor._extract_type_path(for_type)
                print(f"  🎯 Extracted type path: '{type_path}'")
        
        print(f"\n🎯 Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Error during paths cache debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_paths_cache())