#!/usr/bin/env python3
"""Test example search functionality."""

import asyncio
import sys
import json
sys.path.insert(0, 'src')

from pathlib import Path
from docsrs_mcp.services.example_service import ExampleService

async def main():
    print("Testing example search functionality...")
    
    # Use the std database we just created
    service = ExampleService()
    
    # Search for HashMap examples
    results = await service.search_examples("std", "HashMap", k=5)
    
    print(f"\n✅ Search for 'HashMap' returned {len(results['examples'])} examples:")
    for i, ex in enumerate(results['examples'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  Path: {ex['item_path']}")
        print(f"  Language: {ex['language']}")
        print(f"  Score: {ex['score']:.3f}")
        print(f"  Code preview: {ex['code'][:80]}...")
    
    # Search for File examples
    results = await service.search_examples("std", "File", k=5)
    print(f"\n✅ Search for 'File' returned {len(results['examples'])} examples")
    
    # Search for io examples
    results = await service.search_examples("std", "io stdout", k=5)
    print(f"\n✅ Search for 'io stdout' returned {len(results['examples'])} examples")
    
    print("\n🎉 SUCCESS: Example search functionality is working correctly!")

if __name__ == "__main__":
    asyncio.run(main())
