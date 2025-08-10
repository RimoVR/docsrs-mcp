#!/usr/bin/env python3
"""
Debug script to understand why extraction is returning 0 items.
"""

import asyncio
import tarfile
import io
import aiohttp


async def debug_crate_contents():
    """Download and inspect crate contents to debug extraction."""
    
    crate = "lazy_static"
    version = "0.1.0"
    url = f"https://static.crates.io/crates/{crate}/{crate}-{version}.crate"
    
    print(f"Downloading {crate}@{version} from {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
    
    print(f"Downloaded {len(data)} bytes")
    
    # Open and inspect tar contents
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        print("\nFiles in archive:")
        
        rust_files = []
        for member in tar.getmembers():
            print(f"  - {member.name} ({member.size} bytes)")
            if member.name.endswith('.rs'):
                rust_files.append(member)
        
        print(f"\nFound {len(rust_files)} Rust files")
        
        # Examine first Rust file
        if rust_files:
            for rust_file in rust_files[:3]:  # Check first 3 Rust files
                print(f"\n--- Contents of {rust_file.name} ---")
                file_obj = tar.extractfile(rust_file)
                if file_obj:
                    content = file_obj.read().decode('utf-8', errors='ignore')
                    lines = content.split('\n')
                    
                    # Show first 50 lines
                    for i, line in enumerate(lines[:50]):
                        print(f"{i+1:3}: {line}")
                    
                    # Look for pub items
                    print(f"\nSearching for 'pub' items in {rust_file.name}:")
                    pub_count = 0
                    for i, line in enumerate(lines):
                        if 'pub' in line and not line.strip().startswith('//'):
                            print(f"  Line {i+1}: {line.strip()[:80]}")
                            pub_count += 1
                            if pub_count >= 10:
                                print("  ... (showing first 10)")
                                break
                    
                    if pub_count == 0:
                        print("  No 'pub' items found!")
                    
                    file_obj.close()


if __name__ == "__main__":
    asyncio.run(debug_crate_contents())