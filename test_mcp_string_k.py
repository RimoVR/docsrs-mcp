#!/usr/bin/env python3
"""Test search_examples functionality with Python MCP client."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_search_examples():
    """Test the search_examples tool via MCP."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "docsrs-mcp"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                if "search" in tool.name.lower() or "example" in tool.name.lower():
                    print(f"  - {tool.name}: {tool.description[:100]}")
            
            # Test search_examples with 'serde' crate
            print("\n=== Testing search_examples for 'serde' ===")
            
            # Test 1: Search for serialization examples
            result = await session.call_tool(
                "search_examples",
                arguments={
                    "crate_name": "serde",
                    "query": "serialize struct"
                }
            )
            
            print(f"\nSearch for 'serialize struct':")
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        # Parse the JSON response
                        data = json.loads(item.text)
                        print(f"Found {len(data.get('examples', []))} examples")
                        
                        # Show first example
                        if data.get('examples'):
                            first = data['examples'][0]
                            print(f"\nFirst example:")
                            print(f"  Item: {first['item_path']}")
                            print(f"  Language: {first['language']}")
                            print(f"  Code preview: {first['code'][:200]}...")
            
            # Test 2: Search with k parameter
            result = await session.call_tool(
                "search_examples",
                arguments={
                    "crate_name": "serde",
                    "query": "deserialize",
                    "k": "3"
                }
            )
            
            print(f"\n\nSearch for 'deserialize' (k=3):")
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        data = json.loads(item.text)
                        print(f"Found {len(data.get('examples', []))} examples")
            
            # Test 3: Language filter
            result = await session.call_tool(
                "search_examples", 
                arguments={
                    "crate_name": "serde",
                    "query": "derive macro",
                    "language": "rust"
                }
            )
            
            print(f"\n\nSearch for 'derive macro' (rust only):")
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        data = json.loads(item.text)
                        examples = data.get('examples', [])
                        rust_count = sum(1 for e in examples if e.get('language') == 'rust')
                        print(f"Found {rust_count} Rust examples out of {len(examples)} total")


if __name__ == "__main__":
    asyncio.run(test_search_examples())
