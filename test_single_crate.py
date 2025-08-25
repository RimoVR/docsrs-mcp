#!/usr/bin/env python3
"""Test single crate ingestion to debug version issue."""

import asyncio
import sys
sys.path.insert(0, 'src')

from docsrs_mcp.ingestion.ingest_orchestrator import ingest_crate

async def main():
    # Test with an old serde version that definitely has no rustdoc JSON
    result = await ingest_crate("serde", "1.0.50")
    print(f"Result: {result}")

asyncio.run(main())
