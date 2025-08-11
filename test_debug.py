#!/usr/bin/env python3
"""Debug script for testing version comparison error."""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp.version_diff import VersionDiffEngine
from docsrs_mcp.models import CompareVersionsRequest

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def main():
    """Test version comparison that fails."""
    engine = VersionDiffEngine()
    
    request = CompareVersionsRequest(
        crate_name="once_cell",
        version_a="1.19.0",
        version_b="1.21.3"
    )
    
    try:
        result = await engine.compare_versions(request)
        print("SUCCESS: Comparison completed")
        print(f"Total changes: {result.summary.total_changes}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())