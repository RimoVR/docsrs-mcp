#!/usr/bin/env python3
"""Test script to verify re-export auto-discovery works."""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_reexport_discovery():
    """Test that re-export discovery works for serde."""
    try:
        from src.docsrs_mcp.database import get_discovered_reexports
        from src.docsrs_mcp.fuzzy_resolver import resolve_path_alias
        from src.docsrs_mcp.ingest import ingest_crate

        # Test with serde crate
        crate_name = "serde"
        version = "1.0.219"  # Use a specific version for consistency

        logger.info(f"Testing re-export discovery for {crate_name}@{version}")

        # Ingest the crate
        logger.info("Ingesting crate...")
        db_path = await ingest_crate(crate_name, version)
        logger.info(f"Database path: {db_path}")

        # Get discovered re-exports
        logger.info("Loading discovered re-exports...")
        reexports = await get_discovered_reexports(db_path, crate_name)

        # Log discovered re-exports
        if reexports:
            logger.info(f"Found {len(reexports)} re-export mappings:")
            # Show first 10 re-exports
            for i, (alias, actual) in enumerate(list(reexports.items())[:10]):
                logger.info(f"  {alias} -> {actual}")
            if len(reexports) > 10:
                logger.info(f"  ... and {len(reexports) - 10} more")
        else:
            logger.warning("No re-exports discovered!")

        # Test resolution of common aliases
        test_cases = [
            "Deserialize",
            "Serialize",
            "Deserializer",
            "Serializer",
        ]

        logger.info("\nTesting path alias resolution:")
        for item_path in test_cases:
            resolved = await resolve_path_alias(crate_name, item_path, str(db_path))
            logger.info(f"  {item_path} -> {resolved}")

            # Check if it was resolved via re-export
            if resolved != item_path:
                logger.info("    ✓ Successfully resolved!")

        logger.info("\nTest completed successfully!")

        # Return results for verification
        return {
            "crate": crate_name,
            "version": version,
            "reexports_found": len(reexports) if reexports else 0,
            "test_results": {
                path: await resolve_path_alias(crate_name, path, str(db_path))
                for path in test_cases
            },
        }

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    result = asyncio.run(test_reexport_discovery())
    if result:
        print("\n=== Test Results ===")
        print(f"Crate: {result['crate']}@{result['version']}")
        print(f"Re-exports discovered: {result['reexports_found']}")
        print("\nPath resolutions:")
        for path, resolved in result["test_results"].items():
            status = "✓" if resolved != path else "✗"
            print(f"  {status} {path} -> {resolved}")
