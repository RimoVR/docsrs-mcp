#!/usr/bin/env python3
"""
Test extraction with macro crates like lazy_static.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extractors.source_extractor import CratesIoSourceExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_lazy_static():
    """Test extraction from lazy_static macro crate."""

    test_crate = "lazy_static"
    test_version = "0.2.0"  # Try a slightly newer version

    logger.info(f"Testing macro extraction for {test_crate}@{test_version}")

    async with CratesIoSourceExtractor() as extractor:
        try:
            items = await extractor.extract_from_source(test_crate, test_version)

            logger.info(f"Extracted {len(items)} items")

            # Count by type
            type_counts = {}
            macro_items = []
            for item in items:
                item_type = item.get("item_type", "unknown")
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
                if item_type == "macro":
                    macro_items.append(item)

            logger.info("Item type breakdown:")
            for item_type, count in sorted(type_counts.items()):
                logger.info(f"  {item_type}: {count}")

            # Show macro items
            if macro_items:
                logger.info(f"\nFound {len(macro_items)} macros:")
                for macro_item in macro_items:
                    logger.info(f"  - {macro_item['item_path']}")
                    if macro_item.get("docstring"):
                        doc_preview = (
                            macro_item["docstring"][:100] + "..."
                            if len(macro_item["docstring"]) > 100
                            else macro_item["docstring"]
                        )
                        logger.info(f"    Doc: {doc_preview}")
            else:
                logger.info("No macros found - checking for any content")
                if items:
                    for i, item in enumerate(items[:3]):
                        logger.info(f"\nItem {i + 1}:")
                        logger.info(f"  Path: {item['item_path']}")
                        logger.info(f"  Type: {item.get('item_type')}")
                        logger.info(f"  Signature: {item.get('signature', '')[:100]}")

            return len(items) > 0

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            import traceback

            traceback.print_exc()
            return False


async def test_serde_derive():
    """Test with serde_derive which has lots of macros."""

    test_crate = "serde_derive"
    test_version = "1.0.0"

    logger.info(f"\nTesting macro extraction for {test_crate}@{test_version}")

    async with CratesIoSourceExtractor() as extractor:
        try:
            items = await extractor.extract_from_source(test_crate, test_version)

            logger.info(f"Extracted {len(items)} items")

            # Count by type
            type_counts = {}
            for item in items:
                item_type = item.get("item_type", "unknown")
                type_counts[item_type] = type_counts.get(item_type, 0) + 1

            logger.info("Item type breakdown:")
            for item_type, count in sorted(type_counts.items()):
                logger.info(f"  {item_type}: {count}")

            return len(items) > 0

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False


async def main():
    """Run tests."""
    logger.info("Starting macro extraction tests")

    success1 = await test_lazy_static()
    success2 = await test_serde_derive()

    if success1 and success2:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
