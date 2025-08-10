#!/usr/bin/env python3
"""
Test script for fallback extraction functionality.
Tests with an old crate version that likely lacks rustdoc JSON.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extractors.source_extractor import CratesIoSourceExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_fallback_extraction():
    """Test the fallback extraction with an old crate version."""
    
    # Use an old version of a small crate that likely doesn't have rustdoc JSON
    # lazy_static 0.1.0 was published in 2016, well before rustdoc JSON support
    test_crate = "lazy_static"
    test_version = "0.1.0"
    
    logger.info(f"Testing fallback extraction for {test_crate}@{test_version}")
    
    async with CratesIoSourceExtractor() as extractor:
        try:
            # Extract documentation
            items = await extractor.extract_from_source(test_crate, test_version)
            
            logger.info(f"Successfully extracted {len(items)} items")
            
            # Display some sample items
            if items:
                logger.info("\nSample extracted items:")
                for i, item in enumerate(items[:5]):  # Show first 5 items
                    logger.info(f"\n--- Item {i+1} ---")
                    logger.info(f"Path: {item['item_path']}")
                    logger.info(f"Type: {item.get('item_type', 'unknown')}")
                    logger.info(f"Header: {item['header']}")
                    if item.get('docstring'):
                        doc_preview = item['docstring'][:100] + "..." if len(item['docstring']) > 100 else item['docstring']
                        logger.info(f"Doc: {doc_preview}")
                    if item.get('signature'):
                        sig_preview = item['signature'][:100] + "..." if len(item['signature']) > 100 else item['signature']
                        logger.info(f"Signature: {sig_preview}")
            
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False


async def test_with_modern_crate():
    """Test with a more recent crate to compare."""
    
    # Test with a small, simple crate
    test_crate = "once_cell"
    test_version = "1.0.0"
    
    logger.info(f"\nTesting with {test_crate}@{test_version}")
    
    async with CratesIoSourceExtractor() as extractor:
        try:
            items = await extractor.extract_from_source(test_crate, test_version)
            logger.info(f"Extracted {len(items)} items from {test_crate}@{test_version}")
            
            # Count by type
            type_counts = {}
            for item in items:
                item_type = item.get('item_type', 'unknown')
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            logger.info("Item type breakdown:")
            for item_type, count in sorted(type_counts.items()):
                logger.info(f"  {item_type}: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False


async def main():
    """Run tests."""
    logger.info("Starting fallback extraction tests")
    
    # Test with old crate
    success1 = await test_fallback_extraction()
    
    # Test with another crate
    success2 = await test_with_modern_crate()
    
    if success1 and success2:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())