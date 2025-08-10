#!/usr/bin/env python3
"""
Test enhanced macro extraction capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extractors.source_extractor import CratesIoSourceExtractor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_macro_crates():
    """Test extraction from various macro-heavy crates."""
    
    test_cases = [
        ("lazy_static", "1.4.0"),  # Classic macro_rules! crate
        ("serde_derive", "1.0.100"),  # Procedural macro crate
        ("paste", "1.0.0"),  # Complex macro patterns
        ("anyhow", "1.0.50"),  # Has macro_rules! for error handling
    ]
    
    results = {}
    
    async with CratesIoSourceExtractor() as extractor:
        for crate_name, version in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {crate_name}@{version}")
            logger.info(f"{'='*60}")
            
            try:
                items = await extractor.extract_from_source(crate_name, version)
                
                # Analyze results
                type_counts = {}
                macro_items = []
                
                for item in items:
                    item_type = item.get('item_type', 'unknown')
                    type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    
                    if 'macro' in item_type.lower():
                        macro_items.append(item)
                
                # Log summary
                logger.info(f"Total items extracted: {len(items)}")
                logger.info("Item type breakdown:")
                for item_type, count in sorted(type_counts.items()):
                    logger.info(f"  {item_type}: {count}")
                
                # Show macro details
                if macro_items:
                    logger.info(f"\nMacro items ({len(macro_items)}):")
                    for macro in macro_items[:10]:  # Show first 10
                        logger.info(f"  - {macro['item_path']}")
                        if macro.get('signature'):
                            sig_preview = macro['signature'][:100]
                            if len(macro['signature']) > 100:
                                sig_preview += "..."
                            logger.info(f"    Signature: {sig_preview}")
                        if macro.get('docstring'):
                            doc_preview = macro['docstring'].split('\n')[0][:80]
                            logger.info(f"    Doc: {doc_preview}")
                        if macro.get('macro_patterns'):
                            logger.info(f"    Patterns: {len(macro.get('macro_patterns', []))} pattern(s)")
                
                results[crate_name] = {
                    "total": len(items),
                    "macros": len(macro_items),
                    "types": type_counts
                }
                
            except Exception as e:
                logger.error(f"Failed to extract from {crate_name}@{version}: {e}")
                import traceback
                traceback.print_exc()
                results[crate_name] = {"error": str(e)}
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for crate_name, result in results.items():
        if "error" in result:
            logger.error(f"{crate_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{crate_name}: {result['total']} items, {result['macros']} macros")
    
    # Check if we improved macro extraction
    success = all(
        result.get("macros", 0) > 0 
        for crate_name, result in results.items() 
        if "lazy_static" in crate_name and "error" not in result
    )
    
    if success:
        logger.info("\n✅ Enhanced macro extraction working!")
    else:
        logger.warning("\n⚠️ Macro extraction needs more work")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_macro_crates())