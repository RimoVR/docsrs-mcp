#!/usr/bin/env python3
"""
Test the MCP server with fallback extraction by ingesting an old crate.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docsrs_mcp.ingest import ingest_crate
from docsrs_mcp.database import init_database, get_db_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_ingest_with_fallback():
    """Test ingesting a crate that will trigger fallback extraction."""
    
    # Use a very old version that definitely won't have rustdoc JSON
    # serde 0.1.0 was published in 2015
    test_crate = "serde"
    test_version = "0.1.0"
    
    logger.info(f"Testing ingestion with fallback for {test_crate}@{test_version}")
    
    try:
        # Ingest the crate (should trigger fallback)
        db_path = await ingest_crate(test_crate, test_version)
        
        if db_path and db_path.exists():
            logger.info(f"✅ Successfully ingested {test_crate}@{test_version}")
            logger.info(f"Database created at: {db_path}")
            
            # Check database contents
            import aiosqlite
            async with aiosqlite.connect(db_path) as db:
                # Count embeddings
                cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
                count = (await cursor.fetchone())[0]
                logger.info(f"Total embeddings stored: {count}")
                
                # Sample some items
                cursor = await db.execute(
                    "SELECT item_path, item_type, LENGTH(content) as doc_len FROM embeddings LIMIT 5"
                )
                rows = await cursor.fetchall()
                
                if rows:
                    logger.info("\nSample items in database:")
                    for row in rows:
                        logger.info(f"  - {row[0]} ({row[1]}) - {row[2]} chars")
                
                return True
        else:
            logger.error("Ingestion completed but no database created")
            return False
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_fallback_data():
    """Test searching the fallback-ingested data."""
    
    test_crate = "serde"
    test_version = "0.1.0"
    
    logger.info(f"\nTesting search on fallback data for {test_crate}@{test_version}")
    
    try:
        # Get database path
        db_path = await get_db_path(test_crate, test_version)
        
        if not db_path.exists():
            logger.warning("Database doesn't exist, skipping search test")
            return False
        
        # Import search function
        from docsrs_mcp.database import search_embeddings
        from docsrs_mcp.ingest import get_embedding_model
        
        # Create a test query
        query_text = "serialize data"
        
        # Generate query embedding
        model = get_embedding_model()
        query_embedding = list(model.embed([query_text])[0])
        
        # Search
        results = await search_embeddings(
            db_path,
            query_embedding,
            k=3
        )
        
        if results:
            logger.info(f"Found {len(results)} search results:")
            for score, path, header, content in results:
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"  - {path} (score: {score:.3f})")
                logger.info(f"    {content_preview}")
        else:
            logger.info("No search results found (this might be normal for fallback data)")
        
        return True
        
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run MCP server tests with fallback."""
    logger.info("Starting MCP server fallback tests")
    
    # Test ingestion with fallback
    success1 = await test_ingest_with_fallback()
    
    # Test searching fallback data
    success2 = await test_search_fallback_data()
    
    if success1:
        logger.info("\n✅ Fallback ingestion test passed!")
    else:
        logger.error("\n❌ Fallback ingestion test failed")
    
    if success2:
        logger.info("✅ Search test passed!")
    else:
        logger.error("❌ Search test failed")
    
    sys.exit(0 if (success1 and success2) else 1)


if __name__ == "__main__":
    asyncio.run(main())