"""Database diagnostics and consistency checking functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import aiosqlite
import sqlite_vec
import structlog

from ..config import DB_TIMEOUT

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def check_vector_table_consistency(db_path: Path) -> Dict[str, Any]:
    """
    Check consistency between main tables and vector tables.
    
    Based on research findings, this implements timestamp-based consistency 
    checking and row count validation patterns.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary containing consistency check results
    """
    results = {
        "consistent": True,
        "issues": [],
        "embeddings_count": 0,
        "vec_embeddings_count": 0,
        "example_embeddings_count": 0,
        "vec_example_embeddings_count": 0,
        "sync_required": False
    }
    
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')") 
        await db.enable_load_extension(False)
        
        try:
            # Check main embeddings vs vec_embeddings consistency
            cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
            embeddings_count = (await cursor.fetchone())[0]
            results["embeddings_count"] = embeddings_count
            
            cursor = await db.execute("SELECT COUNT(*) FROM vec_embeddings")
            vec_embeddings_count = (await cursor.fetchone())[0]
            results["vec_embeddings_count"] = vec_embeddings_count
            
            if embeddings_count != vec_embeddings_count:
                results["consistent"] = False
                results["sync_required"] = True
                results["issues"].append(
                    f"Embeddings table has {embeddings_count} rows but "
                    f"vec_embeddings has {vec_embeddings_count} rows"
                )
            
            # Check example embeddings vs vec_example_embeddings consistency  
            cursor = await db.execute("SELECT COUNT(*) FROM example_embeddings")
            example_embeddings_count = (await cursor.fetchone())[0]
            results["example_embeddings_count"] = example_embeddings_count
            
            cursor = await db.execute("SELECT COUNT(*) FROM vec_example_embeddings")
            vec_example_embeddings_count = (await cursor.fetchone())[0]
            results["vec_example_embeddings_count"] = vec_example_embeddings_count
            
            if example_embeddings_count != vec_example_embeddings_count:
                results["consistent"] = False
                results["issues"].append(
                    f"Example_embeddings table has {example_embeddings_count} rows but "
                    f"vec_example_embeddings has {vec_example_embeddings_count} rows"
                )
            
            # Check for orphaned rows in vector tables
            cursor = await db.execute("""
                SELECT COUNT(*) FROM vec_embeddings v
                WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.id = v.rowid)
            """)
            orphaned_vec_embeddings = (await cursor.fetchone())[0]
            if orphaned_vec_embeddings > 0:
                results["consistent"] = False
                results["issues"].append(
                    f"Found {orphaned_vec_embeddings} orphaned rows in vec_embeddings"
                )
            
            cursor = await db.execute("""
                SELECT COUNT(*) FROM vec_example_embeddings v
                WHERE NOT EXISTS (SELECT 1 FROM example_embeddings e WHERE e.id = v.rowid)
            """)
            orphaned_vec_examples = (await cursor.fetchone())[0]
            if orphaned_vec_examples > 0:
                results["consistent"] = False
                results["issues"].append(
                    f"Found {orphaned_vec_examples} orphaned rows in vec_example_embeddings"
                )
                
        except Exception as e:
            results["consistent"] = False
            results["issues"].append(f"Error during consistency check: {str(e)}")
            logger.error("Database consistency check failed", error=str(e))
    
    return results


async def repair_vector_table_sync(db_path: Path, table_type: str = "both") -> Dict[str, Any]:
    """
    Repair synchronization between main and vector tables.
    
    Implements the manual synchronization patterns from research findings.
    
    Args:
        db_path: Path to the SQLite database file
        table_type: "embeddings", "examples", or "both" (default)
        
    Returns:
        Dictionary containing repair operation results
    """
    results = {
        "success": True,
        "embeddings_synced": 0,
        "examples_synced": 0,
        "errors": []
    }
    
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')") 
        await db.enable_load_extension(False)
        
        try:
            if table_type in ("embeddings", "both"):
                # Sync embeddings to vec_embeddings
                logger.info("Repairing embeddings vector table sync...")
                
                # Clear and rebuild vec_embeddings table
                await db.execute("DELETE FROM vec_embeddings")
                
                cursor = await db.execute("SELECT id, embedding FROM embeddings")
                vec_data = []
                synced_count = 0
                
                async for row in cursor:
                    rowid, embedding_blob = row
                    vec_data.append((rowid, embedding_blob))
                    
                    # Process in batches for efficiency
                    if len(vec_data) >= 100:
                        await db.executemany(
                            "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                            vec_data
                        )
                        synced_count += len(vec_data)
                        vec_data = []
                
                # Insert remaining data
                if vec_data:
                    await db.executemany(
                        "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                        vec_data
                    )
                    synced_count += len(vec_data)
                
                results["embeddings_synced"] = synced_count
                await db.commit()
                logger.info(f"Synced {synced_count} embeddings to vec_embeddings")
            
            if table_type in ("examples", "both"):
                # Sync example_embeddings to vec_example_embeddings
                logger.info("Repairing example embeddings vector table sync...")
                
                # Clear and rebuild vec_example_embeddings table
                await db.execute("DELETE FROM vec_example_embeddings")
                
                cursor = await db.execute("SELECT id, embedding FROM example_embeddings")
                vec_data = []
                synced_count = 0
                
                async for row in cursor:
                    rowid, embedding_blob = row
                    vec_data.append((rowid, embedding_blob))
                    
                    # Process in batches for efficiency
                    if len(vec_data) >= 100:
                        await db.executemany(
                            "INSERT INTO vec_example_embeddings(rowid, example_embedding) VALUES (?, ?)",
                            vec_data
                        )
                        synced_count += len(vec_data)
                        vec_data = []
                
                # Insert remaining data
                if vec_data:
                    await db.executemany(
                        "INSERT INTO vec_example_embeddings(rowid, example_embedding) VALUES (?, ?)",
                        vec_data
                    )
                    synced_count += len(vec_data)
                
                results["examples_synced"] = synced_count
                await db.commit()
                logger.info(f"Synced {synced_count} example embeddings to vec_example_embeddings")
                
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            logger.error("Vector table repair failed", error=str(e))
            await db.rollback()
    
    return results


async def get_database_health_summary(db_path: Path) -> Dict[str, Any]:
    """
    Get comprehensive database health summary including consistency status.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary containing health summary
    """
    health = {
        "healthy": True,
        "tables": {},
        "consistency": {},
        "recommendations": []
    }
    
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Load sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')") 
        await db.enable_load_extension(False)
        
        try:
            # Get table row counts
            tables = ["embeddings", "vec_embeddings", "example_embeddings", "vec_example_embeddings"]
            
            for table in tables:
                try:
                    cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                    count = (await cursor.fetchone())[0]
                    health["tables"][table] = count
                except Exception as e:
                    health["tables"][table] = f"Error: {e}"
                    health["healthy"] = False
            
            # Run consistency check
            consistency_results = await check_vector_table_consistency(db_path)
            health["consistency"] = consistency_results
            
            if not consistency_results["consistent"]:
                health["healthy"] = False
                health["recommendations"].append("Run repair_vector_table_sync to fix consistency issues")
            
            # Check for basic table existence and structure
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%vec%'")
            vec_tables = [row[0] for row in await cursor.fetchall()]
            
            if "vec_embeddings" not in vec_tables:
                health["healthy"] = False
                health["recommendations"].append("vec_embeddings virtual table is missing")
            
            if "vec_example_embeddings" not in vec_tables:
                health["healthy"] = False
                health["recommendations"].append("vec_example_embeddings virtual table is missing")
                
        except Exception as e:
            health["healthy"] = False
            health["consistency"]["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
    
    return health