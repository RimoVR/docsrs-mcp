"""Migration runner for database schema updates."""

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import structlog

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


async def create_migrations_table(db_path: Path) -> None:
    """Create migrations tracking table if it doesn't exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def is_migration_applied(db_path: Path, migration_name: str) -> bool:
    """Check if a migration has already been applied."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = ?",
            (migration_name,),
        )
        return (await cursor.fetchone())[0] > 0


async def record_migration(db_path: Path, migration_name: str) -> None:
    """Record that a migration has been applied."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES (?)",
            (migration_name,),
        )
        await db.commit()


async def run_migrations(db_path: Path) -> None:
    """Run all pending database migrations in order."""
    # Ensure migrations table exists
    await create_migrations_table(db_path)

    # Get all migration files
    migrations_dir = Path(__file__).parent
    migration_files = sorted(
        f
        for f in migrations_dir.glob("*.py")
        if f.name not in ("__init__.py", "migration_runner.py")
    )

    for migration_file in migration_files:
        migration_name = migration_file.stem

        # Skip if already applied
        if await is_migration_applied(db_path, migration_name):
            logger.debug(f"Skipping already applied migration: {migration_name}")
            continue

        try:
            # Import and run the migration
            logger.info(f"Running migration: {migration_name}")
            module_name = f"docsrs_mcp.database.migrations.{migration_name}"
            migration_module = importlib.import_module(module_name)

            # Run the upgrade function
            if hasattr(migration_module, "upgrade"):
                await migration_module.upgrade(db_path)
                await record_migration(db_path, migration_name)
                logger.info(f"Successfully applied migration: {migration_name}")
            else:
                logger.warning(f"Migration {migration_name} has no upgrade function")

        except Exception as e:
            logger.error(f"Failed to apply migration {migration_name}: {e}")
            raise


async def rollback_migration(db_path: Path, migration_name: str) -> None:
    """Rollback a specific migration."""
    try:
        # Import the migration module
        module_name = f"docsrs_mcp.database.migrations.{migration_name}"
        migration_module = importlib.import_module(module_name)

        # Run the downgrade function if it exists
        if hasattr(migration_module, "downgrade"):
            logger.info(f"Rolling back migration: {migration_name}")
            await migration_module.downgrade(db_path)

            # Remove from migrations table
            async with aiosqlite.connect(db_path) as db:
                await db.execute(
                    "DELETE FROM schema_migrations WHERE migration_name = ?",
                    (migration_name,),
                )
                await db.commit()

            logger.info(f"Successfully rolled back migration: {migration_name}")
        else:
            logger.warning(f"Migration {migration_name} has no downgrade function")

    except Exception as e:
        logger.error(f"Failed to rollback migration {migration_name}: {e}")
        raise
