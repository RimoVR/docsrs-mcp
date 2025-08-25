"""Main orchestration for crate ingestion following service layer pattern.

This module handles:
- Main ingestion pipeline coordination
- Four-tier fallback system
- Recovery mechanisms
- Per-crate locking
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp

from ..database import (
    get_db_path,
    init_database,
    is_ingestion_complete,
    migrate_add_ingestion_tracking,
    set_ingestion_status,
    store_crate_metadata,
    store_modules,
)
from ..models import IngestionTier
from .cache_manager import evict_cache_if_needed
from .code_examples import extract_code_examples, generate_example_embeddings
from .embedding_manager import cleanup_embedding_model
from .rustdoc_parser import (
    parse_rustdoc_items_streaming,
)
from .signature_extractor import (
    extract_deprecated,
    extract_signature,
    extract_visibility,
)
from .storage_manager import (
    generate_embeddings_streaming,
    store_embeddings_streaming,
)
from .version_resolver import (
    RustdocVersionNotFoundError,
    decompress_content,
    download_rustdoc,
    fetch_crate_info,
    is_stdlib_crate,
    resolve_stdlib_version,
    resolve_version,
)

# Import source extractor for Tier 2 fallback
import sys
from pathlib import Path as PathLib
# Add extractors directory to path for import
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent.parent / "extractors"))
try:
    from source_extractor import CratesIoSourceExtractor
except ImportError:
    logger.warning("CratesIoSourceExtractor not available, Tier 2 will be skipped")
    CratesIoSourceExtractor = None

logger = logging.getLogger(__name__)

# Global per-crate lock registry to prevent duplicate ingestion
_crate_locks: dict[str, asyncio.Lock] = {}


async def get_crate_lock(crate_name: str, version: str) -> asyncio.Lock:
    """Get or create a lock for a specific crate@version.

    Args:
        crate_name: Name of the crate
        version: Version of the crate

    Returns:
        asyncio.Lock: Lock for this crate@version
    """
    key = f"{crate_name}@{version}"
    if key not in _crate_locks:
        _crate_locks[key] = asyncio.Lock()
    return _crate_locks[key]


class IngestionOrchestrator:
    """Service class for orchestrating crate ingestion.

    Following the existing service layer pattern from CrateService and IngestionService.
    """

    def __init__(self):
        """Initialize the orchestrator with dependencies."""
        self.logger = logging.getLogger(__name__)
        self.ingestion_tier: IngestionTier | None = None

    async def create_stdlib_fallback_documentation(
        self, crate_name: str, version: str
    ) -> list[dict[str, Any]]:
        """Create enhanced fallback documentation for stdlib crates.

        Args:
            crate_name: Name of the stdlib crate
            version: Rust version

        Returns:
            List[Dict]: Fallback documentation chunks
        """
        self.logger.info(
            f"Creating comprehensive stdlib fallback documentation for {crate_name}@{version}"
        )

        # Enhanced descriptions for stdlib crates
        stdlib_info = {
            "std": {
                "description": "The Rust standard library providing essential functionality",
                "modules": [
                    ("collections", "Collection types like Vec, HashMap, BTreeMap"),
                    ("io", "I/O operations, readers, writers, and buffering"),
                    ("fs", "Filesystem operations and file handling"),
                    ("net", "TCP/UDP networking"),
                    ("thread", "Thread spawning and synchronization"),
                    ("sync", "Synchronization primitives like Arc, Mutex, RwLock"),
                    ("time", "Time measurement and manipulation"),
                    ("process", "Process spawning and management"),
                    ("env", "Environment variables and program arguments"),
                    ("path", "Path manipulation utilities"),
                ],
            },
            "core": {
                "description": "Core functionality available without heap allocation",
                "modules": [
                    ("mem", "Memory manipulation and management"),
                    ("ptr", "Raw pointer operations"),
                    ("slice", "Slice primitive operations"),
                    ("str", "String slice operations"),
                    ("option", "Optional values"),
                    ("result", "Error handling with Result"),
                    ("iter", "Iterator traits and implementations"),
                    ("marker", "Marker traits like Send and Sync"),
                    ("ops", "Operator traits"),
                    ("fmt", "Formatting and display traits"),
                ],
            },
            "alloc": {
                "description": "Heap allocation and collection types",
                "modules": [
                    ("vec", "Dynamic arrays"),
                    ("string", "UTF-8 strings"),
                    ("boxed", "Heap-allocated values"),
                    ("rc", "Reference-counted pointers"),
                    ("arc", "Atomically reference-counted pointers"),
                    ("collections", "BTreeMap, BTreeSet, LinkedList, VecDeque"),
                ],
            },
        }

        chunks = []
        info = stdlib_info.get(
            crate_name,
            {
                "description": f"Rust {crate_name} library",
                "modules": [],
            },
        )

        # Main crate documentation with example code
        main_doc = info["description"]
        if crate_name == "std":
            main_doc += "\n\n## Example\n```rust\nuse std::collections::HashMap;\nuse std::fs::File;\nuse std::io::prelude::*;\n\nfn main() -> std::io::Result<()> {\n    let mut file = File::create(\"hello.txt\")?;\n    file.write_all(b\"Hello, world!\")?;\n    Ok(())\n}\n```"
        
        chunks.append(
            {
                "item_path": crate_name,
                "item_id": f"{crate_name}:root",
                "item_type": "module",
                "header": f"{crate_name} - Rust standard library",
                "doc": main_doc,
                "parent_id": None,
                "examples": extract_code_examples(main_doc),
            }
        )

        # Add module documentation with examples
        for module_name, module_desc in info["modules"]:
            # Add example code for key modules
            doc_with_example = module_desc
            if module_name == "collections" and crate_name == "std":
                doc_with_example += "\n\n## Example\n```rust\nuse std::collections::HashMap;\n\nlet mut map = HashMap::new();\nmap.insert(\"key\", \"value\");\n```"
            elif module_name == "io" and crate_name == "std":
                doc_with_example += "\n\n## Example\n```rust\nuse std::io::{self, Write};\n\nio::stdout().write_all(b\"Hello, world!\\n\")?;\n```"
            elif module_name == "vec" and crate_name == "alloc":
                doc_with_example += "\n\n## Example\n```rust\nlet mut vec = Vec::new();\nvec.push(1);\nvec.push(2);\n```"
            
            chunks.append(
                {
                    "item_path": f"{crate_name}::{module_name}",
                    "item_id": f"{crate_name}::{module_name}",
                    "item_type": "module",
                    "header": f"{crate_name}::{module_name}",
                    "doc": doc_with_example,
                    "parent_id": f"{crate_name}:root",
                    "examples": extract_code_examples(doc_with_example),
                }
            )

        return chunks

    async def enhance_fallback_schema(
        self, chunk: dict[str, Any], ingestion_tier: IngestionTier
    ) -> dict[str, Any]:
        """Enhance fallback ingestion schema with synthesized metadata.

        Args:
            chunk: Raw chunk from fallback ingestion
            ingestion_tier: The tier of ingestion being used

        Returns:
            Dict: Enhanced chunk with synthesized fields
        """
        # Ensure all required fields exist with sensible defaults
        enhanced = chunk.copy()

        # Synthesize missing fields
        if "item_id" not in enhanced:
            enhanced["item_id"] = enhanced.get("item_path", "unknown")

        if "item_type" not in enhanced:
            # Try to infer from content
            path = enhanced.get("item_path", "").lower()
            if "struct" in path:
                enhanced["item_type"] = "struct"
            elif "trait" in path:
                enhanced["item_type"] = "trait"
            elif "fn" in path or "function" in path:
                enhanced["item_type"] = "function"
            elif "enum" in path:
                enhanced["item_type"] = "enum"
            elif "mod" in path or "::" in path:
                enhanced["item_type"] = "module"
            else:
                enhanced["item_type"] = "unknown"

        if "signature" not in enhanced:
            enhanced["signature"] = None

        if "deprecated" not in enhanced:
            enhanced["deprecated"] = False

        if "parent_id" not in enhanced:
            enhanced["parent_id"] = None

        if "tags" not in enhanced:
            enhanced["tags"] = ""

        if "char_start" not in enhanced:
            enhanced["char_start"] = 0

        # Add ingestion tier metadata
        enhanced["_ingestion_tier"] = ingestion_tier.value

        return enhanced


async def ingest_crate(crate_name: str, version: str | None = None) -> Path:
    """Ingest a crate's documentation and return the database path.

    Main entry point for crate ingestion with four-tier fallback system.

    Args:
        crate_name: Name of the crate to ingest
        version: Optional version (defaults to latest)

    Returns:
        Path: Path to the ingested database
    """
    orchestrator = IngestionOrchestrator()
    ingestion_tier = None

    # Normalize inputs
    crate_name = crate_name.strip()
    version = version.strip() if version else "latest"

    # Special handling for standard library crates
    is_stdlib = is_stdlib_crate(crate_name)

    # Evict cache if needed before ingestion
    await evict_cache_if_needed()

    # Get database path
    db_path = await get_db_path(crate_name, version)

    # Acquire per-crate lock to prevent duplicate ingestion
    lock = await get_crate_lock(crate_name, version)

    async with lock:
        # Check if already ingested
        if await is_ingestion_complete(db_path):
            logger.info(f"Crate {crate_name}@{version} already fully ingested")
            return db_path
        elif db_path.exists():
            # Database exists but ingestion is incomplete - delete and retry
            logger.warning(
                f"Found incomplete ingestion for {crate_name}@{version}, reingesting..."
            )
            try:
                os.remove(db_path)
            except Exception as e:
                logger.error(f"Failed to remove incomplete database: {e}")

        # Initialize database
        await init_database(db_path)

        # Default description - will be updated if we can fetch crate info
        description = f"Rust crate {crate_name}"

        # Run migrations
        await migrate_add_ingestion_tracking(db_path)

        # Try the four-tier ingestion system
        async with aiohttp.ClientSession() as session:
            # Fetch crate info to get description
            if not is_stdlib:
                try:
                    crate_info = await fetch_crate_info(session, crate_name)
                    description = crate_info.get("description", description)
                except Exception as e:
                    logger.warning(f"Could not fetch crate info: {e}")

            # Store crate metadata and get crate_id
            crate_id = await store_crate_metadata(
                db_path, crate_name, version, description
            )

            # Set initial status
            await set_ingestion_status(db_path, crate_id, "started")
            
            # Initialize resolved_version outside try block so it's available in fallback
            resolved_version = version
            
            try:
                # Resolve version
                if is_stdlib:
                    resolved_version = await resolve_stdlib_version(session, version)
                else:
                    resolved_version, rustdoc_url = await resolve_version(
                        session, crate_name, version
                    )

                # Try Tier 1: Rustdoc JSON
                try:
                    await set_ingestion_status(db_path, crate_id, "downloading")

                    if is_stdlib:
                        # For stdlib, we know this will likely fail but try anyway
                        rustdoc_url = (
                            f"https://docs.rs/{crate_name}/latest/{crate_name}.json"
                        )

                    raw_content, used_url = await download_rustdoc(
                        session, crate_name, resolved_version, rustdoc_url
                    )

                    json_content = await decompress_content(raw_content, used_url)

                    ingestion_tier = IngestionTier.RUSTDOC_JSON

                    # Parse and store
                    await set_ingestion_status(db_path, crate_id, "processing")

                    chunks = []
                    modules = None

                    async for item in parse_rustdoc_items_streaming(json_content):
                        if "_modules" in item:
                            modules = item["_modules"]
                        else:
                            # Extract metadata
                            item["signature"] = extract_signature(item)
                            item["deprecated"] = extract_deprecated(item)
                            item["visibility"] = extract_visibility(item)
                            item["examples"] = extract_code_examples(
                                item.get("doc", "")
                            )
                            chunks.append(item)

                    # Store modules if found
                    if modules:
                        await store_modules(db_path, crate_id, modules)

                    # Generate and store embeddings
                    chunk_embedding_pairs = generate_embeddings_streaming(chunks)
                    await store_embeddings_streaming(db_path, chunk_embedding_pairs)

                    # Generate example embeddings
                    await generate_example_embeddings(
                        db_path, crate_name, resolved_version
                    )

                    await set_ingestion_status(
                        db_path,
                        crate_id,
                        "completed",
                        ingestion_tier=ingestion_tier.value,
                    )

                    logger.info(
                        f"Successfully ingested {crate_name}@{resolved_version}"
                    )
                    return db_path

                except RustdocVersionNotFoundError:
                    # Expected for stdlib and older versions - fall back
                    logger.info(
                        f"No rustdoc JSON for {crate_name}@{version}, trying fallback"
                    )

                except Exception as e:
                    logger.warning(f"Rustdoc JSON ingestion failed: {e}")

                # Try Tier 2: Source extraction from CDN
                if CratesIoSourceExtractor and not is_stdlib:
                    try:
                        logger.info(f"Attempting source extraction for {crate_name}@{resolved_version}")
                        await set_ingestion_status(db_path, crate_id, "processing")
                        
                        extractor = CratesIoSourceExtractor(session=session)
                        chunks = await extractor.extract_from_source(
                            crate_name, resolved_version
                        )
                        
                        if chunks and len(chunks) > 0:
                            ingestion_tier = IngestionTier.SOURCE_EXTRACTION
                            
                            # Process chunks - add metadata and examples
                            for chunk in chunks:
                                # Extract examples from documentation
                                chunk["examples"] = extract_code_examples(chunk.get("doc", ""))
                                # Add signature extraction
                                chunk["signature"] = extract_signature(chunk)
                                chunk["deprecated"] = extract_deprecated(chunk)
                                chunk["visibility"] = extract_visibility(chunk)
                            
                            # Generate and store embeddings
                            chunk_embedding_pairs = generate_embeddings_streaming(chunks)
                            await store_embeddings_streaming(db_path, chunk_embedding_pairs)
                            
                            # Generate example embeddings
                            try:
                                await generate_example_embeddings(
                                    db_path, crate_name, resolved_version
                                )
                            except Exception as e:
                                logger.warning(f"Failed to generate example embeddings: {e}")
                            
                            await set_ingestion_status(
                                db_path, crate_id, "completed",
                                ingestion_tier=ingestion_tier.value
                            )
                            
                            logger.info(
                                f"Successfully ingested {crate_name}@{resolved_version} via source extraction"
                            )
                            return db_path
                            
                    except Exception as e:
                        logger.warning(f"Source extraction failed: {e}")

                # Tier 3: Fallback to latest version
                if version != "latest":
                    try:
                        logger.info(f"Falling back to latest version for {crate_name}")
                        latest_path = await ingest_crate(crate_name, "latest")
                        # Copy data from latest version
                        # (simplified - would copy relevant data)
                        return latest_path
                    except Exception as e:
                        logger.warning(f"Latest version fallback failed: {e}")

                # Tier 4: Description-only fallback
                ingestion_tier = IngestionTier.DESCRIPTION_ONLY

                if is_stdlib:
                    chunks = await orchestrator.create_stdlib_fallback_documentation(
                        crate_name, resolved_version
                    )
                else:
                    # Fetch basic crate info
                    crate_info = await fetch_crate_info(session, crate_name)
                    description = crate_info.get(
                        "description", f"Rust crate {crate_name}"
                    )

                    chunks = [
                        {
                            "item_path": "crate",
                            "item_id": "crate",
                            "item_type": "module",
                            "header": crate_name,
                            "doc": description,
                            "parent_id": None,
                        }
                    ]

                # Enhance and store
                for i, chunk in enumerate(chunks):
                    # Extract code examples from documentation
                    chunk["examples"] = extract_code_examples(chunk.get("doc", ""))
                    chunks[i] = await orchestrator.enhance_fallback_schema(
                        chunk, ingestion_tier
                    )

                chunk_embedding_pairs = generate_embeddings_streaming(chunks)
                await store_embeddings_streaming(db_path, chunk_embedding_pairs)

                # Generate example embeddings even in fallback mode
                # This fixes the issue where examples weren't processed in description_only tier
                try:
                    await generate_example_embeddings(
                        db_path, crate_name, resolved_version
                    )
                    logger.info(
                        f"Generated example embeddings for {crate_name}@{resolved_version} in fallback mode"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate example embeddings in fallback mode: {e}"
                    )
                    # Continue without failing the entire ingestion

                await set_ingestion_status(
                    db_path, crate_id, "completed", ingestion_tier=ingestion_tier.value
                )

                logger.info(
                    f"Successfully ingested {crate_name}@{version} with fallback"
                )
                return db_path

            except Exception as e:
                logger.error(f"Ingestion failed for {crate_name}@{version}: {e}")
                await set_ingestion_status(
                    db_path, crate_id, "failed", error_message=str(e)
                )
                raise
            finally:
                # Cleanup resources
                cleanup_embedding_model()


async def recover_incomplete_ingestion(
    crate_name: str, version: str, db_path: Path
) -> Path:
    """Recover from an incomplete ingestion by re-ingesting the crate.

    Args:
        crate_name: Name of the crate
        version: Version of the crate
        db_path: Path to the incomplete database

    Returns:
        Path to the recovered database
    """
    logger.info(
        f"Attempting to recover incomplete ingestion for {crate_name}@{version}"
    )

    # Check if already recovered
    if await is_ingestion_complete(db_path):
        logger.info(f"Ingestion already recovered for {crate_name}@{version}")
        return db_path

    # Remove incomplete database
    try:
        if db_path.exists():
            os.remove(db_path)
            logger.info(f"Removed incomplete database: {db_path}")
    except Exception as e:
        logger.error(f"Failed to remove incomplete database: {e}")
        # Continue anyway - ingest_crate will handle existing file

    # Re-ingest the crate
    logger.info(f"Re-ingesting {crate_name}@{version}")
    return await ingest_crate(crate_name, version)
