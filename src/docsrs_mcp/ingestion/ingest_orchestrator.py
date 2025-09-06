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

import aiofiles

# Import source extractor for Tier 2 fallback
import sys
from pathlib import Path
from pathlib import Path as PathLib
from typing import Any

import aiohttp

from ..cargo import parse_cargo_toml
from ..database import (
    get_db_path,
    init_database,
    is_ingestion_complete,
    migrate_add_ingestion_tracking,
    set_ingestion_status,
    store_crate_dependencies,
    store_crate_metadata,
    store_modules,
)
from ..models import IngestionTier
from ..rustup_detector import RustupDetector
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

# Initialize logger before any usage
logger = logging.getLogger(__name__)

# Add extractors directory to path for import
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent.parent / "extractors"))
try:
    from source_extractor import CratesIoSourceExtractor
except ImportError:
    logger.warning("CratesIoSourceExtractor not available, Tier 2 will be skipped")
    CratesIoSourceExtractor = None

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

        # Enhanced descriptions for stdlib crates with comprehensive coverage
        stdlib_info = {
            "std": {
                "description": "The Rust standard library providing essential functionality for system programming",
                "modules": [
                    ("collections", "Collection types like Vec, HashMap, BTreeMap, VecDeque, LinkedList, BinaryHeap"),
                    ("io", "I/O operations, readers, writers, buffering, stdin/stdout/stderr"),
                    ("fs", "Filesystem operations, File, OpenOptions, Permissions, DirEntry"),
                    ("net", "TCP/UDP networking, TcpListener, TcpStream, UdpSocket, SocketAddr"),
                    ("thread", "Thread spawning, JoinHandle, ThreadId, sleep, yield_now"),
                    ("sync", "Synchronization primitives Arc, Mutex, RwLock, Condvar, Once, Barrier"),
                    ("time", "Time measurement Duration, Instant, SystemTime"),
                    ("process", "Process spawning Command, Child, Output, Stdio"),
                    ("env", "Environment variables, program arguments, current_dir"),
                    ("path", "Path manipulation Path, PathBuf, Component"),
                    ("hash", "Hash traits and implementations, HashMap, HashSet"),
                    ("fmt", "Formatting traits Display, Debug, Write"),
                    ("str", "String manipulation and searching"),
                    ("string", "Owned UTF-8 strings String"),
                    ("vec", "Dynamic arrays Vec"),
                    ("option", "Optional values Option<T>"),
                    ("result", "Error handling Result<T, E>"),
                    ("iter", "Iterators Iterator, IntoIterator, collect"),
                    ("convert", "Type conversions From, Into, TryFrom, TryInto"),
                    ("ops", "Operator overloading Add, Sub, Mul, Div, Index, Deref"),
                    ("cmp", "Comparison traits Ord, Eq, PartialEq, PartialOrd"),
                    ("clone", "Cloning Clone trait"),
                    ("default", "Default values Default trait"),
                    ("marker", "Marker traits Send, Sync, Copy, Sized"),
                    ("mem", "Memory utilities size_of, align_of, replace, swap"),
                    ("ptr", "Raw pointer operations NonNull, unique_ptr"),
                    ("slice", "Slice operations and methods"),
                    ("any", "Dynamic typing Any trait"),
                    ("panic", "Panic handling and recovery"),
                    ("error", "Error trait and error handling"),
                    ("ffi", "Foreign function interface CString, CStr, OsString"),
                    ("os", "OS-specific functionality"),
                ],
            },
            "core": {
                "description": "Core functionality available in no_std environments without heap allocation",
                "modules": [
                    ("mem", "Memory manipulation size_of, align_of, replace, swap, forget"),
                    ("ptr", "Raw pointer operations NonNull, read, write, offset"),
                    ("slice", "Slice primitive operations from_raw_parts, len, as_ptr"),
                    ("str", "String slice operations from_utf8, chars, bytes"),
                    ("option", "Optional values Option<T>, Some, None, map, unwrap"),
                    ("result", "Error handling Result<T, E>, Ok, Err, map, and_then"),
                    ("iter", "Iterator traits Iterator, IntoIterator, collect, map, filter"),
                    ("marker", "Marker traits Send, Sync, Copy, Sized, PhantomData"),
                    ("ops", "Operator traits Add, Sub, Mul, Div, Index, Deref, Drop"),
                    ("fmt", "Formatting Display, Debug, Write, Arguments"),
                    ("cmp", "Comparison Ord, Eq, PartialEq, PartialOrd, Ordering"),
                    ("convert", "Type conversions From, Into, TryFrom, TryInto, AsRef"),
                    ("clone", "Cloning Clone trait and implementations"),
                    ("default", "Default values Default trait"),
                    ("hash", "Hashing Hash, Hasher traits"),
                    ("any", "Dynamic typing Any, TypeId"),
                    ("panic", "Panic handling PanicInfo"),
                    ("num", "Numeric types and operations"),
                    ("char", "Character manipulation and properties"),
                    ("primitive", "Primitive type documentation"),
                    ("cell", "Interior mutability Cell, RefCell"),
                    ("pin", "Pinned pointers Pin"),
                    ("task", "Async task abstractions Context, Poll, Waker"),
                    ("future", "Future trait and async operations"),
                    ("array", "Array utilities and implementations"),
                    ("ascii", "ASCII character operations"),
                ],
            },
            "alloc": {
                "description": "Heap allocation and collection types for no_std with allocator",
                "modules": [
                    ("vec", "Dynamic arrays Vec<T>, push, pop, extend, drain"),
                    ("string", "UTF-8 strings String, push_str, format!"),
                    ("boxed", "Heap-allocated values Box<T>, into_raw, from_raw"),
                    ("rc", "Reference-counted pointers Rc<T>, clone, strong_count"),
                    ("arc", "Atomically reference-counted Arc<T>, clone, strong_count"),
                    ("collections", "BTreeMap, BTreeSet, LinkedList, VecDeque, BinaryHeap"),
                    ("slice", "Slice utilities to_vec, sort, binary_search"),
                    ("str", "String slice utilities to_owned, split, replace"),
                    ("borrow", "Borrowing Cow, ToOwned, Borrow"),
                    ("fmt", "Formatting format! macro and utilities"),
                    ("sync", "Arc and synchronization primitives"),
                    ("task", "Task abstractions for async"),
                ],
            },
            "proc_macro": {
                "description": "Procedural macro support for compile-time code generation",
                "modules": [
                    ("TokenStream", "Token stream manipulation"),
                    ("TokenTree", "Token tree operations"),
                    ("Span", "Source code span tracking"),
                    ("Diagnostic", "Compilation error reporting"),
                    ("Group", "Token grouping"),
                    ("Ident", "Identifier handling"),
                    ("Literal", "Literal value handling"),
                    ("Punct", "Punctuation token handling"),
                ],
            },
            "test": {
                "description": "Testing support for unit and integration tests",
                "modules": [
                    ("test", "Test harness and benchmarking"),
                    ("bench", "Benchmark support"),
                    ("assert", "Assertion macros"),
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

        # Main crate documentation with example code and setup guidance
        main_doc = info["description"]

        # Add comprehensive setup instructions for complete documentation
        main_doc += "\n\n## ⚠️ Limited Documentation Available"
        main_doc += "\n\nThis is fallback documentation with basic coverage. For complete standard library documentation:"
        main_doc += "\n\n### Get Complete Documentation"
        main_doc += "\n1. Install the rust-docs-json component:"
        main_doc += "\n   ```bash"
        main_doc += "\n   rustup component add --toolchain nightly rust-docs-json"
        main_doc += "\n   ```"
        main_doc += "\n2. Restart your MCP server to enable full stdlib support"
        main_doc += "\n3. Access thousands of documented stdlib items instead of this limited fallback"
        main_doc += "\n\n### Alternative: Use docs.rs"
        main_doc += f"\nFor web browsing, visit: https://doc.rust-lang.org/{crate_name}/"

        if crate_name == "std":
            main_doc += "\n\n## Example Usage\n```rust\nuse std::collections::HashMap;\nuse std::fs::File;\nuse std::io::prelude::*;\n\nfn main() -> std::io::Result<()> {\n    let mut file = File::create(\"hello.txt\")?;\n    file.write_all(b\"Hello, world!\")?;\n    \n    let mut map = HashMap::new();\n    map.insert(\"key\", \"value\");\n    println!(\"Value: {:?}\", map.get(\"key\"));\n    Ok(())\n}\n```"
        elif crate_name == "core":
            main_doc += "\n\n## Example Usage\n```rust\n#![no_std]\nuse core::{option::Option, result::Result};\n\nfn safe_divide(a: i32, b: i32) -> Result<i32, &'static str> {\n    if b == 0 {\n        Err(\"Division by zero\")\n    } else {\n        Ok(a / b)\n    }\n}\n```"
        elif crate_name == "alloc":
            main_doc += "\n\n## Example Usage\n```rust\n#![no_std]\nextern crate alloc;\nuse alloc::{vec::Vec, string::String};\n\nfn main() {\n    let mut vec = Vec::new();\n    vec.push(\"Hello\");\n    vec.push(\"World\");\n    \n    let s = String::from(\"Heap allocated string\");\n}\n```"

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

            # Fetch and store crate dependencies from Cargo.toml
            if not is_stdlib:
                try:
                    # Download Cargo.toml for this crate
                    cargo_url = f"https://docs.rs/crate/{crate_name}/{version if version != 'latest' else ''}/download"

                    async with session.get(cargo_url) as response:
                        if response.status == 200:
                            import io
                            import tarfile
                            import tempfile

                            # Download crate source
                            crate_data = await response.read()

                            # Extract Cargo.toml from the tarball
                            with tarfile.open(fileobj=io.BytesIO(crate_data), mode='r:gz') as tar:
                                # Find Cargo.toml in the archive
                                cargo_toml_member = None
                                for member in tar.getmembers():
                                    if member.name.endswith('/Cargo.toml') and member.name.count('/') == 1:
                                        cargo_toml_member = member
                                        break

                                if cargo_toml_member:
                                    # Extract and parse Cargo.toml
                                    cargo_toml_file = tar.extractfile(cargo_toml_member)
                                    if cargo_toml_file:
                                        # Write to temp file and parse
                                        with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as tmp:
                                            tmp.write(cargo_toml_file.read())
                                            tmp_path = Path(tmp.name)

                                        try:
                                            # Parse dependencies
                                            cargo_data = parse_cargo_toml(tmp_path)
                                            dependencies = []

                                            # Extract crate names and versions
                                            for crate_spec in cargo_data.get("crates", []):
                                                if "@" in crate_spec:
                                                    name, ver = crate_spec.split("@", 1)
                                                    dependencies.append({"name": name, "version": ver})
                                                else:
                                                    dependencies.append({"name": crate_spec, "version": "latest"})

                                            # Store dependencies
                                            if dependencies:
                                                await store_crate_dependencies(
                                                    db_path, crate_id, crate_name, dependencies
                                                )
                                                logger.info(f"Stored {len(dependencies)} dependencies for {crate_name}")
                                        finally:
                                            # Clean up temp file
                                            tmp_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.warning(f"Could not fetch or parse Cargo.toml for {crate_name}: {e}")
                    # Continue without failing ingestion

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

                # For stdlib crates, try RUST_LANG_STDLIB tier first
                if is_stdlib:
                    try:
                        await set_ingestion_status(db_path, crate_id, "downloading")

                        # Check for local stdlib JSON using rustup detector
                        rustup_detector = RustupDetector()
                        local_json_path = rustup_detector.get_stdlib_json_path(crate_name, "nightly")

                        if local_json_path and local_json_path.exists():
                            logger.info(f"Found local stdlib JSON for {crate_name}: {local_json_path}")
                            await set_ingestion_status(db_path, crate_id, "processing")

                            # Read and process local stdlib JSON
                            async with aiofiles.open(local_json_path, encoding='utf-8') as f:
                                json_content = await f.read()

                            ingestion_tier = IngestionTier.RUST_LANG_STDLIB

                            # Parse and store
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
                                f"Successfully ingested {crate_name}@{resolved_version} from local stdlib JSON"
                            )
                            return db_path
                        else:
                            logger.info(f"Local stdlib JSON not found for {crate_name}, will fall back to remote/fallback")

                    except Exception as e:
                        logger.warning(f"Local stdlib ingestion failed for {crate_name}: {e}, falling back")

                # Try Tier 1: Rustdoc JSON (for non-stdlib or when stdlib local failed)
                try:
                    await set_ingestion_status(db_path, crate_id, "downloading")

                    if is_stdlib:
                        # For stdlib, try docs.rs URL (expected to fail but worth trying)
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
