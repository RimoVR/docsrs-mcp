"""Ingestion pipeline modules for Rust crate documentation.

This package splits the monolithic ingest.py into focused modules under 500 LOC each,
while maintaining backward compatibility through re-exports.
"""

# Import all public functions and classes from sub-modules
# Re-export constants and types for backward compatibility
from .cache_manager import (
    CACHE_DIR,
    MAX_CACHE_SIZE,
    calculate_cache_size,
    evict_cache_if_needed,
)
from .code_examples import (
    batch_examples,
    calculate_example_hash,
    extract_code_examples,
    format_example_for_embedding,
    generate_example_embeddings,
    normalize_code,
)
from .embedding_manager import (
    EMBEDDING_MODEL_NAME,
    cleanup_embedding_model,
    get_embedding_model,
    get_warmup_status,
    warmup_embedding_model,
)
from .ingest_orchestrator import (
    IngestionOrchestrator,
    get_crate_lock,
    ingest_crate,
    recover_incomplete_ingestion,
)
from .rustdoc_parser import (
    build_module_hierarchy,
    parse_rustdoc_items,
    parse_rustdoc_items_streaming,
    resolve_parent_id,
)
from .signature_extractor import (
    extract_deprecated,
    extract_generics,
    extract_signature,
    extract_type_name,
    extract_visibility,
    extract_where_clause,
    extract_where_predicate,
    format_signature,
)
from .storage_manager import (
    clean_existing_embeddings,
    generate_embeddings,
    generate_embeddings_streaming,
    store_embeddings,
    store_embeddings_streaming,
)
from .version_resolver import (
    COMPRESSED_RUSTDOC_SUPPORTED,
    HTTP_TIMEOUT,
    RUST_VERSION_MANIFEST_URL,
    STDLIB_CRATES,
    construct_stdlib_url,
    decompress_content,
    download_rustdoc,
    fetch_crate_info,
    fetch_current_stable_version,
    is_stdlib_crate,
    resolve_stdlib_version,
    resolve_version,
    resolve_version_from_crate_info,
)

__all__ = [
    # Orchestrator
    "IngestionOrchestrator",
    "ingest_crate",
    "recover_incomplete_ingestion",
    "get_crate_lock",
    # Embedding management
    "get_embedding_model",
    "cleanup_embedding_model",
    "warmup_embedding_model",
    "get_warmup_status",
    # Version resolution
    "is_stdlib_crate",
    "fetch_current_stable_version",
    "resolve_stdlib_version",
    "resolve_version",
    "fetch_crate_info",
    "resolve_version_from_crate_info",
    "download_rustdoc",
    "decompress_content",
    "construct_stdlib_url",
    # Rustdoc parsing
    "parse_rustdoc_items_streaming",
    "parse_rustdoc_items",
    "build_module_hierarchy",
    "resolve_parent_id",
    # Signature extraction
    "format_signature",
    "extract_signature",
    "extract_type_name",
    "extract_visibility",
    "extract_generics",
    "extract_deprecated",
    "extract_where_clause",
    "extract_where_predicate",
    # Code examples
    "extract_code_examples",
    "normalize_code",
    "calculate_example_hash",
    "batch_examples",
    "format_example_for_embedding",
    "generate_example_embeddings",
    # Storage management
    "generate_embeddings_streaming",
    "generate_embeddings",
    "store_embeddings_streaming",
    "store_embeddings",
    "clean_existing_embeddings",
    # Cache management
    "calculate_cache_size",
    "evict_cache_if_needed",
    # Constants
    "STDLIB_CRATES",
    "RUST_VERSION_MANIFEST_URL",
    "HTTP_TIMEOUT",
    "COMPRESSED_RUSTDOC_SUPPORTED",
    "CACHE_DIR",
    "MAX_CACHE_SIZE",
    "EMBEDDING_MODEL_NAME",
]
