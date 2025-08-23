"""Ingestion pipeline for Rust crate documentation.

This module now delegates to the focused modules in the ingestion package.
All functionality has been preserved through re-exports for backward compatibility.
"""

# Re-export all public functions and classes from the ingestion package
# Re-export constants for backward compatibility
from .ingestion import (
    CACHE_DIR,
    COMPRESSED_RUSTDOC_SUPPORTED,
    EMBEDDING_MODEL_NAME,
    HTTP_TIMEOUT,
    MAX_CACHE_SIZE,
    RUST_VERSION_MANIFEST_URL,
    STDLIB_CRATES,
    # Main orchestration
    IngestionOrchestrator,
    # Code examples
    batch_examples,
    # Rustdoc parsing
    build_module_hierarchy,
    # Cache management
    calculate_cache_size,
    calculate_example_hash,
    # Storage management
    clean_existing_embeddings,
    # Embedding management
    cleanup_embedding_model,
    # Version resolution
    construct_stdlib_url,
    decompress_content,
    download_rustdoc,
    evict_cache_if_needed,
    extract_code_examples,
    # Signature extraction
    extract_deprecated,
    extract_generics,
    extract_signature,
    extract_type_name,
    extract_visibility,
    extract_where_clause,
    extract_where_predicate,
    fetch_crate_info,
    fetch_current_stable_version,
    format_example_for_embedding,
    format_signature,
    generate_embeddings,
    generate_embeddings_streaming,
    generate_example_embeddings,
    get_crate_lock,
    get_embedding_model,
    get_warmup_status,
    ingest_crate,
    is_stdlib_crate,
    normalize_code,
    parse_rustdoc_items,
    parse_rustdoc_items_streaming,
    recover_incomplete_ingestion,
    resolve_parent_id,
    resolve_stdlib_version,
    resolve_version,
    resolve_version_from_crate_info,
    store_embeddings,
    store_embeddings_streaming,
    warmup_embedding_model,
)

# This allows the module to be used exactly as before
__all__ = [
    # Main orchestration
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
