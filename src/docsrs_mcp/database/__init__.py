"""Database operations with SQLite and sqlite-vec.

This is a facade module that re-exports all public functions from the
modularized database package to maintain backward compatibility.
"""

from __future__ import annotations

# Re-export config values that were previously imported from database.py
from ..config import CACHE_DIR, EMBEDDING_DIM

# Re-export from connection module
from .connection import (
    DB_TIMEOUT,
    RetryableTransaction,
    execute_with_retry,
    get_db_path,
    load_sqlite_vec_extension,
    performance_timer,
    prepared_statements,
)

# Re-export from ingestion module
from .ingestion import (
    compute_item_hash,
    detect_stalled_ingestions,
    find_incomplete_ingestions,
    get_ingestion_status,
    is_ingestion_complete,
    reset_ingestion_status,
    set_ingestion_status,
)

# Re-export from retrieval module
from .retrieval import (
    get_all_item_paths,
    get_all_items_for_version,
    get_cross_references,
    get_discovered_reexports,
    get_item_signatures_batch,
    get_module_by_path,
    get_module_items,
    get_module_tree,
    query_associated_items,
    query_generic_constraints,
    query_method_signatures,
    query_trait_implementations,
)

# Re-export from schema module
from .schema import (
    init_database,
    migrate_add_generics_metadata,
    migrate_add_ingestion_tracking,
    migrate_database_duplicates,
    migrate_reexports_for_crossrefs,
)

# Re-export from search module
from .search import (
    _apply_mmr_diversification,
    cross_crate_search,
    get_see_also_suggestions,
    search_embeddings,
    search_example_embeddings,
)

# Re-export from storage module
from .storage import (
    store_crate_metadata,
    store_crate_dependencies,
    store_modules,
    store_reexports,
)

# Define __all__ for explicit public API
__all__ = [
    # Connection module
    "DB_TIMEOUT",
    "RetryableTransaction",
    "execute_with_retry",
    "get_db_path",
    "load_sqlite_vec_extension",
    "performance_timer",
    "prepared_statements",
    # Schema module
    "init_database",
    "migrate_add_generics_metadata",
    "migrate_add_ingestion_tracking",
    "migrate_database_duplicates",
    "migrate_reexports_for_crossrefs",
    # Storage module
    "store_crate_metadata",
    "store_crate_dependencies",
    "store_modules",
    "store_reexports",
    # Search module
    "_apply_mmr_diversification",
    "cross_crate_search",
    "get_see_also_suggestions",
    "search_embeddings",
    "search_example_embeddings",
    # Retrieval module
    "get_all_item_paths",
    "get_all_items_for_version",
    "get_cross_references",
    "get_discovered_reexports",
    "get_item_signatures_batch",
    "get_module_by_path",
    "get_module_items",
    "get_module_tree",
    "query_trait_implementations",
    "query_method_signatures",
    "query_associated_items",
    "query_generic_constraints",
    # Ingestion module
    "compute_item_hash",
    "detect_stalled_ingestions",
    "find_incomplete_ingestions",
    "get_ingestion_status",
    "is_ingestion_complete",
    "reset_ingestion_status",
    "set_ingestion_status",
    # Config values
    "CACHE_DIR",
    "EMBEDDING_DIM",
]
