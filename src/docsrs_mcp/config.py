"""Simple configuration for docsrs-mcp."""

import os
import warnings
from pathlib import Path

# Cache configuration
CACHE_DIR = Path(os.getenv("DOCSRS_CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = os.getenv("DOCSRS_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # for bge-small-en-v1.5

# Database configuration
DB_TIMEOUT = float(os.getenv("DOCSRS_DB_TIMEOUT", "30.0"))

# HTTP configuration
HTTP_TIMEOUT = float(os.getenv("DOCSRS_HTTP_TIMEOUT", "120.0"))
MAX_DOWNLOAD_SIZE = int(
    os.getenv("DOCSRS_MAX_DOWNLOAD_SIZE", str(30 * 1024 * 1024))
)  # 30MB
MAX_DECOMPRESSED_SIZE = int(
    os.getenv("DOCSRS_MAX_DECOMPRESSED_SIZE", str(100 * 1024 * 1024))
)  # 100MB

# Search configuration
DEFAULT_K = int(os.getenv("DOCSRS_DEFAULT_K", "5"))

# Ranking configuration
RANKING_VECTOR_WEIGHT = float(os.getenv("DOCSRS_RANKING_VECTOR_WEIGHT", "0.6"))
RANKING_TYPE_WEIGHT = float(os.getenv("DOCSRS_RANKING_TYPE_WEIGHT", "0.15"))
RANKING_QUALITY_WEIGHT = float(os.getenv("DOCSRS_RANKING_QUALITY_WEIGHT", "0.1"))
RANKING_EXAMPLES_WEIGHT = float(os.getenv("DOCSRS_RANKING_EXAMPLES_WEIGHT", "0.15"))

# MMR Diversification configuration
RANKING_DIVERSITY_LAMBDA = float(os.getenv("DOCSRS_RANKING_DIVERSITY_LAMBDA", "0.6"))
RANKING_DIVERSITY_WEIGHT = float(os.getenv("DOCSRS_RANKING_DIVERSITY_WEIGHT", "0.1"))
MODULE_DIVERSITY_WEIGHT = float(os.getenv("DOCSRS_MODULE_DIVERSITY_WEIGHT", "0.15"))

# Validate that ranking weights sum to approximately 1.0
_weight_sum = (
    RANKING_VECTOR_WEIGHT
    + RANKING_TYPE_WEIGHT
    + RANKING_QUALITY_WEIGHT
    + RANKING_EXAMPLES_WEIGHT
)
if not 0.99 <= _weight_sum <= 1.01:
    warnings.warn(
        f"Ranking weights sum to {_weight_sum:.2f} instead of 1.0. "
        f"Weights will be normalized during ranking.",
        stacklevel=2,
    )

# Type-specific weights for ranking
TYPE_WEIGHTS = {
    "function": float(os.getenv("DOCSRS_TYPE_WEIGHT_FUNCTION", "1.2")),
    "trait": float(os.getenv("DOCSRS_TYPE_WEIGHT_TRAIT", "1.15")),
    "struct": float(os.getenv("DOCSRS_TYPE_WEIGHT_STRUCT", "1.1")),
    "module": float(os.getenv("DOCSRS_TYPE_WEIGHT_MODULE", "0.9")),
    "enum": float(os.getenv("DOCSRS_TYPE_WEIGHT_ENUM", "1.0")),
    "type": float(os.getenv("DOCSRS_TYPE_WEIGHT_TYPE", "1.0")),
    "trait_impl": float(os.getenv("DOCSRS_TYPE_WEIGHT_TRAIT_IMPL", "1.1")),
}

# Fuzzy matching configuration
FUZZY_WEIGHTS = {
    "token_set_ratio": float(os.getenv("DOCSRS_FUZZY_TOKEN_SET_WEIGHT", "0.4")),
    "token_sort_ratio": float(os.getenv("DOCSRS_FUZZY_TOKEN_SORT_WEIGHT", "0.3")),
    "partial_ratio": float(os.getenv("DOCSRS_FUZZY_PARTIAL_WEIGHT", "0.3")),
    "path_component_bonus": float(os.getenv("DOCSRS_FUZZY_PATH_BONUS", "0.15")),
    "partial_component_bonus": float(os.getenv("DOCSRS_FUZZY_PARTIAL_BONUS", "0.08")),
}

# Caching configuration
CACHE_SIZE = int(os.getenv("DOCSRS_CACHE_SIZE", "1000"))
CACHE_TTL = int(os.getenv("DOCSRS_CACHE_TTL", "900"))  # 15 minutes in seconds
CACHE_ADAPTIVE_TTL_ENABLED = (
    os.getenv("DOCSRS_CACHE_ADAPTIVE_TTL_ENABLED", "true").lower() == "true"
)

# Batch processing
EMBEDDING_BATCH_SIZE = int(os.getenv("DOCSRS_EMBEDDING_BATCH_SIZE", "32"))
DB_BATCH_SIZE = int(os.getenv("DOCSRS_DB_BATCH_SIZE", "999"))  # SQLite parameter limit

# Memory management
MEMORY_THRESHOLD_HIGH = int(
    os.getenv("DOCSRS_MEMORY_THRESHOLD_HIGH", "80")
)  # Percentage
MEMORY_THRESHOLD_CRITICAL = int(
    os.getenv("DOCSRS_MEMORY_THRESHOLD_CRITICAL", "90")
)  # Percentage
MIN_BATCH_SIZE = int(os.getenv("DOCSRS_MIN_BATCH_SIZE", "16"))
MAX_BATCH_SIZE = int(os.getenv("DOCSRS_MAX_BATCH_SIZE", "512"))
PARSE_CHUNK_SIZE = int(
    os.getenv("DOCSRS_PARSE_CHUNK_SIZE", "100")
)  # Items to process before yielding

# Rate limiting
RATE_LIMIT_PER_SECOND = int(os.getenv("DOCSRS_RATE_LIMIT", "30"))

# Cache management
CACHE_MAX_SIZE_BYTES = int(
    os.getenv("DOCSRS_CACHE_MAX_SIZE_BYTES", str(2 * 1024**3))
)  # 2GB

# Priority-aware cache eviction configuration
PRIORITY_CACHE_EVICTION_ENABLED = (
    os.getenv("DOCSRS_PRIORITY_CACHE_EVICTION_ENABLED", "true").lower() == "true"
)
CACHE_PRIORITY_WEIGHT = float(os.getenv("DOCSRS_CACHE_PRIORITY_WEIGHT", "0.7"))

# Rustdoc processing
RUSTDOC_CHUNK_MAX_SIZE = int(os.getenv("DOCSRS_RUSTDOC_CHUNK_MAX_SIZE", "4096"))

# Download configuration
DOWNLOAD_CHUNK_SIZE = int(os.getenv("DOCSRS_DOWNLOAD_CHUNK_SIZE", "8192"))

# Standard library configuration
STDLIB_CRATES = {"std", "core", "alloc", "proc_macro", "test"}
RUST_CHANNEL_BASE = "https://static.rust-lang.org"
RUST_VERSION_MANIFEST_URL = f"{RUST_CHANNEL_BASE}/version"

# Rust-specific term expansions for query preprocessing
RUST_TERM_EXPANSIONS = {
    "async": ["async", "asynchronous"],
    "impl": ["impl", "implementation"],
    "fn": ["fn", "function"],
    "mut": ["mut", "mutable"],
    "ref": ["ref", "reference"],
    "deref": ["deref", "dereference"],
    "struct": ["struct", "structure"],
    "enum": ["enum", "enumeration"],
    "trait": ["trait", "interface"],
    "mod": ["mod", "module"],
    "pub": ["pub", "public"],
    "priv": ["priv", "private"],
    "const": ["const", "constant"],
    "static": ["static", "global"],
    "iter": ["iter", "iterator", "iteration"],
    "vec": ["vec", "vector"],
    "str": ["str", "string"],
    "arc": ["arc", "atomic reference counted"],
    "rc": ["rc", "reference counted"],
    "box": ["box", "heap allocated"],
}

# Server configuration
PORT = int(os.getenv("DOCSRS_PORT", "8000"))
if not 1024 <= PORT <= 65535:
    warnings.warn(f"Port {PORT} out of range (1024-65535), using 8000", stacklevel=2)
    PORT = 8000

# Pre-ingestion configuration
PRE_INGEST_ENABLED = os.getenv("DOCSRS_PRE_INGEST_ENABLED", "false").lower() == "true"

# Embeddings warmup configuration
EMBEDDINGS_WARMUP_ENABLED = (
    os.getenv("DOCSRS_EMBEDDINGS_WARMUP_ENABLED", "true").lower() == "true"
)
CONCURRENCY = int(
    os.getenv("DOCSRS_CONCURRENCY", os.getenv("DOCSRS_PRE_INGEST_CONCURRENCY", "3"))
)
if not 1 <= CONCURRENCY <= 10:
    warnings.warn(
        f"Concurrency {CONCURRENCY} out of range (1-10), using 3", stacklevel=2
    )
    CONCURRENCY = 3
# Keep backward compatibility alias
PRE_INGEST_CONCURRENCY = CONCURRENCY

# Background scheduler configuration
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
SCHEDULER_INTERVAL_HOURS = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "6"))
SCHEDULER_JITTER_PERCENT = int(os.getenv("SCHEDULER_JITTER_PERCENT", "10"))

# Popular crates configuration with validation
_popular_count = int(os.getenv("DOCSRS_POPULAR_CRATES_COUNT", "100"))
POPULAR_CRATES_COUNT = max(100, min(500, _popular_count))  # Clamp to 100-500 range
if _popular_count != POPULAR_CRATES_COUNT:
    print(
        f"Warning: POPULAR_CRATES_COUNT clamped from {_popular_count} to {POPULAR_CRATES_COUNT}"
    )

POPULAR_CRATES_URL = "https://crates.io/api/v1/crates?sort=downloads&per_page={}"
POPULAR_CRATES_REFRESH_HOURS = int(
    os.getenv("DOCSRS_POPULAR_CRATES_REFRESH_HOURS", "24")
)
POPULAR_CRATES_CACHE_FILE = CACHE_DIR / "popular_crates.msgpack"
POPULAR_CRATES_REFRESH_THRESHOLD = float(
    os.getenv("DOCSRS_POPULAR_CRATES_REFRESH_THRESHOLD", "0.75")
)  # Refresh at 75% of TTL

# Hardcoded fallback list of essential popular crates
FALLBACK_POPULAR_CRATES = [
    "serde",
    "tokio",
    "clap",
    "syn",
    "anyhow",
    "thiserror",
    "rand",
    "log",
    "regex",
    "async-trait",
    "futures",
    "bytes",
    "chrono",
    "reqwest",
    "once_cell",
    "tracing",
    "serde_json",
    "quote",
    "proc-macro2",
    "itertools",
]

# Version for User-Agent header
VERSION = "0.1.0"

# Batch processing enhancements configuration
FASTEMBED_MAX_TEXT_LENGTH = int(os.getenv("FASTEMBED_MAX_TEXT_LENGTH", "100"))
FASTEMBED_MAX_BATCHES = int(os.getenv("FASTEMBED_MAX_BATCHES", "50"))
TRANSACTION_MAX_RETRIES = int(os.getenv("TRANSACTION_MAX_RETRIES", "3"))
TRANSACTION_BUSY_TIMEOUT = int(os.getenv("TRANSACTION_BUSY_TIMEOUT", "5000"))
BATCH_MEMORY_TREND_WINDOW = int(os.getenv("BATCH_MEMORY_TREND_WINDOW", "10"))
BATCH_PROCESSOR_MAX_BATCHES = int(os.getenv("BATCH_PROCESSOR_MAX_BATCHES", "50"))
