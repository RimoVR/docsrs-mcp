"""Simple configuration for docsrs-mcp."""

import os
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

# Caching configuration
CACHE_SIZE = int(os.getenv("DOCSRS_CACHE_SIZE", "1000"))
CACHE_TTL = int(os.getenv("DOCSRS_CACHE_TTL", "900"))  # 15 minutes in seconds

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

# Pre-ingestion configuration
PRE_INGEST_ENABLED = os.getenv("DOCSRS_PRE_INGEST_ENABLED", "false").lower() == "true"
PRE_INGEST_CONCURRENCY = int(os.getenv("DOCSRS_PRE_INGEST_CONCURRENCY", "3"))

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
