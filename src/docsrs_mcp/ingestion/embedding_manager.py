"""Embedding model lifecycle management for ONNX-based text embeddings.

This module handles:
- Lazy loading of embedding models to save memory
- Model warmup for cold-start elimination
- ONNX session optimization settings
- Resource cleanup and memory management
"""

import asyncio
import logging
import os

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global model instance (singleton pattern)
_embedding_model: TextEmbedding | None = None
_embeddings_warmed = False


def get_embedding_model() -> TextEmbedding:
    """Get or create the embedding model instance with optimized ONNX settings.

    This function implements lazy loading to save ~500MB startup memory.
    The model is only loaded when first needed, not at import time.

    ONNX optimizations applied:
    - Disables CPU memory arena for smaller models
    - Disables memory pattern optimization to reduce overhead

    Returns:
        TextEmbedding: The singleton embedding model instance
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

        # Configure ONNX Runtime environment variables for memory optimization
        # These settings reduce memory usage for smaller models
        os.environ["ORT_DISABLE_CPU_ARENA_ALLOCATOR"] = "1"
        os.environ["ORT_DISABLE_MEMORY_PATTERN"] = "1"

        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model


def cleanup_embedding_model() -> None:
    """Clean up the embedding model to free memory.

    This function releases the model resources and triggers garbage collection
    to reclaim memory. Useful when switching between memory-intensive operations.
    """
    global _embedding_model
    if _embedding_model is not None:
        logger.debug("Cleaning up embedding model to free memory")
        try:
            # Attempt to clean up the model
            del _embedding_model
            _embedding_model = None

            # Force garbage collection to reclaim memory
            import gc

            gc.collect()

            # Trigger additional memory cleanup if needed
            try:
                from ..memory_utils import trigger_gc_if_needed

                trigger_gc_if_needed()
            except ImportError:
                # memory_utils might not be available in all contexts
                pass
        except Exception as e:
            logger.warning(f"Error during embedding model cleanup: {e}")


async def warmup_embedding_model() -> None:
    """Warm up the embedding model to eliminate cold-start latency.

    This function pre-loads the model and runs sample embeddings to:
    - Load the ONNX model into memory
    - JIT compile any optimizations
    - Reduce first-request latency by ~400ms

    The warmup runs in the background to avoid blocking startup.
    """
    from .. import config

    if not config.EMBEDDINGS_WARMUP_ENABLED:
        return

    try:
        # Create background task to not block startup
        warmup_task = asyncio.create_task(_perform_warmup())
        # Fire-and-forget pattern for non-blocking warmup
        logger.info("Starting embedding model warmup in background")
    except Exception as e:
        logger.warning(f"Failed to start embedding warmup: {e}")
        # Non-critical - continue without warmup


async def _perform_warmup() -> None:
    """Perform actual warmup in background.

    Runs 3-5 representative embeddings covering different text lengths
    to ensure the model is fully warmed up for various input sizes.
    """
    global _embeddings_warmed
    try:
        # Trigger model loading via existing function
        model = get_embedding_model()

        # Perform representative embeddings for different text lengths
        warmup_texts = [
            "async runtime spawn tasks",  # Short text (~4 tokens)
            "The Rust programming language provides memory safety without garbage collection through its ownership system and borrow checker",  # Medium text (~20 tokens)
            " ".join(["token"] * 100),  # Long text for edge case (~100 tokens)
        ]

        for text in warmup_texts:
            # Use asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(model.embed, [text])

        # Set warmup status to true on success
        _embeddings_warmed = True
        logger.info("Embedding model warmup completed successfully")
    except Exception as e:
        logger.warning(f"Embedding warmup failed: {e}")
        # Non-critical failure - service continues


def get_warmup_status() -> bool:
    """Get the current warmup status for health endpoint.

    Returns:
        bool: True if model has been warmed up, False otherwise
    """
    return _embeddings_warmed


def configure_onnx_session() -> None:
    """Configure ONNX Runtime session options for optimal performance.

    This function sets additional ONNX Runtime environment variables
    for memory arena optimization and performance tuning.
    """
    # Enable memory arena for session sharing
    os.environ["ORT_ENABLE_CPU_MEM_ARENA"] = "1"

    # Set arena extend strategy for better memory usage
    os.environ["ORT_ARENA_EXTEND_STRATEGY"] = "1"

    # Configure thread pool size based on CPU cores
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Use half the CPU cores for embedding generation to leave room for other operations
    thread_count = max(1, cpu_count // 2)
    os.environ["ORT_INTER_OP_NUM_THREADS"] = str(thread_count)
    os.environ["ORT_INTRA_OP_NUM_THREADS"] = str(thread_count)

    logger.debug(f"Configured ONNX Runtime with {thread_count} threads")
