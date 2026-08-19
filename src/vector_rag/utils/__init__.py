"""Utilities package for Vector RAG."""

from vector_rag.utils.errors import (
    ChunkingError,
    ConfigurationError,
    DocumentError,
    DocumentParseError,
    EmbeddingError,
    GenerationError,
    RerankingError,
    RetrievalError,
    StorageError,
    UnsupportedFileTypeError,
    VectorRAGError,
    VectorStoreError,
)
from vector_rag.utils.logging import (
    clear_request_id,
    get_logger,
    set_request_id,
    setup_logging,
)

__all__ = [
    "VectorRAGError",
    "ConfigurationError",
    "DocumentError",
    "UnsupportedFileTypeError",
    "DocumentParseError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "StorageError",
    "RetrievalError",
    "RerankingError",
    "GenerationError",
    "setup_logging",
    "get_logger",
    "set_request_id",
    "clear_request_id",
]
