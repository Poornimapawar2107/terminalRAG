"""Custom domain exceptions for Vector RAG."""


class VectorRAGError(Exception):
    """Base exception for all Vector RAG errors."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} (Hint: {self.hint})"
        return self.message


class ConfigurationError(VectorRAGError):
    """Raised when configuration validation or loading fails."""


class DocumentError(VectorRAGError):
    """Base exception for document processing errors."""


class UnsupportedFileTypeError(DocumentError):
    """Raised when an unsupported file type is provided for ingestion."""


class DocumentParseError(DocumentError):
    """Raised when parsing document contents fails."""


class ChunkingError(VectorRAGError):
    """Raised when document chunking fails."""


class EmbeddingError(VectorRAGError):
    """Raised when generating embeddings fails."""


class VectorStoreError(VectorRAGError):
    """Raised when interacting with vector store (ChromaDB) fails."""


class StorageError(VectorRAGError):
    """Raised when interacting with relational/metadata storage (SQLite) fails."""


class RetrievalError(VectorRAGError):
    """Raised when retrieval operations fail."""


class RerankingError(VectorRAGError):
    """Raised when reranking candidate chunks fails."""


class GenerationError(VectorRAGError):
    """Raised when LLM text generation fails."""
