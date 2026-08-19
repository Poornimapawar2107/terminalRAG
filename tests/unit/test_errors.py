"""Unit tests for domain error hierarchy and hint propagation."""

import pytest
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


class TestErrorHierarchy:
    """Validate the exception class hierarchy."""

    def test_base_error_str_with_hint(self):
        err = VectorRAGError("Something failed", hint="Try again")
        assert str(err) == "Something failed (Hint: Try again)"
        assert err.message == "Something failed"
        assert err.hint == "Try again"

    def test_base_error_str_without_hint(self):
        err = VectorRAGError("Just an error")
        assert str(err) == "Just an error"
        assert err.hint is None

    def test_all_errors_are_vectorrag_error(self):
        """Every domain exception inherits from VectorRAGError."""
        error_classes = [
            ConfigurationError,
            DocumentError,
            UnsupportedFileTypeError,
            DocumentParseError,
            ChunkingError,
            EmbeddingError,
            VectorStoreError,
            StorageError,
            RetrievalError,
            RerankingError,
            GenerationError,
        ]
        for cls in error_classes:
            err = cls("test msg")
            assert isinstance(err, VectorRAGError), f"{cls.__name__} not a VectorRAGError"
            assert isinstance(err, Exception)

    def test_document_error_subtypes(self):
        """UnsupportedFileTypeError and DocumentParseError are subtypes of DocumentError."""
        assert issubclass(UnsupportedFileTypeError, DocumentError)
        assert issubclass(DocumentParseError, DocumentError)

    def test_hint_propagation_through_hierarchy(self):
        """Hints set on child classes propagate correctly."""
        err = UnsupportedFileTypeError(
            "Unsupported format '.docx'",
            hint="Supported formats: .txt, .pdf"
        )
        assert "Unsupported format" in str(err)
        assert "Supported formats" in str(err)
        assert err.hint == "Supported formats: .txt, .pdf"

    def test_errors_catchable_by_base(self):
        """Domain errors can be caught as VectorRAGError."""
        with pytest.raises(VectorRAGError):
            raise GenerationError("LLM failed")

        with pytest.raises(VectorRAGError):
            raise StorageError("SQLite error")

    def test_generation_error_with_chained_exception(self):
        """GenerationError preserves chained exception context."""
        original = RuntimeError("Out of memory")
        try:
            raise GenerationError("Model load failed") from original
        except GenerationError as e:
            assert e.__cause__ is original
            assert "Model load failed" in str(e)
