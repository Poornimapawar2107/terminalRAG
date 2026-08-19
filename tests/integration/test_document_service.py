"""Integration tests for DocumentService: re-ingestion and edge cases."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.ingestion.service import DocumentService
from vector_rag.storage.sqlite import SQLiteStorage
from vector_rag.utils.errors import DocumentError
from vector_rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def doc_service(tmp_path: Path) -> DocumentService:
    db_file = tmp_path / "app.db"
    chroma_dir = tmp_path / "chroma"
    storage = SQLiteStorage(db_file)
    vector_store = ChromaVectorStore(persist_directory=chroma_dir)
    embedder = MockEmbedder(dimension=64)

    return DocumentService(
        storage=storage,
        vector_store=vector_store,
        embedder=embedder,
    )


def test_reingest_modified_file(doc_service: DocumentService, tmp_path: Path):
    """When a file is modified (different hash), old chunks/vectors are replaced."""
    test_file = tmp_path / "dynamic.txt"
    test_file.write_text("Version 1: Original content about databases.", encoding="utf-8")

    res1 = doc_service.ingest_path(test_file)
    assert res1.parsed == 1
    assert res1.chunks_created == 1
    assert doc_service.vector_store.count() == 1

    doc_id_v1 = res1.documents[0].document_id

    # Modify file content
    test_file.write_text(
        "Version 2: Updated content about distributed systems and consensus algorithms.",
        encoding="utf-8",
    )

    res2 = doc_service.ingest_path(test_file)
    assert res2.parsed == 1
    assert res2.skipped == 0
    # Old vectors cleaned up, new vectors added
    assert doc_service.vector_store.count() == 1

    # New document should have a different ID since old was deleted and re-created
    docs = doc_service.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "dynamic.txt"


def test_ingest_invalid_path(doc_service: DocumentService):
    """Attempting to ingest a nonexistent path raises DocumentError."""
    with pytest.raises(DocumentError, match="does not exist"):
        doc_service.ingest_path("/nonexistent/path/file.txt")


def test_ingest_force_reingestion(doc_service: DocumentService, tmp_path: Path):
    """Force flag re-ingests even when hash is unchanged."""
    test_file = tmp_path / "stable.txt"
    test_file.write_text("Content that stays the same.", encoding="utf-8")

    res1 = doc_service.ingest_path(test_file)
    assert res1.parsed == 1
    assert res1.skipped == 0

    res2 = doc_service.ingest_path(test_file, force=True)
    assert res2.parsed == 1
    assert res2.skipped == 0


def test_list_documents_empty(doc_service: DocumentService):
    """Empty service returns no documents."""
    assert doc_service.list_documents() == []


def test_delete_nonexistent_document(doc_service: DocumentService):
    """Deleting a non-existent document returns False."""
    result = doc_service.delete_document("nonexistent-doc-id")
    assert result is False


def test_ingest_empty_directory(doc_service: DocumentService, tmp_path: Path):
    """Ingesting an empty directory returns zero discovered."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = doc_service.ingest_path(empty_dir)
    assert result.discovered == 0
    assert result.parsed == 0
    assert result.chunks_created == 0


def test_ingest_mixed_supported_unsupported(doc_service: DocumentService, tmp_path: Path):
    """Ingesting a directory ignores unsupported file types gracefully."""
    doc_dir = tmp_path / "mixed"
    doc_dir.mkdir()

    (doc_dir / "valid.txt").write_text("Valid text document.", encoding="utf-8")
    (doc_dir / "image.png").write_bytes(b"PNGDATA")
    (doc_dir / "binary.exe").write_bytes(b"EXEDATA")

    result = doc_service.ingest_path(doc_dir)
    assert result.discovered == 1  # only .txt
    assert result.parsed == 1
    assert doc_service.vector_store.count() == 1
