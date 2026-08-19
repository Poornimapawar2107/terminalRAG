"""Integration tests for DocumentService ingestion pipeline."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.ingestion.service import DocumentService
from vector_rag.storage.sqlite import SQLiteStorage
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


def test_ingest_single_file_and_deduplication(doc_service: DocumentService, tmp_path: Path):
    test_file = tmp_path / "operating_systems.txt"
    test_file.write_text(
        "Operating systems manage memory, process scheduling, and disk block allocation.",
        encoding="utf-8",
    )

    # 1. First Ingestion
    res1 = doc_service.ingest_path(test_file)
    assert res1.discovered == 1
    assert res1.parsed == 1
    assert res1.skipped == 0
    assert res1.chunks_created == 1
    assert res1.vectors_indexed == 1

    # Verify SQLite record
    docs = doc_service.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "operating_systems.txt"

    # Verify SQLite chunks
    chunks = doc_service.get_document_chunks(docs[0].document_id)
    assert len(chunks) == 1
    assert "Operating systems manage" in chunks[0].text

    # Verify Chroma vector count
    assert doc_service.vector_store.count() == 1

    # 2. Second Ingestion (Identical File Hash -> should skip)
    res2 = doc_service.ingest_path(test_file)
    assert res2.discovered == 1
    assert res2.skipped == 1
    assert res2.parsed == 0
    assert doc_service.vector_store.count() == 1


def test_ingest_directory_batch(doc_service: DocumentService, tmp_path: Path):
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()

    (doc_dir / "doc1.txt").write_text("Introduction to data structures.", encoding="utf-8")
    (doc_dir / "doc2.txt").write_text("B-trees and hash tables in database systems.", encoding="utf-8")

    res = doc_service.ingest_path(doc_dir)
    assert res.discovered == 2
    assert res.parsed == 2
    assert res.chunks_created == 2
    assert res.vectors_indexed == 2
    assert doc_service.vector_store.count() == 2


def test_delete_document_service(doc_service: DocumentService, tmp_path: Path):
    test_file = tmp_path / "to_delete.txt"
    test_file.write_text("Text to be deleted.", encoding="utf-8")

    res = doc_service.ingest_path(test_file)
    doc_id = res.documents[0].document_id

    assert len(doc_service.list_documents()) == 1
    assert doc_service.vector_store.count() == 1

    deleted = doc_service.delete_document(doc_id)
    assert deleted is True
    assert len(doc_service.list_documents()) == 0
    assert doc_service.vector_store.count() == 0
