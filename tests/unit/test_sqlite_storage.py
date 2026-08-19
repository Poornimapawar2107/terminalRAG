"""Unit and integration tests for SQLiteStorage."""

from pathlib import Path
import pytest
from vector_rag.models.chunk import Chunk
from vector_rag.models.document import DocumentMetadata
from vector_rag.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path: Path) -> SQLiteStorage:
    db_file = tmp_path / "test_app.db"
    return SQLiteStorage(db_file)


def test_init_and_save_document(storage: SQLiteStorage):
    doc_meta = DocumentMetadata(
        filename="kernel.pdf",
        file_path="/docs/kernel.pdf",
        file_type="pdf",
        file_size=1024,
        content_hash="abc123hash",
        title="Kernel Architecture",
        page_count=5,
    )

    storage.save_document(doc_meta)

    # Lookup by ID
    fetched_by_id = storage.get_document_by_id(doc_meta.document_id)
    assert fetched_by_id is not None
    assert fetched_by_id.filename == "kernel.pdf"
    assert fetched_by_id.page_count == 5

    # Lookup by Hash
    fetched_by_hash = storage.get_document_by_hash("abc123hash")
    assert fetched_by_hash is not None
    assert fetched_by_hash.document_id == doc_meta.document_id

    # Lookup by Path
    fetched_by_path = storage.get_document_by_path("/docs/kernel.pdf")
    assert fetched_by_path is not None
    assert fetched_by_path.title == "Kernel Architecture"


def test_list_and_delete_documents(storage: SQLiteStorage):
    doc1 = DocumentMetadata(
        filename="file1.txt",
        file_path="/docs/file1.txt",
        file_type="txt",
        file_size=100,
        content_hash="hash1",
    )
    doc2 = DocumentMetadata(
        filename="file2.txt",
        file_path="/docs/file2.txt",
        file_type="txt",
        file_size=200,
        content_hash="hash2",
    )

    storage.save_document(doc1)
    storage.save_document(doc2)

    docs = storage.list_documents()
    assert len(docs) == 2

    # Delete doc1
    deleted = storage.delete_document(doc1.document_id)
    assert deleted is True
    assert storage.get_document_by_id(doc1.document_id) is None
    assert len(storage.list_documents()) == 1


def test_chunks_crud_and_cascade(storage: SQLiteStorage):
    doc = DocumentMetadata(
        filename="manual.txt",
        file_path="/docs/manual.txt",
        file_type="txt",
        file_size=500,
        content_hash="manualhash",
    )
    storage.save_document(doc)

    chunk1 = Chunk(
        document_id=doc.document_id,
        chunk_index=0,
        text="Section 1 of manual",
        page=1,
        metadata={"section": "intro"},
    )
    chunk2 = Chunk(
        document_id=doc.document_id,
        chunk_index=1,
        text="Section 2 of manual",
        page=1,
        metadata={"section": "body"},
    )

    storage.save_chunks([chunk1, chunk2])

    chunks = storage.get_chunks_by_document_id(doc.document_id)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Section 1 of manual"
    assert chunks[0].metadata == {"section": "intro"}

    # Cascade deletion when document is deleted
    storage.delete_document(doc.document_id)
    assert len(storage.get_chunks_by_document_id(doc.document_id)) == 0
