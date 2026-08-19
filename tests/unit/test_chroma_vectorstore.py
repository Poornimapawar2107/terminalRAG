"""Unit and integration tests for ChromaVectorStore."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.models.chunk import Chunk
from vector_rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def chroma_store(tmp_path: Path) -> ChromaVectorStore:
    chroma_dir = tmp_path / "chroma_test"
    return ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="test_collection",
    )


def test_chroma_add_and_query(chroma_store: ChromaVectorStore):
    embedder = MockEmbedder(dimension=64)

    chunk1 = Chunk(
        document_id="doc-linux",
        chunk_index=0,
        text="Linux kernel inodes store metadata about files.",
        page=42,
        metadata={"filename": "linux.pdf", "file_type": "pdf"},
    )
    chunk2 = Chunk(
        document_id="doc-db",
        chunk_index=0,
        text="Relational databases use B-Trees for indexing.",
        page=12,
        metadata={"filename": "db.pdf", "file_type": "pdf"},
    )

    chunks = [chunk1, chunk2]
    embeddings = embedder.embed_documents([c.text for c in chunks])

    chroma_store.add_chunks(chunks, embeddings)
    assert chroma_store.count() == 2

    # Query with chunk1 text vector
    query_vec = embedder.embed_query("Tell me about Linux kernel inodes")
    results = chroma_store.query_vectors(query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id in [chunk1.id, chunk2.id]
    assert results[0].page is not None
    assert results[0].filename in ["linux.pdf", "db.pdf"]


def test_chroma_delete_by_document(chroma_store: ChromaVectorStore):
    embedder = MockEmbedder(dimension=64)

    chunk1 = Chunk(
        document_id="doc-to-delete",
        chunk_index=0,
        text="This text belongs to a deleted doc.",
    )
    chunk2 = Chunk(
        document_id="doc-to-keep",
        chunk_index=0,
        text="This text should remain in index.",
    )

    chunks = [chunk1, chunk2]
    embeddings = embedder.embed_documents([c.text for c in chunks])
    chroma_store.add_chunks(chunks, embeddings)
    assert chroma_store.count() == 2

    # Delete first document
    chroma_store.delete_by_document_id("doc-to-delete")
    assert chroma_store.count() == 1

    query_vec = embedder.embed_query("text")
    results = chroma_store.query_vectors(query_vec, top_k=5)
    assert len(results) == 1
    assert results[0].document_id == "doc-to-keep"


def test_chroma_reset(chroma_store: ChromaVectorStore):
    embedder = MockEmbedder(dimension=64)
    chunk = Chunk(document_id="doc1", chunk_index=0, text="sample text")
    emb = embedder.embed_documents([chunk.text])

    chroma_store.add_chunks([chunk], emb)
    assert chroma_store.count() == 1

    chroma_store.reset()
    assert chroma_store.count() == 0
