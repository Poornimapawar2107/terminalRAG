"""Unit tests for VectorRetriever and RetrievalService."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.models.chunk import Chunk
from vector_rag.models.retrieval import RetrievalRequest
from vector_rag.retrieval.retriever import VectorRetriever
from vector_rag.retrieval.service import RetrievalService
from vector_rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def retrieval_setup(tmp_path: Path):
    chroma_dir = tmp_path / "chroma_retrieval"
    vector_store = ChromaVectorStore(persist_directory=chroma_dir)
    embedder = MockEmbedder(dimension=64)

    # Seed sample chunks
    chunks = [
        Chunk(
            document_id="doc1",
            chunk_index=0,
            text="Inodes store permissions, owner ID, size, and data block pointers.",
            page=1,
            metadata={"filename": "linux_fs.pdf"},
        ),
        Chunk(
            document_id="doc2",
            chunk_index=0,
            text="Relational databases use B-Trees for primary and secondary indexing.",
            page=5,
            metadata={"filename": "db_internals.pdf"},
        ),
    ]
    embeddings = embedder.embed_documents([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder, default_top_k=2)
    service = RetrievalService(vector_store=vector_store, embedder=embedder, retriever=retriever)

    return service, retriever


def test_vector_retriever_query(retrieval_setup):
    _, retriever = retrieval_setup

    results = retriever.retrieve("What data does an inode store?", top_k=2)
    assert len(results) == 2
    assert results[0].score >= 0.0
    assert results[0].filename in ["linux_fs.pdf", "db_internals.pdf"]


def test_retrieval_service(retrieval_setup):
    service, _ = retrieval_setup

    results = service.retrieve(query="indexing in databases", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id is not None
    assert results[0].text is not None


def test_empty_query(retrieval_setup):
    service, _ = retrieval_setup
    results = service.retrieve(query="   ")
    assert results == []
