"""Integration tests for two-stage retrieval (Vector search + Cross-Encoder reranking)."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.models.chunk import Chunk
from vector_rag.retrieval.reranker import MockReranker
from vector_rag.retrieval.retriever import VectorRetriever
from vector_rag.retrieval.service import RetrievalService
from vector_rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def two_stage_service(tmp_path: Path) -> RetrievalService:
    chroma_dir = tmp_path / "chroma_rerank"
    vector_store = ChromaVectorStore(persist_directory=chroma_dir)
    embedder = MockEmbedder(dimension=64)
    reranker = MockReranker(top_n=2)

    chunks = [
        Chunk(
            document_id="doc1",
            chunk_index=0,
            text="Inodes contain metadata such as file size, access permissions, and data block pointers.",
            page=1,
            metadata={"filename": "linux.txt"},
        ),
        Chunk(
            document_id="doc2",
            chunk_index=0,
            text="Virtual memory allows executing processes that are not completely in physical memory.",
            page=2,
            metadata={"filename": "memory.txt"},
        ),
        Chunk(
            document_id="doc3",
            chunk_index=0,
            text="CPU scheduling algorithms include Round Robin, First-Come-First-Served, and Shortest-Job-First.",
            page=3,
            metadata={"filename": "cpu.txt"},
        ),
    ]

    embeddings = embedder.embed_documents([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder, default_top_k=3)

    return RetrievalService(
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
    )


def test_two_stage_search_and_rerank(two_stage_service: RetrievalService):
    result = two_stage_service.search_and_rerank(
        query="Explain filesystem inodes and block allocation",
        top_k=3,
        top_n=2,
    )

    assert len(result.candidates) == 3
    assert len(result.reranked) == 2

    top_chunk = result.reranked[0]
    assert top_chunk.retrieval_score > 0.0
    assert top_chunk.rerank_score >= top_chunk.retrieval_score
    assert top_chunk.filename in ["linux.txt", "memory.txt", "cpu.txt"]
