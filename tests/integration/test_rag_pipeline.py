"""End-to-End Integration test for complete RAG pipeline (Ingest -> Two-Stage Search -> LLM -> Citations)."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.generation.citation import CitationExtractor
from vector_rag.generation.context import ContextBuilder
from vector_rag.generation.llm import MockLLM
from vector_rag.generation.rag_service import RAGService
from vector_rag.ingestion.service import DocumentService
from vector_rag.retrieval.reranker import MockReranker
from vector_rag.retrieval.retriever import VectorRetriever
from vector_rag.retrieval.service import RetrievalService
from vector_rag.storage.sqlite import SQLiteStorage
from vector_rag.vectorstore.chroma import ChromaVectorStore


def test_full_rag_pipeline(tmp_path: Path):
    db_path = tmp_path / "rag_e2e.db"
    chroma_dir = tmp_path / "rag_chroma"

    storage = SQLiteStorage(db_path)
    vector_store = ChromaVectorStore(chroma_dir)
    embedder = MockEmbedder(dimension=64)
    reranker = MockReranker(top_n=2)
    llm = MockLLM(
        default_response="Linux systems store file attributes inside inodes [1] while directories map names to inode IDs [2]."
    )

    doc_service = DocumentService(
        storage=storage,
        vector_store=vector_store,
        embedder=embedder,
    )
    retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        context_builder=ContextBuilder(),
        llm=llm,
        citation_extractor=CitationExtractor(),
    )

    # Ingest document
    doc_file = tmp_path / "linux_storage.txt"
    doc_file.write_text(
        "Linux filesystems store permissions and block pointers in inodes.\n\n"
        "Directory entries link human readable file names to inode numbers.",
        encoding="utf-8",
    )
    doc_service.ingest_path(doc_file)

    # Execute complete RAG query
    response = rag_service.query("How do Linux filesystems work?", top_k=5, top_n=2)

    assert response.query == "How do Linux filesystems work?"
    assert len(response.answer) > 0
    assert len(response.citations) > 0
    assert response.citations[0].filename == "linux_storage.txt"
    assert response.request_id is not None
