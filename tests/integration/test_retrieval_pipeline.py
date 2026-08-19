"""End-to-End Integration test for Phase 7: Document Ingestion -> Vector Retrieval."""

from pathlib import Path
import pytest
from vector_rag.embeddings.embedder import MockEmbedder
from vector_rag.ingestion.service import DocumentService
from vector_rag.retrieval.service import RetrievalService
from vector_rag.storage.sqlite import SQLiteStorage
from vector_rag.vectorstore.chroma import ChromaVectorStore


def test_e2e_ingestion_and_retrieval_pipeline(tmp_path: Path):
    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "test_chroma"

    storage = SQLiteStorage(db_path)
    vector_store = ChromaVectorStore(chroma_path)
    embedder = MockEmbedder(dimension=64)

    doc_service = DocumentService(
        storage=storage,
        vector_store=vector_store,
        embedder=embedder,
    )
    retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedder=embedder,
    )

    # 1. Create and ingest documents
    doc1 = tmp_path / "linux_kernel.txt"
    doc1.write_text(
        "Linux kernel architecture handles memory management, process scheduling, and block I/O.",
        encoding="utf-8",
    )
    doc2 = tmp_path / "networking.txt"
    doc2.write_text(
        "TCP uses a three-way handshake SYN, SYN-ACK, ACK to establish a reliable stream.",
        encoding="utf-8",
    )

    ingest_result = doc_service.ingest_path(tmp_path)
    assert ingest_result.parsed == 2
    assert ingest_result.vectors_indexed == 2
    assert vector_store.count() == 2

    # 2. Query retrieval pipeline
    results = retrieval_service.retrieve("How does TCP establish connections?", top_k=2)
    assert len(results) == 2
    assert results[0].filename in ["linux_kernel.txt", "networking.txt"]
    assert results[0].score > 0.0

    # 3. Clean up / delete doc1
    docs = storage.list_documents()
    linux_doc = next(d for d in docs if d.filename == "linux_kernel.txt")
    doc_service.delete_document(linux_doc.document_id)

    # Verify vector store updated
    assert vector_store.count() == 1
    post_delete_results = retrieval_service.retrieve("TCP connection", top_k=5)
    assert len(post_delete_results) == 1
    assert post_delete_results[0].filename == "networking.txt"
