"""Unit tests for RAGService orchestration logic."""

from unittest.mock import MagicMock
import pytest
from vector_rag.generation.citation import CitationExtractor
from vector_rag.generation.context import ContextBuilder
from vector_rag.generation.llm import MockLLM
from vector_rag.generation.rag_service import RAGService
from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk
from vector_rag.retrieval.service import RetrievalService
from vector_rag.utils.errors import GenerationError


def _make_mock_retrieval_service(candidates, reranked):
    """Create a mock RetrievalService that returns preset results."""
    mock_service = MagicMock(spec=RetrievalService)
    result = MagicMock()
    result.candidates = candidates
    result.reranked = reranked
    mock_service.search_and_rerank.return_value = result
    return mock_service


def _make_retrieved(idx, filename, page, text, score=0.85):
    return RetrievedChunk(
        chunk_id=f"chk-{idx}",
        document_id=f"doc-{idx}",
        filename=filename,
        text=text,
        score=score,
        page=page,
    )


def _make_reranked(idx, filename, page, text):
    return RerankedChunk(
        chunk_id=f"chk-{idx}",
        document_id=f"doc-{idx}",
        filename=filename,
        page=page,
        text=text,
        retrieval_score=0.85,
        rerank_score=0.92,
    )


def test_rag_query_returns_answer_with_citations():
    """Full pipeline with mock components returns answer and resolved citations."""
    r1 = _make_retrieved(1, "linux.pdf", 42, "Inodes store metadata like permissions.")
    r2 = _make_retrieved(2, "fs.pdf", 10, "Superblocks define filesystem geometry.")
    rr1 = _make_reranked(1, "linux.pdf", 42, "Inodes store metadata like permissions.")
    rr2 = _make_reranked(2, "fs.pdf", 10, "Superblocks define filesystem geometry.")

    retrieval_svc = _make_mock_retrieval_service(
        candidates=[r1, r2],
        reranked=[rr1, rr2],
    )
    llm = MockLLM(
        default_response="Inodes store metadata [1] and superblocks manage geometry [2]."
    )

    rag = RAGService(
        retrieval_service=retrieval_svc,
        context_builder=ContextBuilder(),
        llm=llm,
        citation_extractor=CitationExtractor(),
    )

    response = rag.query("How do filesystems work?", top_k=5, top_n=2)

    assert response.query == "How do filesystems work?"
    assert "[1]" in response.answer
    assert len(response.citations) == 2
    assert response.citations[0].filename == "linux.pdf"
    assert response.citations[1].filename == "fs.pdf"
    assert response.request_id is not None
    retrieval_svc.search_and_rerank.assert_called_once()


def test_rag_query_no_results():
    """When no candidates are found, returns a fallback 'no docs' answer."""
    retrieval_svc = _make_mock_retrieval_service(candidates=[], reranked=[])

    rag = RAGService(
        retrieval_service=retrieval_svc,
        context_builder=ContextBuilder(),
        llm=MockLLM(),
        citation_extractor=CitationExtractor(),
    )

    response = rag.query("Unknown topic with no documents")

    assert "No relevant documents" in response.answer
    assert response.citations == []
    assert response.request_id is not None


def test_rag_query_uses_candidates_when_no_reranked():
    """When reranking returns empty but candidates exist, candidates are used."""
    r1 = _make_retrieved(1, "data.txt", 1, "Some data about algorithms.")
    retrieval_svc = _make_mock_retrieval_service(
        candidates=[r1],
        reranked=[],
    )
    llm = MockLLM(default_response="Algorithms handle sorting [1].")

    rag = RAGService(
        retrieval_service=retrieval_svc,
        context_builder=ContextBuilder(),
        llm=llm,
        citation_extractor=CitationExtractor(),
    )

    response = rag.query("Tell me about algorithms")

    assert len(response.answer) > 0
    assert "[1]" in response.answer
    assert response.retrieved_chunks == [r1]


def test_rag_query_llm_error_propagates():
    """When the LLM raises an error, it propagates through RAGService."""
    r1 = _make_retrieved(1, "test.txt", 1, "Test content.")
    rr1 = _make_reranked(1, "test.txt", 1, "Test content.")
    retrieval_svc = _make_mock_retrieval_service(candidates=[r1], reranked=[rr1])

    error_llm = MagicMock(spec=MockLLM)
    error_llm.generate.side_effect = GenerationError("LLM crashed")

    rag = RAGService(
        retrieval_service=retrieval_svc,
        context_builder=ContextBuilder(),
        llm=error_llm,
        citation_extractor=CitationExtractor(),
    )

    with pytest.raises(GenerationError, match="LLM crashed"):
        rag.query("query")
