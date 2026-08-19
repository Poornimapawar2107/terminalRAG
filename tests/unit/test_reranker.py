"""Unit tests for CrossEncoderReranker and MockReranker."""

import pytest
from vector_rag.models.retrieval import RetrievedChunk
from vector_rag.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    create_reranker,
)
from vector_rag.utils.errors import RerankingError


def make_retrieved_chunk(chunk_id: str, text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        filename="linux.pdf",
        text=text,
        score=score,
        page=1,
    )


def test_mock_reranker():
    reranker = MockReranker(top_n=2)
    c1 = make_retrieved_chunk("c1", "Linux kernel inode explanation", 0.85)
    c2 = make_retrieved_chunk("c2", "Database query optimization", 0.70)
    c3 = make_retrieved_chunk("c3", "Network socket programming", 0.60)

    results = reranker.rerank(
        query="What is an inode?",
        chunks=[c1, c2, c3],
        top_n=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].retrieval_score == 0.85
    assert results[0].rerank_score >= 0.85


def test_reranker_empty_list():
    reranker = MockReranker()
    assert reranker.rerank("query", []) == []


def test_factory_mock():
    reranker = create_reranker(mock=True, top_n=3)
    assert isinstance(reranker, MockReranker)
    assert reranker.top_n == 3


def test_cross_encoder_lazy_load_error():
    reranker = CrossEncoderReranker(model_name="non_existent_cross_encoder_model_999")
    chunk = make_retrieved_chunk("c1", "Sample text")

    with pytest.raises(RerankingError) as exc_info:
        reranker.rerank("Query", [chunk])

    assert "Failed to load CrossEncoder model" in str(exc_info.value)
