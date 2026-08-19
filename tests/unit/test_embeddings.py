"""Unit tests for embedding generation."""

import pytest
from vector_rag.embeddings.embedder import (
    MockEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from vector_rag.utils.errors import EmbeddingError


def test_mock_embedder():
    embedder = MockEmbedder(dimension=256)
    assert embedder.dimension == 256

    query_vec = embedder.embed_query("How does an inode work?")
    assert len(query_vec) == 256
    assert isinstance(query_vec[0], float)

    # Test determinism
    query_vec_2 = embedder.embed_query("How does an inode work?")
    assert query_vec == query_vec_2

    # Test batch docs
    doc_vecs = embedder.embed_documents(["Document 1", "Document 2"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 256
    assert len(doc_vecs[1]) == 256


def test_mock_embedder_empty_list():
    embedder = MockEmbedder()
    assert embedder.embed_documents([]) == []


def test_factory_mock():
    embedder = create_embedder(mock=True)
    assert isinstance(embedder, MockEmbedder)
    assert embedder.dimension == 384


def test_sentence_transformer_lazy_load_error():
    # Invalid model name should raise domain EmbeddingError when triggered
    embedder = SentenceTransformerEmbedder(model_name="non_existent_model_12345_xyz")
    with pytest.raises(EmbeddingError) as exc_info:
        embedder.embed_query("Test query")

    assert "Failed to load SentenceTransformer" in str(exc_info.value)
    assert exc_info.value.hint is not None
