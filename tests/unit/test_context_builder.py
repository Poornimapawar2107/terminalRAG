"""Unit tests for ContextBuilder."""

import pytest
from vector_rag.generation.context import ContextBuilder
from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk


def make_reranked_chunk(idx: int, filename: str, page: int, text: str) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=f"chk-{idx}",
        document_id=f"doc-{idx}",
        filename=filename,
        page=page,
        text=text,
        retrieval_score=0.85,
        rerank_score=0.92,
    )


def test_build_context_basic():
    builder = ContextBuilder(max_context_tokens=1000)
    c1 = make_reranked_chunk(1, "linux.pdf", 42, "Inodes hold permissions and block pointers.")
    c2 = make_reranked_chunk(2, "fs.pdf", 18, "Superblocks describe the filesystem layout.")

    package = builder.build_context("How do filesystems work?", [c1, c2])

    assert len(package.chunks) == 2
    assert package.chunks[0].source_id == 1
    assert package.chunks[0].filename == "linux.pdf"
    assert package.chunks[0].page == 42
    assert package.chunks[1].source_id == 2

    # Check prompt context formatting
    prompt_str = package.format_prompt_context()
    assert "[Source 1] File: linux.pdf | Page: 42" in prompt_str
    assert "Inodes hold permissions and block pointers." in prompt_str
    assert "[Source 2] File: fs.pdf | Page: 18" in prompt_str


def test_build_context_token_budget_truncation():
    # Set a tiny token limit (e.g. 15 tokens ~ 60 chars)
    builder = ContextBuilder(max_context_tokens=15, approx_chars_per_token=4.0)
    c1 = make_reranked_chunk(1, "doc1.txt", 1, "Short chunk number one.")
    c2 = make_reranked_chunk(2, "doc2.txt", 1, "Second chunk that should exceed the tiny token budget and be dropped.")

    package = builder.build_context("query", [c1, c2])
    assert len(package.chunks) == 1
    assert package.chunks[0].filename == "doc1.txt"


def test_citation_mapping():
    builder = ContextBuilder()
    c1 = make_reranked_chunk(1, "network.pdf", 5, "TCP handshake SYN, SYN-ACK, ACK.")
    package = builder.build_context("TCP", [c1])

    citation_map = builder.get_citation_map(package)
    assert 1 in citation_map
    citation = citation_map[1]
    assert citation.source_id == 1
    assert citation.filename == "network.pdf"
    assert citation.page == 5
    assert "TCP handshake" in citation.snippet


def test_empty_chunks():
    builder = ContextBuilder()
    package = builder.build_context("query", [])
    assert package.chunks == []
    assert package.total_tokens == 0
    assert package.format_prompt_context() == ""
