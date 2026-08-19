"""Unit tests for LLM generation and citation resolution."""

import pytest
from vector_rag.generation.citation import CitationExtractor
from vector_rag.generation.context import ContextBuilder
from vector_rag.generation.llm import MockLLM, create_llm
from vector_rag.models.retrieval import RerankedChunk


def test_mock_llm():
    llm = create_llm(mock=True)
    assert isinstance(llm, MockLLM)

    resp = llm.generate("What is an inode?")
    assert "[1]" in resp
    assert "inode" in resp.lower()


def test_citation_extractor_regex():
    extractor = CitationExtractor()

    text1 = "Inodes store permissions [1] and block pointers [2]."
    assert extractor.extract_referenced_source_ids(text1) == [1, 2]

    text2 = "Composite citations [1, 2] and [Source 3]."
    assert extractor.extract_referenced_source_ids(text2) == [1, 2, 3]

    text3 = "No citations here."
    assert extractor.extract_referenced_source_ids(text3) == []


def test_resolve_citations():
    builder = ContextBuilder()
    extractor = CitationExtractor()

    c1 = RerankedChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        filename="linux.pdf",
        page=42,
        text="Linux inode implementation details.",
        retrieval_score=0.9,
        rerank_score=0.95,
    )
    c2 = RerankedChunk(
        chunk_id="chk-2",
        document_id="doc-2",
        filename="fs.pdf",
        page=12,
        text="Filesystem superblock structure.",
        retrieval_score=0.8,
        rerank_score=0.88,
    )

    pkg = builder.build_context("query", [c1, c2])

    answer = "Inodes hold Linux metadata [1]."
    citations = extractor.resolve_citations(answer, pkg)

    assert len(citations) == 1
    assert citations[0].source_id == 1
    assert citations[0].filename == "linux.pdf"
    assert citations[0].page == 42
