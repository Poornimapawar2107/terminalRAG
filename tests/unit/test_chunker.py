"""Unit tests for text chunking implementations."""

import pytest
from vector_rag.ingestion.chunker import RecursiveCharacterChunker, create_chunker
from vector_rag.models.document import Document, DocumentMetadata
from vector_rag.utils.errors import ChunkingError


def make_doc(text: str, pages: dict = None) -> Document:
    meta = DocumentMetadata(
        filename="test_doc.txt",
        file_path="/docs/test_doc.txt",
        file_type="txt",
        file_size=len(text),
        content_hash="testcontenthash",
        page_count=len(pages) if pages else 1,
    )
    return Document(metadata=meta, raw_text=text, pages=pages or {1: text})


def test_chunker_validation():
    with pytest.raises(ChunkingError):
        RecursiveCharacterChunker(chunk_size=0)

    with pytest.raises(ChunkingError):
        RecursiveCharacterChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ChunkingError):
        create_chunker(strategy="unknown_strategy")


def test_chunk_short_text():
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    doc = make_doc("Short content.")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Short content."
    assert chunks[0].char_count == len("Short content.")
    assert chunks[0].page == 1


def test_chunk_long_text_and_overlap():
    paragraphs = [
        "First paragraph introducing the concept of operating system kernels.",
        "Second paragraph explaining how processes and virtual memory operate.",
        "Third paragraph discussing filesystem inodes and block storage pointers.",
        "Fourth paragraph discussing device drivers and system call interfaces.",
    ]
    full_text = "\n\n".join(paragraphs)

    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=30)
    doc = make_doc(full_text)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    # Ensure all chunks are non-empty and sequential indices
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert len(c.text) > 0
        assert c.document_id == doc.document_id


def test_page_aware_chunking():
    pages = {
        1: "This is page one discussing initial definitions and introduction.",
        2: "This is page two detailing memory management algorithms and cache replacement policies in modern systems.",
    }
    raw_text = "\n\n".join(pages.values())
    doc = make_doc(raw_text, pages=pages)

    chunker = RecursiveCharacterChunker(chunk_size=70, chunk_overlap=15)
    chunks = chunker.chunk_document(doc)

    page_numbers = {c.page for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    for c in chunks:
        assert c.metadata["page"] == c.page
        assert c.metadata["filename"] == "test_doc.txt"
