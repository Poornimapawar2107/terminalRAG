"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError
from vector_rag.models.chunk import Chunk
from vector_rag.models.document import Document, DocumentMetadata
from vector_rag.models.response import ContextChunk, ContextPackage, RAGResponse


def test_document_metadata_and_hash():
    content = "Hello Vector RAG world!"
    content_hash = DocumentMetadata.compute_content_hash(content)
    assert len(content_hash) == 64  # SHA256 length

    doc_meta = DocumentMetadata(
        filename="test.txt",
        file_path="/path/test.txt",
        file_type="txt",
        file_size=len(content),
        content_hash=content_hash,
    )
    assert doc_meta.filename == "test.txt"
    assert doc_meta.page_count == 1

    doc = Document(metadata=doc_meta, raw_text=content)
    assert doc.document_id == doc_meta.document_id
    assert doc.filename == "test.txt"


def test_chunk_char_count_auto():
    chunk = Chunk(
        document_id="doc-123",
        chunk_index=0,
        text="Sample chunk content text",
    )
    assert chunk.char_count == len("Sample chunk content text")
    assert chunk.token_count is None


def test_context_package_formatting():
    chunk1 = ContextChunk(
        source_id=1,
        chunk_id="chk-1",
        document_id="doc-1",
        filename="linux.pdf",
        page=42,
        text="Linux uses inodes to store filesystem metadata.",
        score=0.95,
    )
    chunk2 = ContextChunk(
        source_id=2,
        chunk_id="chk-2",
        document_id="doc-2",
        filename="os.pdf",
        page=10,
        text="Virtual memory maps physical pages.",
        score=0.88,
    )

    pkg = ContextPackage(
        query="What is an inode?",
        chunks=[chunk1, chunk2],
    )

    formatted = pkg.format_prompt_context()
    assert "[Source 1] File: linux.pdf | Page: 42" in formatted
    assert "Linux uses inodes to store filesystem metadata." in formatted
    assert "[Source 2] File: os.pdf | Page: 10" in formatted


def test_validation_errors():
    with pytest.raises(ValidationError):
        # Missing required fields
        DocumentMetadata(filename="test.txt")  # type: ignore
