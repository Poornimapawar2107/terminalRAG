"""Unit tests for DocumentLoader and file parsers."""

from pathlib import Path
import pytest
from vector_rag.ingestion.loader import DocumentLoader
from vector_rag.ingestion.parser import TextParser
from vector_rag.utils.errors import (
    DocumentError,
    DocumentParseError,
    UnsupportedFileTypeError,
)


def test_text_parser(tmp_path: Path):
    txt_file = tmp_path / "hello.txt"
    content = "Hello World! This is a test document."
    txt_file.write_text(content, encoding="utf-8")

    parser = TextParser()
    raw_text, page_map, page_count = parser.parse(txt_file)

    assert raw_text == content
    assert page_count == 1
    assert page_map[1] == content


def test_load_text_file(tmp_path: Path):
    txt_file = tmp_path / "linux_inodes.txt"
    content = "Linux uses inodes to store filesystem metadata."
    txt_file.write_text(content, encoding="utf-8")

    loader = DocumentLoader()
    doc = loader.load_file(txt_file)

    assert doc.filename == "linux_inodes.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.page_count == 1
    assert doc.metadata.title == "Linux Inodes"
    assert doc.raw_text == content
    assert len(doc.metadata.content_hash) == 64
    assert doc.pages[1] == content


def test_load_directory(tmp_path: Path):
    (tmp_path / "file1.txt").write_text("Content 1", encoding="utf-8")
    (tmp_path / "file2.txt").write_text("Content 2", encoding="utf-8")
    # Unsupported file should be ignored during directory scan
    (tmp_path / "image.png").write_bytes(b"PNGDATA")

    loader = DocumentLoader()
    docs = loader.load_directory(tmp_path)

    assert len(docs) == 2
    filenames = [d.filename for d in docs]
    assert "file1.txt" in filenames
    assert "file2.txt" in filenames


def test_unsupported_file_type_error(tmp_path: Path):
    unsupported = tmp_path / "document.docx"
    unsupported.write_text("test", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        loader.load_file(unsupported)

    assert "Unsupported file format '.docx'" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_nonexistent_file_error():
    loader = DocumentLoader()
    with pytest.raises(DocumentError) as exc_info:
        loader.load_file("non_existent_file.txt")

    assert "File not found" in str(exc_info.value)


def test_fixture_loading():
    fixture_path = Path("tests/fixtures/sample.txt")
    assert fixture_path.exists()

    loader = DocumentLoader()
    doc = loader.load_file(fixture_path)
    assert "inode" in doc.raw_text
    assert doc.metadata.file_type == "txt"
