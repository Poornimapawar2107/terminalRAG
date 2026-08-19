"""Comprehensive CLI unit tests exercising all subcommands with mocked services."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from vector_rag.cli.main import (
    cmd_delete,
    cmd_ingest,
    cmd_list,
    cmd_query,
    cmd_status,
    main,
    print_error_panel,
)
from vector_rag.ingestion.service import IngestionResult
from vector_rag.models.document import DocumentMetadata
from vector_rag.models.response import Citation, RAGResponse
from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk
from vector_rag.utils.errors import (
    ConfigurationError,
    DocumentError,
    GenerationError,
    VectorRAGError,
)


# ─── Fixtures ──────────────────────────────────────────────────────


def _make_args(**kwargs):
    """Create a mock argparse Namespace with defaults."""
    defaults = {"config": "config/config.yaml", "command": None}
    defaults.update(kwargs)
    return MagicMock(**defaults)


def _make_doc_meta(filename="test.txt"):
    from datetime import datetime

    return DocumentMetadata(
        filename=filename,
        file_path=f"/docs/{filename}",
        file_type="txt",
        file_size=100,
        content_hash="abc123",
        page_count=1,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    )


# ─── Help and Argument Parsing ─────────────────────────────────────


def test_main_no_command(capsys):
    """Running without a command prints help."""
    with patch("sys.argv", ["vector-rag"]):
        main()
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower() or "vector-rag" in captured.out.lower()


def test_main_query_subcommand_help(capsys):
    """Query subcommand --help exits cleanly."""
    with patch("sys.argv", ["vector-rag", "query", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_main_ingest_subcommand_help(capsys):
    """Ingest subcommand --help exits cleanly."""
    with patch("sys.argv", ["vector-rag", "ingest", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


# ─── Error Handling ────────────────────────────────────────────────


def test_print_error_panel_with_hint(capsys):
    """Error panel renders hint text."""
    err = DocumentError("File missing: data.pdf", hint="Check the file path.")
    print_error_panel(err)
    captured = capsys.readouterr()
    assert "DocumentError" in captured.out
    assert "File missing" in captured.out


def test_print_error_panel_without_hint(capsys):
    """Error panel renders cleanly without a hint."""
    err = VectorRAGError("Generic error occurred")
    print_error_panel(err)
    captured = capsys.readouterr()
    assert "Generic error occurred" in captured.out


def test_print_error_panel_non_domain_error(capsys):
    """Non-VectorRAG exceptions are displayed as 'Unexpected Error'."""
    err = RuntimeError("Something unexpected")
    print_error_panel(err)
    captured = capsys.readouterr()
    assert "Unexpected Error" in captured.out
    assert "Something unexpected" in captured.out


# ─── cmd_ingest ────────────────────────────────────────────────────


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_ingest_success(mock_settings, mock_logging, mock_svc_cls, capsys, tmp_path):
    """Ingest command displays summary table on success."""
    mock_settings.return_value = MagicMock()

    mock_svc = MagicMock()
    mock_svc.ingest_path.return_value = IngestionResult(
        discovered=2,
        parsed=2,
        skipped=0,
        chunks_created=4,
        embeddings_generated=4,
        vectors_indexed=4,
    )
    mock_svc.vector_store.count.return_value = 4
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="ingest", path=str(tmp_path), force=False)
    cmd_ingest(args)

    captured = capsys.readouterr()
    assert "Ingestion Complete" in captured.out
    assert "2" in captured.out  # discovered or parsed count


# ─── cmd_list ──────────────────────────────────────────────────────


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_list_with_docs(mock_settings, mock_logging, mock_svc_cls, capsys):
    """List command displays document table."""
    mock_settings.return_value = MagicMock()
    mock_svc = MagicMock()
    mock_svc.list_documents.return_value = [
        _make_doc_meta("file1.txt"),
        _make_doc_meta("file2.txt"),
    ]
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="list")
    cmd_list(args)

    captured = capsys.readouterr()
    assert "file1.txt" in captured.out
    assert "file2.txt" in captured.out
    assert "2 total" in captured.out


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_list_no_docs(mock_settings, mock_logging, mock_svc_cls, capsys):
    """List command with no documents shows empty message."""
    mock_settings.return_value = MagicMock()
    mock_svc = MagicMock()
    mock_svc.list_documents.return_value = []
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="list")
    cmd_list(args)

    captured = capsys.readouterr()
    assert "No documents" in captured.out


# ─── cmd_delete ────────────────────────────────────────────────────


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_delete_by_filename(mock_settings, mock_logging, mock_svc_cls, capsys):
    """Delete command by filename succeeds."""
    mock_settings.return_value = MagicMock()
    doc = _make_doc_meta("report.txt")
    mock_svc = MagicMock()
    mock_svc.list_documents.return_value = [doc]
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="delete", identifier="report.txt")
    cmd_delete(args)

    captured = capsys.readouterr()
    assert "Document Deleted" in captured.out
    assert "report.txt" in captured.out
    mock_svc.delete_document.assert_called_once_with(doc.document_id)


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_delete_not_found(mock_settings, mock_logging, mock_svc_cls, capsys):
    """Delete command for non-existent doc shows not-found message."""
    mock_settings.return_value = MagicMock()
    mock_svc = MagicMock()
    mock_svc.list_documents.return_value = []
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="delete", identifier="nonexistent.pdf")
    cmd_delete(args)

    captured = capsys.readouterr()
    assert "not found" in captured.out


# ─── cmd_status ────────────────────────────────────────────────────


@patch("vector_rag.cli.main.DocumentService")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_cmd_status(mock_settings, mock_svc_cls, capsys):
    """Status command prints system info."""
    mock_cfg = MagicMock()
    mock_cfg.embedding.model = "BAAI/bge-small-en-v1.5"
    mock_cfg.reranker.model = "ms-marco-MiniLM"
    mock_cfg.generation.model = "Qwen/Qwen2.5-0.5B-Instruct"
    mock_cfg.chunking.strategy = "recursive"
    mock_cfg.chunking.chunk_size = 800
    mock_cfg.chunking.chunk_overlap = 120
    mock_settings.return_value = mock_cfg

    mock_svc = MagicMock()
    mock_svc.list_documents.return_value = [_make_doc_meta()]
    mock_svc.vector_store.count.return_value = 5
    mock_svc_cls.return_value = mock_svc

    args = _make_args(command="status")
    cmd_status(args)

    captured = capsys.readouterr()
    assert "System Status" in captured.out
    assert "1 document" in captured.out
    assert "5 vector" in captured.out


# ─── Main exception handler ───────────────────────────────────────


@patch("vector_rag.cli.main.cmd_ingest")
@patch("vector_rag.cli.main.setup_logging")
@patch("vector_rag.cli.main.Settings.load_from_yaml")
def test_main_catches_domain_errors(mock_settings, mock_logging, mock_cmd, capsys):
    """Main function catches VectorRAGError and exits with code 1."""
    mock_cmd.side_effect = ConfigurationError("bad config", hint="Fix YAML")

    with patch("sys.argv", ["vector-rag", "ingest", "test_path"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
