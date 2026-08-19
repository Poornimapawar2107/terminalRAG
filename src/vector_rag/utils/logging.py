"""Logging utilities for Vector RAG."""

import contextvars
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import uuid

# Context variable for correlating operations (e.g. queries, ingest batches)
current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_request_id", default=None
)


class RequestIdFilter(logging.Filter):
    """Logging filter that injects the current request_id into the log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        req_id = current_request_id.get()
        record.request_id = req_id if req_id else "-"
        return True


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the active request_id context variable."""
    req_id = request_id or str(uuid.uuid4())[:8]
    current_request_id.set(req_id)
    return req_id


def clear_request_id() -> None:
    """Clear active request_id."""
    current_request_id.set(None)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure application-wide logging.

    All output goes to ``log_file`` only — no console handler is attached,
    so the terminal remains clean for Rich UI output.
    Third-party Python warnings (e.g. from transformers) are also redirected
    to the logging system instead of being printed to stderr.
    """
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on re-init
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    req_filter = RequestIdFilter()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [req:%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File-only handler — keeps terminal clean
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(req_filter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    else:
        # Fallback: discard all log output when no file is configured
        root_logger.addHandler(logging.NullHandler())

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "chromadb", "httpx", "httpcore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Redirect Python warnings (transformers UserWarnings etc.) to log file
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=UserWarning)

    logger = logging.getLogger("vector_rag")
    logger.info("Logging configured. Level=%s, File=%s", level, log_file or "None")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger namespaced under vector_rag."""
    if not name.startswith("vector_rag"):
        name = f"vector_rag.{name}"
    return logging.getLogger(name)
