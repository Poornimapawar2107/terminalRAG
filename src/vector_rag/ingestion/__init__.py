"""Document ingestion package."""

from vector_rag.ingestion.chunker import (
    BaseChunker,
    RecursiveCharacterChunker,
    create_chunker,
)
from vector_rag.ingestion.loader import DocumentLoader
from vector_rag.ingestion.parser import BaseParser, PDFParser, TextParser
from vector_rag.ingestion.service import DocumentService, IngestionResult

__all__ = [
    "BaseParser",
    "TextParser",
    "PDFParser",
    "DocumentLoader",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "create_chunker",
    "DocumentService",
    "IngestionResult",
]
