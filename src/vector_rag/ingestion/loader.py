"""Document loader for discovering, validating, and reading files."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from vector_rag.ingestion.parser import BaseParser, PDFParser, TextParser
from vector_rag.models.document import Document, DocumentMetadata
from vector_rag.utils.errors import DocumentError, UnsupportedFileTypeError
from vector_rag.utils.logging import get_logger

logger = get_logger("ingestion.loader")


class DocumentLoader:
    """Discovers and loads documents from file paths and directories."""

    SUPPORTED_EXTENSIONS: Dict[str, Type[BaseParser]] = {
        ".txt": TextParser,
        ".pdf": PDFParser,
    }

    def __init__(self, parsers: Optional[Dict[str, Type[BaseParser]]] = None) -> None:
        self.parsers = parsers or self.SUPPORTED_EXTENSIONS

    def load_file(self, file_path: str | Path) -> Document:
        """Load and parse a single document file."""
        path = Path(file_path).resolve()

        if not path.is_file():
            raise DocumentError(
                f"File not found: '{file_path}'",
                hint="Verify that the path points to an existing file.",
            )

        suffix = path.suffix.lower()
        if suffix not in self.parsers:
            supported = ", ".join(self.parsers.keys())
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{suffix}' for file '{path.name}'.",
                hint=f"Supported formats are: {supported}",
            )

        file_size = path.stat().st_size
        file_bytes = path.read_bytes()
        content_hash = DocumentMetadata.compute_content_hash(file_bytes)

        parser_cls = self.parsers[suffix]
        parser = parser_cls()
        raw_text, pages_map, page_count = parser.parse(path)

        metadata = DocumentMetadata(
            filename=path.name,
            file_path=str(path),
            file_type=suffix.lstrip("."),
            file_size=file_size,
            content_hash=content_hash,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            page_count=page_count,
        )

        doc = Document(
            metadata=metadata,
            raw_text=raw_text,
            pages=pages_map,
        )
        logger.info(
            "Loaded document '%s' (type=%s, size=%d bytes, pages=%d, hash=%s)",
            metadata.filename,
            metadata.file_type,
            metadata.file_size,
            metadata.page_count,
            metadata.content_hash[:8],
        )
        return doc

    def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = False,
    ) -> List[Document]:
        """Discover and load all supported documents in a directory."""
        directory = Path(dir_path).resolve()

        if not directory.is_dir():
            raise DocumentError(
                f"Directory not found: '{dir_path}'",
                hint="Verify that the path points to a valid directory.",
            )

        documents: List[Document] = []
        pattern = "**/*" if recursive else "*"

        candidate_files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.parsers
        ]

        logger.info(
            "Discovered %d candidate document(s) in '%s' (recursive=%s)",
            len(candidate_files),
            directory,
            recursive,
        )

        for file_path in sorted(candidate_files):
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error("Failed to load document '%s': %s", file_path.name, e)
                raise

        return documents
