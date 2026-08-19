"""Document parsers for extracting text and structure from raw files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

from vector_rag.utils.errors import DocumentParseError


class BaseParser(ABC):
    """Abstract interface for file parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> Tuple[str, Dict[int, str], int]:
        """
        Parse file and extract content.

        Returns:
            Tuple of (full_raw_text, page_map, page_count)
            where page_map is {page_number: page_text}.
        """


class TextParser(BaseParser):
    """Parser for plain text files (.txt)."""

    def parse(self, file_path: Path) -> Tuple[str, Dict[int, str], int]:
        try:
            # Try utf-8 first, fallback to latin-1 if needed
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="latin-1")

            page_map = {1: text}
            return text, page_map, 1
        except Exception as e:
            raise DocumentParseError(
                f"Failed to parse text file '{file_path.name}': {e}",
                hint="Verify that the file is a readable text file.",
            ) from e


class PDFParser(BaseParser):
    """Parser for PDF documents (.pdf) using pypdf."""

    def parse(self, file_path: Path) -> Tuple[str, Dict[int, str], int]:
        try:
            import pypdf
        except ImportError as e:
            raise DocumentParseError(
                "pypdf package is required for PDF parsing.",
                hint="Install pypdf via 'pip install pypdf' or 'uv pip install pypdf'.",
            ) from e

        try:
            reader = pypdf.PdfReader(str(file_path))
            if reader.is_encrypted:
                try:
                    # Attempt empty password decryption
                    reader.decrypt("")
                except Exception as dec_err:
                    raise DocumentParseError(
                        f"PDF file '{file_path.name}' is password encrypted.",
                        hint="Provide an unencrypted PDF document.",
                    ) from dec_err

            pages_text: Dict[int, str] = {}
            full_text_parts = []

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages_text[idx] = page_text
                if page_text.strip():
                    full_text_parts.append(page_text)

            full_text = "\n\n".join(full_text_parts)
            page_count = len(reader.pages) or 1

            return full_text, pages_text, page_count
        except DocumentParseError:
            raise
        except Exception as e:
            raise DocumentParseError(
                f"Failed to parse PDF file '{file_path.name}': {e}",
                hint="Check whether the PDF is corrupted or formatted improperly.",
            ) from e
