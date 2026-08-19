"""Document chunking strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional

from vector_rag.models.chunk import Chunk
from vector_rag.models.document import Document
from vector_rag.utils.errors import ChunkingError
from vector_rag.utils.logging import get_logger

logger = get_logger("ingestion.chunker")


class BaseChunker(ABC):
    """Abstract base class for chunking implementations."""

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into a sequence of Chunk objects."""


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text into chunks using a list of hierarchical separators.
    
    Preserves document page boundaries when per-page text is available on the document.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size}).",
                hint="Set chunk_overlap to roughly 10-20% of chunk_size.",
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        if not separator:
            return list(text)
        return text.split(separator)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text trying separators in decreasing order of hierarchy."""
        final_chunks: List[str] = []
        # Find the first valid separator present in the text
        separator = separators[-1]
        for s in separators:
            if s == "" or s in text:
                separator = s
                break

        splits = self._split_text_with_separator(text, separator)
        new_separators = separators[separators.index(separator) + 1 :] if separator != "" else []

        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Combine smaller pieces up to chunk_size, applying overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        sep = separator if separator != "" else ""

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(sep) if current_doc else 0

            if total + piece_len + sep_len > self.chunk_size and current_doc:
                doc_text = sep.join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)

                # Backtrack for overlap
                while total > self.chunk_overlap and len(current_doc) > 1:
                    removed = current_doc.pop(0)
                    total -= len(removed) + (len(sep) if current_doc else 0)

            current_doc.append(piece)
            total += piece_len + (len(sep) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = sep.join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs

    def chunk_text(self, text: str) -> List[str]:
        """Split a raw text string into chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_text(text, self.separators)

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        If page mapping is present, processes page by page to tag exact page numbers.
        """
        chunks: List[Chunk] = []
        global_index = 0

        if document.pages:
            for page_num in sorted(document.pages.keys()):
                page_text = document.pages[page_num].strip()
                if not page_text:
                    continue

                page_chunks = self.chunk_text(page_text)
                for txt in page_chunks:
                    chunk = Chunk(
                        document_id=document.document_id,
                        chunk_index=global_index,
                        text=txt,
                        page=page_num,
                        metadata={
                            "filename": document.filename,
                            "page": page_num,
                            "file_type": document.metadata.file_type,
                        },
                    )
                    chunks.append(chunk)
                    global_index += 1
        else:
            raw_chunks = self.chunk_text(document.raw_text)
            for txt in raw_chunks:
                chunk = Chunk(
                    document_id=document.document_id,
                    chunk_index=global_index,
                    text=txt,
                    page=1,
                    metadata={
                        "filename": document.filename,
                        "file_type": document.metadata.file_type,
                    },
                )
                chunks.append(chunk)
                global_index += 1

        logger.info(
            "Chunked document '%s' into %d chunk(s) (size=%d, overlap=%d)",
            document.filename,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> BaseChunker:
    """Factory function for creating chunker instances."""
    if strategy.lower() == "recursive":
        return RecursiveCharacterChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    raise ChunkingError(
        f"Unsupported chunking strategy: '{strategy}'",
        hint="Supported strategies: 'recursive'",
    )
