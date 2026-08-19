"""Document ingestion service coordinating loaders, chunkers, embedders, and storage."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from vector_rag.config.settings import Settings
from vector_rag.embeddings.embedder import BaseEmbedder, create_embedder
from vector_rag.ingestion.chunker import BaseChunker, create_chunker
from vector_rag.ingestion.loader import DocumentLoader
from vector_rag.models.document import Document, DocumentMetadata
from vector_rag.storage.sqlite import SQLiteStorage
from vector_rag.utils.errors import DocumentError
from vector_rag.utils.logging import get_logger, set_request_id
from vector_rag.vectorstore.chroma import ChromaVectorStore

logger = get_logger("services.document")


@dataclass
class IngestionResult:
    """Summary of document ingestion workflow."""

    discovered: int = 0
    parsed: int = 0
    skipped: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_indexed: int = 0
    documents: List[DocumentMetadata] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = []


class DocumentService:
    """High-level service orchestrating document ingestion and storage."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        storage: Optional[SQLiteStorage] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        chunker: Optional[BaseChunker] = None,
        loader: Optional[DocumentLoader] = None,
    ) -> None:
        self.settings = settings or Settings.load_from_yaml()
        self.storage = storage or SQLiteStorage(self.settings.storage.sqlite_path)
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=self.settings.storage.chroma_path
        )
        self.embedder = embedder or create_embedder(
            model_name=self.settings.embedding.model,
            batch_size=self.settings.embedding.batch_size,
        )
        self.chunker = chunker or create_chunker(
            strategy=self.settings.chunking.strategy,
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
        )
        self.loader = loader or DocumentLoader()

    def ingest_path(self, target_path: str | Path, force: bool = False) -> IngestionResult:
        """
        Ingest a file or all files in a directory.
        
        Args:
            target_path: Path to file or directory.
            force: If True, re-ingests even if content hash is unchanged.
        """
        set_request_id()
        path = Path(target_path).resolve()
        result = IngestionResult()

        if path.is_file():
            raw_docs = [self.loader.load_file(path)]
        elif path.is_dir():
            raw_docs = self.loader.load_directory(path, recursive=True)
        else:
            raise DocumentError(
                f"Path '{target_path}' does not exist.",
                hint="Provide a valid file or directory path.",
            )

        result.discovered = len(raw_docs)

        for doc in raw_docs:
            existing = self.storage.get_document_by_path(doc.metadata.file_path)

            if existing and existing.content_hash == doc.metadata.content_hash and not force:
                logger.info("Skipping '%s': identical content hash already ingested.", doc.filename)
                result.skipped += 1
                result.documents.append(existing)
                continue

            # If file was modified, clean up previous chunks and vectors first
            if existing:
                logger.info("File '%s' was modified. Re-ingesting...", doc.filename)
                self.vector_store.delete_by_document_id(existing.document_id)
                self.storage.delete_document(existing.document_id)

            # 1. Save metadata to SQLite
            self.storage.save_document(doc.metadata)
            result.parsed += 1

            # 2. Chunk document text
            chunks = self.chunker.chunk_document(doc)
            result.chunks_created += len(chunks)

            if chunks:
                # 3. Save chunks in SQLite
                self.storage.save_chunks(chunks)

                # 4. Generate embeddings
                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedder.embed_documents(chunk_texts)
                result.embeddings_generated += len(embeddings)

                # 5. Index into ChromaDB
                self.vector_store.add_chunks(chunks, embeddings)
                result.vectors_indexed += len(chunks)

            result.documents.append(doc.metadata)

        logger.info(
            "Ingestion completed: Discovered=%d, Parsed=%d, Skipped=%d, Chunks=%d, Vectors=%d",
            result.discovered,
            result.parsed,
            result.skipped,
            result.chunks_created,
            result.vectors_indexed,
        )
        return result

    def list_documents(self) -> List[DocumentMetadata]:
        """List all ingested documents in SQLite."""
        return self.storage.list_documents()

    def get_document_chunks(self, document_id: str):
        """Fetch all chunks for a specific document from SQLite."""
        return self.storage.get_chunks_by_document_id(document_id)

    def delete_document(self, document_id: str) -> bool:
        """Delete document from SQLite and ChromaDB."""
        self.vector_store.delete_by_document_id(document_id)
        return self.storage.delete_document(document_id)
