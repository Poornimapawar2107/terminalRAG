"""SQLite storage repository for document metadata and chunks."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from vector_rag.models.chunk import Chunk
from vector_rag.models.document import DocumentMetadata
from vector_rag.utils.errors import StorageError
from vector_rag.utils.logging import get_logger

logger = get_logger("storage.sqlite")


class SQLiteStorage:
    """Relational SQLite storage manager for documents and chunks."""

    def __init__(self, db_path: str | Path = "data/app.db") -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except Exception as e:
            raise StorageError(
                f"Failed to connect to SQLite database at '{self.db_path}': {e}",
                hint="Check database path and write permissions.",
            ) from e

    def init_db(self) -> None:
        """Create tables and indexes if they do not already exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL UNIQUE,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        title TEXT,
                        page_count INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                    ON documents(content_hash)
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        token_count INTEGER,
                        page INTEGER,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id)
                    """
                )
                conn.commit()
            logger.debug("Initialized SQLite schema at '%s'", self.db_path)
        except Exception as e:
            raise StorageError(
                f"Failed to initialize SQLite database tables: {e}",
                hint="Verify database integrity.",
            ) from e

    def save_document(self, doc: DocumentMetadata) -> None:
        """Insert or replace a document metadata record."""
        query = """
            INSERT INTO documents (
                id, filename, file_path, file_type, file_size,
                content_hash, title, page_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename,
                file_path=excluded.file_path,
                file_type=excluded.file_type,
                file_size=excluded.file_size,
                content_hash=excluded.content_hash,
                title=excluded.title,
                page_count=excluded.page_count,
                updated_at=excluded.updated_at
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    query,
                    (
                        doc.document_id,
                        doc.filename,
                        doc.file_path,
                        doc.file_type,
                        doc.file_size,
                        doc.content_hash,
                        doc.title,
                        doc.page_count,
                        doc.created_at.isoformat(),
                        doc.updated_at.isoformat(),
                    ),
                )
                conn.commit()
            logger.info("Saved document metadata for '%s' (id=%s)", doc.filename, doc.document_id)
        except Exception as e:
            raise StorageError(
                f"Failed to save document metadata for '{doc.filename}': {e}"
            ) from e

    def get_document_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """Fetch document metadata by primary ID."""
        query = "SELECT * FROM documents WHERE id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                row = cursor.execute(query, (document_id,)).fetchone()
                if not row:
                    return None
                return self._row_to_document_metadata(row)
        except Exception as e:
            raise StorageError(f"Failed to fetch document by ID '{document_id}': {e}") from e

    def get_document_by_path(self, file_path: str) -> Optional[DocumentMetadata]:
        """Fetch document metadata by file path."""
        query = "SELECT * FROM documents WHERE file_path = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                row = cursor.execute(query, (str(file_path),)).fetchone()
                if not row:
                    return None
                return self._row_to_document_metadata(row)
        except Exception as e:
            raise StorageError(f"Failed to fetch document by path '{file_path}': {e}") from e

    def get_document_by_hash(self, content_hash: str) -> Optional[DocumentMetadata]:
        """Fetch document metadata by SHA256 content hash."""
        query = "SELECT * FROM documents WHERE content_hash = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                row = cursor.execute(query, (content_hash,)).fetchone()
                if not row:
                    return None
                return self._row_to_document_metadata(row)
        except Exception as e:
            raise StorageError(f"Failed to fetch document by hash '{content_hash}': {e}") from e

    def list_documents(self) -> List[DocumentMetadata]:
        """List all ingested document metadata records."""
        query = "SELECT * FROM documents ORDER BY created_at DESC"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(query).fetchall()
                return [self._row_to_document_metadata(r) for r in rows]
        except Exception as e:
            raise StorageError(f"Failed to list documents: {e}") from e

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated chunks."""
        query = "DELETE FROM documents WHERE id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (document_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            if deleted:
                logger.info("Deleted document and cascading chunks for ID '%s'", document_id)
            return deleted
        except Exception as e:
            raise StorageError(f"Failed to delete document '{document_id}': {e}") from e

    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Batch insert chunk records."""
        if not chunks:
            return

        query = """
            INSERT INTO chunks (
                id, document_id, chunk_index, text, char_count,
                token_count, page, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        records = [
            (
                c.id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.char_count,
                c.token_count,
                c.page,
                json.dumps(c.metadata) if c.metadata else None,
                c.created_at.isoformat(),
            )
            for c in chunks
        ]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, records)
                conn.commit()
            logger.info("Saved %d chunks into SQLite storage.", len(chunks))
        except Exception as e:
            raise StorageError(f"Failed to save chunks: {e}") from e

    def get_chunks_by_document_id(self, document_id: str) -> List[Chunk]:
        """Fetch all chunks belonging to a document ordered by index."""
        query = "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(query, (document_id,)).fetchall()
                return [self._row_to_chunk(r) for r in rows]
        except Exception as e:
            raise StorageError(
                f"Failed to fetch chunks for document '{document_id}': {e}"
            ) from e

    def delete_chunks_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        query = "DELETE FROM chunks WHERE document_id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (document_id,))
                count = cursor.rowcount
                conn.commit()
            return count
        except Exception as e:
            raise StorageError(
                f"Failed to delete chunks for document '{document_id}': {e}"
            ) from e

    @staticmethod
    def _row_to_document_metadata(row: sqlite3.Row) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            title=row["title"],
            page_count=row["page_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            char_count=row["char_count"],
            token_count=row["token_count"],
            page=row["page"],
            metadata=meta,
            created_at=row["created_at"],
        )
