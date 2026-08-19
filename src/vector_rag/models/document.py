"""Document models for Vector RAG."""

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Optional
import uuid
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing a parsed document."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    file_type: str
    file_size: int
    content_hash: str
    title: Optional[str] = None
    page_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def compute_content_hash(cls, content: bytes | str) -> str:
        """Compute SHA256 hash of document raw content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()


class Document(BaseModel):
    """Full in-memory representation of a document during ingestion."""

    metadata: DocumentMetadata
    raw_text: str
    pages: dict[int, str] = Field(default_factory=dict)

    @property
    def document_id(self) -> str:
        return self.metadata.document_id

    @property
    def filename(self) -> str:
        return self.metadata.filename


class IngestionRequest(BaseModel):
    """Request payload to ingest one or more documents."""

    source_path: Path
    recursive: bool = False
