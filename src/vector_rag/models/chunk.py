"""Chunk model definitions."""

from datetime import datetime
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents an extracted text chunk from a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    text: str
    char_count: int = 0
    token_count: Optional[int] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count and self.text:
            self.char_count = len(self.text)
