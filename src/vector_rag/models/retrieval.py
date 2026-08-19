"""Retrieval request and result models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Query payload for vector retrieval."""

    query: str
    top_k: int = Field(default=10, ge=1)
    filter_metadata: Optional[Dict[str, Any]] = None


class RetrievedChunk(BaseModel):
    """Chunk retrieved from vector search with similarity score."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RerankedChunk(BaseModel):
    """Chunk after passing through cross-encoder reranker."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    retrieval_score: float
    rerank_score: float
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
