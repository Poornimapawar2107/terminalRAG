"""Response and Context models for Vector RAG."""

from typing import List, Optional
from pydantic import BaseModel, Field

from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk


class ContextChunk(BaseModel):
    """Context chunk packaged with an assigned source ID for prompt injection."""

    source_id: int
    chunk_id: str
    document_id: str
    filename: str
    page: Optional[int] = None
    text: str
    score: float


class ContextPackage(BaseModel):
    """Structured context ready for LLM prompt."""

    query: str
    chunks: List[ContextChunk]
    total_tokens: Optional[int] = None

    def format_prompt_context(self) -> str:
        """Format chunks with explicit Source IDs for the LLM."""
        formatted_blocks = []
        for chunk in self.chunks:
            page_info = f" | Page: {chunk.page}" if chunk.page is not None else ""
            header = f"[Source {chunk.source_id}] File: {chunk.filename}{page_info}"
            formatted_blocks.append(f"{header}\n{chunk.text}")
        return "\n\n".join(formatted_blocks)


class Citation(BaseModel):
    """Source citation mapping back to document and page."""

    source_id: int
    filename: str
    page: Optional[int] = None
    chunk_id: str
    snippet: Optional[str] = None


class GenerationRequest(BaseModel):
    """Input payload for LLM generation."""

    query: str
    context: ContextPackage
    system_prompt: Optional[str] = None


class RAGResponse(BaseModel):
    """Complete final response returned by RAGService."""

    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: List[RerankedChunk] = Field(default_factory=list)
    request_id: Optional[str] = None
