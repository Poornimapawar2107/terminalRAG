"""Context construction layer for preparing structured, budgeted prompt context for LLMs."""

import math
from typing import Dict, List, Optional, Union

from vector_rag.models.response import Citation, ContextChunk, ContextPackage
from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk
from vector_rag.utils.logging import get_logger

logger = get_logger("generation.context")

AnyRetrievedChunk = Union[RerankedChunk, RetrievedChunk]


class ContextBuilder:
    """
    Transforms retrieved/reranked chunks into structured ContextPackages.
    
    Assigns clear Source IDs, enforces token budgets, and produces source citation maps.
    """

    def __init__(
        self,
        max_context_tokens: int = 2000,
        approx_chars_per_token: float = 4.0,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.approx_chars_per_token = approx_chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on average character-to-token ratio."""
        return max(1, math.ceil(len(text) / self.approx_chars_per_token))

    def build_context(
        self,
        query: str,
        chunks: List[AnyRetrievedChunk],
        max_tokens: Optional[int] = None,
    ) -> ContextPackage:
        """
        Package chunks into a prompt-ready ContextPackage with Source IDs and token limits.
        
        Args:
            query: User's original search query.
            chunks: List of retrieved or reranked chunks.
            max_tokens: Optional override for context token budget.
        """
        token_budget = max_tokens or self.max_context_tokens
        context_chunks: List[ContextChunk] = []
        accumulated_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            chunk_score = (
                chunk.rerank_score
                if hasattr(chunk, "rerank_score")
                else chunk.score
            )

            # Estimate token footprint of chunk
            estimated_tokens = self._estimate_tokens(chunk.text)
            if accumulated_tokens + estimated_tokens > token_budget and context_chunks:
                logger.info(
                    "Context token budget (%d) reached at chunk %d. Truncating context.",
                    token_budget,
                    idx,
                )
                break

            context_chunk = ContextChunk(
                source_id=idx,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                text=chunk.text.strip(),
                score=chunk_score,
            )
            context_chunks.append(context_chunk)
            accumulated_tokens += estimated_tokens

        package = ContextPackage(
            query=query,
            chunks=context_chunks,
            total_tokens=accumulated_tokens,
        )

        logger.info(
            "Built ContextPackage with %d chunk(s), ~%d tokens for query '%s'",
            len(context_chunks),
            accumulated_tokens,
            query[:50],
        )
        return package

    def extract_citations(self, package: ContextPackage) -> List[Citation]:
        """Generate Citation objects for all sources in the context package."""
        citations: List[Citation] = []
        for chunk in package.chunks:
            snippet = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            citations.append(
                Citation(
                    source_id=chunk.source_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                )
            )
        return citations

    def get_citation_map(self, package: ContextPackage) -> Dict[int, Citation]:
        """Return a mapping of source_id -> Citation."""
        citations = self.extract_citations(package)
        return {c.source_id: c for c in citations}
