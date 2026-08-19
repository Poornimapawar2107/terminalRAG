"""Reranking model abstraction and cross-encoder implementations."""

from abc import ABC, abstractmethod
import math
from typing import List, Optional

from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk
from vector_rag.utils.errors import RerankingError
from vector_rag.utils.logging import get_logger

logger = get_logger("retrieval.reranker")


class BaseReranker(ABC):
    """Abstract interface for cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RerankedChunk]:
        """Rerank candidate retrieved chunks using cross-encoder relevance scores."""


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit testing."""

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RerankedChunk]:
        if not chunks:
            return []

        n = top_n or self.top_n
        reranked: List[RerankedChunk] = []

        # Sort slightly adjusting retrieval score to simulate reranking effect
        for idx, chunk in enumerate(chunks):
            pseudo_score = round(min(1.0, chunk.score * 1.05), 4)
            reranked.append(
                RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    retrieval_score=chunk.score,
                    rerank_score=pseudo_score,
                    page=chunk.page,
                    metadata=chunk.metadata,
                )
            )

        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:n]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker leveraging SentenceTransformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self._model = None

    def _get_model(self):
        """Lazy load the cross encoder model."""
        if self._model is None:
            try:
                import os
                from sentence_transformers import CrossEncoder

                # Suppress HuggingFace/tqdm noise
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                try:
                    from transformers import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except ImportError:
                    pass

                logger.info(
                    "Loading CrossEncoder model '%s' (device=%s)...",
                    self.model_name,
                    self.device or "auto",
                )
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                raise RerankingError(
                    f"Failed to load CrossEncoder model '{self.model_name}': {e}",
                    hint="Verify cross-encoder model name or internet access for downloading weights.",
                ) from e
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Map raw logits into a calibrated [0, 1] probability range."""
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RerankedChunk]:
        if not chunks:
            return []

        n = top_n or self.top_n
        model = self._get_model()

        try:
            pairs = [[query, chunk.text] for chunk in chunks]
            raw_scores = model.predict(pairs, show_progress_bar=False)

            reranked: List[RerankedChunk] = []
            for chunk, score in zip(chunks, raw_scores):
                # Normalize logit to [0, 1]
                norm_score = round(self._sigmoid(float(score)), 4)
                reranked.append(
                    RerankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        text=chunk.text,
                        retrieval_score=chunk.score,
                        rerank_score=norm_score,
                        page=chunk.page,
                        metadata=chunk.metadata,
                    )
                )

            # Sort descending by cross-encoder score
            reranked.sort(key=lambda x: x.rerank_score, reverse=True)
            result = reranked[:n]

            logger.info(
                "Reranked %d candidates down to top_n=%d chunks.",
                len(chunks),
                len(result),
            )
            return result
        except Exception as e:
            raise RerankingError(
                f"Cross-encoder reranking failed for query '{query}': {e}"
            ) from e


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
    mock: bool = False,
) -> BaseReranker:
    """Factory to instantiate the appropriate reranker."""
    if mock:
        return MockReranker(top_n=top_n)
    return CrossEncoderReranker(model_name=model_name, top_n=top_n)
