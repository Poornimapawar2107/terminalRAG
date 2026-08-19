"""High-level retrieval service coordinating vector search and cross-encoder reranking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vector_rag.config.settings import Settings
from vector_rag.embeddings.embedder import BaseEmbedder, create_embedder
from vector_rag.models.retrieval import RerankedChunk, RetrievedChunk
from vector_rag.retrieval.reranker import BaseReranker, create_reranker
from vector_rag.retrieval.retriever import VectorRetriever
from vector_rag.utils.logging import get_logger, set_request_id
from vector_rag.vectorstore.chroma import ChromaVectorStore

logger = get_logger("services.retrieval")


@dataclass
class SearchResult:
    """Consolidated result containing both initial vector candidates and reranked chunks."""

    query: str
    candidates: List[RetrievedChunk]
    reranked: List[RerankedChunk]


class RetrievalService:
    """Application-level service orchestrating query retrieval and reranking workflows."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        retriever: Optional[VectorRetriever] = None,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self.settings = settings or Settings.load_from_yaml()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=self.settings.storage.chroma_path
        )
        self.embedder = embedder or create_embedder(
            model_name=self.settings.embedding.model,
            batch_size=self.settings.embedding.batch_size,
        )
        self.retriever = retriever or VectorRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            default_top_k=self.settings.retrieval.top_k,
        )
        self.reranker = reranker or create_reranker(
            model_name=self.settings.reranker.model,
            top_n=self.settings.reranker.top_n,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Execute pure vector search retrieval for a query."""
        set_request_id()
        k = top_k or self.settings.retrieval.top_k
        logger.info("Executing vector retrieval (query='%s', top_k=%d)", query, k)
        return self.retriever.retrieve(
            request=query,
            top_k=k,
            filter_metadata=filter_metadata,
        )

    def search_and_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Execute two-stage retrieval:
        1. Vector search -> top_k candidates
        2. Cross-Encoder reranking -> top_n reranked chunks
        """
        set_request_id()
        k = top_k or self.settings.retrieval.top_k
        n = top_n or self.settings.reranker.top_n

        logger.info(
            "Executing two-stage search (query='%s', top_k=%d, top_n=%d)",
            query,
            k,
            n,
        )

        candidates = self.retriever.retrieve(
            request=query,
            top_k=k,
            filter_metadata=filter_metadata,
        )

        reranked = self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_n=n,
        )

        return SearchResult(
            query=query,
            candidates=candidates,
            reranked=reranked,
        )
