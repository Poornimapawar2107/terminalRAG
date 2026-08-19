"""Vector similarity retriever implementation."""

from typing import Any, Dict, List, Optional

from vector_rag.embeddings.embedder import BaseEmbedder
from vector_rag.models.retrieval import RetrievalRequest, RetrievedChunk
from vector_rag.utils.errors import RetrievalError
from vector_rag.utils.logging import get_logger
from vector_rag.vectorstore.chroma import ChromaVectorStore

logger = get_logger("retrieval.retriever")


class VectorRetriever:
    """Retrieves top-K most similar text chunks for a query from vector storage."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: BaseEmbedder,
        default_top_k: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_top_k = default_top_k

    def retrieve(
        self,
        request: RetrievalRequest | str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve nearest neighbor chunks for a query.
        
        Args:
            request: Either a string query or a RetrievalRequest object.
            top_k: Number of candidates to retrieve.
            filter_metadata: Optional metadata filter dict for ChromaDB.
        """
        if isinstance(request, str):
            req = RetrievalRequest(
                query=request,
                top_k=top_k or self.default_top_k,
                filter_metadata=filter_metadata,
            )
        else:
            req = request
            if top_k is not None:
                req.top_k = top_k
            if filter_metadata is not None:
                req.filter_metadata = filter_metadata

        query_str = req.query.strip()
        if not query_str:
            return []

        try:
            logger.info("Embedding query for vector retrieval: '%s'", query_str[:60])
            query_embedding = self.embedder.embed_query(query_str)

            logger.info(
                "Querying ChromaDB for top_k=%d candidate chunks...", req.top_k
            )
            chunks = self.vector_store.query_vectors(
                query_embedding=query_embedding,
                top_k=req.top_k,
                filter_metadata=req.filter_metadata,
            )
            logger.info("Vector retrieval found %d matching chunk(s).", len(chunks))
            return chunks
        except Exception as e:
            raise RetrievalError(
                f"Vector retrieval failed for query '{query_str}': {e}"
            ) from e
