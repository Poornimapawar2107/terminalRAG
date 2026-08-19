"""ChromaDB vector store implementation for managing dense vector embeddings and metadata."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from vector_rag.models.chunk import Chunk
from vector_rag.models.retrieval import RetrievedChunk
from vector_rag.utils.errors import VectorStoreError
from vector_rag.utils.logging import get_logger

logger = get_logger("vectorstore.chroma")


class ChromaVectorStore:
    """Persistent ChromaDB client and collection wrapper."""

    DEFAULT_COLLECTION_NAME = "vector_rag_collection"

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy load Chroma persistent client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}",
                    hint="Check disk permissions and ensure chromadb is installed.",
                ) from e
        return self._client

    def _get_collection(self):
        """Retrieve or create the active vector collection."""
        if self._collection is None:
            client = self._get_client()
            try:
                # Use cosine distance metric for normalized vector embeddings
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to get or create ChromaDB collection '{self.collection_name}': {e}"
                ) from e
        return self._collection

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """Insert or update chunk vectors and their metadata into ChromaDB."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        collection = self._get_collection()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)

            meta: Dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "filename": chunk.metadata.get("filename", ""),
                "file_type": chunk.metadata.get("file_type", ""),
            }
            if chunk.page is not None:
                meta["page"] = int(chunk.page)

            metadatas.append(meta)

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Indexed %d chunk vectors into ChromaDB collection '%s'", len(chunks), self.collection_name)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert chunks into ChromaDB: {e}"
            ) from e

    def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Query top-K nearest neighbors from ChromaDB given a query vector."""
        collection = self._get_collection()

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata if filter_metadata else None,
                include=["documents", "metadatas", "distances"],
            )

            retrieved: List[RetrievedChunk] = []

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                # For cosine distance (in range [0, 2]), similarity = 1 - (dist / 2) or 1 - dist
                similarity_score = max(0.0, 1.0 - float(dist))

                page_val = meta.get("page")
                page_int = int(page_val) if page_val is not None else None

                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        document_id=str(meta.get("document_id", "")),
                        filename=str(meta.get("filename", "")),
                        text=doc_text,
                        score=round(similarity_score, 4),
                        page=page_int,
                        metadata=meta,
                    )
                )

            logger.info("ChromaDB query returned %d nearest chunks.", len(retrieved))
            return retrieved
        except Exception as e:
            raise VectorStoreError(f"Failed to query ChromaDB collection: {e}") from e

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunk vectors associated with a specific document ID."""
        collection = self._get_collection()
        try:
            collection.delete(where={"document_id": document_id})
            logger.info("Deleted vector records for document_id '%s' from ChromaDB.", document_id)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete vectors for document '{document_id}': {e}"
            ) from e

    def count(self) -> int:
        """Get total number of vectors in the collection."""
        collection = self._get_collection()
        try:
            return collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to count vectors in collection: {e}") from e

    def reset(self) -> None:
        """Clear all records from the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info("Reset ChromaDB collection '%s'.", self.collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to reset collection: {e}") from e
