"""Embedding model abstraction layer and concrete implementations."""

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List, Optional

from vector_rag.utils.errors import EmbeddingError
from vector_rag.utils.logging import get_logger

logger = get_logger("embeddings.embedder")


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vector embeddings for a list of document strings."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute dense vector embedding for a single user query string."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of this embedding model."""


class MockEmbedder(BaseEmbedder):
    """
    Deterministic in-memory mock embedder for fast unit testing.
    
    Generates normalized pseudo-embeddings derived from SHA256 hashes of text.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> List[float]:
        """Convert string to deterministic normalized vector."""
        vec = []
        for i in range(self._dim):
            h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "big") / (2**32 - 1)) * 2.0 - 1.0
            vec.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Concrete embedder leveraging HuggingFace sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dim: Optional[int] = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                import os
                from sentence_transformers import SentenceTransformer
                
                # Suppress HuggingFace/tqdm noise
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                try:
                    from transformers import logging as hf_logging
                    hf_logging.set_verbosity_error()
                except ImportError:
                    pass

                logger.info(
                    "Loading embedding model '%s' (device=%s)...",
                    self.model_name,
                    self.device or "auto",
                )
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dim = self._model.get_embedding_dimension()
                else:
                    self._dim = self._model.get_sentence_embedding_dimension()
                logger.info("Loaded embedding model '%s' with dimension=%d", self.model_name, self._dim)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {e}",
                    hint="Ensure internet connection for model download or verify model name.",
                ) from e
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._get_model()
        return self._dim or 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )
            return [e.tolist() for e in embeddings]
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings for {len(texts)} document chunks: {e}"
            ) from e

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        try:
            embedding = model.encode(
                text,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
            )
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}") from e


def create_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    device: Optional[str] = None,
    mock: bool = False,
) -> BaseEmbedder:
    """Factory to instantiate the appropriate embedding model."""
    if mock:
        return MockEmbedder()
    return SentenceTransformerEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
