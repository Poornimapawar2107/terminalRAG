"""Configuration models and settings loader for Vector RAG."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from vector_rag.utils.errors import ConfigurationError


class ApplicationConfig(BaseModel):
    """Application level configuration."""

    name: str = "vector-rag"
    environment: str = "development"


class StorageConfig(BaseModel):
    """Paths for database and vector storage."""

    sqlite_path: str = "data/app.db"
    chroma_path: str = "data/chroma"


class ChunkingConfig(BaseModel):
    """Text chunking strategy and sizes."""

    strategy: str = "recursive"
    chunk_size: int = Field(default=800, ge=50, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)


class RetrievalConfig(BaseModel):
    """Candidate retrieval configuration."""

    top_k: int = Field(default=10, ge=1, le=100)


class RerankerConfig(BaseModel):
    """Reranker cross-encoder model configuration."""

    top_n: int = Field(default=5, ge=1, le=50)
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(default=32, ge=1)


class GenerationConfig(BaseModel):
    """LLM generation configuration."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=50, le=4096)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: Optional[str] = "logs/vector-rag.log"


class Settings(BaseSettings):
    """Global application settings combining YAML config and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VECTOR_RAG_",
        extra="ignore",
    )

    # API Keys / secrets via env
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")

    # Modular configs
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path = "config/config.yaml") -> "Settings":
        """Load settings from a YAML file, overlaid with environment variables."""
        path = Path(yaml_path)
        yaml_data: Dict[str, Any] = {}

        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content and isinstance(content, dict):
                        yaml_data = content
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to read configuration file at '{yaml_path}': {e}",
                    hint="Check YAML syntax and formatting.",
                ) from e
        elif str(yaml_path) != "config/config.yaml":
            # If a custom path was provided and doesn't exist, raise error
            raise ConfigurationError(
                f"Configuration file '{yaml_path}' not found.",
                hint="Verify that the config path exists.",
            )

        try:
            return cls(**yaml_data)
        except Exception as e:
            raise ConfigurationError(
                f"Configuration validation failed: {e}",
                hint="Check config values against expected types and constraints.",
            ) from e
