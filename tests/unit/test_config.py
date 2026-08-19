"""Unit tests for configuration loading."""

from pathlib import Path
import pytest
from vector_rag.config.settings import Settings
from vector_rag.utils.errors import ConfigurationError


def test_default_settings():
    settings = Settings()
    assert settings.application.name == "vector-rag"
    assert settings.storage.sqlite_path == "data/app.db"
    assert settings.chunking.chunk_size == 800
    assert settings.retrieval.top_k == 10
    assert settings.reranker.top_n == 5


def test_load_from_yaml_valid(tmp_path: Path):
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(
        """
application:
  name: "custom-rag"
chunking:
  chunk_size: 500
  chunk_overlap: 50
retrieval:
  top_k: 20
""",
        encoding="utf-8",
    )

    settings = Settings.load_from_yaml(yaml_file)
    assert settings.application.name == "custom-rag"
    assert settings.chunking.chunk_size == 500
    assert settings.chunking.chunk_overlap == 50
    assert settings.retrieval.top_k == 20
    # defaults preserved
    assert settings.storage.sqlite_path == "data/app.db"


def test_load_from_yaml_invalid_path():
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.load_from_yaml("non_existent_file.yaml")
    assert "not found" in str(exc_info.value)


def test_invalid_chunk_size(tmp_path: Path):
    yaml_file = tmp_path / "invalid_config.yaml"
    yaml_file.write_text(
        """
chunking:
  chunk_size: 10  # Below minimum 50
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.load_from_yaml(yaml_file)
    assert "Configuration validation failed" in str(exc_info.value)
