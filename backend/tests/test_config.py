"""
Tests for the configuration module.
"""

import os

from backend.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_settings_load(self):
        """Settings should load with default values."""
        settings = Settings(
            _env_file=None,
            database_url="postgresql+asyncpg://test:test@localhost:5432/test_db",
            database_url_sync="postgresql+psycopg2://test:test@localhost:5432/test_db",
        )
        assert settings.app_name == "Question Paper Analysis & Generation System"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_settings_from_env(self, monkeypatch):
        """Settings should read from environment variables."""
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.90")

        settings = Settings(_env_file=None)
        assert settings.app_name == "Test App"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.similarity_threshold == 0.90

    def test_upload_path_creates_directory(self, tmp_path, monkeypatch):
        """upload_path property should create the directory if it doesn't exist."""
        upload_dir = tmp_path / "test_uploads"
        monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

        settings = Settings(_env_file=None)
        path = settings.upload_path

        assert path.exists()
        assert path.is_dir()

    def test_output_path_creates_directory(self, tmp_path, monkeypatch):
        """output_path property should create the directory if it doesn't exist."""
        output_dir = tmp_path / "test_outputs"
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

        settings = Settings(_env_file=None)
        path = settings.output_path

        assert path.exists()
        assert path.is_dir()

    def test_get_settings_returns_same_instance(self):
        """get_settings should return the same cached instance."""
        # Clear the cache to test from scratch
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()

    def test_default_thresholds(self):
        """Default threshold values should be set correctly."""
        settings = Settings(_env_file=None)
        assert settings.similarity_threshold == 0.75
        assert settings.duplicate_threshold == 0.90

    def test_default_model_names(self):
        """Default NLP model names should be set correctly."""
        settings = Settings(_env_file=None)
        assert settings.spacy_model == "en_core_web_sm"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
