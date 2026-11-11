"""
Application Configuration Module.

Centralizes all configuration using pydantic-settings.
Values are loaded from environment variables and .env files.
"""

import os
from pathlib import Path
from functools import lru_cache

# Ensure cache directories default to workspace paddle cache or writeable locations
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_PADDLE_CACHE = _WORKSPACE_ROOT / ".paddle_cache"
_MODEL_CACHE = _WORKSPACE_ROOT / ".model_cache"

if _MODEL_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(_MODEL_CACHE))

if _PADDLE_CACHE.exists():
    os.environ.setdefault("PADDLE_HOME", str(_PADDLE_CACHE))
    os.environ.setdefault("PADDLEX_HOME", str(_PADDLE_CACHE))
    os.environ.setdefault("HOME", str(_PADDLE_CACHE))
else:
    if "HOME" not in os.environ or not os.access(os.environ.get("HOME", "/"), os.W_OK):
        os.environ["HOME"] = "/tmp"
    os.environ.setdefault("PADDLE_HOME", "/tmp/.paddle")
    os.environ.setdefault("PADDLEX_HOME", "/tmp/.paddlex")

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Question Paper Analysis & Generation System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://qpaper_user:qpaper_pass@localhost:5432/qpaper_db"
    database_url_sync: str = "postgresql+psycopg2://qpaper_user:qpaper_pass@localhost:5432/qpaper_db"

    # --- Gemini API ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # --- Mathpix OCR API (Optional) ---
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

    # --- File Storage ---
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"

    # --- NLP Models ---
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Similarity Thresholds ---
    similarity_threshold: float = 0.75
    duplicate_threshold: float = 0.90

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Computed Properties ---
    @property
    def upload_path(self) -> Path:
        """Resolved upload directory path."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_path(self) -> Path:
        """Resolved output directory path."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """
    Return cached application settings.

    Uses lru_cache so the Settings object is created only once
    and reused across the application lifetime.
    """
    settings = Settings()
    if not settings.gemini_api_key:
        # Check standard fallback locations for gemini_api_key.txt
        possible_key_paths = [
            _WORKSPACE_ROOT.parent / "API" / "gemini_api_key.txt",
            _WORKSPACE_ROOT / "API" / "gemini_api_key.txt",
            Path("/home/persistence/Projects/archive/API/gemini_api_key.txt"),
        ]
        for key_path in possible_key_paths:
            if key_path.exists():
                try:
                    loaded_key = key_path.read_text(encoding="utf-8").strip()
                    if loaded_key:
                        settings.gemini_api_key = loaded_key
                        break
                except Exception:
                    pass
    return settings
