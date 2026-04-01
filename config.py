"""
config.py — Central configuration for the AI system.
All environment-sensitive values live here. Override via .env or environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ─── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "LocalAI System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ─── Ollama / LLM ──────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "mistral"          # swap to "llama3" if preferred
    OLLAMA_TIMEOUT: int = 120               # seconds per request
    OLLAMA_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096

    # ─── FastAPI ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # ─── Storage ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./ai_system.db"
    MAX_STORED_RESULTS: int = 1000          # auto-prune after this

    # ─── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/ai_system.log"
    LOG_MAX_BYTES: int = 5_242_880          # 5 MB
    LOG_BACKUP_COUNT: int = 3

    # ─── Analysis ──────────────────────────────────────────────────────────────
    MIN_KEYWORD_SCORE: float = 0.5          # TF-IDF minimum relevance
    SIMILARITY_THRESHOLD: float = 0.6      # flag resume if below this

    # ─── Parsing ───────────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["pdf", "docx", "txt"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
