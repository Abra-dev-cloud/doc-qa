"""
app/config.py — Centralised settings loaded from environment / .env file.
All other modules import from here; never read os.environ directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # Chunking
    chunk_size: int = 512        # tokens
    chunk_overlap: int = 200     # tokens

    # Retrieval
    default_top_k: int = 5

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
