"""Central configuration, loaded from environment variables / .env."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store: str = "chroma"
    storage_dir: str = "./storage"
    collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    openai_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
