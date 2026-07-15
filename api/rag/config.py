"""Application settings for the RAG pipeline.

Loads configuration (OpenAI, Qdrant) from the environment and ``.env`` files
via pydantic-settings, and exposes a single shared :data:`settings` instance.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the RAG pipeline.

    Values are read from environment variables, falling back to ``.env`` and
    ``.env.local``. Unknown variables are ignored.

    Attributes:
        openai_api_key: API key for OpenAI (embeddings and chat).
        openai_embedding_model: Name of the OpenAI embedding model.
        llm_model: Name of the OpenAI chat model used for generation.
        qdrant_url: Base URL of the Qdrant instance.
        qdrant_api_key: API key for the Qdrant instance.
        qdrant_collection_name: Name of the target Qdrant collection.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    openai_embedding_model: str
    llm_model: str = "gpt-4.1-mini"

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str


settings = Settings()  # type: ignore[call-arg]
