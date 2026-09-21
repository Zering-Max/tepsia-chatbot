"""Application settings for the RAG pipeline.

Loads configuration (Mistral, Qdrant) from the environment and ``.env`` files
via pydantic-settings, and exposes a single shared :data:`settings` instance.

The OpenAI fields are optional: the adapters are kept in the codebase to allow
rolling back, but nothing builds them at startup any more, so the pipeline must
boot without an OpenAI key.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the RAG pipeline.

    Values are read from environment variables, falling back to ``.env`` and
    ``.env.local``. Unknown variables are ignored.

    Attributes:
        mistral_api_key: API key for Mistral (embeddings and chat).
        mistral_embedding_model: Name of the Mistral embedding model.
        mistral_llm_model: Name of the Mistral chat model used for generation.
        openai_api_key: API key for OpenAI, only needed by the unused adapters.
        openai_embedding_model: Name of the OpenAI embedding model.
        llm_model: Name of the OpenAI chat model.
        qdrant_url: Base URL of the Qdrant instance.
        qdrant_api_key: API key for the Qdrant instance.
        qdrant_collection_name: Name of the target Qdrant collection.
        qdrant_cache_collection_name: Name of the Qdrant collection used by the semantic cache.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mistral_api_key: str
    mistral_embedding_model: str = "mistral-embed"
    mistral_llm_model: str = "mistral-medium-latest"

    openai_api_key: str | None = None
    openai_embedding_model: str | None = None
    llm_model: str = "gpt-4.1-mini"

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    qdrant_cache_collection_name: str


settings = Settings()  # type: ignore[call-arg]
