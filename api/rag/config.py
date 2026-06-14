from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
