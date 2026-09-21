import logging
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the ``.env`` file via pydantic-settings.

    All secrets (API keys, URLs) must be provided through environment variables
    or the ``.env`` file — never hard-coded.

    Attributes:
        mistral_api_key: Mistral secret key.
        mistral_embedding_model: Mistral embedding model identifier (1024 dims).
        qdrant_url: Qdrant Cloud cluster URL.
        qdrant_api_key: Qdrant Cloud API key.
        qdrant_collection_name: Name of the default target Qdrant collection.
        gemini_api_key: Google Gemini API key.
        gemini_parsing_model: Gemini model used for PDF parsing.
        kdrive_token: Infomaniak kDrive Bearer token.
        kdrive_drive_id: Numeric kDrive drive identifier.
        kdrive_share_uuid: UUID of the kDrive public share link.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Mistral
    mistral_api_key: str
    mistral_embedding_model: str = "mistral-embed"

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str

    # Gemini
    gemini_api_key: str
    gemini_parsing_model: str

    # kDrive
    kdrive_token: str
    kdrive_drive_id: str
    kdrive_share_uuid: str


settings = Settings()  # type: ignore[call-arg]


class LoggingConfig:
    """Configures application-wide logging (console + ``logs/ingestion.log``)."""

    LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"
    LOG_FILE = Path("logs") / "ingestion.log"

    @staticmethod
    def setup(level: int = logging.INFO) -> None:
        """Sets up the root logger with a console handler and a file handler.

        Third-party HTTP libraries are lowered to WARNING so that the console
        only shows the pipeline's own progress.

        Args:
            level: Minimum level for the root logger.
        """
        # Windows consoles default to a legacy code page that cannot print "—" or "↳".
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

        LoggingConfig.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=LoggingConfig.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LoggingConfig.LOG_FILE, encoding="utf-8"),
            ],
        )
        for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "google.auth"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
