from abc import ABC, abstractmethod
from pathlib import Path

from domain.models import IngestionReport


class IngestionUseCase(ABC):
    """Port: orchestrates the ingestion of documents into the vector store."""

    @abstractmethod
    async def ingest(
        self,
        file_path: Path,
        link_preview: str | None = None,
        debug_dir: Path | None = None,
    ) -> bool:
        """Ingests a single file into the vector store.

        Args:
            file_path: Path to the source file to ingest.
            link_preview: Optional public URL to attach as a preview link.
            debug_dir: When set, the parsed Markdown and the chunks are dumped
                there for inspection.

        Returns:
            True if the file was indexed, False if it was already present.

        Raises:
            IngestionStageError: If a pipeline stage fails.
        """
        ...

    @abstractmethod
    async def ingest_from_manifest(
        self,
        manifest_path: Path,
        debug_dir: Path | None = None,
    ) -> IngestionReport:
        """Ingests all files listed in a manifest JSON file.

        A failing file is recorded in the report and does not stop the batch.

        Args:
            manifest_path: Path to the manifest JSON produced by kdrive_downloader.
            debug_dir: When set, debug dumps are written there (one sub-folder
                per document).

        Returns:
            A report listing ingested, skipped and failed files.
        """
        ...
