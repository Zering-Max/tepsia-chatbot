import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import LoggingConfig
from container import DENSE_VECTOR_SIZE, ServiceContainer
from domain.exceptions import CollectionNotFoundError, IngestionStageError

logger = logging.getLogger(__name__)


class CLI:
    """Entry point for the TepsIA ingestion command-line interface."""

    @staticmethod
    def _debug_dir(enabled: bool) -> Path | None:
        """Creates a timestamped debug directory when debug dumps are requested.

        Args:
            enabled: Whether ``--debug`` was passed.

        Returns:
            The directory ``debug/<timestamp>``, or None when disabled.
        """
        if not enabled:
            return None
        path = Path("debug") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    async def _run_create_collection(name: str) -> int:
        """Creates a Qdrant collection with the schema expected by ingestion.

        Args:
            name: Name of the collection to create.

        Returns:
            Process exit code (0 on success).
        """
        vector_store = ServiceContainer.build_vector_store(name)
        await vector_store.create_collection(dense_size=DENSE_VECTOR_SIZE)
        return 0

    @staticmethod
    async def _run_ingestion(args: argparse.Namespace) -> int:
        """Ingests a single PDF file.

        Args:
            args: Parsed CLI arguments.

        Returns:
            Process exit code: 1 if ingestion failed, 0 otherwise.
        """
        service = await ServiceContainer.build_ingestion_service(args.collection)
        try:
            await service.ingest(
                Path(args.ingest),
                link_preview=args.link_preview,
                debug_dir=CLI._debug_dir(args.debug),
            )
        except IngestionStageError as e:
            logger.error("%s", e)
            return 1
        return 0

    @staticmethod
    async def _run_ingestion_manifest(args: argparse.Namespace) -> int:
        """Ingests all files listed in a kDrive manifest JSON.

        Args:
            args: Parsed CLI arguments.

        Returns:
            Process exit code: 1 if at least one file failed, 0 otherwise.
        """
        service = await ServiceContainer.build_ingestion_service(args.collection)
        report = await service.ingest_from_manifest(
            Path(args.ingest_manifest),
            debug_dir=CLI._debug_dir(args.debug),
        )
        for name, error in report.failed:
            logger.error("FAILED %s — %s", name, error)
        return 1 if report.failed else 0

    @staticmethod
    def run() -> int:
        """Parses CLI arguments and dispatches to the appropriate handler.

        Returns:
            Process exit code.
        """
        parser = argparse.ArgumentParser(description="TepsIA — document ingestion into Qdrant.")
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--create-collection", metavar="NAME", help="Create a Qdrant collection and exit")
        action.add_argument("--ingest", metavar="FILE", help="Path to a PDF to ingest")
        action.add_argument("--ingest-manifest", metavar="MANIFEST", help="Path to a kDrive manifest JSON to ingest")
        parser.add_argument("--collection", metavar="NAME", help="Target collection (default: QDRANT_COLLECTION_NAME from .env)")
        parser.add_argument("--link-preview", metavar="URL", help="Preview URL attached to the file (with --ingest)")
        parser.add_argument("--debug", action="store_true", help="Dump parsed Markdown and chunks into debug/<timestamp>/")
        args = parser.parse_args()

        LoggingConfig.setup()

        try:
            if args.create_collection:
                return asyncio.run(CLI._run_create_collection(args.create_collection))
            if args.ingest:
                return asyncio.run(CLI._run_ingestion(args))
            return asyncio.run(CLI._run_ingestion_manifest(args))
        except (CollectionNotFoundError, ValueError) as e:
            logger.error("%s", e)
            return 1


if __name__ == "__main__":
    sys.exit(CLI.run())
