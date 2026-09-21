import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, TypeVar

from domain.exceptions import IngestionStageError
from domain.identity import Identity
from domain.models import Document, EmbeddedChunk, IngestionReport, TextChunk
from ports.chunker import TextChunker
from ports.document_parser import DocumentParser
from ports.embedding import DenseEmbedder
from ports.ingestion import IngestionUseCase
from ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The Gemini parser only handles PDFs; other files listed in a manifest are skipped.
SUPPORTED_EXTENSIONS = {".pdf"}
DEFAULT_UPSERT_BATCH_SIZE = 64


@dataclass
class IngestionService(IngestionUseCase):
    """Ingests documents: existence check → parse → chunk → embed → upsert.

    Documents are identified by the SHA-256 of their content, so a file that is
    already indexed is skipped, and re-running a manifest only processes what is
    new. Files are processed one at a time to stay within the Gemini rate limits.

    Attributes:
        parser: Converts a PDF into a structured Document.
        chunker: Splits a Document into text chunks.
        embedder: Computes dense embeddings for chunks.
        vector_store: Destination store (also computes BM25 vectors server-side).
        upsert_batch_size: Number of points sent per upsert request.
    """

    parser: DocumentParser
    chunker: TextChunker
    embedder: DenseEmbedder
    vector_store: VectorStore
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE

    async def ingest(
        self,
        file_path: Path,
        link_preview: str | None = None,
        debug_dir: Path | None = None,
    ) -> bool:
        """Ingests a single PDF into the vector store.

        Args:
            file_path: Path to the PDF to ingest.
            link_preview: Optional public URL attached to every chunk of the file.
            debug_dir: When set, the parsed Markdown and the chunks are written there.

        Returns:
            True if the file was indexed, False if it was already present.

        Raises:
            IngestionStageError: If a pipeline stage fails.
        """
        name = file_path.name

        doc_id = self._stage("hash", name, Identity.document_id, file_path)
        if await self._stage_async("existence check", name, self.vector_store.document_exists(doc_id)):
            logger.info("Already indexed, skipping: %s", name)
            return False

        document = await self._stage_async("parsing", name, self.parser.parse(file_path))
        if link_preview:
            document = document.with_link_preview(link_preview)

        chunks = self._stage("chunking", name, self.chunker.chunk, document)
        if not chunks:
            raise IngestionStageError("chunking", name, ValueError("No text extracted from the document."))
        if debug_dir:
            self._dump_debug(debug_dir, document, chunks)

        embeddings = await self._stage_async("embedding", name, self.embedder.embed(chunks))
        if len(embeddings) != len(chunks):
            raise IngestionStageError(
                "embedding", name,
                ValueError(f"{len(embeddings)} embeddings for {len(chunks)} chunks."),
            )
        items = [EmbeddedChunk(chunk=c, dense=e) for c, e in zip(chunks, embeddings)]

        for start in range(0, len(items), self.upsert_batch_size):
            batch = items[start : start + self.upsert_batch_size]
            await self._stage_async("upsert", name, self.vector_store.upsert_points(batch))

        logger.info("Ingested %s: %d chunks.", name, len(chunks))
        return True

    async def ingest_from_manifest(
        self,
        manifest_path: Path,
        debug_dir: Path | None = None,
    ) -> IngestionReport:
        """Ingests every PDF listed in a kDrive manifest.

        Each manifest entry provides ``local_path`` (downloaded file) and
        ``share_url`` (public preview link). A failing file is recorded in the
        report and the batch goes on.

        Args:
            manifest_path: Path to the manifest JSON produced by kdrive_downloader.
            debug_dir: When set, debug dumps are written in one sub-folder per document.

        Returns:
            A report listing ingested, skipped and failed files.
        """
        entries: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = IngestionReport()
        logger.info("Manifest %s: %d entries.", manifest_path, len(entries))

        for position, entry in enumerate(entries, start=1):
            file_path = Path(entry["local_path"])
            name = entry.get("name", file_path.name)
            logger.info("[%d/%d] %s", position, len(entries), name)

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("Unsupported file type, skipping: %s", name)
                report.skipped.append(name)
                continue
            if not file_path.is_file():
                logger.error("File not found on disk: %s", file_path)
                report.failed.append((name, f"File not found: {file_path}"))
                continue

            try:
                ingested = await self.ingest(
                    file_path,
                    link_preview=entry.get("share_url"),
                    debug_dir=debug_dir / file_path.stem if debug_dir else None,
                )
            except IngestionStageError as e:
                logger.error("%s", e)
                report.failed.append((name, str(e)))
                continue

            (report.ingested if ingested else report.skipped).append(name)

        logger.info(
            "Manifest done: %d ingested, %d skipped, %d failed.",
            len(report.ingested), len(report.skipped), len(report.failed),
        )
        return report

    @staticmethod
    def _stage(stage: str, file_name: str, func, *args):
        """Runs a synchronous pipeline stage, wrapping any error in IngestionStageError."""
        try:
            return func(*args)
        except Exception as e:
            raise IngestionStageError(stage, file_name, e) from e

    @staticmethod
    async def _stage_async(stage: str, file_name: str, awaitable: Awaitable[T]) -> T:
        """Awaits an asynchronous pipeline stage, wrapping any error in IngestionStageError."""
        try:
            return await awaitable
        except Exception as e:
            raise IngestionStageError(stage, file_name, e) from e

    @staticmethod
    def _dump_debug(debug_dir: Path, document: Document, chunks: list[TextChunk]) -> None:
        """Writes the parsed Markdown and the chunks to ``debug_dir`` for inspection.

        Args:
            debug_dir: Destination directory (created if missing).
            document: Parsed document.
            chunks: Chunks produced from it.
        """
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "parsed.md").write_text(document.content, encoding="utf-8")
        (debug_dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Debug files written to %s", debug_dir)
