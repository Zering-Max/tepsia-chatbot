from pathlib import Path
from dataclasses import dataclass
import asyncio
import tempfile

from pypdf import PdfReader, PdfWriter
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log

from domain.identity import Identity
from domain.models import Document, DocumentMetadata, Page, Prompt, ParsedDocumentResponse
from ports.document_parser import DocumentParser
from google.genai import Client as GeminiAPIClient
from google.genai import types
from google.genai.types import File as GeminiFile
from domain.exceptions import ParsingFileError
import logging

logger = logging.getLogger(__name__)

_POLLING_INTERVAL_SECONDS = 5
_POLLING_MAX_ATTEMPTS = 12  # 60 s total before giving up


@dataclass(frozen=True)
class _PageBatch:
    """A contiguous slice of a source PDF to be parsed in a single API call.

    Attributes:
        path: Local PDF holding only the pages of this batch. Equal to the
            source file when the document is small enough to be sent whole.
        page_offset: Number of source pages preceding this batch, added to the
            page numbers returned by the model to keep them document-absolute.
    """

    path: Path
    page_offset: int

def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Returns True for transient server-side errors that are safe to retry.

    Matches HTTP 503 / UNAVAILABLE responses caused by temporary Gemini overload.
    Defined at module level so it can be referenced by tenacity decorators at class
    definition time.

    Args:
        exc: The exception to evaluate.

    Returns:
        True if the error is transient and a retry is appropriate.
    """
    msg = str(exc).lower()
    return "503" in msg or "unavailable" in msg


@dataclass
class GeminiDocumentParser(DocumentParser):
    """DocumentParser backed by the Gemini Files + GenerateContent API.

    Long documents are split into page batches before upload: a single
    GenerateContent call cannot emit more than the model's output token budget,
    and a dense PDF of more than a few dozen pages exceeds it, which truncates
    the JSON mid-document. Each batch is uploaded, parsed and deleted in turn,
    then the pages are stitched back together with absolute page numbers.

    Attributes:
        gemini_client: Authenticated Gemini API client.
        parsing_prompt: Prompt sent alongside the uploaded file for extraction.
        model: Gemini model identifier used for content generation.
        max_pages_per_batch: Largest number of source pages sent in one call.
    """

    gemini_client: GeminiAPIClient
    parsing_prompt: Prompt
    model: str = "gemini-2.5-flash"
    max_pages_per_batch: int = 40

    async def parse(self, file_path: Path) -> Document:
        """Parses a file into a structured Document using the Gemini API.

        Splits the document into page batches, then for each batch uploads the
        file, waits for Gemini to finish processing it, calls GenerateContent
        with a JSON response schema, and cleans up the remote file.

        Args:
            file_path: Path to the local file to parse.

        Returns:
            A Document containing extracted pages and full concatenated text.

        Raises:
            ParsingFileError: If any step (splitting, upload, polling, generation,
                or schema validation) fails.
        """
        pages: list[Page] = []

        with tempfile.TemporaryDirectory(prefix="tepsai-parse-") as temp_dir:
            batches = self._split_into_batches(file_path, Path(temp_dir))
            for index, batch in enumerate(batches, start=1):
                logger.info(
                    "Parsing %s — batch %d/%d (pages %d+).",
                    file_path.name,
                    index,
                    len(batches),
                    batch.page_offset + 1,
                )
                pages.extend(await self._parse_batch(file_path, batch))

        return self._build_document(file_path, pages)

    def _split_into_batches(self, file_path: Path, temp_dir: Path) -> list[_PageBatch]:
        """Slices a PDF into batches of at most ``max_pages_per_batch`` pages.

        Documents that already fit in a single batch are returned as-is, so no
        temporary file is written for them.

        Args:
            file_path: Path to the source PDF.
            temp_dir: Directory receiving the split PDFs.

        Returns:
            The ordered batches covering the whole document.

        Raises:
            ParsingFileError: If the PDF cannot be read or split.
        """
        try:
            reader = PdfReader(file_path)
            page_count = len(reader.pages)

            if page_count <= self.max_pages_per_batch:
                return [_PageBatch(path=file_path, page_offset=0)]

            batches: list[_PageBatch] = []
            for offset in range(0, page_count, self.max_pages_per_batch):
                writer = PdfWriter()
                for page_index in range(offset, min(offset + self.max_pages_per_batch, page_count)):
                    writer.add_page(reader.pages[page_index])

                batch_path = temp_dir / f"{file_path.stem}_p{offset + 1}.pdf"
                with batch_path.open("wb") as handle:
                    writer.write(handle)
                batches.append(_PageBatch(path=batch_path, page_offset=offset))

            logger.info(
                "Split %s (%d pages) into %d batch(es) of up to %d pages.",
                file_path.name,
                page_count,
                len(batches),
                self.max_pages_per_batch,
            )
            return batches

        except Exception as e:
            raise ParsingFileError(
                file_path=str(file_path),
                reason="Could not read or split the PDF into page batches.",
                original_error=e,
            )

    async def _parse_batch(self, file_path: Path, batch: _PageBatch) -> list[Page]:
        """Uploads, parses and cleans up a single page batch.

        Args:
            file_path: Original local path, used in error messages.
            batch: The batch to parse.

        Returns:
            The extracted pages, renumbered relative to the whole document.

        Raises:
            ParsingFileError: If upload, polling or generation fails.
        """
        uploaded_file = await self._upload_file(batch.path)
        assert uploaded_file.name is not None
        file_name: str = uploaded_file.name

        try:
            await self._wait_for_processing(file_path, file_name)
            parsed_data = await self._generate_content(file_path, uploaded_file)
        finally:
            await self._delete_remote_file(file_name)

        return [
            Page(
                page_number=page.page_number + batch.page_offset,
                page_content=page.page_content,
            )
            for page in parsed_data.pages
        ]

    @retry(
        retry=retry_if_exception(_is_retryable_gemini_error),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _upload_file(self, file_path: Path) -> GeminiFile:
        """Uploads a local file to the Gemini Files API.

        Args:
            file_path: Path to the file to upload.

        Returns:
            The uploaded file object returned by the API.

        Raises:
            ParsingFileError: If the upload fails or the returned object has no name.
        """
        try:
            # Upload an open handle rather than the path: given a path, the SDK
            # sends the file name in an HTTP header, which must be ASCII and
            # breaks on accented names such as "Fiche de sécurité.pdf".
            with file_path.open("rb") as handle:
                uploaded_file = await self.gemini_client.aio.files.upload(
                    file=handle,
                    config=types.UploadFileConfig(mime_type="application/pdf"),
                )
        except Exception as e:
            if _is_retryable_gemini_error(e):
                raise  # tenacity will catch and retry
            raise ParsingFileError(
                file_path=str(file_path),
                reason="Network error or API limits reached during upload.",
                original_error=e,
            )

        if not hasattr(uploaded_file, "name") or uploaded_file.name is None:
            raise ParsingFileError(
                str(file_path),
                "Invalid upload response (empty or corrupted file).",
            )

        return uploaded_file

    async def _wait_for_processing(self, file_path: Path, file_name: str) -> None:
        """Polls the Gemini Files API until the file reaches a terminal state.

        Retries up to ``_POLLING_MAX_ATTEMPTS`` times with a fixed interval of
        ``_POLLING_INTERVAL_SECONDS`` seconds between each attempt.

        Args:
            file_path: Original local path, used in error messages.
            file_name: Gemini file resource name to poll.

        Raises:
            ParsingFileError: If processing fails, a polling error occurs, or the
                maximum number of attempts is exceeded.
        """
        try:
            for attempt in range(_POLLING_MAX_ATTEMPTS):
                file_info = await self.gemini_client.aio.files.get(name=file_name)
                status = file_info.state

                if status == "ACTIVE":
                    logger.info("File ready for generation: %s", file_name)
                    return

                if status == "FAILED":
                    raise ParsingFileError(str(file_path), "Gemini file processing failed.")

                logger.debug(
                    "File %s status=%s — retrying in %ds (attempt %d/%d).",
                    file_name,
                    status,
                    _POLLING_INTERVAL_SECONDS,
                    attempt + 1,
                    _POLLING_MAX_ATTEMPTS,
                )
                await asyncio.sleep(_POLLING_INTERVAL_SECONDS)

        except ParsingFileError:
            raise
        except Exception as e:
            raise ParsingFileError(
                str(file_path),
                f"Could not poll processing status for {file_name}.",
                e,
            )

        raise ParsingFileError(
            str(file_path),
            f"File {file_name} did not become ACTIVE after {_POLLING_MAX_ATTEMPTS} attempts "
            f"({_POLLING_MAX_ATTEMPTS * _POLLING_INTERVAL_SECONDS}s).",
        )

    @retry(
        retry=retry_if_exception(_is_retryable_gemini_error),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _generate_content(self, file_path: Path, uploaded_file) -> ParsedDocumentResponse:
        """Calls GenerateContent to extract structured page data from the uploaded file.

        Args:
            file_path: Original local path, used in error messages.
            uploaded_file: Gemini file object passed as generation input.

        Returns:
            A ParsedDocumentResponse containing the extracted pages.

        Raises:
            ParsingFileError: If the API call fails or the response does not match
                the expected JSON schema.
        """
        try:
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model,
                contents=[uploaded_file, self.parsing_prompt.content],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ParsedDocumentResponse,
                    temperature=0.0,
                    # Reasoning tokens are billed against the same output budget as
                    # the JSON itself; disabling them leaves it all for the Markdown.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            if _is_retryable_gemini_error(e):
                raise  # tenacity will catch and retry
            raise ParsingFileError(
                file_path=str(file_path),
                reason="Gemini API error during content generation.",
                original_error=e,
            )

        if response.parsed is None:
            finish_reason = next(
                (candidate.finish_reason for candidate in response.candidates or []),
                None,
            )
            raise ParsingFileError(
                file_path=str(file_path),
                reason=(
                    "Model responded but output could not be parsed as the expected JSON "
                    f"schema (finish_reason={finish_reason}). A MAX_TOKENS finish_reason "
                    "means the output was truncated: lower max_pages_per_batch."
                ),
            )

        if not isinstance(response.parsed, ParsedDocumentResponse):
            raise ParsingFileError(
                file_path=str(file_path),
                reason=f"Unexpected response type: {type(response.parsed)}",
            )

        return response.parsed

    async def _delete_remote_file(self, file_name: str) -> None:
        """Deletes the uploaded file from Gemini storage to avoid quota exhaustion.

        Failures are logged as warnings rather than raised, so that the caller's
        ``finally`` block does not swallow the original exception.

        Args:
            file_name: Gemini file resource name to delete.
        """
        try:
            await self.gemini_client.aio.files.delete(name=file_name)
            logger.debug("Deleted remote file: %s", file_name)
        except Exception:
            logger.warning(
                "Failed to delete remote file %s — manual cleanup may be required.",
                file_name,
            )

    def _build_document(self, file_path: Path, pages: list[Page]) -> Document:
        """Assembles a domain Document from parsed page data.

        Args:
            file_path: Path to the original local file.
            pages: Extracted pages, ordered and numbered document-absolutely.

        Returns:
            A frozen Document with metadata and full concatenated text content.
        """
        full_text = "\n\n".join(
            f"<!-- page:{page.page_number} -->\n{page.page_content}"
            for page in pages
        )
        metadata = DocumentMetadata(
            file_name=file_path.name,
            file_path=file_path,
            doctype="pdf",
            parser_name="GeminiDocumentParser",
        )

        return Document(
            id=Identity.document_id(file_path),
            pages=tuple(pages),
            metadata=metadata,
            content=full_text,
        )
