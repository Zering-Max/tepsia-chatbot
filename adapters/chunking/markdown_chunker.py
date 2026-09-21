import re
import logging
from dataclasses import dataclass
from collections.abc import Iterator

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from domain.models import TextChunk, TextChunkMetadata, Document
from domain.identity import Identity
from ports.chunker import TextChunker

logger = logging.getLogger(__name__)

_PAGE_MARKER = re.compile(r"<!-- page:(\d+) -->\n?")
# Matches one or more consecutive lines starting with | (markdown table)
_TABLE_BLOCK = re.compile(r"((?:\|[^\n]*\n?)+)", re.MULTILINE)

_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


@dataclass
class MarkdownChunker(TextChunker):
    """Three-stage chunker: header structure → table extraction → RCTS.

    Stage 1 — MarkdownHeaderTextSplitter splits on #/##/### boundaries and
    records the section hierarchy (section_title) for each chunk.

    Stage 2 — Within each section, markdown tables are isolated as standalone
    chunks. Each table chunk is prefixed with the last paragraph that precedes
    it in the same section, so the retrieval context includes the intro sentence.

    Stage 3 — Non-table text is further split by RecursiveCharacterTextSplitter
    to enforce a maximum chunk size.

    Page numbers are recovered from <!-- page:N --> markers that survive all
    three stages unchanged and are stripped only at the final step.
    """

    chunk_size: int = 1500
    chunk_overlap: int = 200

    def chunk(self, document: Document) -> list[TextChunk]:
        """Splits a document into text chunks using the three-stage pipeline.

        Args:
            document: Fully parsed document whose ``content`` field is Markdown
                with embedded ``<!-- page:N -->`` markers.

        Returns:
            Ordered list of TextChunks with page range and section metadata.
        """
        md_splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON)
        rc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[TextChunk] = []
        last_page: int | None = None
        chunk_index = 0

        for lc_chunk in md_splitter.split_text(document.content):
            section_title = self._section_title(lc_chunk.metadata)

            for block_type, raw_block, context_prefix in self._iter_blocks(lc_chunk.page_content, rc_splitter):
                clean_block, raw_page_start, raw_page_end, has_lead, last_marker_page = (
                    self._extract_and_clean_pages(raw_block)
                )

                if block_type == "table":
                    content = (context_prefix + "\n\n" + clean_block).strip() if context_prefix else clean_block.strip()
                else:
                    content = clean_block.strip()

                if not content:
                    # Marker-only chunk: advance last_page even though no text is emitted.
                    if last_marker_page is not None:
                        last_page = last_marker_page
                    continue

                if raw_page_start is None:
                    page_start = page_end = last_page
                else:
                    if has_lead:
                        if last_page is not None:
                            page_start = last_page
                        elif raw_page_start > 1:
                            page_start = raw_page_start - 1
                        else:
                            page_start = raw_page_start
                    else:
                        page_start = raw_page_start
                    # raw_page_end is None when all markers are trailing (no content
                    # after them): the chunk's content lives entirely on page_start.
                    page_end = raw_page_end if raw_page_end is not None else page_start
                    last_page = last_marker_page

                metadata = TextChunkMetadata(
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    file_name=document.metadata.file_name,
                    file_path=str(document.metadata.file_path),
                    link_preview=document.metadata.link_preview,
                )
                chunks.append(TextChunk(
                    id=Identity.chunk_id(document.id, chunk_index),
                    document_id=document.id,
                    content=content,
                    chunk_index=chunk_index,
                    metadata=metadata,
                ))
                chunk_index += 1

        logger.info("Chunking done: %d chunks for %s", chunk_index, document.metadata.file_name)
        return chunks

    def _iter_blocks(
        self,
        text: str,
        rc_splitter: RecursiveCharacterTextSplitter,
    ) -> Iterator[tuple[str, str, str]]:
        """Splits a section into (block_type, raw_block, context_prefix) tuples.

        re.split with a capturing group produces alternating segments:
          [text0, table1, text2, table3, ...]
        Even indices are plain text; odd indices are captured table blocks.

        For text blocks  → each RCTS sub-chunk is yielded as ("text", sub, "").
        For table blocks → yielded as ("table", raw_table, last_paragraph_of_preceding_text).

        The context_prefix for tables is built from the cleaned (marker-free)
        version of the immediately preceding text block, so page markers from
        that text do not pollute the table chunk's page tracking.
        """
        parts = _TABLE_BLOCK.split(text)
        preceding_clean = ""

        for i, part in enumerate(parts):
            if i % 2 == 0:  # plain text segment
                if part.strip():
                    for sub in rc_splitter.split_text(part):
                        yield ("text", sub, "")
                    clean, _, _, _, _ = self._extract_and_clean_pages(part)
                    preceding_clean = self._last_paragraph(clean)
                else:
                    preceding_clean = ""
            else:  # table block (captured group)
                if part.strip():
                    yield ("table", part, preceding_clean)
                    preceding_clean = ""

    @staticmethod
    def _extract_and_clean_pages(
        text: str,
    ) -> tuple[str, int | None, int | None, bool, int | None]:
        """Strips <!-- page:N --> markers and returns page tracking info.

        Returns:
            clean           — text with all markers removed.
            raw_page_start  — page number of the first marker (None if no markers).
            page_end        — page number of the LAST marker that has non-whitespace
                              content after it; None if every marker is trailing
                              (signals a page transition with no content on that page
                              yet in this chunk).
            has_lead        — True when non-whitespace content precedes the first marker
                              (that content belongs to the previous page).
            last_marker_page — page number of the very last marker seen, used to advance
                               last_page even when page_end is None.
        """
        clean = _PAGE_MARKER.sub("", text)
        matches = list(_PAGE_MARKER.finditer(text))
        if not matches:
            return clean, None, None, False, None

        first_match = matches[0]
        has_lead = bool(text[:first_match.start()].strip())
        raw_page_start = int(first_match.group(1))
        last_marker_page = int(matches[-1].group(1))

        # page_end: last marker whose segment (up to the next marker or end) has content.
        page_end: int | None = None
        for i, m in enumerate(matches):
            seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            if text[m.end():seg_end].strip():
                page_end = int(m.group(1))

        return clean, raw_page_start, page_end, has_lead, last_marker_page

    @staticmethod
    def _last_paragraph(text: str) -> str:
        """Returns the last non-empty block of text, used as context prefix for tables.

        Strips trailing whitespace first so a trailing blank line doesn't make
        rfind return an empty match. Tries \n\n (paragraph) then \n (line).
        """
        stripped = text.rstrip()
        for sep in ("\n\n", "\n"):
            idx = stripped.rfind(sep)
            if idx != -1:
                last = stripped[idx + len(sep):].strip()
                if last:
                    return last
        return stripped.strip()

    @staticmethod
    def _section_title(lc_metadata: dict) -> str | None:
        """Joins header levels into a full section path, e.g. 'Ch.1 > Art.3'."""
        parts = [
            lc_metadata[key]
            for key in ("Header 1", "Header 2", "Header 3")
            if key in lc_metadata
        ]
        return " > ".join(parts) if parts else None
