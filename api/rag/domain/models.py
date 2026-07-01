from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable metadata attached to a parsed document.

    Attributes:
        file_name: Original file name (e.g. ``"report.pdf"``).
        file_path: Absolute path to the source file on disk.
        doctype: Media type of the source file (e.g. ``"pdf"``).
        parser_name: Name of the adapter that produced this document.
        parser_version: Version string of the parser, if available.
        link_preview: Public preview URL for the source file (e.g. kDrive share link).
    """

    file_name: str
    file_path: Path
    doctype: str
    parser_name: str | None = None
    parser_version: str | None = None
    link_preview: str | None = None


@dataclass(frozen=True)
class Page:
    """A single page extracted from a source document.

    Attributes:
        page_number: 1-based page number as reported by the parser.
        page_content: Markdown text extracted from this page.
    """

    page_number: int
    page_content: str


@dataclass(frozen=True)
class ParsedDocumentResponse:
    """Raw structured output returned by the document parser.

    Attributes:
        pages: Ordered list of extracted pages.
    """

    pages: list[Page]


@dataclass(frozen=True)
class Document:
    """A fully parsed source document, ready for chunking.

    Attributes:
        id: SHA-256 hash of the source file, used for deduplication.
        metadata: File-level metadata (name, path, parser, preview URL).
        content: Full concatenated Markdown text of all pages, with
            ``<!-- page:N -->`` markers at page boundaries.
        pages: Immutable tuple of individual page objects.
    """

    id: str
    metadata: DocumentMetadata
    content: str
    pages: tuple[Page, ...] = ()

    def with_link_preview(self, link_preview: str) -> "Document":
        """Returns a copy of this document with the given preview URL.

        Args:
            link_preview: Public URL for the source file.

        Returns:
            A new frozen Document with the updated metadata.
        """
        return replace(self, metadata=replace(self.metadata, link_preview=link_preview))


@dataclass(frozen=True)
class TextChunkMetadata:
    """Metadata attached to a text chunk, carried into the vector store payload.

    Attributes:
        chunk_index: 0-based position of this chunk within the document.
        section_title: Markdown header path of the section this chunk belongs to
            (e.g. ``"Ch. 1 > Art. 3"``), or ``None`` if at the top level.
        file_name: Name of the source file.
        file_path: String representation of the source file path.
        link_preview: Public preview URL inherited from the parent document.
        page_start: First page this chunk's content comes from.
        page_end: Last page this chunk's content comes from.
    """

    chunk_index: int
    section_title: str | None
    file_name: str
    file_path: str
    link_preview: str | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A slice of document text produced by a TextChunker.

    Attributes:
        content: Plain text of this chunk (page markers stripped).
        id: UUID5 derived from the document id and chunk index.
        document_id: SHA-256 hash of the parent document.
        chunk_index: 0-based position within the parent document.
        metadata: Structural and source metadata for retrieval.
    """

    content: str
    id: str | None = None
    document_id: str | None = None
    chunk_index: int | None = None
    metadata: TextChunkMetadata | None = None


@dataclass(frozen=True)
class DenseEmbedding:
    """Dense vector representation of a text chunk.

    Attributes:
        chunk_id: UUID of the associated TextChunk.
        vector: Float vector produced by the embedding model.
    """

    chunk_id: str
    vector: list[float]


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse (BM25-style) vector representation of a text chunk.

    Attributes:
        chunk_id: UUID of the associated TextChunk.
        indices: Non-zero dimension indices.
        values: Corresponding non-zero values.
    """

    chunk_id: str
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddedChunk:
    """A text chunk bundled with its vector representations, ready for upsert.

    Attributes:
        chunk: The original text chunk with all metadata.
        dense: Dense embedding vector (always present).
        sparse: Sparse BM25 embedding (optional; omitted when using server-side inference).
    """

    chunk: TextChunk
    dense: DenseEmbedding
    sparse: SparseEmbedding | None = None


@dataclass(frozen=True)
class SearchResult:
    """A single result returned by a vector store query.

    Attributes:
        chunk: The retrieved text chunk with all metadata.
        score: Relevance score assigned by the vector store (higher is better).
    """

    chunk: TextChunk
    score: float


Query = str


@dataclass(frozen=True)
class Answer:
    """The final answer produced by the retrieval pipeline.

    Attributes:
        query: The original user question.
        sources: Retrieved chunks that grounded the answer.
        generated_answer: LLM-generated response text, including inline citations.
    """

    query: Query
    sources: list[SearchResult]
    generated_answer: str


@dataclass(frozen=True)
class CitedSource:
    """A single source actually cited (via ``[N]``) in a generated answer.

    Attributes:
        index: The ``[N]`` citation number as it appears in the answer text.
        file_name: Name of the cited source file.
        link_preview: Public preview URL for the cited source, if any.
        page_start: First page this citation refers to.
        page_end: Last page this citation refers to.
    """

    index: int
    file_name: str
    link_preview: str | None
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class TextDeltaEvent:
    """A newly available fragment of the streamed answer text."""

    delta: str


@dataclass(frozen=True)
class SourcesEvent:
    """The final list of cited sources, available once generation is complete."""

    sources: list[CitedSource]


@dataclass(frozen=True)
class QuestionsEvent:
    """Suggested follow-up questions, available once generation is complete."""

    questions: list[str]


StreamEvent = TextDeltaEvent | SourcesEvent | QuestionsEvent


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt string used throughout the pipeline.

    Attributes:
        version: Semantic version of the prompt (e.g. ``"1.0"``).
        content: Raw prompt text sent to the model.
    """

    version: str
    content: str
