from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    file_name: str
    file_path: Path
    doctype: str
    parser_name: str | None = None
    parser_version: str | None = None


@dataclass(frozen=True)
class Page:
    page_number: int
    page_content: str

@dataclass(frozen=True)
class ParsedDocumentResponse:
    pages: list[Page]


@dataclass(frozen=True)
class Document:
    id: str
    metadata: DocumentMetadata
    content: str
    pages: tuple[Page, ...] = ()

@dataclass(frozen=True)
class TextChunkMetadata:
    chunk_index: int
    section_title: str | None
    file_name: str
    file_path: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class TextChunk:
    content: str
    id: str | None = None
    document_id: str | None = None
    chunk_index: int | None = None
    metadata: TextChunkMetadata | None = None

@dataclass(frozen=True)
class DenseEmbedding:
    chunk_id: str
    vector: list[float]


@dataclass(frozen=True)
class SparseEmbedding:
    chunk_id: str
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: TextChunk
    dense: DenseEmbedding
    sparse: SparseEmbedding | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


Query = str


@dataclass(frozen=True)
class Answer:
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
    version: str
    content: str
