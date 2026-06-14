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
class Prompt:
    version: str
    content: str
