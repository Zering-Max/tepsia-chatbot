from abc import ABC, abstractmethod

from domain.models import Document, TextChunk


class TextChunker(ABC):
    """Port: splits a Document into an ordered list of TextChunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[TextChunk]:
        """Splits a document into text chunks.

        Args:
            document: Fully parsed document to split.

        Returns:
            Ordered list of chunks derived from the document.
        """
        ...
