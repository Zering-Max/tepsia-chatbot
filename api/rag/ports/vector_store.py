"""Port for vector storage and retrieval.

Defines the interface for indexing embedded chunks and searching them, so the
pipeline stays independent of any specific vector database.
"""

from abc import ABC, abstractmethod

from ..domain.models import EmbeddedChunk, SearchResult


class VectorStore(ABC):
    """Stores embedded chunks and retrieves them by similarity."""

    @abstractmethod
    async def upsert_points(self, items: list[EmbeddedChunk]) -> None:
        """Inserts or overwrites embedded chunks in the store.

        Args:
            items: Embedded chunks to index.
        """
        ...

    @abstractmethod
    async def document_exists(self, doc_id: str) -> bool:
        """Checks whether any chunk of a given document is already indexed.

        Args:
            doc_id: Identifier of the source document.

        Returns:
            True if at least one chunk with this document id exists.
        """
        ...

    @abstractmethod
    async def dense_search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        """Searches the store using dense vector similarity.

        Args:
            query_vector: Dense embedding of the query.
            k: Number of results to return.

        Returns:
            Top-k results ordered by descending similarity.
        """
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        k: int,
        prefetch_limit: int | None = None,
    ) -> list[SearchResult]:
        """Searches using both dense and sparse (BM25) signals, then fuses them.

        Args:
            query_vector: Dense embedding of the query.
            query_text: Raw query text, used for the sparse/BM25 branch.
            k: Number of results to return.
            prefetch_limit: Candidates retrieved per branch before fusion.

        Returns:
            Top-k results ordered by descending fused score.
        """
        pass
