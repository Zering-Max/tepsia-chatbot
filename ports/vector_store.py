from abc import ABC, abstractmethod

from domain.models import EmbeddedChunk


class VectorStore(ABC):
    """Port: persists embedded chunks in a vector database."""

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Checks whether the target collection exists.

        Returns:
            True if the collection exists, False otherwise.
        """
        ...

    @abstractmethod
    async def create_collection(self, dense_size: int) -> None:
        """Creates the target collection with the schema expected by ingestion.

        The collection holds one dense vector and one sparse (BM25) vector per
        point, plus the payload indexes used for filtering.

        Args:
            dense_size: Dimension of the dense embeddings (1024 for mistral-embed).

        Raises:
            ValueError: If the collection already exists.
        """
        ...

    @abstractmethod
    async def upsert_points(self, items: list[EmbeddedChunk]) -> None:
        """Inserts or overwrites a batch of embedded chunks.

        Args:
            items: Chunks with their vector representations to store.
        """
        ...

    @abstractmethod
    async def document_exists(self, doc_id: str) -> bool:
        """Checks whether a document is already indexed.

        Args:
            doc_id: Unique identifier (SHA-256 hash) of the document.

        Returns:
            True if at least one chunk from this document exists, False otherwise.
        """
        ...
