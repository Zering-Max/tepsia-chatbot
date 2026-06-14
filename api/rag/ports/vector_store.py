from abc import ABC, abstractmethod

from ..domain.models import EmbeddedChunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    async def upsert_points(self, items: list[EmbeddedChunk]) -> None:
        ...

    @abstractmethod
    async def document_exists(self, doc_id: str) -> bool:
        ...

    @abstractmethod
    async def dense_search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        ...
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        k: int,
        prefetch_limit: int | None = None,
    ) -> list[SearchResult]:
        pass