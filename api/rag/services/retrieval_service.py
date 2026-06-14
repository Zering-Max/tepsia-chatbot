from ..domain.models import TextChunk, DenseEmbedding, SearchResult
from ..ports.embedding import DenseEmbedder
from ..ports.vector_store import VectorStore


class RetrievalService:
    def __init__(
        self,
        embedder: DenseEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve(self, query: str) -> list[SearchResult]:
        query_embedded = await self._embedder.embed(chunks=[TextChunk(content=query)])
        result: DenseEmbedding = query_embedded[0]
        return await self._vector_store.hybrid_search(
            query_vector=result.vector, 
            query_text=query,
            k=8
        )
