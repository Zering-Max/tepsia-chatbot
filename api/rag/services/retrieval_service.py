"""Retrieval orchestration service.

Coordinates the embedding port and the vector store port to turn a raw query
into a ranked list of relevant chunks via hybrid search.
"""

from ..domain.models import TextChunk, DenseEmbedding, SearchResult
from ..ports.embedding import DenseEmbedder
from ..ports.vector_store import VectorStore


class RetrievalService:
    """Orchestrates hybrid search over the vector store.

    Attributes:
        _embedder: Encodes the query into a dense vector.
        _vector_store: Executes hybrid search and returns ranked chunks.
        _top_k: Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        embedder: DenseEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        """Wires the service with its required port implementations.

        Args:
            embedder: Dense embedder used to encode the query.
            vector_store: Vector store used for hybrid search.
            top_k: Number of chunks to retrieve per query.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve(self, query: str) -> list[SearchResult]:
        """Encodes a query and retrieves the most relevant chunks.

        Args:
            query: Natural-language question from the user.

        Returns:
            Top-k search results ordered by relevance.
        """
        query_embedded = await self._embedder.embed(chunks=[TextChunk(content=query)])
        result: DenseEmbedding = query_embedded[0]

        return await self._vector_store.hybrid_search(
            query_vector=result.vector,
            query_text=query,
            k=self._top_k,
        )
