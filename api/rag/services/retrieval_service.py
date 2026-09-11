"""Retrieval orchestration service.

Coordinates the embedding port and the vector store port to turn a raw query
into a ranked list of relevant chunks via hybrid search.
"""

import logging

from ..domain.models import TextChunk, DenseEmbedding, SearchResult
from ..ports.embedding import DenseEmbedder
from ..ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


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

    async def retrieve(
        self, query: str, seed_chunk_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """Encodes a query and retrieves the most relevant chunks.

        When `seed_chunk_ids` is given, those chunks are fetched directly and
        merged after the fresh search results (deduplicated by chunk id).
        This carries forward the passages that grounded an earlier turn (e.g.
        a suggested follow-up question) so they stay available even if a
        fresh search on the new query text would not resurface them. A
        failure to fetch seed chunks is logged and ignored — the fresh
        results are still returned.

        Args:
            query: Natural-language question from the user.
            seed_chunk_ids: Chunk ids to guarantee in the result, in addition
                to the fresh search.

        Returns:
            Fresh search results first, then any seed chunks not already
            present among them, deduplicated by chunk id.
        """
        query_embedded = await self._embedder.embed(chunks=[TextChunk(content=query)])
        result: DenseEmbedding = query_embedded[0]

        fresh_results = await self._vector_store.hybrid_search(
            query_vector=result.vector,
            query_text=query,
            k=self._top_k,
        )

        if not seed_chunk_ids:
            return fresh_results

        try:
            seed_results = await self._vector_store.get_by_ids(seed_chunk_ids)
        except Exception:
            logger.exception("Failed to fetch seed chunks by id; using fresh results only.")
            return fresh_results

        seen_ids = {r.chunk.id for r in fresh_results}
        merged = list(fresh_results)
        for seed in seed_results:
            if seed.chunk.id not in seen_ids:
                merged.append(seed)
                seen_ids.add(seed.chunk.id)
        return merged
