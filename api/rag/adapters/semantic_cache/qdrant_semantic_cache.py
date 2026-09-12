"""Qdrant-backed semantic cache adapter.

Implements the :class:`SemanticCache` port against a dedicated Qdrant
collection (separate from the documentary collection): one plain dense
vector per cached answer, no named vectors, no sparse/BM25 branch.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, ScoredPoint

from ...domain.models import CachedAnswer, CitedSource, DenseEmbedding, SourcesEvent
from ...ports.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class QdrantSemanticCache(SemanticCache):
    """SemanticCache backed by a dedicated Qdrant collection.

    Attributes:
        async_qdrant_client: Async Qdrant client used for all operations.
        collection_name: Name of the target Qdrant collection.
    """

    async_qdrant_client: AsyncQdrantClient
    collection_name: str

    async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None:
        """Searches for the closest cached question and returns it if similar enough.

        Args:
            embedded_query: Dense embedding of the incoming question.

        Returns:
            The nearest cached answer if its score is at least
            `SIMILARITY_THRESHOLD`, otherwise None.
        """
        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedded_query.vector,
            with_payload=True,
            limit=1,
        )
        if not response.points or response.points[0].score < self.SIMILARITY_THRESHOLD:
            return None
        cached = self._to_cached_answer(response.points[0], embedded_query)
        logger.info(
            "Semantic cache hit for query %r (score=%.4f).",
            cached.query,
            response.points[0].score,
        )
        return cached

    async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None:
        """Upserts a cached answer, keyed by its own id.

        Args:
            cached_answer: The entry to store.
        """
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[self._prepare_point(cached_answer)],
        )
        logger.info(
            "Cached answer for query %r into '%s'.", cached_answer.query, self.collection_name
        )

    @staticmethod
    def _prepare_point(cached_answer: CachedAnswer) -> PointStruct:
        """Converts a CachedAnswer into a Qdrant PointStruct.

        Args:
            cached_answer: The entry to convert.

        Returns:
            A PointStruct ready for upsert.
        """
        payload = {
            "date": cached_answer.date.isoformat(),
            "query": cached_answer.query,
            "retrieved_chunk_ids": cached_answer.retrieved_chunk_ids,
            "cited_sources": [
                {
                    "index": source.index,
                    "file_name": source.file_name,
                    "link_preview": source.link_preview,
                    "page_start": source.page_start,
                    "page_end": source.page_end,
                }
                for source in cached_answer.cited_sources.sources
            ],
            "generated_answer": cached_answer.generated_answer,
            "questions": cached_answer.questions,
        }
        return PointStruct(
            id=cached_answer.id,
            vector=cached_answer.embedded_query.vector,
            payload=payload,
        )

    @staticmethod
    def _to_cached_answer(point: ScoredPoint, embedded_query: DenseEmbedding) -> CachedAnswer:
        """Reconstructs a CachedAnswer from a Qdrant scored point.

        Args:
            point: The top match returned by `query_points`.
            embedded_query: The embedding of the *incoming* question. A hit
                carries this rather than the stored vector — nothing
                downstream needs the original cached vector, and fetching it
                would require `with_vectors=True` on every lookup.

        Returns:
            The cached answer reconstructed from the point's payload.
        """
        p = point.payload
        cited_sources = SourcesEvent(
            sources=[
                CitedSource(
                    index=s["index"],
                    file_name=s["file_name"],
                    link_preview=s["link_preview"],
                    page_start=s["page_start"],
                    page_end=s["page_end"],
                )
                for s in p["cited_sources"]
            ]
        )
        return CachedAnswer(
            id=str(point.id),
            date=datetime.fromisoformat(p["date"]),
            query=p["query"],
            embedded_query=embedded_query,
            retrieved_chunk_ids=p["retrieved_chunk_ids"],
            cited_sources=cited_sources,
            generated_answer=p["generated_answer"],
            questions=p["questions"],
        )
