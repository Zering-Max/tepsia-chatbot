import logging
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, ScoredPoint

from ...domain.models import EmbeddedChunk, SearchResult, TextChunk, TextChunkMetadata
from ...ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"



@dataclass
class QdrantVectorStore(VectorStore):
    async_qdrant_client: AsyncQdrantClient
    collection_name: str

    async def upsert_points(self, items: list[EmbeddedChunk]) -> None:
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=self._prepare_points(items),
        )
        logger.info("Upserted %d points into '%s'.", len(items), self.collection_name)

    async def document_exists(self, doc_id: str) -> bool:
        points, _ = await self.async_qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
        )
        return len(points) > 0

    async def dense_search(self, query_vector: list[float], k: int) -> list[SearchResult]:
        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            with_payload=True,
            limit=k,
        )
        results = [self._to_search_result(point) for point in response.points]
        logger.info("Dense search returned %d results from '%s'.", len(results), self.collection_name)
        return results
    
    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        k: int,
        prefetch_limit: int | None = None,
    ) -> list[SearchResult]:
        """Searches the collection using both dense and sparse (BM25) vectors,
        fused with Reciprocal Rank Fusion (RRF).

        Args:
            query_vector: Dense embedding of the query (1536-dim).
            query_text: Raw query text, used for server-side BM25 inference.
            k: Number of results to return.
            prefetch_limit: Number of candidates retrieved per branch before
                fusion. Defaults to max(k * 5, 50).

        Returns:
            Top-k results ordered by descending fused score.
        """
        prefetch_limit = prefetch_limit or max(k * 5, 50)

        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model="Qdrant/bm25"),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=k,
        )

        results = [self._to_search_result(point) for point in response.points]
        logger.info("Hybrid search returned %d results from '%s'.", len(results), self.collection_name)
        return results

    @staticmethod
    def _to_search_result(point: ScoredPoint) -> SearchResult:
        if point.payload is None:
            raise Exception("Qdrant point has no payload")
        p = point.payload
        metadata = TextChunkMetadata(
            chunk_index=p["chunk_index"],
            page_start=p["page_start"],
            page_end=p["page_end"],
            section_title=p["section_title"],
            file_name=p["file_name"],
            file_path=p["file_path"],
        )
        chunk = TextChunk(
            id=str(point.id),
            document_id=p["doc_id"],
            content=p["text"],
            chunk_index=p["chunk_index"],
            metadata=metadata,
        )
        return SearchResult(chunk=chunk, score=point.score)

    @staticmethod
    def _prepare_points(items: list[EmbeddedChunk]) -> list[PointStruct]:
        return [QdrantVectorStore._prepare_single_point(item) for item in items]

    @staticmethod
    def _prepare_single_point(item: EmbeddedChunk) -> PointStruct:
        payload = {
            "doc_id": item.chunk.document_id,
            "chunk_index": item.chunk.chunk_index,
            "text": item.chunk.content,
            "page_start": item.chunk.metadata.page_start,
            "page_end": item.chunk.metadata.page_end,
            "section_title": item.chunk.metadata.section_title,
            "file_name": item.chunk.metadata.file_name,
            "file_path": str(item.chunk.metadata.file_path),
        }
        return PointStruct(
            id=item.chunk.id,
            vector={DENSE_VECTOR_NAME: item.dense.vector, SPARSE_VECTOR_NAME: models.Document(text=item.chunk.content, model="Qdrant/bm25")},
            payload=payload,
        )
