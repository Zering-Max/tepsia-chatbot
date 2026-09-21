import logging
from dataclasses import dataclass

from domain.models import EmbeddedChunk
from ports.vector_store import VectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"


@dataclass
class QdrantVectorStore(VectorStore):
    """VectorStore backed by Qdrant Cloud.

    Attributes:
        async_qdrant_client: Async Qdrant client used for all operations.
        collection_name: Name of the target Qdrant collection.
    """

    async_qdrant_client: AsyncQdrantClient
    collection_name: str

    async def collection_exists(self) -> bool:
        """Checks whether the target collection exists in Qdrant.

        Returns:
            True if the collection exists, False otherwise.
        """
        return await self.async_qdrant_client.collection_exists(
            collection_name=self.collection_name
        )

    async def create_collection(self, dense_size: int) -> None:
        """Creates the collection with a dense vector, a BM25 sparse vector and payload indexes.

        The sparse vector uses the IDF modifier, which BM25 scoring requires.

        Args:
            dense_size: Dimension of the dense embeddings.

        Raises:
            ValueError: If the collection already exists.
        """
        if await self.collection_exists():
            raise ValueError(f"Collection '{self.collection_name}' already exists.")

        await self.async_qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=24, ef_construct=256),
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                    modifier=models.Modifier.IDF,
                )
            },
        )

        indexes: dict[str, models.PayloadSchemaType | models.TextIndexParams] = {
            "doc_id": models.PayloadSchemaType.KEYWORD,
            "file_name": models.PayloadSchemaType.KEYWORD,
            "section_title": models.PayloadSchemaType.KEYWORD,
            "link_preview": models.PayloadSchemaType.KEYWORD,
            "chunk_index": models.PayloadSchemaType.INTEGER,
            "page_start": models.PayloadSchemaType.INTEGER,
            "page_end": models.PayloadSchemaType.INTEGER,
            "text": models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WHITESPACE,
                lowercase=True,
                phrase_matching=True,
            ),
        }
        for field_name, field_schema in indexes.items():
            await self.async_qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        logger.info("Created collection '%s' (dense size %d).", self.collection_name, dense_size)

    async def upsert_points(self, items: list[EmbeddedChunk]) -> None:
        """Inserts or overwrites embedded chunks in Qdrant.

        Waits for the server to confirm the write before returning.

        Args:
            items: Embedded chunks to insert or overwrite.

        Raises:
            UnexpectedResponse: If Qdrant rejects the operation.
        """
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=self._prepare_points(items),
        )
        logger.info("Upserted %d points into '%s'.", len(items), self.collection_name)

    async def document_exists(self, doc_id: str) -> bool:
        """Checks whether any chunk from a given document is already indexed.

        Args:
            doc_id: SHA-256 hash of the source file, stored in the payload as ``doc_id``.

        Returns:
            True if at least one point with this doc_id exists, False otherwise.
        """
        points, _ = await self.async_qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
        )
        return len(points) > 0

    @staticmethod
    def _prepare_points(items: list[EmbeddedChunk]) -> list[PointStruct]:
        """Converts a list of EmbeddedChunks into Qdrant PointStructs.

        Args:
            items: Embedded chunks to convert.

        Returns:
            List of PointStructs ready for upsert.
        """
        return [QdrantVectorStore._prepare_single_point(item) for item in items]

    @staticmethod
    def _prepare_single_point(item: EmbeddedChunk) -> PointStruct:
        """Converts a single EmbeddedChunk into a Qdrant PointStruct.

        Uses named-vector format required by collections with multiple vector spaces.

        Args:
            item: Embedded chunk containing text, metadata, and dense vector.

        Returns:
            A PointStruct ready for upsert.
        """
        payload = {
            "doc_id": item.chunk.document_id,
            "chunk_index": item.chunk.chunk_index,
            "text": item.chunk.content,
            "page_start": item.chunk.metadata.page_start,
            "page_end": item.chunk.metadata.page_end,
            "section_title": item.chunk.metadata.section_title,
            "file_name": item.chunk.metadata.file_name,
            'link_preview': item.chunk.metadata.link_preview
        }
        return PointStruct(
            id=item.chunk.id,
            vector={DENSE_VECTOR_NAME: item.dense.vector, SPARSE_VECTOR_NAME: models.Document(text=item.chunk.content, model="Qdrant/bm25")},
            payload=payload,
        )

