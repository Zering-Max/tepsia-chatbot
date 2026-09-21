import logging

from google.genai import Client as GeminiAPIClient
from mistralai.client import Mistral
from qdrant_client import AsyncQdrantClient

from adapters.chunking.markdown_chunker import MarkdownChunker
from adapters.embedding.mistral_embedder import MistralDenseEmbedder
from adapters.parsing.gemini_document_parser import GeminiDocumentParser
from adapters.vectorstore.qdrant_store import QdrantVectorStore
from config import settings
from domain.exceptions import CollectionNotFoundError
from domain.prompts import PARSING_PROMPT
from ports.embedding import DenseEmbedder
from services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# Output dimension of the Mistral embedder; a collection must be created with it.
DENSE_VECTOR_SIZE = 1024


class ServiceContainer:
    """Factory class that wires adapters to ports and builds ready-to-use services.

    Active providers: Gemini for parsing, Mistral for dense embeddings, Qdrant
    for storage (BM25 sparse vectors are computed server-side by Qdrant Cloud).
    Changing the embedder means changing ``_build_embedder`` and ``DENSE_VECTOR_SIZE``,
    and creating a collection with the matching dimension.
    """

    @staticmethod
    async def build_ingestion_service(collection_name: str | None = None) -> IngestionService:
        """Wires all ingestion adapters and returns a ready-to-use IngestionService.

        Args:
            collection_name: Target Qdrant collection. Defaults to
                ``QDRANT_COLLECTION_NAME`` from the ``.env`` file.

        Returns:
            A fully wired IngestionService instance.

        Raises:
            CollectionNotFoundError: If the target collection does not exist.
        """
        parser = GeminiDocumentParser(
            gemini_client=GeminiAPIClient(api_key=settings.gemini_api_key),
            parsing_prompt=PARSING_PROMPT,
            model=settings.gemini_parsing_model,
        )
        vector_store = ServiceContainer.build_vector_store(collection_name)
        if not await vector_store.collection_exists():
            raise CollectionNotFoundError(vector_store.collection_name)
        logger.info("Target collection: '%s'", vector_store.collection_name)
        return IngestionService(
            parser=parser,
            chunker=MarkdownChunker(),
            embedder=ServiceContainer._build_embedder(),
            vector_store=vector_store,
        )

    @staticmethod
    def build_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
        """Builds the vector store client for a given collection.

        ``cloud_inference`` is enabled so that BM25 sparse vectors are computed
        server-side by Qdrant Cloud rather than locally.

        Args:
            collection_name: Target collection. Defaults to ``QDRANT_COLLECTION_NAME``.

        Returns:
            A VectorStore backed by Qdrant Cloud.
        """
        return QdrantVectorStore(
            async_qdrant_client=AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                cloud_inference=True,
            ),
            collection_name=collection_name or settings.qdrant_collection_name,
        )

    @staticmethod
    def _build_embedder() -> DenseEmbedder:
        """Builds the dense embedder.

        Returns:
            A DenseEmbedder backed by the Mistral Embeddings API (1024 dims).
        """
        return MistralDenseEmbedder(
            mistral_client=Mistral(api_key=settings.mistral_api_key),
            model=settings.mistral_embedding_model,
        )
