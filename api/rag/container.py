"""Dependency wiring for the RAG pipeline.

Composition root that builds the concrete adapters from :data:`settings` and
assembles them into the services the API depends on. Embeddings and generation
both run on Mistral; the Mistral adapters open their own Langfuse observations,
since the ``langfuse.openai`` drop-in used previously does not cover this SDK.
"""

from mistralai.client import Mistral
from qdrant_client import AsyncQdrantClient

from .adapters.embedding.mistral_embedder import MistralDenseEmbedder
from .adapters.llm.mistral_llm import MistralLLMProvider
from .adapters.vectorstore.qdrant_store import QdrantVectorStore
from .config import settings
from .ports.llm import LLMProvider
from .services.retrieval_service import RetrievalService


async def build_retrieval_service() -> RetrievalService:
    """Builds the retrieval service with its Mistral and Qdrant adapters.

    Returns:
        A :class:`RetrievalService` wired with a Mistral dense embedder and a
        Qdrant vector store, configured from :data:`settings`.
    """
    embedder = MistralDenseEmbedder(
        mistral_client=Mistral(api_key=settings.mistral_api_key),
        model=settings.mistral_embedding_model,
    )
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    vector_store = QdrantVectorStore(
        async_qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
    )
    return RetrievalService(embedder=embedder, vector_store=vector_store, top_k=8)


def build_llm_provider() -> LLMProvider:
    """Builds the Mistral-backed LLM provider.

    Returns:
        A :class:`MistralLLMProvider` configured from :data:`settings`.
    """
    return MistralLLMProvider(
        mistral_client=Mistral(api_key=settings.mistral_api_key),
        model=settings.mistral_llm_model,
    )
