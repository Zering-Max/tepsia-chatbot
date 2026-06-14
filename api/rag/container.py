from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from .adapters.embedding.openai_embedder import OpenAIDenseEmbedder
from .adapters.llm.openai_llm import OpenAILLMProvider
from .adapters.vectorstore.qdrant_store import QdrantVectorStore
from .config import settings
from .services.retrieval_service import RetrievalService


async def build_retrieval_service() -> RetrievalService:
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIDenseEmbedder(
        client_openai=openai_client,
        model=settings.openai_embedding_model,
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


def build_llm_provider() -> OpenAILLMProvider:
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return OpenAILLMProvider(
        async_openai_client=openai_client,
        model=settings.llm_model,
    )
