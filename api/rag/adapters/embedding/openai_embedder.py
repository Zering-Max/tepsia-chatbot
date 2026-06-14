import asyncio
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from ...domain.models import DenseEmbedding, TextChunk
from ...ports.embedding import DenseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class OpenAIDenseEmbedder(DenseEmbedder):
    client_openai: AsyncOpenAI
    model: str

    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        logger.info("Embedding %d chunks with model '%s'", len(chunks), self.model)
        tasks = [self.single_embed(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def single_embed(self, chunk: TextChunk) -> DenseEmbedding:
        response = await self.client_openai.embeddings.create(
            input=chunk.content,
            model=self.model
        )
        vector: list[float] = response.data[0].embedding
        return DenseEmbedding(chunk.id, vector=vector)
