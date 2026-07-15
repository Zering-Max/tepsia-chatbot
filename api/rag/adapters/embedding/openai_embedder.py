"""OpenAI-backed dense embedding adapter.

Implements the :class:`DenseEmbedder` port using OpenAI's embeddings API,
embedding chunks concurrently for throughput.
"""

import asyncio
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from ...domain.models import DenseEmbedding, TextChunk
from ...ports.embedding import DenseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class OpenAIDenseEmbedder(DenseEmbedder):
    """Dense embedder backed by the OpenAI embeddings API.

    Attributes:
        client_openai: Async OpenAI client used for embedding calls.
        model: Name of the OpenAI embedding model (e.g. ``"text-embedding-3-small"``).
    """

    client_openai: AsyncOpenAI
    model: str

    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        """Embeds a batch of chunks concurrently.

        Args:
            chunks: Text chunks to encode.

        Returns:
            One dense embedding per input chunk, in the same order.
        """
        logger.info("Embedding %d chunks with model '%s'", len(chunks), self.model)
        tasks = [self.single_embed(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def single_embed(self, chunk: TextChunk) -> DenseEmbedding:
        """Embeds a single chunk with one API call.

        Args:
            chunk: The text chunk to encode.

        Returns:
            The dense embedding for the chunk, keyed by its id.
        """
        response = await self.client_openai.embeddings.create(
            input=chunk.content,
            model=self.model
        )
        vector: list[float] = response.data[0].embedding
        return DenseEmbedding(chunk.id, vector=vector)
