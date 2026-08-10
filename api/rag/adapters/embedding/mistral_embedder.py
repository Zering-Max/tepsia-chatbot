"""Mistral-backed dense embedding adapter.

Implements the :class:`DenseEmbedder` port using Mistral's embeddings API,
batching chunks per request and sending batches concurrently.
"""

import asyncio
import logging
from dataclasses import dataclass

from mistralai.client import Mistral

from ...domain.models import DenseEmbedding, TextChunk
from ...ports.embedding import DenseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CONCURRENCY = 4


@dataclass
class MistralDenseEmbedder(DenseEmbedder):
    """DenseEmbedder backed by the Mistral Embeddings API.

    Unlike the OpenAI adapter, which issues one request per chunk, this adapter
    groups chunks into batches because the Mistral Embeddings API accepts a list
    of inputs per call. Batches are sent concurrently, bounded by a semaphore to
    stay within rate limits.

    Attributes:
        mistral_client: Authenticated Mistral client.
        model: Embeddings model identifier (e.g. 'mistral-embed', 1024 dims).
        batch_size: Number of chunks sent per API request.
        max_concurrency: Maximum number of in-flight requests.
    """

    mistral_client: Mistral
    model: str
    batch_size: int = DEFAULT_BATCH_SIZE
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        """Embeds a list of chunks in concurrent batches using the Mistral API.

        Args:
            chunks: Text chunks to embed.

        Returns:
            Dense embeddings in the same order as the input chunks.
        """
        if not chunks:
            return []

        batches = [
            chunks[i : i + self.batch_size]
            for i in range(0, len(chunks), self.batch_size)
        ]
        logger.info(
            "Embedding %d chunks with model '%s' (%d batch(es) of up to %d).",
            len(chunks),
            self.model,
            len(batches),
            self.batch_size,
        )

        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = await asyncio.gather(
            *[self._embed_batch(batch, semaphore) for batch in batches]
        )
        return [embedding for batch in results for embedding in batch]

    async def _embed_batch(
        self, batch: list[TextChunk], semaphore: asyncio.Semaphore
    ) -> list[DenseEmbedding]:
        """Fetches embedding vectors for a single batch of chunks.

        Args:
            batch: Text chunks sent together in one API request.
            semaphore: Guard limiting the number of concurrent requests.

        Returns:
            Dense embeddings in the same order as the input batch.

        Raises:
            ValueError: If the API returns a different number of vectors than
                requested, or a vector is missing.
        """
        async with semaphore:
            response = await self.mistral_client.embeddings.create_async(
                model=self.model,
                inputs=[chunk.content for chunk in batch],
            )

        if len(response.data) != len(batch):
            raise ValueError(
                f"Mistral returned {len(response.data)} embeddings for {len(batch)} inputs."
            )

        # The API echoes each input's position in `index`; sort on it rather than
        # trusting the response order.
        ordered = sorted(response.data, key=lambda item: item.index or 0)

        embeddings: list[DenseEmbedding] = []
        for chunk, item in zip(batch, ordered):
            if item.embedding is None:
                raise ValueError(
                    f"Mistral returned an empty embedding at index {item.index}."
                )
            embeddings.append(DenseEmbedding(chunk.id, vector=item.embedding))
        return embeddings
