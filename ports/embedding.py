from abc import ABC, abstractmethod

from domain.models import DenseEmbedding, TextChunk


class DenseEmbedder(ABC):
    """Port: encodes TextChunks into dense vector representations."""

    @abstractmethod
    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        """Computes dense embeddings for a list of chunks.

        Args:
            chunks: Text chunks to embed.

        Returns:
            Dense embeddings in the same order as the input.
        """
        ...
