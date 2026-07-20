"""Port for dense text embedding.

Defines the interface used to turn text chunks into dense vectors, decoupling
the pipeline from any specific embedding provider.
"""

from abc import ABC, abstractmethod

from ..domain.models import DenseEmbedding, TextChunk


class DenseEmbedder(ABC):
    """Encodes text chunks into dense embedding vectors."""

    @abstractmethod
    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        """Embeds a batch of text chunks into dense vectors.

        Args:
            chunks: Text chunks to encode.

        Returns:
            One dense embedding per input chunk, in the same order.
        """
        ...
