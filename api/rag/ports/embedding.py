from abc import ABC, abstractmethod

from ..domain.models import DenseEmbedding, TextChunk


class DenseEmbedder(ABC):
    @abstractmethod
    async def embed(self, chunks: list[TextChunk]) -> list[DenseEmbedding]:
        ...
