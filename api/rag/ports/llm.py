from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..domain.models import Answer, SearchResult


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, query: str, sources: list[SearchResult]) -> Answer:
        ...

    @abstractmethod
    async def generate_stream(
        self, query: str, sources: list[SearchResult]
    ) -> AsyncIterator[str]:
        ...
