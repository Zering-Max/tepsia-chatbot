from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..domain.models import Answer, QuestionsEvent, SearchResult, StreamEvent


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, query: str, sources: list[SearchResult]) -> Answer:
        ...

    @abstractmethod
    async def generate_stream(
        self, query: str, sources: list[SearchResult]
    ) -> AsyncIterator[StreamEvent]:
        ...

    @abstractmethod
    async def generate_followup_questions(
        self, query: str, answer: str, sources: list[SearchResult]
    ) -> QuestionsEvent:
        ...
