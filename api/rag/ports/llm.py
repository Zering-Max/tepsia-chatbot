"""Port for large language model generation.

Defines the interface for producing answers (batch and streamed) and follow-up
questions from a query and its retrieved sources, keeping the pipeline
independent of any specific LLM provider.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..domain.models import Answer, QuestionsEvent, SearchResult, StreamEvent


class LLMProvider(ABC):
    """Generates grounded answers and follow-up questions from retrieved sources."""

    @abstractmethod
    async def generate(self, query: str, sources: list[SearchResult]) -> Answer:
        """Generates a complete answer grounded in the retrieved sources.

        Args:
            query: The user's question.
            sources: Retrieved chunks used to ground the answer.

        Returns:
            The generated answer with its citations and sources.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self, query: str, sources: list[SearchResult]
    ) -> AsyncIterator[StreamEvent]:
        """Streams the answer as it is generated.

        Args:
            query: The user's question.
            sources: Retrieved chunks used to ground the answer.

        Yields:
            Stream events: text fragments followed by the cited sources.
        """
        ...

    @abstractmethod
    async def generate_followup_questions(
        self, query: str, answer: str, sources: list[SearchResult]
    ) -> QuestionsEvent:
        """Proposes follow-up questions answerable from the retrieved sources.

        Args:
            query: The original user question.
            answer: The answer that was generated.
            sources: Retrieved chunks the questions must be answerable from.

        Returns:
            An event holding up to three follow-up questions (possibly empty).
        """
        ...
