"""Port for the semantic answer cache.

Defines the interface for looking up and storing previously generated
answers by semantic similarity of their question, so the pipeline can skip
retrieval and generation entirely on a close-enough repeat question.
"""

from abc import ABC, abstractmethod

from ..domain.models import CachedAnswer, DenseEmbedding


class SemanticCache(ABC):
    """Looks up and stores answers keyed by the semantic similarity of their question."""

    SIMILARITY_THRESHOLD: float = 0.97

    @abstractmethod
    async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None:
        """Returns the closest cached answer if it is similar enough.

        Args:
            embedded_query: Dense embedding of the incoming question.

        Returns:
            The nearest cached answer if its similarity score is at least
            `SIMILARITY_THRESHOLD`, otherwise None.
        """
        ...

    @abstractmethod
    async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None:
        """Inserts or overwrites a cached answer.

        Args:
            cached_answer: The entry to store, keyed by `cached_answer.id`.
        """
        ...
