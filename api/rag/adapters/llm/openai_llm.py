"""OpenAI-backed LLM provider adapter.

Implements the :class:`LLMProvider` port using OpenAI's chat completions API.
Builds the grounded prompt from retrieved passages, generates answers (batch or
streamed), extracts the ``[N]`` citations back into structured sources, and
produces follow-up questions.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI
from ...domain.models import (
    Answer,
    CitedSource,
    Query,
    QuestionsEvent,
    SearchResult,
    SourcesEvent,
    StreamEvent,
    TextChunkMetadata,
    TextDeltaEvent,
)
from ...domain.prompts import FOLLOWUP_QUESTIONS_PROMPT, NO_INFO_SENTINEL, RAG_SYSTEM_PROMPT
from ...ports.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLMProvider(LLMProvider):
    """LLM provider backed by the OpenAI chat completions API.

    Attributes:
        async_openai_client: Async OpenAI client used for chat completions.
        model: Name of the chat model (e.g. ``"gpt-4.1-mini"``).
    """

    async_openai_client: AsyncOpenAI
    model: str

    _CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    async def generate(self, query: Query, sources: list[SearchResult]) -> Answer:
        """Generates a complete, non-streamed answer grounded in the sources.

        Deduplicates the sources, calls the chat model once, and appends a
        formatted list of the sources actually cited in the answer.

        Args:
            query: The user's question.
            sources: Retrieved chunks used to ground the answer.

        Returns:
            The generated answer with its cited-sources footer.
        """
        unique_sources = self._deduplicate(sources)
        response = await self.async_openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT.content},
                {"role": "user", "content": self._build_user_message(query, unique_sources)},
            ],
        )
        generated_answer = response.choices[0].message.content or ""
        generated_answer += self._build_cited_sources(generated_answer, unique_sources)
        logger.info("LLM generated answer (%d chars).", len(generated_answer))
        return Answer(query=query, sources=unique_sources, generated_answer=generated_answer)

    async def generate_stream(
        self, query: Query, sources: list[SearchResult]
    ) -> AsyncIterator[StreamEvent]:
        """Streams the answer token by token, then emits the cited sources.

        Deduplicates the sources and streams the chat completion, yielding a
        :class:`TextDeltaEvent` per content fragment. Once the stream ends, the
        full text is scanned for ``[N]`` citations and a single
        :class:`SourcesEvent` is emitted with the matching sources.

        Args:
            query: The user's question.
            sources: Retrieved chunks used to ground the answer.

        Yields:
            A :class:`TextDeltaEvent` per token, followed by one
            :class:`SourcesEvent` with the cited sources.
        """
        unique_sources = self._deduplicate(sources)
        stream = await self.async_openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT.content},
                {"role": "user", "content": self._build_user_message(query, unique_sources)},
            ],
            stream=True,
        )
        full_text = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield TextDeltaEvent(delta=delta)
        yield SourcesEvent(sources=self._extract_cited_sources(full_text, unique_sources))

    async def generate_followup_questions(
        self, query: str, answer: str, sources: list[SearchResult]
    ) -> QuestionsEvent:
        """Generates follow-up questions answerable from the retrieved passages.

        Skips the API call entirely when the answer contains the no-info
        sentinel. Failures (API error, malformed JSON) are logged and yield
        an empty event so the caller can degrade gracefully.

        Args:
            query: The original user question.
            answer: The generated answer text.
            sources: Retrieved passages the questions must be answerable from.

        Returns:
            A QuestionsEvent with up to 3 follow-up questions (possibly empty).
        """
        if NO_INFO_SENTINEL in answer:
            return QuestionsEvent(questions=[])
        try:
            unique_sources = self._deduplicate(sources)
            response = await self.async_openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": FOLLOWUP_QUESTIONS_PROMPT.content},
                    {
                        "role": "user",
                        "content": (
                            f"{self._build_user_message(query, unique_sources)}"
                            f"\n\nRéponse : {answer}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            questions = [q for q in payload.get("questions", []) if isinstance(q, str) and q.strip()]
            return QuestionsEvent(questions=questions[:3])
        except Exception:
            logger.exception("Failed to generate follow-up questions.")
            return QuestionsEvent(questions=[])

    @staticmethod
    def _build_user_message(query: Query, sources: list[SearchResult]) -> str:
        passages = []
        for i, result in enumerate(sources, 1):
            chunk = result.chunk
            meta = chunk.metadata
            file_name = meta.file_name if meta else "document inconnu"
            page_info = OpenAILLMProvider._format_page_range(meta)
            header = f"[{i}] {file_name}" + (f", {page_info}" if page_info else "")
            passages.append(f"{header}\n{chunk.content}")
        context = "\n\n".join(passages)
        return f"Passages :\n{context}\n\nQuestion : {query}"

    @classmethod
    def _extract_cited_indices(cls, text: str) -> set[int]:
        """Collects the ``[N]`` (and grouped ``[N, M]``) citation numbers in a text."""
        indices: set[int] = set()
        for group in cls._CITATION_PATTERN.findall(text):
            indices.update(int(n) for n in group.split(","))
        return indices

    @classmethod
    def _extract_cited_sources(
        cls, text: str, sources: list[SearchResult]
    ) -> list[CitedSource]:
        cited: list[CitedSource] = []
        for idx in sorted(cls._extract_cited_indices(text)):
            if not 1 <= idx <= len(sources):
                continue
            meta = sources[idx - 1].chunk.metadata
            cited.append(
                CitedSource(
                    index=idx,
                    file_name=meta.file_name if meta else "document inconnu",
                    link_preview=meta.link_preview if meta else None,
                    page_start=meta.page_start if meta else None,
                    page_end=meta.page_end if meta else None,
                )
            )
        return cited

    @staticmethod
    def _format_page_range(meta: TextChunkMetadata | None) -> str:
        if meta and meta.page_start and meta.page_end and meta.page_start != meta.page_end:
            return f"p. {meta.page_start}–{meta.page_end}"
        if meta and meta.page_start:
            return f"p. {meta.page_start}"
        return ""

    @classmethod
    def _build_cited_sources(cls, text: str, sources: list[SearchResult]) -> str:
        lines = ["\n\n---\n**Sources :**"]
        for idx in sorted(cls._extract_cited_indices(text)):
            if 1 <= idx <= len(sources):
                meta = sources[idx - 1].chunk.metadata
                file_name = meta.file_name if meta else "document inconnu"
                page_info = cls._format_page_range(meta)
                suffix = f", {page_info}" if page_info else ""
                lines.append(f"[{idx}] {file_name}{suffix}")
        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _deduplicate(sources: list[SearchResult]) -> list[SearchResult]:
        seen: set[tuple] = set()
        unique: list[SearchResult] = []
        for result in sources:
            meta = result.chunk.metadata
            key = (
                meta.file_name if meta else None,
                meta.page_start if meta else None,
                meta.page_end if meta else None,
            )
            if key not in seen:
                seen.add(key)
                unique.append(result)
        return unique
