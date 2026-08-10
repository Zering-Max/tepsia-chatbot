"""Mistral-backed LLM provider adapter.

Implements the :class:`LLMProvider` port using Mistral's chat completions API.
Builds the grounded prompt from retrieved passages, generates answers (batch or
streamed), extracts the ``[N]`` citations back into structured sources, and
produces follow-up questions.

Unlike the OpenAI adapter, which the ``langfuse.openai`` drop-in traces
automatically, this adapter opens its Langfuse generation observations
explicitly: no equivalent drop-in exists for the Mistral SDK.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langfuse import get_client
from mistralai.client import Mistral

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
class MistralLLMProvider(LLMProvider):
    """LLM provider backed by the Mistral chat completions API.

    Attributes:
        mistral_client: Mistral client used for chat completions.
        model: Name of the chat model (e.g. ``"mistral-medium-latest"``).
    """

    mistral_client: Mistral
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
        messages = self._build_messages(
            RAG_SYSTEM_PROMPT.content, self._build_user_message(query, unique_sources)
        )
        with get_client().start_as_current_observation(
            name="mistral-generate", as_type="generation", model=self.model, input=messages
        ) as generation:
            response = await self.mistral_client.chat.complete_async(
                model=self.model,
                messages=messages,
            )
            generated_answer = self._first_choice_text(response)
            generation.update(
                output=generated_answer,
                usage_details=self._usage_details(response.usage),
            )
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
        messages = self._build_messages(
            RAG_SYSTEM_PROMPT.content, self._build_user_message(query, unique_sources)
        )
        full_text = ""
        with get_client().start_as_current_observation(
            name="mistral-generate-stream",
            as_type="generation",
            model=self.model,
            input=messages,
        ) as generation:
            usage = None
            stream = await self.mistral_client.chat.stream_async(
                model=self.model,
                messages=messages,
            )
            async with stream as event_stream:
                async for event in event_stream:
                    chunk = event.data
                    # Mistral reports usage on the terminal chunk, which carries
                    # no choices; keep the last non-empty value seen.
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = self._as_text(chunk.choices[0].delta.content)
                    if delta:
                        full_text += delta
                        yield TextDeltaEvent(delta=delta)
            generation.update(output=full_text, usage_details=self._usage_details(usage))
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
            messages = self._build_messages(
                FOLLOWUP_QUESTIONS_PROMPT.content,
                f"{self._build_user_message(query, unique_sources)}\n\nRéponse : {answer}",
            )
            with get_client().start_as_current_observation(
                name="mistral-followup-questions",
                as_type="generation",
                model=self.model,
                input=messages,
            ) as generation:
                response = await self.mistral_client.chat.complete_async(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                raw = self._first_choice_text(response)
                generation.update(
                    output=raw, usage_details=self._usage_details(response.usage)
                )
            payload = json.loads(raw or "{}")
            questions = [q for q in payload.get("questions", []) if isinstance(q, str) and q.strip()]
            return QuestionsEvent(questions=questions[:3])
        except Exception:
            logger.exception("Failed to generate follow-up questions.")
            return QuestionsEvent(questions=[])

    @staticmethod
    def _build_messages(system_prompt: str, user_message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    @classmethod
    def _first_choice_text(cls, response: Any) -> str:
        """Extracts the assistant text from a non-streamed chat response."""
        if not response.choices:
            return ""
        return cls._as_text(response.choices[0].message.content)

    @staticmethod
    def _as_text(content: Any) -> str:
        """Normalises Mistral message content to plain text.

        The SDK types both message and delta content as
        ``str | list[ContentChunk] | None``. The list form appears when the model
        returns structured chunks; only the text ones carry answer content, so
        anything else (images, references) is dropped rather than stringified.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _usage_details(usage: Any) -> dict[str, int]:
        """Maps Mistral's token counts onto Langfuse's usage keys."""
        if usage is None:
            return {}
        return {
            "input": usage.prompt_tokens or 0,
            "output": usage.completion_tokens or 0,
            "total": usage.total_tokens or 0,
        }

    @staticmethod
    def _build_user_message(query: Query, sources: list[SearchResult]) -> str:
        passages = []
        for i, result in enumerate(sources, 1):
            chunk = result.chunk
            meta = chunk.metadata
            file_name = meta.file_name if meta else "document inconnu"
            page_info = MistralLLMProvider._format_page_range(meta)
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
