import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from ...domain.models import Answer, Query, SearchResult
from ...domain.prompts import RAG_SYSTEM_PROMPT
from ...ports.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLMProvider(LLMProvider):
    async_openai_client: AsyncOpenAI
    model: str

    async def generate(self, query: Query, sources: list[SearchResult]) -> Answer:
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
    ) -> AsyncIterator[str]:
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
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield delta
        yield self._build_cited_sources(full_text, unique_sources)

    @staticmethod
    def _build_user_message(query: Query, sources: list[SearchResult]) -> str:
        passages = []
        for i, result in enumerate(sources, 1):
            chunk = result.chunk
            meta = chunk.metadata
            file_name = meta.file_name if meta else "document inconnu"
            if meta and meta.page_start and meta.page_end and meta.page_start != meta.page_end:
                page_info = f"p. {meta.page_start}–{meta.page_end}"
            elif meta and meta.page_start:
                page_info = f"p. {meta.page_start}"
            else:
                page_info = ""
            header = f"[{i}] {file_name}" + (f", {page_info}" if page_info else "")
            passages.append(f"{header}\n{chunk.content}")
        context = "\n\n".join(passages)
        return f"Passages :\n{context}\n\nQuestion : {query}"

    @staticmethod
    def _build_cited_sources(text: str, sources: list[SearchResult]) -> str:
        cited = {int(n) for n in re.findall(r'\[(\d+)\]', text)}
        lines = ["\n\n---\n**Sources :**"]
        for idx in sorted(cited):
            if 1 <= idx <= len(sources):
                meta = sources[idx - 1].chunk.metadata
                file_name = meta.file_name if meta else "document inconnu"
                if meta and meta.page_start and meta.page_end and meta.page_start != meta.page_end:
                    page_info = f", p. {meta.page_start}–{meta.page_end}"
                elif meta and meta.page_start:
                    page_info = f", p. {meta.page_start}"
                else:
                    page_info = ""
                lines.append(f"[{idx}] {file_name}{page_info}")
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
