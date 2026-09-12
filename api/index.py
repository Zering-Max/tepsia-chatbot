"""FastAPI entry point for the Teps'out RAG chatbot.

Exposes the ``/api/chat`` endpoint, which retrieves the passages relevant to the
user's question, streams a grounded answer as Server-Sent Events (Vercel AI SDK
protocol), then appends the cited sources and suggested follow-up questions.
Application startup/shutdown — building the services and checking Langfuse
tracing — is handled by the lifespan context manager.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request as FastAPIRequest
from langfuse import get_client
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from datetime import datetime, timezone

from .rag.container import build_llm_provider, build_retrieval_service, build_semantic_cache
from .rag.domain.models import CachedAnswer, DenseEmbedding, SourcesEvent, TextDeltaEvent, compute_cache_id
from .rag.domain.prompts import NO_INFO_SENTINEL
from .rag.ports.llm import LLMProvider
from .rag.ports.semantic_cache import SemanticCache
from .rag.services.retrieval_service import RetrievalService
from .utils.prompt import ClientMessage
from .utils.stream import patch_response_with_headers

from vercel.headers import set_headers

load_dotenv(".env.local")

logger = logging.getLogger(__name__)

retrieval_service: RetrievalService | None = None
llm_provider: LLMProvider | None = None
semantic_cache: SemanticCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Builds the services on startup and flushes Langfuse traces on shutdown.

    Instantiates the retrieval service and LLM provider once and stores them as
    module globals. Runs a Langfuse auth check (logged, non-fatal) so tracing
    problems are visible early. On shutdown, flushes any buffered traces.
    """
    global retrieval_service, llm_provider, semantic_cache
    retrieval_service = await build_retrieval_service()
    llm_provider = build_llm_provider()
    semantic_cache = build_semantic_cache()
    try:
        if get_client().auth_check():
            logger.info("Langfuse connecté — tracing actif.")
        else:
            logger.warning("Langfuse : échec d'authentification — tracing inactif.")
    except Exception:
        logger.warning("Langfuse injoignable — tracing inactif.", exc_info=True)
    yield
    get_client().flush()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _vercel_set_headers(request: FastAPIRequest, call_next):
    """Exposes the incoming request headers to the Vercel runtime helpers."""
    set_headers(dict(request.headers))
    return await call_next(request)


class Request(BaseModel):
    """Request body for ``/api/chat``.

    Attributes:
        messages: The chat history sent by the client, oldest first.
    """

    messages: List[ClientMessage]


def _extract_last_user_query(messages: List[ClientMessage]) -> str:
    """Returns the text of the most recent user message.

    Falls back to concatenating the message's text parts when it has no plain
    ``content`` field (Vercel AI SDK message shape).

    Args:
        messages: The chat history, oldest first.

    Returns:
        The latest user query, or an empty string if none is found.
    """
    for message in reversed(messages):
        if message.role == "user":
            if message.content:
                return message.content
            if message.parts:
                texts = [p.text for p in message.parts if p.type == "text" and p.text]
                if texts:
                    return " ".join(texts)
    return ""


def _extract_seed_chunk_ids(messages: List[ClientMessage]) -> list[str]:
    """Returns the seed chunk ids attached to the most recent user message.

    These come from `metadata.seedChunkIds` on a clicked follow-up question
    (see `components/message.tsx`), identifying the passages that grounded
    that question so they can be carried into its retrieval. A freely typed
    question, or malformed metadata, yields an empty list — the caller then
    falls back to a plain fresh search.

    Args:
        messages: The chat history, oldest first.

    Returns:
        The seed chunk ids, or an empty list if none are present/valid.
    """
    for message in reversed(messages):
        if message.role == "user":
            if not message.metadata:
                return []
            seed_ids = message.metadata.get("seedChunkIds")
            if isinstance(seed_ids, list) and all(isinstance(i, str) for i in seed_ids):
                return seed_ids
            return []
    return []


def _split_into_deltas(text: str, chunk_size: int = 40) -> list[str]:
    """Splits text into chunks for a simulated text-delta stream.

    Used to replay a cached answer as a sequence of `text-delta` frames, so a
    cache hit renders with the same incremental appearance as a fresh
    generation. Splits on single spaces and groups words up to `chunk_size`
    characters per delta; no artificial delay is added between yields.

    Args:
        text: The already-generated answer to replay.
        chunk_size: Approximate number of characters per delta.

    Returns:
        Text fragments that, concatenated in order, reproduce `text` exactly.
    """
    words = text.split(" ")
    deltas: list[str] = []
    current = ""
    for i, word in enumerate(words):
        piece = word if i == 0 else " " + word
        if current and len(current) + len(piece) > chunk_size:
            deltas.append(current)
            current = piece
        else:
            current += piece
    if current:
        deltas.append(current)
    return deltas


async def _rag_stream(query: str, seed_chunk_ids: list[str]):
    """Runs the RAG pipeline and yields the answer as SSE frames.

    Emits, in order: a ``start`` frame, the answer text as ``text-delta`` frames,
    the cited sources (``data-sources``), the follow-up questions
    (``data-questions``), and a terminal ``finish``/``[DONE]``. An empty query
    yields a short prompt-for-input message. Any failure is logged, recorded on
    the Langfuse span, and surfaced to the client as an ``error`` frame. The
    whole run is wrapped in a ``rag-chat`` span and flushed on completion.

    Args:
        query: The user's question (already extracted from the request).
        seed_chunk_ids: Chunk ids to carry forward from a clicked follow-up
            question, or an empty list.

    Yields:
        SSE-formatted strings following the Vercel AI SDK UI message protocol.
    """
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    langfuse = get_client()
    message_id = f"msg-{uuid.uuid4().hex}"
    yield sse({"type": "start", "messageId": message_id})
    yield sse({"type": "text-start", "id": "text-1"})

    with langfuse.start_as_current_observation(
        name="rag-chat", as_type="span", input={"query": query}
    ) as span:
        full_answer = ""
        try:
            if not query.strip():
                yield sse({
                    "type": "text-delta",
                    "id": "text-1",
                    "delta": "Veuillez poser une question pour que je puisse vous aider.",
                })
                yield sse({"type": "text-end", "id": "text-1"})
            else:
                embedded_query: DenseEmbedding | None = None
                cached: CachedAnswer | None = None
                if not seed_chunk_ids:
                    try:
                        embedded_query = await retrieval_service.embed_query(query)
                        cached = await semantic_cache.check_similar_questions(embedded_query)
                    except Exception:
                        logger.exception("Semantic cache lookup failed for query: %r", query)
                        cached = None

                if cached is not None:
                    for delta in _split_into_deltas(cached.generated_answer):
                        full_answer += delta
                        yield sse({"type": "text-delta", "id": "text-1", "delta": delta})
                    yield sse({"type": "text-end", "id": "text-1"})
                    cached_cited_sources = [asdict(source) for source in cached.cited_sources.sources]
                    if cached_cited_sources:
                        yield sse({"type": "data-sources", "data": cached_cited_sources})
                    if cached.questions:
                        yield sse({"type": "data-questions", "data": cached.questions})
                    yield sse({
                        "type": "finish",
                        "messageMetadata": {
                            "finishReason": "stop",
                            "seedChunkIds": cached.retrieved_chunk_ids,
                        },
                    })
                    span.update(output=full_answer, metadata={"cache_hit": True})
                    yield "data: [DONE]\n\n"
                    return

                sources = await retrieval_service.retrieve(query, seed_chunk_ids)
                cited_sources: list[dict] = []
                cited_sources_event: SourcesEvent | None = None
                async for event in llm_provider.generate_stream(query, sources):
                    if isinstance(event, TextDeltaEvent):
                        full_answer += event.delta
                        yield sse({"type": "text-delta", "id": "text-1", "delta": event.delta})
                    elif isinstance(event, SourcesEvent):
                        cited_sources_event = event
                        cited_sources = [asdict(source) for source in event.sources]
                yield sse({"type": "text-end", "id": "text-1"})
                if cited_sources:
                    yield sse({"type": "data-sources", "data": cited_sources})
                questions_event = await llm_provider.generate_followup_questions(
                    query, full_answer, sources
                )
                if questions_event.questions:
                    yield sse({"type": "data-questions", "data": questions_event.questions})
                retrieved_chunk_ids = list(dict.fromkeys(r.chunk.id for r in sources))
                yield sse({
                    "type": "finish",
                    "messageMetadata": {
                        "finishReason": "stop",
                        "seedChunkIds": retrieved_chunk_ids,
                    },
                })
                span.update(output=full_answer, metadata={"cache_hit": False})

                if full_answer.strip() and NO_INFO_SENTINEL not in full_answer:
                    try:
                        cache_vector = embedded_query or await retrieval_service.embed_query(query)
                        cached_answer = CachedAnswer(
                            id=compute_cache_id(query),
                            date=datetime.now(timezone.utc),
                            query=query,
                            embedded_query=cache_vector,
                            retrieved_chunk_ids=retrieved_chunk_ids,
                            cited_sources=cited_sources_event or SourcesEvent(sources=[]),
                            generated_answer=full_answer,
                            questions=questions_event.questions,
                        )
                        await semantic_cache.send_to_semantic_cache(cached_answer)
                    except Exception:
                        logger.exception("Semantic cache write failed for query: %r", query)

                yield "data: [DONE]\n\n"
                return
            yield sse({"type": "finish", "messageMetadata": {"finishReason": "stop"}})
            span.update(output=full_answer)
        except Exception:
            logger.exception("RAG stream failed for query: %r", query)
            span.update(level="ERROR", status_message="RAG stream failed")
            yield sse({
                "type": "error",
                "errorText": "Une erreur est survenue lors de la génération de la réponse.",
            })
        finally:
            langfuse.flush()
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    """Streams a grounded answer to the latest user message.

    Args:
        request: The chat request body containing the message history.
        protocol: The Vercel AI protocol to advertise in the response headers.

    Returns:
        A streaming ``text/event-stream`` response carrying the RAG answer.
    """
    query = _extract_last_user_query(request.messages)
    seed_chunk_ids = _extract_seed_chunk_ids(request.messages)
    response = StreamingResponse(
        _rag_stream(query, seed_chunk_ids),
        media_type="text/event-stream",
    )
    return patch_response_with_headers(response, protocol)
