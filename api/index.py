import json
import uuid
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .rag.adapters.llm.openai_llm import OpenAILLMProvider
from .rag.container import build_llm_provider, build_retrieval_service
from .rag.services.retrieval_service import RetrievalService
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.stream import patch_response_with_headers

from vercel.headers import set_headers

load_dotenv(".env.local")

retrieval_service: RetrievalService | None = None
llm_provider: OpenAILLMProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_service, llm_provider
    retrieval_service = await build_retrieval_service()
    llm_provider = build_llm_provider()
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _vercel_set_headers(request: FastAPIRequest, call_next):
    set_headers(dict(request.headers))
    return await call_next(request)


class Request(BaseModel):
    messages: List[ClientMessage]


def _extract_last_user_query(messages: List[ClientMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            if message.content:
                return message.content
            if message.parts:
                texts = [p.text for p in message.parts if p.type == "text" and p.text]
                if texts:
                    return " ".join(texts)
    return ""


async def _rag_stream(query: str, protocol: str):
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

    message_id = f"msg-{uuid.uuid4().hex}"
    yield sse({"type": "start", "messageId": message_id})
    yield sse({"type": "text-start", "id": "text-1"})

    sources = await retrieval_service.retrieve(query)
    async for token in llm_provider.generate_stream(query, sources):
        yield sse({"type": "text-delta", "id": "text-1", "delta": token})

    yield sse({"type": "text-end", "id": "text-1"})
    yield sse({"type": "finish", "messageMetadata": {"finishReason": "stop"}})
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    query = _extract_last_user_query(request.messages)
    response = StreamingResponse(
        _rag_stream(query, protocol),
        media_type="text/event-stream",
    )
    return patch_response_with_headers(response, protocol)
