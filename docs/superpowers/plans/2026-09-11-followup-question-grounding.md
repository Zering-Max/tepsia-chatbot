# Fiabiliser les questions de suivi (seed chunks) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a user clicks a suggested follow-up question, the passages that justified generating that question must still be available to the LLM answering it, even if a fresh search on the question's own text would not surface them.

**Architecture:** The backend already returns, per user turn, the exact set of retrieved passages it used to generate the answer and the follow-up questions. We now also echo their chunk IDs to the client as message metadata (`seedChunkIds`), on the existing `finish` SSE frame. When the user clicks a follow-up question, the client resends those IDs on the new user message's metadata. The backend fetches those specific chunks by ID and merges them (deduplicated) into the fresh search results for the new query, before calling the LLM. The API stays stateless — no server-side session/conversation store.

**Tech Stack:** FastAPI + Qdrant (`qdrant-client` async) + Mistral (`mistralai`) on the backend; Next.js + `@ai-sdk/react` (AI SDK v5, `useChat`) on the frontend. No test framework is installed in this repo (confirmed: no `pytest`, no `tests/` dir, no JS test runner in `package.json`) — this plan follows the project's existing convention of small throwaway verification scripts run directly against the live Qdrant/Mistral instances configured in `.env.local`, plus manual smoke tests through the running app.

**Spec:** [docs/superpowers/specs/2026-09-11-followup-question-grounding-design.md](../specs/2026-09-11-followup-question-grounding-design.md)

## Global Constraints

- No test framework exists in this repo — do not add one. Pure-logic changes get a small standalone Python script using `assert` (run with `python`, not `pytest`), deleted after it passes. Anything touching real Qdrant/Mistral gets a manual smoke test against the credentials already in `.env.local`.
- Seed chunk fetch failures (`get_by_ids`) must never break the response: log and fall back to fresh-search-only results.
- No additional cap on merged results beyond deduplication (spec decision — up to ~16 passages, no truncation).
- Merge order: fresh search results first, then seed chunks not already present, deduplicated by `chunk.id`.
- A message with no `seedChunkIds` (freely typed question, not a clicked suggestion) must behave exactly as today.

---

### Task 1: `VectorStore.get_by_ids` (port + Qdrant adapter)

**Files:**
- Modify: `api/rag/ports/vector_store.py`
- Modify: `api/rag/adapters/vectorstore/qdrant_store.py`

**Interfaces:**
- Produces: `VectorStore.get_by_ids(self, ids: list[str]) -> list[SearchResult]` (async, abstract on the port; concrete on `QdrantVectorStore`). Empty `ids` returns `[]` without a network call. IDs not found in the collection are silently skipped (no exception). Every returned `SearchResult.score` is `1.0` — a synthetic value, since this is a fetch-by-ID, not a similarity search.

- [ ] **Step 1: Add the abstract method to the port**

Edit `api/rag/ports/vector_store.py`, add after `hybrid_search`:

```python
    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Fetches specific chunks by id, without a similarity query.

        Used to re-include passages that grounded an earlier turn (e.g. the
        passages behind a suggested follow-up question) when they might not
        resurface in a fresh search on different query text.

        Args:
            ids: Chunk ids to fetch. Ids not present in the store are
                silently skipped.

        Returns:
            The matching chunks, each with a synthetic score of 1.0 (there
            is no similarity score for a direct fetch). Order is not
            guaranteed to match the input order.
        """
        ...
```

- [ ] **Step 2: Implement it on `QdrantVectorStore`**

Edit `api/rag/adapters/vectorstore/qdrant_store.py`.

First, add the import for `Record` next to the existing `qdrant_client.models` imports:

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, Record, ScoredPoint
```

Add a module-level constant near `DENSE_VECTOR_NAME`/`SPARSE_VECTOR_NAME`:

```python
SEED_SCORE = 1.0
```

Generalize `_to_search_result` to accept either a `ScoredPoint` (has `.score`) or a `Record` (no `.score`, so the caller passes one in). Replace the existing `_to_search_result` static method with:

```python
    @staticmethod
    def _to_search_result(point: ScoredPoint | Record, score: float | None = None) -> SearchResult:
        """Converts a Qdrant point into a domain SearchResult.

        Args:
            point: A scored point from a query, or a plain record from a
                by-id fetch.
            score: Score to use when `point` has none (a `Record` from
                `get_by_ids`). Ignored for `ScoredPoint`, which carries its
                own score.

        Returns:
            A SearchResult containing the reconstructed TextChunk and score.

        Raises:
            Exception: If the point has no payload.
        """
        if point.payload is None:
            raise Exception("Qdrant point has no payload")
        p = point.payload
        metadata = TextChunkMetadata(
            chunk_index=p["chunk_index"],
            page_start=p["page_start"],
            page_end=p["page_end"],
            section_title=p["section_title"],
            file_name=p["file_name"],
            link_preview=p.get("link_preview"),
        )
        chunk = TextChunk(
            id=str(point.id),
            document_id=p["doc_id"],
            content=p["text"],
            chunk_index=p["chunk_index"],
            metadata=metadata,
        )
        resolved_score = point.score if isinstance(point, ScoredPoint) else score
        return SearchResult(chunk=chunk, score=resolved_score)
```

Add the new method, placed after `hybrid_search`:

```python
    async def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Fetches specific chunks by id, without a similarity query.

        Args:
            ids: Chunk ids to fetch.

        Returns:
            The matching chunks, scored at SEED_SCORE. Ids not found in the
            collection are skipped.
        """
        if not ids:
            return []
        records = await self.async_qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
        )
        results = [self._to_search_result(record, score=SEED_SCORE) for record in records]
        logger.info("Fetched %d/%d requested chunks by id from '%s'.", len(results), len(ids), self.collection_name)
        return results
```

- [ ] **Step 3: Write a throwaway live verification script**

Create `scratch_verify_get_by_ids.py` at the repo root (temporary, not committed):

```python
import asyncio
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env.local")

from api.rag.container import build_retrieval_service


async def main():
    retrieval_service = await build_retrieval_service()

    # First, grab two real chunk ids from a live search.
    seed_results = await retrieval_service.retrieve("projet Tepsa GNL Reichstett")
    ids = [r.chunk.id for r in seed_results[:2]]
    print("Fetching by id:", ids)

    fetched = await retrieval_service._vector_store.get_by_ids(ids)
    assert len(fetched) == 2, f"expected 2 results, got {len(fetched)}"
    fetched_ids = {r.chunk.id for r in fetched}
    assert fetched_ids == set(ids), f"expected {set(ids)}, got {fetched_ids}"
    assert all(r.score == 1.0 for r in fetched), "expected synthetic score 1.0"
    assert all(r.chunk.content for r in fetched), "expected non-empty content"
    print("OK: get_by_ids returned the exact chunks requested, with content.")

    # Unknown id: should be skipped, not raise.
    empty = await retrieval_service._vector_store.get_by_ids(["00000000-0000-0000-0000-000000000000"])
    assert empty == [], f"expected empty list for unknown id, got {empty}"
    print("OK: unknown id is silently skipped.")

    # Empty input: no network call, empty result.
    assert await retrieval_service._vector_store.get_by_ids([]) == []
    print("OK: empty ids list returns [] without erroring.")


asyncio.run(main())
```

- [ ] **Step 4: Run it and verify all assertions pass**

Run: `python scratch_verify_get_by_ids.py`
Expected output ends with the three "OK:" lines and no traceback.

- [ ] **Step 5: Delete the throwaway script**

```bash
rm scratch_verify_get_by_ids.py
```

- [ ] **Step 6: Commit**

```bash
git add api/rag/ports/vector_store.py api/rag/adapters/vectorstore/qdrant_store.py
git commit -m "feat: add VectorStore.get_by_ids for fetching chunks by id"
```

---

### Task 2: `RetrievalService.retrieve` merges seed chunks

**Files:**
- Modify: `api/rag/services/retrieval_service.py`

**Interfaces:**
- Consumes: `VectorStore.get_by_ids(ids: list[str]) -> list[SearchResult]` (Task 1)
- Produces: `RetrievalService.retrieve(self, query: str, seed_chunk_ids: list[str] | None = None) -> list[SearchResult]` — existing callers passing only `query` keep working unchanged (new parameter defaults to `None`).

- [ ] **Step 1: Write a throwaway unit-test script using fake ports**

Create `scratch_verify_retrieve_merge.py` at the repo root:

```python
import asyncio

from api.rag.domain.models import DenseEmbedding, SearchResult, TextChunk
from api.rag.services.retrieval_service import RetrievalService


class FakeEmbedder:
    async def embed(self, chunks):
        return [DenseEmbedding(chunk_id="q", vector=[0.0]) for _ in chunks]


def chunk(id_: str, content: str) -> TextChunk:
    return TextChunk(content=content, id=id_)


class FakeVectorStore:
    def __init__(self, fresh_results, seed_results_by_id):
        self._fresh_results = fresh_results
        self._seed_results_by_id = seed_results_by_id

    async def hybrid_search(self, query_vector, query_text, k, prefetch_limit=None):
        return self._fresh_results[:k]

    async def get_by_ids(self, ids):
        return [self._seed_results_by_id[i] for i in ids if i in self._seed_results_by_id]


async def main():
    fresh = [SearchResult(chunk=chunk("a", "fresh A"), score=0.9)]
    seed_only = SearchResult(chunk=chunk("b", "seed B, not in fresh results"), score=1.0)
    seed_overlap = SearchResult(chunk=chunk("a", "duplicate of fresh A"), score=1.0)

    store = FakeVectorStore(
        fresh_results=fresh,
        seed_results_by_id={"b": seed_only, "a": seed_overlap},
    )
    service = RetrievalService(embedder=FakeEmbedder(), vector_store=store, top_k=8)

    # No seeds: behaves exactly like today.
    results_no_seeds = await service.retrieve("query")
    assert [r.chunk.id for r in results_no_seeds] == ["a"], results_no_seeds
    print("OK: no seed_chunk_ids -> fresh results only, unchanged behaviour.")

    # Seeds provided: fresh results first, then seeds not already present, deduplicated.
    results_with_seeds = await service.retrieve("query", seed_chunk_ids=["a", "b"])
    ids = [r.chunk.id for r in results_with_seeds]
    assert ids == ["a", "b"], f"expected ['a', 'b'], got {ids}"
    # "a" must come from the fresh result (score 0.9), not the seed duplicate (score 1.0).
    assert results_with_seeds[0].score == 0.9, "expected fresh result to win over duplicate seed"
    print("OK: seeds merged after fresh results, deduplicated by chunk id.")

    # Seed fetch failure must not break retrieval.
    class FailingVectorStore(FakeVectorStore):
        async def get_by_ids(self, ids):
            raise RuntimeError("Qdrant unavailable")

    failing_service = RetrievalService(embedder=FakeEmbedder(), vector_store=FailingVectorStore(fresh, {}), top_k=8)
    results_after_failure = await failing_service.retrieve("query", seed_chunk_ids=["b"])
    assert [r.chunk.id for r in results_after_failure] == ["a"], results_after_failure
    print("OK: get_by_ids failure falls back to fresh results only.")


asyncio.run(main())
```

- [ ] **Step 2: Run it and verify it fails (the new parameter doesn't exist yet)**

Run: `python scratch_verify_retrieve_merge.py`
Expected: `TypeError: RetrievalService.retrieve() got an unexpected keyword argument 'seed_chunk_ids'`

- [ ] **Step 3: Implement `seed_chunk_ids` merging in `retrieve`**

Replace the `retrieve` method in `api/rag/services/retrieval_service.py`:

```python
    async def retrieve(
        self, query: str, seed_chunk_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """Encodes a query and retrieves the most relevant chunks.

        When `seed_chunk_ids` is given, those chunks are fetched directly and
        merged after the fresh search results (deduplicated by chunk id).
        This carries forward the passages that grounded an earlier turn (e.g.
        a suggested follow-up question) so they stay available even if a
        fresh search on the new query text would not resurface them. A
        failure to fetch seed chunks is logged and ignored — the fresh
        results are still returned.

        Args:
            query: Natural-language question from the user.
            seed_chunk_ids: Chunk ids to guarantee in the result, in addition
                to the fresh search.

        Returns:
            Fresh search results first, then any seed chunks not already
            present among them, deduplicated by chunk id.
        """
        query_embedded = await self._embedder.embed(chunks=[TextChunk(content=query)])
        result: DenseEmbedding = query_embedded[0]

        fresh_results = await self._vector_store.hybrid_search(
            query_vector=result.vector,
            query_text=query,
            k=self._top_k,
        )

        if not seed_chunk_ids:
            return fresh_results

        try:
            seed_results = await self._vector_store.get_by_ids(seed_chunk_ids)
        except Exception:
            logger.exception("Failed to fetch seed chunks by id; using fresh results only.")
            return fresh_results

        seen_ids = {r.chunk.id for r in fresh_results}
        merged = list(fresh_results)
        for seed in seed_results:
            if seed.chunk.id not in seen_ids:
                merged.append(seed)
                seen_ids.add(seed.chunk.id)
        return merged
```

Add the `logging` import and module logger at the top of the file (next to the existing docstring), matching the style used in `qdrant_store.py`:

```python
import logging

from ..domain.models import TextChunk, DenseEmbedding, SearchResult
from ..ports.embedding import DenseEmbedder
from ..ports.vector_store import VectorStore

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run it again and verify all assertions pass**

Run: `python scratch_verify_retrieve_merge.py`
Expected output ends with the three "OK:" lines and no traceback.

- [ ] **Step 5: Delete the throwaway script**

```bash
rm scratch_verify_retrieve_merge.py
```

- [ ] **Step 6: Commit**

```bash
git add api/rag/services/retrieval_service.py
git commit -m "feat: merge seed chunks into RetrievalService.retrieve"
```

---

### Task 3: `ClientMessage.metadata` field

**Files:**
- Modify: `api/utils/prompt.py`

**Interfaces:**
- Produces: `ClientMessage.metadata: dict | None` (defaults to `None` — existing request bodies without a `metadata` field keep parsing exactly as before).

- [ ] **Step 1: Write a throwaway verification script**

Create `scratch_verify_client_message.py` at the repo root:

```python
from api.utils.prompt import ClientMessage

# A request body without metadata must still parse (backward compatible).
msg_no_metadata = ClientMessage.model_validate({"role": "user", "content": "hello"})
assert msg_no_metadata.metadata is None
print("OK: message without metadata parses, metadata defaults to None.")

# A request body with metadata carrying seedChunkIds must parse it through.
msg_with_metadata = ClientMessage.model_validate({
    "role": "user",
    "content": "Quels médias nationaux ont relayé la mobilisation ?",
    "metadata": {"seedChunkIds": ["a", "b", "c"]},
})
assert msg_with_metadata.metadata == {"seedChunkIds": ["a", "b", "c"]}
print("OK: message with metadata.seedChunkIds parses correctly.")
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python scratch_verify_client_message.py`
Expected: `AssertionError` on the second assertion (extra `metadata` field is currently dropped, so `msg_with_metadata.metadata` doesn't exist / is `None`) — or a `pydantic.ValidationError` if the model forbids extra fields. Either way, it must not print both "OK:" lines.

- [ ] **Step 3: Add the field**

In `api/utils/prompt.py`, edit the `ClientMessage` class:

```python
class ClientMessage(BaseModel):
    role: str
    content: Optional[str] = None
    parts: Optional[List[ClientMessagePart]] = None
    experimental_attachments: Optional[List[ClientAttachment]] = None
    toolInvocations: Optional[List[ToolInvocation]] = None
    metadata: Optional[dict] = None
```

- [ ] **Step 4: Run it again and verify it passes**

Run: `python scratch_verify_client_message.py`
Expected output: both "OK:" lines, no traceback.

- [ ] **Step 5: Delete the throwaway script**

```bash
rm scratch_verify_client_message.py
```

- [ ] **Step 6: Commit**

```bash
git add api/utils/prompt.py
git commit -m "feat: accept metadata on incoming chat messages"
```

---

### Task 4: Wire seed chunk ids through `api/index.py`

**Files:**
- Modify: `api/index.py`

**Interfaces:**
- Consumes: `ClientMessage.metadata` (Task 3), `RetrievalService.retrieve(query, seed_chunk_ids=...)` (Task 2)
- Produces: `_extract_seed_chunk_ids(messages: List[ClientMessage]) -> list[str]`. The `finish` SSE frame's `messageMetadata` gains a `seedChunkIds` key.

- [ ] **Step 1: Write a throwaway verification script for the extraction function**

Create `scratch_verify_extract_seeds.py` at the repo root:

```python
from api.index import _extract_seed_chunk_ids
from api.utils.prompt import ClientMessage

# No messages at all.
assert _extract_seed_chunk_ids([]) == []

# Last user message has no metadata (freely typed question).
messages = [ClientMessage(role="user", content="une question libre")]
assert _extract_seed_chunk_ids(messages) == []

# Last user message has seedChunkIds metadata (clicked follow-up).
messages = [
    ClientMessage(role="user", content="q1"),
    ClientMessage(role="assistant", content="a1"),
    ClientMessage(role="user", content="q2 (follow-up)", metadata={"seedChunkIds": ["x", "y"]}),
]
assert _extract_seed_chunk_ids(messages) == ["x", "y"]

# Malformed metadata (not a list of strings) must degrade to no seeds, not crash.
messages = [ClientMessage(role="user", content="q", metadata={"seedChunkIds": "not-a-list"})]
assert _extract_seed_chunk_ids(messages) == []

messages = [ClientMessage(role="user", content="q", metadata={"seedChunkIds": [1, 2]})]
assert _extract_seed_chunk_ids(messages) == []

print("OK: _extract_seed_chunk_ids handles empty, absent, valid, and malformed metadata.")
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python scratch_verify_extract_seeds.py`
Expected: `ImportError: cannot import name '_extract_seed_chunk_ids' from 'api.index'`

- [ ] **Step 3: Implement `_extract_seed_chunk_ids`**

In `api/index.py`, add this function right after `_extract_last_user_query`:

```python
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
```

- [ ] **Step 4: Run it again and verify it passes**

Run: `python scratch_verify_extract_seeds.py`
Expected output: the single "OK:" line, no traceback.

- [ ] **Step 5: Delete the throwaway script**

```bash
rm scratch_verify_extract_seeds.py
```

- [ ] **Step 6: Wire it into `_rag_stream` and the `handle_chat_data` endpoint**

In `api/index.py`, change the `_rag_stream` signature and its retrieval call:

```python
async def _rag_stream(query: str, seed_chunk_ids: list[str]):
```

(update the docstring's `Args:` section to add `seed_chunk_ids: Chunk ids to carry forward from a clicked follow-up question, or an empty list.`)

Replace:
```python
                sources = await retrieval_service.retrieve(query)
```
with:
```python
                sources = await retrieval_service.retrieve(query, seed_chunk_ids)
```

Replace the `finish` frame:
```python
            yield sse({"type": "finish", "messageMetadata": {"finishReason": "stop"}})
```
with:
```python
            seed_ids_for_next_turn = list(dict.fromkeys(r.chunk.id for r in sources))
            yield sse({
                "type": "finish",
                "messageMetadata": {
                    "finishReason": "stop",
                    "seedChunkIds": seed_ids_for_next_turn,
                },
            })
```

Note `sources` is only bound inside the `if query.strip():` branch — the empty-query branch above it returns early in the `if`/`else`, so this line must stay inside the `else:` branch, after the existing `if cited_sources:` block, at the same indentation as the other frame `yield`s inside `else:`. Since `sources` isn't defined in the empty-query branch, keep the original unconditional `yield sse({"type": "finish", ...})` for that branch unchanged (it doesn't have a `seedChunkIds` field, which is fine — there's nothing to seed from a prompt-for-input turn) and only add `seedChunkIds` to the `finish` frame emitted at the end of the `else:` branch. Concretely, the `else:` branch's tail (from `yield sse({"type": "text-end"...})` onward) becomes:

```python
            else:
                sources = await retrieval_service.retrieve(query, seed_chunk_ids)
                cited_sources: list[dict] = []
                async for event in llm_provider.generate_stream(query, sources):
                    if isinstance(event, TextDeltaEvent):
                        full_answer += event.delta
                        yield sse({"type": "text-delta", "id": "text-1", "delta": event.delta})
                    elif isinstance(event, SourcesEvent):
                        cited_sources = [asdict(source) for source in event.sources]
                yield sse({"type": "text-end", "id": "text-1"})
                if cited_sources:
                    yield sse({"type": "data-sources", "data": cited_sources})
                questions_event = await llm_provider.generate_followup_questions(
                    query, full_answer, sources
                )
                if questions_event.questions:
                    yield sse({"type": "data-questions", "data": questions_event.questions})
                yield sse({
                    "type": "finish",
                    "messageMetadata": {
                        "finishReason": "stop",
                        "seedChunkIds": list(dict.fromkeys(r.chunk.id for r in sources)),
                    },
                })
                span.update(output=full_answer)
                yield "data: [DONE]\n\n"
                return
            yield sse({"type": "finish", "messageMetadata": {"finishReason": "stop"}})
            span.update(output=full_answer)
```

This keeps the empty-query path's original single `finish` frame untouched, and gives the real-answer path its own `finish` frame carrying `seedChunkIds`, without duplicating the `finish`/`span.update` emission — the `return` after the early `yield "data: [DONE]\n\n"` skips the trailing shared lines, which still serve the empty-query branch. (The `finally: langfuse.flush()` still runs on `return`, since it's a `finally` block around this whole `try`.)

- [ ] **Step 7: Update `handle_chat_data` to compute and pass `seed_chunk_ids`**

Replace:
```python
    query = _extract_last_user_query(request.messages)
    response = StreamingResponse(
        _rag_stream(query),
        media_type="text/event-stream",
    )
```
with:
```python
    query = _extract_last_user_query(request.messages)
    seed_chunk_ids = _extract_seed_chunk_ids(request.messages)
    response = StreamingResponse(
        _rag_stream(query, seed_chunk_ids),
        media_type="text/event-stream",
    )
```

- [ ] **Step 8: Manual live smoke test — reproduce the original bug and confirm the fix**

Start the API: `python -m uvicorn api.index:app --reload` (from the repo root, with `.env.local` present).

In a second terminal, replay the exact reported repro as two separate HTTP calls, carrying the `seedChunkIds` from the first response's `finish` frame into the second request's last message metadata:

```bash
curl -s -N -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"parle moi de l'\''objectif du projet tepsout"}]}'
```

Read the `data-questions` and `finish` frames from the output; copy the `seedChunkIds` array from the `finish` frame's `messageMetadata`. Then:

```bash
curl -s -N -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"parle moi de l'\''objectif du projet tepsout"},{"role":"assistant","content":"..."},{"role":"user","content":"Quels médias nationaux ont relayé la mobilisation du collectif Teps'\''out ?","metadata":{"seedChunkIds":[PASTE_THE_IDS_HERE]}}]}'
```

Expected: the streamed `text-delta` frames now form an answer mentioning **Les Echos**, not the `NO_INFO_SENTINEL` refusal ("Cette information n'est pas disponible..."). This is the exact case root-caused during debugging — confirm it's fixed before moving on.

- [ ] **Step 9: Commit**

```bash
git add api/index.py
git commit -m "feat: carry seed chunk ids from clicked follow-up questions into retrieval"
```

---

### Task 5: Propagate `seedChunkIds` from the frontend on follow-up click

**Files:**
- Modify: `components/message.tsx`

**Interfaces:**
- Consumes: `message.metadata` on the assistant `UIMessage` (populated by the AI SDK from the backend's `finish` frame `messageMetadata`, Task 4) — read as `(message.metadata as { seedChunkIds?: string[] } | undefined)?.seedChunkIds`.

- [ ] **Step 1: Update the follow-up question button's click handler**

In `components/message.tsx`, find the follow-up questions block (around the `data-questions` part rendering, currently):

```tsx
                    {part.data.map((question: string, questionIndex: number) => (
                      <Button
                        key={`followup-${questionIndex}`}
                        variant="outline"
                        size="sm"
                        disabled={isLoading}
                        onClick={() => sendMessage({ text: question })}
                        className="h-auto whitespace-normal text-left justify-start py-2"
                      >
                        {question}
                      </Button>
```

Replace the `onClick` with:

```tsx
                    {part.data.map((question: string, questionIndex: number) => (
                      <Button
                        key={`followup-${questionIndex}`}
                        variant="outline"
                        size="sm"
                        disabled={isLoading}
                        onClick={() => {
                          const seedChunkIds =
                            (message.metadata as { seedChunkIds?: string[] } | undefined)
                              ?.seedChunkIds ?? [];
                          sendMessage({ text: question, metadata: { seedChunkIds } });
                        }}
                        className="h-auto whitespace-normal text-left justify-start py-2"
                      >
                        {question}
                      </Button>
```

- [ ] **Step 2: Manual browser verification**

Start both dev servers (two terminals, from the repo root):

```bash
npm run fastapi-dev
```
```bash
npm run next-dev
```

Open the app, ask: `parle moi de l'objectif du projet tepsout`. Once the answer streams in with its "Pour approfondir :" suggestions, open the browser's Network tab, click the follow-up question about national media coverage (or whichever suggestion appears — the underlying `seedChunkIds` mechanism applies to any of them). Inspect the outgoing `POST /api/chat` request body and confirm the last message's `metadata.seedChunkIds` is a non-empty array. Confirm the rendered answer is a real answer, not the "Cette information n'est pas disponible dans les documents fournis." refusal.

Also verify the non-regression case: type a completely unrelated free-form question directly into the input box (not a clicked suggestion). Confirm it still works exactly as before (no `metadata` sent, or `seedChunkIds: []`, normal fresh-search behavior).

- [ ] **Step 3: Commit**

```bash
git add components/message.tsx
git commit -m "feat: propagate seed chunk ids when a follow-up question is clicked"
```
