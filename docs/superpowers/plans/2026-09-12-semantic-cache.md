# Semantic Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Short-circuit retrieval + generation in the RAG chat pipeline by serving a previously generated answer when the incoming question is semantically close (dense similarity) to one already cached, using a dedicated Qdrant collection.

**Architecture:** A new `SemanticCache` port, backed by a `QdrantSemanticCache` adapter against its own Qdrant collection (separate from the documentary collection — different payload shape, no BM25/named vectors). `api/index.py::_rag_stream` checks the cache before retrieval on freely typed questions, replays a hit as a simulated `text-delta` stream, and always writes a fresh generation to the cache afterward. Questions carrying `seed_chunk_ids` (a clicked follow-up) bypass the cache read entirely but still get written.

**Tech Stack:** Python 3, FastAPI, `qdrant-client` (async), `pydantic-settings`, no test framework (this repo has none — see Global Constraints).

**Spec:** `docs/superpowers/specs/2026-09-12-semantic-cache-design.md`

## Global Constraints

- No automated tests in this repo (no pytest, no jest/vitest, no `tests/` directory) — the user chose manual verification over introducing a test framework for this feature. Every task ends with a manual verification step (Python one-liner, or exercising the running dev server), not a test suite.
- `SemanticCache.SIMILARITY_THRESHOLD` is a class-level constant (0.97 to start), not a `Settings` field.
- Cache read is bypassed whenever the incoming request carries `seed_chunk_ids` (non-empty); cache write always happens after a fresh generation, seeds or not.
- `CachedAnswer.id` is deterministic: `uuid5` over the normalized (`strip().lower()`) question text, so an identical question overwrites its existing cache entry.
- New `Settings` field `qdrant_cache_collection_name: str` is **required** (no default) — the app will fail to boot (`pydantic_settings` raises at import time) once Task 4 lands, until this env var is set in `.env.local`. Flag this to the user before Task 4.
- The Qdrant *collection itself* (vector size 1024 for `mistral-embed`, `Distance.COSINE`) is not created by application code — it's an ops step, done once via a throwaway script before end-to-end verification (Task 9), not part of any committed file.

---

### Task 1: Fix `CachedAnswer` domain model and add its id helper

**Files:**
- Modify: `api/rag/domain/models.py`

**Interfaces:**
- Produces: `compute_cache_id(query: str) -> str`, `CachedAnswer` dataclass with fields `id: str`, `date: datetime`, `query: Query`, `embedded_query: DenseEmbedding`, `retrieved_chunk_ids: list[str]`, `cited_sources: SourcesEvent`, `generated_answer: str`, `questions: list[str]`.

The current `CachedAnswer` (lines 202-210) is invalid Python — `id:` and `date:` have no type, just a comment — so the module doesn't import today. It also sits *before* `SourcesEvent` is defined, which it now needs as a field type. This task removes the broken stub, adds the id-generation helper, and re-adds `CachedAnswer` after `SourcesEvent`.

- [ ] **Step 1: Add the `uuid` and `datetime` imports**

At the top of `api/rag/domain/models.py`, change:

```python
from dataclasses import dataclass, replace
from pathlib import Path
```

to:

```python
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
```

- [ ] **Step 2: Remove the broken `CachedAnswer` stub**

Delete these lines (currently right after the `Answer` dataclass):

```python
@dataclass(frozen=True)
class CachedAnswer:
    id: # faire une méthode similaire à celle déjà existance pour calculer un ID 
    date: # à compléter
    query: Query
    embedded_query: DenseEmbedding
    sources: SourcesEvent
    generated_answer: str
    questions: list[str]
```

- [ ] **Step 3: Add `compute_cache_id` and the fixed `CachedAnswer`, after the `SourcesEvent` class**

`SourcesEvent` is currently defined right after `QuestionsEvent`'s neighbor `TextDeltaEvent`, before `QuestionsEvent`. Insert this new block immediately after the `SourcesEvent` class body (i.e. between `SourcesEvent` and `QuestionsEvent`):

```python
CACHE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "tepsia-chatbot/semantic-cache")


def compute_cache_id(query: str) -> str:
    """Derives a deterministic id for a cached answer from its question.

    Two calls with the same question (after trimming and lowercasing) return
    the same id, so re-caching an identical question overwrites the existing
    entry instead of accumulating duplicates.

    Args:
        query: The raw user question.

    Returns:
        A UUID5 string derived from the normalized question.
    """
    normalized = query.strip().lower()
    return str(uuid.uuid5(CACHE_NAMESPACE, normalized))


@dataclass(frozen=True)
class CachedAnswer:
    """A previously generated answer stored in the semantic cache.

    Attributes:
        id: Deterministic id from `compute_cache_id`, unique per normalized question.
        date: UTC timestamp of when this entry was cached.
        query: The original question this answer was generated for.
        embedded_query: Dense embedding of `query`, used as the cache's search vector.
        retrieved_chunk_ids: Ids of the chunks used to generate the answer, replayed
            as `seedChunkIds` on a cache hit so follow-up grounding keeps working.
        cited_sources: Sources actually cited in `generated_answer`.
        generated_answer: The LLM-generated response text.
        questions: Suggested follow-up questions for this answer.
    """

    id: str
    date: datetime
    query: Query
    embedded_query: DenseEmbedding
    retrieved_chunk_ids: list[str]
    cited_sources: SourcesEvent
    generated_answer: str
    questions: list[str]
```

- [ ] **Step 4: Verify manually**

Run:

```bash
python -c "
from api.rag.domain.models import CachedAnswer, DenseEmbedding, SourcesEvent, compute_cache_id
from datetime import datetime, timezone

id1 = compute_cache_id('Quel est le tarif ?')
id2 = compute_cache_id('  quel est le tarif ?  ')
id3 = compute_cache_id('Une autre question')
assert id1 == id2, 'normalization should make these equal'
assert id1 != id3
print('ids ok:', id1)

ca = CachedAnswer(
    id=id1,
    date=datetime.now(timezone.utc),
    query='Quel est le tarif ?',
    embedded_query=DenseEmbedding(chunk_id=None, vector=[0.1, 0.2]),
    retrieved_chunk_ids=['chunk-1'],
    cited_sources=SourcesEvent(sources=[]),
    generated_answer='42 euros.',
    questions=['Et pour les etudiants ?'],
)
print('CachedAnswer constructed ok:', ca.id)
"
```

Expected: prints both `ids ok: ...` and `CachedAnswer constructed ok: ...` with no traceback.

- [ ] **Step 5: Commit**

```bash
git add api/rag/domain/models.py
git commit -m "fix: repair CachedAnswer model and add deterministic cache id"
```

---

### Task 2: Rewrite the `SemanticCache` port contract

**Files:**
- Modify: `api/rag/ports/semantic_cache.py`

**Interfaces:**
- Consumes: `CachedAnswer`, `DenseEmbedding` from `api/rag/domain/models.py` (Task 1).
- Produces: `SemanticCache` ABC with `SIMILARITY_THRESHOLD: float = 0.97`, `async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None`, `async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None`.

- [ ] **Step 1: Replace the file contents**

```python
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
```

- [ ] **Step 2: Verify manually**

```bash
python -c "
from api.rag.ports.semantic_cache import SemanticCache
try:
    SemanticCache()
    print('ERROR: should not be instantiable')
except TypeError as e:
    print('ok, abstract as expected:', e)
print('threshold:', SemanticCache.SIMILARITY_THRESHOLD)
"
```

Expected: prints `ok, abstract as expected: ...` and `threshold: 0.97`.

- [ ] **Step 3: Commit**

```bash
git add api/rag/ports/semantic_cache.py
git commit -m "feat: define SemanticCache port contract"
```

---

### Task 3: Implement the `QdrantSemanticCache` adapter

**Files:**
- Create: `api/rag/adapters/semantic_cache/__init__.py` (empty, matches sibling adapter packages)
- Create: `api/rag/adapters/semantic_cache/qdrant_semantic_cache.py`

**Interfaces:**
- Consumes: `SemanticCache` port (Task 2), `CachedAnswer`, `CitedSource`, `DenseEmbedding`, `SourcesEvent` from `api/rag/domain/models.py`.
- Produces: `QdrantSemanticCache(async_qdrant_client: AsyncQdrantClient, collection_name: str)` implementing both port methods.

- [ ] **Step 1: Create the package init**

`api/rag/adapters/semantic_cache/__init__.py`:

```python
```

(empty file, same as `api/rag/adapters/embedding/__init__.py` and `api/rag/adapters/llm/__init__.py`)

- [ ] **Step 2: Write the adapter**

`api/rag/adapters/semantic_cache/qdrant_semantic_cache.py`:

```python
"""Qdrant-backed semantic cache adapter.

Implements the :class:`SemanticCache` port against a dedicated Qdrant
collection (separate from the documentary collection): one plain dense
vector per cached answer, no named vectors, no sparse/BM25 branch.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, ScoredPoint

from ...domain.models import CachedAnswer, CitedSource, DenseEmbedding, SourcesEvent
from ...ports.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class QdrantSemanticCache(SemanticCache):
    """SemanticCache backed by a dedicated Qdrant collection.

    Attributes:
        async_qdrant_client: Async Qdrant client used for all operations.
        collection_name: Name of the target Qdrant collection.
    """

    async_qdrant_client: AsyncQdrantClient
    collection_name: str

    async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None:
        """Searches for the closest cached question and returns it if similar enough.

        Args:
            embedded_query: Dense embedding of the incoming question.

        Returns:
            The nearest cached answer if its score is at least
            `SIMILARITY_THRESHOLD`, otherwise None.
        """
        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedded_query.vector,
            with_payload=True,
            limit=1,
        )
        if not response.points or response.points[0].score < self.SIMILARITY_THRESHOLD:
            return None
        cached = self._to_cached_answer(response.points[0], embedded_query)
        logger.info(
            "Semantic cache hit for query %r (score=%.4f).",
            cached.query,
            response.points[0].score,
        )
        return cached

    async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None:
        """Upserts a cached answer, keyed by its own id.

        Args:
            cached_answer: The entry to store.
        """
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[self._prepare_point(cached_answer)],
        )
        logger.info(
            "Cached answer for query %r into '%s'.", cached_answer.query, self.collection_name
        )

    @staticmethod
    def _prepare_point(cached_answer: CachedAnswer) -> PointStruct:
        """Converts a CachedAnswer into a Qdrant PointStruct.

        Args:
            cached_answer: The entry to convert.

        Returns:
            A PointStruct ready for upsert.
        """
        payload = {
            "date": cached_answer.date.isoformat(),
            "query": cached_answer.query,
            "retrieved_chunk_ids": cached_answer.retrieved_chunk_ids,
            "cited_sources": [
                {
                    "index": source.index,
                    "file_name": source.file_name,
                    "link_preview": source.link_preview,
                    "page_start": source.page_start,
                    "page_end": source.page_end,
                }
                for source in cached_answer.cited_sources.sources
            ],
            "generated_answer": cached_answer.generated_answer,
            "questions": cached_answer.questions,
        }
        return PointStruct(
            id=cached_answer.id,
            vector=cached_answer.embedded_query.vector,
            payload=payload,
        )

    @staticmethod
    def _to_cached_answer(point: ScoredPoint, embedded_query: DenseEmbedding) -> CachedAnswer:
        """Reconstructs a CachedAnswer from a Qdrant scored point.

        Args:
            point: The top match returned by `query_points`.
            embedded_query: The embedding of the *incoming* question. A hit
                carries this rather than the stored vector — nothing
                downstream needs the original cached vector, and fetching it
                would require `with_vectors=True` on every lookup.

        Returns:
            The cached answer reconstructed from the point's payload.
        """
        p = point.payload
        cited_sources = SourcesEvent(
            sources=[
                CitedSource(
                    index=s["index"],
                    file_name=s["file_name"],
                    link_preview=s["link_preview"],
                    page_start=s["page_start"],
                    page_end=s["page_end"],
                )
                for s in p["cited_sources"]
            ]
        )
        return CachedAnswer(
            id=str(point.id),
            date=datetime.fromisoformat(p["date"]),
            query=p["query"],
            embedded_query=embedded_query,
            retrieved_chunk_ids=p["retrieved_chunk_ids"],
            cited_sources=cited_sources,
            generated_answer=p["generated_answer"],
            questions=p["questions"],
        )
```

- [ ] **Step 3: Verify manually (payload round-trip, no live Qdrant needed)**

```bash
python -c "
from datetime import datetime, timezone
from api.rag.adapters.semantic_cache.qdrant_semantic_cache import QdrantSemanticCache
from api.rag.domain.models import CachedAnswer, CitedSource, DenseEmbedding, SourcesEvent

ca = CachedAnswer(
    id='fixed-id',
    date=datetime.now(timezone.utc),
    query='Quel est le tarif ?',
    embedded_query=DenseEmbedding(chunk_id=None, vector=[0.1, 0.2, 0.3]),
    retrieved_chunk_ids=['chunk-1', 'chunk-2'],
    cited_sources=SourcesEvent(sources=[CitedSource(index=1, file_name='tarifs.pdf', link_preview=None, page_start=1, page_end=1)]),
    generated_answer='42 euros.',
    questions=['Et pour les etudiants ?'],
)
point = QdrantSemanticCache._prepare_point(ca)
assert point.id == 'fixed-id'
assert point.vector == [0.1, 0.2, 0.3]
assert point.payload['query'] == 'Quel est le tarif ?'
print('prepare_point ok')

class FakePoint:
    id = 'fixed-id'
    payload = point.payload

back = QdrantSemanticCache._to_cached_answer(FakePoint(), embedded_query=ca.embedded_query)
assert back.query == ca.query
assert back.generated_answer == ca.generated_answer
assert back.cited_sources.sources[0].file_name == 'tarifs.pdf'
assert back.retrieved_chunk_ids == ['chunk-1', 'chunk-2']
print('round-trip ok')
"
```

Expected: prints `prepare_point ok` and `round-trip ok`, no traceback.

- [ ] **Step 4: Commit**

```bash
git add api/rag/adapters/semantic_cache/
git commit -m "feat: add QdrantSemanticCache adapter"
```

---

### Task 4: Add the cache collection setting

**Files:**
- Modify: `api/rag/config.py`

**Interfaces:**
- Produces: `Settings.qdrant_cache_collection_name: str`

- [ ] **Step 1: Add the field**

In `api/rag/config.py`, change:

```python
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str


settings = Settings()  # type: ignore[call-arg]
```

to:

```python
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    qdrant_cache_collection_name: str


settings = Settings()  # type: ignore[call-arg]
```

Also update the class docstring's `Attributes:` list to add:

```
        qdrant_cache_collection_name: Name of the Qdrant collection used by the semantic cache.
```

- [ ] **Step 2: Add the env var to `.env.local`**

This field has no default, so `Settings()` now raises at import time until it's set. Add a line to `.env.local` (create the file if it doesn't already have Qdrant settings — check first, since `qdrant_url`/`qdrant_api_key`/`qdrant_collection_name` should already be there):

```
QDRANT_CACHE_COLLECTION_NAME=semantic_cache
```

- [ ] **Step 3: Verify manually**

```bash
python -c "
from api.rag.config import settings
print('cache collection:', settings.qdrant_cache_collection_name)
"
```

Expected: prints `cache collection: semantic_cache` with no traceback. If this raises a `pydantic_settings` validation error, the env var from Step 2 isn't being picked up — double check `.env.local` is at the repo root and matches the existing Qdrant vars' file.

- [ ] **Step 4: Commit**

```bash
git add api/rag/config.py
git commit -m "feat: add qdrant_cache_collection_name setting"
```

(`.env.local` is expected to be gitignored — do not add it to this commit; if `git status` shows it as untracked-and-ignored, that's correct.)

---

### Task 5: Add `RetrievalService.embed_query`

**Files:**
- Modify: `api/rag/services/retrieval_service.py`

**Interfaces:**
- Consumes: existing `self._embedder: DenseEmbedder` (already wired in `__init__`).
- Produces: `async def embed_query(self, query: str) -> DenseEmbedding`

This lets `_rag_stream` (Task 7) get the query's embedding *before* deciding whether to call `retrieve()`, without duplicating the `TextChunk(content=query)` wrapping logic or reaching into `RetrievalService`'s private embedder.

- [ ] **Step 1: Add the method**

In `api/rag/services/retrieval_service.py`, add this method to the `RetrievalService` class, right before `retrieve`:

```python
    async def embed_query(self, query: str) -> DenseEmbedding:
        """Embeds a bare query string, without running a search.

        Args:
            query: Natural-language question from the user.

        Returns:
            The dense embedding of `query`.
        """
        embedded = await self._embedder.embed(chunks=[TextChunk(content=query)])
        return embedded[0]
```

- [ ] **Step 2: Verify manually**

This needs a real embedder (Mistral API key from `.env.local`), so run it against the actual dependency:

```bash
python -c "
import asyncio
from mistralai.client import Mistral
from api.rag.adapters.embedding.mistral_embedder import MistralDenseEmbedder
from api.rag.config import settings

async def main():
    embedder = MistralDenseEmbedder(
        mistral_client=Mistral(api_key=settings.mistral_api_key),
        model=settings.mistral_embedding_model,
    )
    from api.rag.services.retrieval_service import RetrievalService
    service = RetrievalService(embedder=embedder, vector_store=None, top_k=1)
    result = await service.embed_query('Quel est le tarif ?')
    print('vector length:', len(result.vector))

asyncio.run(main())
"
```

Expected: prints `vector length: 1024` (mistral-embed's dimension), no traceback. `vector_store=None` is fine here — `embed_query` never touches it.

- [ ] **Step 3: Commit**

```bash
git add api/rag/services/retrieval_service.py
git commit -m "feat: add RetrievalService.embed_query for cache lookups"
```

---

### Task 6: Wire `build_semantic_cache` into the container

**Files:**
- Modify: `api/rag/container.py`

**Interfaces:**
- Consumes: `QdrantSemanticCache` (Task 3), `settings.qdrant_cache_collection_name` (Task 4).
- Produces: `def build_semantic_cache() -> SemanticCache`

- [ ] **Step 1: Add the import and the builder function**

In `api/rag/container.py`, add to the imports:

```python
from .adapters.semantic_cache.qdrant_semantic_cache import QdrantSemanticCache
from .ports.semantic_cache import SemanticCache
```

Then add this function after `build_retrieval_service`:

```python
def build_semantic_cache() -> SemanticCache:
    """Builds the Qdrant-backed semantic cache.

    Returns:
        A :class:`QdrantSemanticCache` configured from :data:`settings`.
    """
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    return QdrantSemanticCache(
        async_qdrant_client=qdrant_client,
        collection_name=settings.qdrant_cache_collection_name,
    )
```

- [ ] **Step 2: Verify manually**

```bash
python -c "
from api.rag.container import build_semantic_cache
cache = build_semantic_cache()
print('built:', type(cache).__name__, cache.collection_name)
"
```

Expected: prints `built: QdrantSemanticCache semantic_cache` (or whatever `.env.local` sets), no traceback. This only constructs the client, no network call yet.

- [ ] **Step 3: Commit**

```bash
git add api/rag/container.py
git commit -m "feat: wire build_semantic_cache into the composition root"
```

---

### Task 7: Integrate the cache into `_rag_stream`

**Files:**
- Modify: `api/index.py`

**Interfaces:**
- Consumes: `retrieval_service.embed_query` (Task 5), `semantic_cache.check_similar_questions` / `.send_to_semantic_cache` (Tasks 2-3), `build_semantic_cache` (Task 6), `CachedAnswer` / `compute_cache_id` (Task 1).
- Produces: module global `semantic_cache: SemanticCache | None`, `_split_into_deltas(text: str, chunk_size: int = 40) -> list[str]`.

- [ ] **Step 1: Update imports and the module globals**

Change:

```python
from .rag.container import build_llm_provider, build_retrieval_service
from .rag.domain.models import SourcesEvent, TextDeltaEvent
from .rag.ports.llm import LLMProvider
from .rag.services.retrieval_service import RetrievalService
from .utils.prompt import ClientMessage
from .utils.stream import patch_response_with_headers

from vercel.headers import set_headers

load_dotenv(".env.local")

logger = logging.getLogger(__name__)

retrieval_service: RetrievalService | None = None
llm_provider: LLMProvider | None = None
```

to:

```python
from datetime import datetime, timezone

from .rag.container import build_llm_provider, build_retrieval_service, build_semantic_cache
from .rag.domain.models import CachedAnswer, DenseEmbedding, SourcesEvent, TextDeltaEvent, compute_cache_id
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
```

- [ ] **Step 2: Build the cache in `lifespan`**

Change:

```python
    global retrieval_service, llm_provider
    retrieval_service = await build_retrieval_service()
    llm_provider = build_llm_provider()
```

to:

```python
    global retrieval_service, llm_provider, semantic_cache
    retrieval_service = await build_retrieval_service()
    llm_provider = build_llm_provider()
    semantic_cache = build_semantic_cache()
```

- [ ] **Step 3: Add `_split_into_deltas`, right after `_extract_seed_chunk_ids`**

```python
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
            current = word
        else:
            current += piece
    if current:
        deltas.append(current)
    return deltas
```

- [ ] **Step 4: Insert the cache check and hit-rendering into `_rag_stream`**

The current `else` branch (non-empty query) reads:

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
```

Replace it with:

```python
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
                    span.update(output=full_answer)
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
                span.update(output=full_answer)

                if full_answer.strip():
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
```

- [ ] **Step 5: Verify manually — module imports cleanly**

```bash
python -c "import api.index; print('import ok')"
```

Expected: prints `import ok`, no traceback (this alone catches typos/syntax errors in the rewritten block; full behavioral verification is Task 9, once the Qdrant cache collection exists).

- [ ] **Step 6: Commit**

```bash
git add api/index.py
git commit -m "feat: check and populate the semantic cache in the chat stream"
```

---

### Task 8: Frontend — carry `seedChunkIds` on a follow-up click

**Files:**
- Modify: `components/message.tsx`

**Interfaces:**
- Consumes: `message.metadata?.seedChunkIds` (already populated server-side by the existing `finish.messageMetadata.seedChunkIds`, per the 2026-09-11 grounding work).

- [ ] **Step 1: Update the follow-up button handler**

In `components/message.tsx`, change:

```tsx
                        onClick={() => sendMessage({ text: question })}
```

to:

```tsx
                        onClick={() =>
                          sendMessage({
                            text: question,
                            metadata: { seedChunkIds: message.metadata?.seedChunkIds ?? [] },
                          })
                        }
```

- [ ] **Step 2: Verify manually**

```bash
npx tsc --noEmit
```

Expected: no new type errors introduced by this change (pre-existing errors elsewhere, if any, are not this task's concern — compare against a run before this edit if unsure).

- [ ] **Step 3: Commit**

```bash
git add components/message.tsx
git commit -m "fix: carry seedChunkIds when a follow-up question is clicked"
```

---

### Task 9: End-to-end manual verification

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Create the Qdrant cache collection (one-time, ad hoc — not committed)**

```bash
python -c "
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from api.rag.config import settings

async def main():
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    name = settings.qdrant_cache_collection_name
    if await client.collection_exists(name):
        print('already exists:', name)
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print('created:', name)

asyncio.run(main())
"
```

Expected: prints `created: <name>` (or `already exists: <name>` if re-run).

- [ ] **Step 2: Start the dev server**

```bash
python -m uvicorn api.index:app --reload
```

Expected: starts without error, logs show retrieval service, LLM provider, and no crash from `Settings()`.

- [ ] **Step 3: Fresh question — expect a cache miss, then a write**

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Quel est le tarif de l'\''adhesion ?"}]}'
```

Expected: a normal SSE stream (`text-delta` frames building up an answer, `data-sources`, possibly `data-questions`, `finish`). Server logs show `Cached answer for query ... into '<collection>'.` — confirms the write path ran.

- [ ] **Step 4: Same question repeated — expect a cache hit**

Re-run the exact same `curl` command from Step 3.

Expected: server logs show `Semantic cache hit for query ... (score=...)`. The SSE stream should still look like a normal answer (simulated streaming), with the same `data-sources`/`data-questions` content as Step 3's answer, and `finish.messageMetadata.seedChunkIds` matching Step 3's chunk ids.

- [ ] **Step 5: Follow-up with `seedChunkIds` — expect the cache to be bypassed**

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Quel est le tarif de l'\''adhesion ?","metadata":{"seedChunkIds":["some-chunk-id"]}}]}'
```

Expected: server logs show **no** `Semantic cache hit`/lookup log line for this request (the `if not seed_chunk_ids:` guard skips the check entirely) — a fresh retrieval + generation runs instead, and a `Cached answer for query ...` write log still appears afterward.

- [ ] **Step 6: Browser check — click a real follow-up**

Run `npm run next-dev` alongside the FastAPI server (or the combined dev script if one exists), ask a question in the UI, wait for suggested follow-up buttons to appear, click one, and confirm in the FastAPI server logs that the request's payload included `metadata.seedChunkIds` (no cache-hit log line for that request) — confirms Task 8's frontend fix reaches the backend end-to-end.

- [ ] **Step 7: No commit for this task** (verification only — if any step reveals a bug, fix it in the relevant task's file and amend that task's commit message context by creating a small follow-up commit, e.g. `fix: correct semantic cache hit rendering`).
