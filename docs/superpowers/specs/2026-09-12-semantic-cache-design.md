# Cache sémantique (Qdrant)

Date : 2026-09-12
Branche : `feature-semantic_cache`

## Objectif

Éviter de relancer recherche + génération LLM quand une question sémantiquement
proche d'une question déjà posée existe en cache, en renvoyant directement la
réponse déjà générée. Recherche dense dans une collection Qdrant dédiée,
séparée de la collection documentaire.

## Décisions

| Sujet | Décision |
|---|---|
| Lecture du cache sur une question de suivi cliquée (`seed_chunk_ids` non vide) | Bypass complet — pas de lookup, génération fraîche comme aujourd'hui |
| Écriture du cache sur une réponse ancrée par seeds | Toujours effectuée (une future question sans seeds peut légitimement la retrouver) |
| Rendu d'un cache hit côté client | Streaming simulé (découpage du texte déjà généré en `text-delta`), pour rester cohérent avec l'UX d'une génération fraîche |
| Seuil de similarité | Constante de classe `SemanticCache.SIMILARITY_THRESHOLD`, pas de champ dans `Settings` |
| Top-k de la recherche cache | 1 (plus proche voisin uniquement, comparé au seuil) |
| Identité d'une entrée | `id` déterministe (uuid5 sur la question normalisée) — une question identique reposée réécrit l'entrée existante plutôt que d'empiler des doublons |
| Nettoyage périodique du cache | Hors périmètre de ce chantier ; `date` est présent sur chaque entrée pour le permettre plus tard |
| Panne du cache (Qdrant indisponible, etc.) | Non bloquante : loggée, dégrade vers le flux normal (lecture *et* écriture) |
| Fix frontend `seedChunkIds` sur les follow-ups | Inclus dans ce chantier (gap déjà identifié dans `docs/superpowers/specs/2026-09-11-followup-question-grounding-design.md`, jamais appliqué) |

## Pourquoi un port séparé de `VectorStore`

`VectorStore` est façonné autour de `EmbeddedChunk`/`TextChunk` : payload
chunk-only (`doc_id`, `chunk_index`, pages, `section_title`...), et des
méthodes (`document_exists`, `hybrid_search` avec BM25, `get_by_ids` par id de
chunk) qui n'ont pas de sens pour une entrée de cache. Le port `SemanticCache`
reste distinct, avec sa propre collection Qdrant et son propre adapter,
suivant le même style que `QdrantVectorStore` (dataclass, `AsyncQdrantClient` +
`collection_name`) mais sans réutiliser son interface.

## Contrat du port (`api/rag/ports/semantic_cache.py`)

```python
class SemanticCache(ABC):
    SIMILARITY_THRESHOLD: float = 0.97  # à affiner empiriquement

    @abstractmethod
    async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None:
        """Cherche le voisin le plus proche ; renvoie l'entrée si score >= SIMILARITY_THRESHOLD, sinon None."""

    @abstractmethod
    async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None:
        """Upsert l'entrée (id déterministe => une question identique écrase l'ancienne)."""
```

Le port ne dépend pas d'un embedder : l'appelant (`_rag_stream`) fournit le
vecteur déjà calculé, à la fois pour la lecture et pour l'écriture. Pas de
`top_k` ni de `threshold` en paramètre — recherche top-1 fixe, seuil en
constante de classe comme décidé.

## Modèle `CachedAnswer` (`api/rag/domain/models.py`)

```python
@dataclass(frozen=True)
class CachedAnswer:
    id: str
    date: datetime                  # UTC, moment de mise en cache
    query: Query
    embedded_query: DenseEmbedding
    retrieved_chunk_ids: list[str]  # sources utilisées pour générer la réponse
    cited_sources: SourcesEvent     # sources effectivement citées, pour re-emit data-sources
    generated_answer: str
    questions: list[str]            # follow-up questions suggérées, pour re-emit data-questions
```

Deux champs vont au-delà de l'ébauche initiale :

- **`retrieved_chunk_ids`** (et non seulement les sources citées) : nécessaire
  pour que `finish.messageMetadata.seedChunkIds` reste correct sur un cache
  hit, afin que la chaîne de grounding des follow-ups continue de fonctionner
  même après une réponse servie depuis le cache.
- **`cited_sources` renommé** (au lieu de `sources`) pour clarifier qu'il
  s'agit des citations affichées (`SourcesEvent`), distinct de
  `retrieved_chunk_ids`.

### Génération de l'id

Pas de helper UUID5 existant à réutiliser dans ce repo (le schéma documenté
sur `TextChunk.id` est implémenté dans un pipeline d'ingestion externe). Ajout
dans `domain/models.py` :

```python
CACHE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "tepsia-chatbot/semantic-cache")

def compute_cache_id(query: str) -> str:
    normalized = query.strip().lower()
    return str(uuid.uuid5(CACHE_NAMESPACE, normalized))
```

## Adapter Qdrant (`api/rag/adapters/vectorstore/qdrant_semantic_cache.py`)

Nouveau fichier, même style que `QdrantVectorStore` :

```python
@dataclass
class QdrantSemanticCache(SemanticCache):
    async_qdrant_client: AsyncQdrantClient
    collection_name: str

    async def check_similar_questions(self, embedded_query: DenseEmbedding) -> CachedAnswer | None:
        response = await self.async_qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedded_query.vector,
            with_payload=True,
            limit=1,
        )
        if not response.points or response.points[0].score < self.SIMILARITY_THRESHOLD:
            return None
        return self._to_cached_answer(response.points[0])

    async def send_to_semantic_cache(self, cached_answer: CachedAnswer) -> None:
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[self._prepare_point(cached_answer)],
        )
```

Payload stocké : `date` (isoformat), `query`, `retrieved_chunk_ids`,
`cited_sources` (liste de dicts `asdict(CitedSource)`), `generated_answer`,
`questions`. Pas de vecteur nommé (une seule collection à un seul espace
vectoriel, pas besoin du pattern named-vectors de la collection documentaire).

La création de la collection Qdrant du cache (dimension du vecteur, distance
cosine) est une tâche opérationnelle hors périmètre de ce document — à faire
manuellement, comme pour la collection documentaire.

## Config (`api/rag/config.py`)

Ajout d'un seul champ :

```python
qdrant_cache_collection_name: str
```

Pas de champ de seuil (constante de classe, cf. plus haut).

## Wiring (`api/rag/container.py`)

```python
def build_semantic_cache() -> SemanticCache:
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantSemanticCache(
        async_qdrant_client=qdrant_client,
        collection_name=settings.qdrant_cache_collection_name,
    )
```

Même pattern que `build_retrieval_service` : un client Qdrant dédié, pas de
partage avec celui de `RetrievalService` (cohérent avec le fait que chaque
`build_*` construit ses propres dépendances aujourd'hui).

## `RetrievalService` : une méthode ajoutée, pas de rupture

Pour permettre à `_rag_stream` d'obtenir le vecteur de la question *avant* de
savoir s'il faut appeler `retrieve()`, sans dupliquer la construction du
`TextChunk` d'embedding ni exposer l'embedder en dehors du service :

```python
async def embed_query(self, query: str) -> DenseEmbedding:
    """Embeds a bare query string, without a search."""
    return (await self._embedder.embed(chunks=[TextChunk(content=query)]))[0]
```

`retrieve()` n'est pas modifié : sur un cache miss, il réembete la question en
interne (léger doublon d'appel d'embedding, jugé négligeable — une seule
petite chaîne — face à la complexité d'un partage du vecteur entre les deux
appels).

## Flux dans `_rag_stream` (`api/index.py`)

```
si query vide -> comportement actuel, inchangé

si seed_chunk_ids est vide ET query non vide :
    essaie :
        embedded_query = await retrieval_service.embed_query(query)
        cached = await semantic_cache.check_similar_questions(embedded_query)
    sauf Exception :
        log, cached = None   # dégrade vers génération fraîche

sinon :
    embedded_query = None
    cached = None            # bypass explicite (seeds présents)

si cached is not None :
    yield text-start
    pour chaque morceau de cached.generated_answer :        # streaming simulé
        yield text-delta
    yield text-end
    si cached.cited_sources.sources : yield data-sources
    si cached.questions : yield data-questions
    yield finish { seedChunkIds: cached.retrieved_chunk_ids }
    return

sinon (comportement actuel, inchangé) :
    sources = await retrieval_service.retrieve(query, seed_chunk_ids)
    ... génération + streaming réel, cited_sources_event capturé (pas seulement les dicts) ...
    yield finish { seedChunkIds: [...] }

    # écriture cache, toujours tentée si une réponse a été produite (même bypass)
    si full_answer.strip() :
        essaie :
            cache_vector = embedded_query or await retrieval_service.embed_query(query)
            cached_answer = CachedAnswer(
                id=compute_cache_id(query),
                date=datetime.now(timezone.utc),
                query=query,
                embedded_query=cache_vector,
                retrieved_chunk_ids=[dédupliqué depuis `sources`],
                cited_sources=cited_sources_event or SourcesEvent(sources=[]),
                generated_answer=full_answer,
                questions=questions_event.questions,
            )
            await semantic_cache.send_to_semantic_cache(cached_answer)
        sauf Exception :
            log, ignore   # non bloquant
```

Point d'implémentation : la boucle `async for event in llm_provider.generate_stream(...)`
capture aujourd'hui seulement `cited_sources: list[dict]` (via `asdict`). Il
faut aussi garder l'objet `SourcesEvent` brut (`cited_sources_event`) pour le
réinjecter tel quel dans `CachedAnswer.cited_sources` sans reconstruction.

### Streaming simulé

Petit helper privé dans `api/index.py`, pas de dépendance externe :

```python
def _split_into_deltas(text: str, chunk_size: int = 40) -> list[str]:
    """Splits text into chunks for a simulated text-delta stream."""
```

Découpage simple par tranches de caractères (pas de délai artificiel entre les
`yield` — la latence réseau/sérialisation suffit à donner un effet de
streaming, pas besoin d'`asyncio.sleep`).

## `messages.tsx` : fix du gap seedChunkIds

```tsx
onClick={() =>
  sendMessage({
    text: question,
    metadata: { seedChunkIds: message.metadata?.seedChunkIds ?? [] },
  })
}
```

Change déjà spécifié dans le design du 2026-09-11, jamais appliqué au code —
sans ce fix, `_extract_seed_chunk_ids` ne reçoit jamais rien et la règle de
bypass n'est testable qu'en construisant la requête manuellement.

## Hors périmètre

- Nettoyage périodique du cache (comptage des questions, purge par âge...) —
  aucun mécanisme de tâche planifiée n'existe dans ce repo (déploiement
  serverless Vercel, pas de cron/APScheduler/BackgroundTasks). À concevoir
  séparément une fois cette itération en place.
- Création/dimensionnement de la collection Qdrant du cache — tâche opérationnelle.
- Invalidation du cache quand le corpus documentaire change (une entrée
  cachée peut référencer des `retrieved_chunk_ids` qui n'existent plus après
  une ré-ingestion) — dégradation silencieuse acceptée pour cette itération,
  comme pour `get_by_ids` dans le flux seed_chunk_ids existant.
- Tuning du seuil de similarité — valeur de départ à ajuster empiriquement une
  fois en place, pas figée par ce document.

## Risques

- **Faux positifs de similarité** : deux questions textuellement proches mais
  sémantiquement différentes (ex. deux articles différents évoqués en des
  termes similaires) pourraient dépasser le seuil et servir la mauvaise
  réponse. Mitigé par un seuil volontairement haut (0.97 de départ) ; pas de
  garde-fou supplémentaire dans cette itération.
- **Staleness** : une réponse cachée reste servie même si le document source
  a changé, jusqu'au nettoyage (non conçu dans ce chantier) ou à une question
  reformulée qui échappe au seuil.
- **Coût d'un embedding supplémentaire** : chaque question sans seeds déclenche
  un appel d'embedding pour le lookup cache, même en cas de miss (où
  `retrieve()` réembete ensuite). Impact négligeable (une chaîne courte, un
  appel Mistral) mais réel.
