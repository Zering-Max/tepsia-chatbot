# Migration du RAG vers Mistral

Date : 2026-08-09
Branche : `mistral`

## Objectif

Basculer le pipeline RAG d'OpenAI vers Mistral, pour l'embedding des requêtes
(retrieval) comme pour la génération des réponses. Les adapters OpenAI restent
dans le repo mais ne sont plus câblés.

## Contexte

Le pipeline suit une architecture hexagonale : les ports `DenseEmbedder`,
`VectorStore` et `LLMProvider` isolent le domaine des fournisseurs. La bascule
se joue donc surtout dans le composition root (`container.py`) et dans un
nouvel adapter LLM.

`mistral_embedder.py` existe déjà (ajouté hors de ce chantier) et implémente
`DenseEmbedder` par batches concurrents. `mistral_llm.py` est vide.

L'ingestion des documents vit hors de ce repo. Une collection Qdrant réindexée
avec `mistral-embed` (1024 dims, contre 1536 pour OpenAI) existe déjà : seul le
nom de collection change côté API.

## Décisions

| Sujet | Décision |
|---|---|
| Portée | Bascule totale du câblage ; `openai_llm.py` et `openai_embedder.py` restent dans le repo, non câblés, pour permettre un retour arrière |
| Modèle de chat | `mistral-medium-latest` par défaut, surchargeable par `MISTRAL_LLM_MODEL` |
| Tracing | Observations Langfuse manuelles (`start_as_current_observation(as_type="generation")`), le drop-in `langfuse.openai` ne couvrant pas Mistral |
| Helpers de citation | Dupliqués dans `mistral_llm.py` ; `openai_llm.py` n'est pas modifié |
| Collection Qdrant | Déjà réindexée hors repo ; `QDRANT_COLLECTION_NAME` à repointer |

## API Mistral (vérifiée sur `mistralai` 2.9.1)

Signatures confirmées par introspection du SDK installé, pas de mémoire :

- Import : `from mistralai.client import Mistral` (pas de `Mistral` à la racine du package en 2.x)
- Génération : `await client.chat.complete_async(model=..., messages=[...])`
- Streaming : `await client.chat.stream_async(...)` renvoie un `EventStreamAsync[CompletionEvent]`,
  utilisable en `async with` puis `async for` ; le texte est en `event.data.choices[0].delta.content`
- Embeddings : `await client.embeddings.create_async(model=..., inputs=[...])`
- JSON forcé : `response_format={"type": "json_object"}` est supporté

Point d'attention : `delta.content` et `message.content` sont typés
`str | list[ContentChunk] | None`. Un helper de normalisation est nécessaire,
sinon des objets sont concaténés dans la réponse.

## Périmètre

### `requirements.txt`

`mistralai` 2.9.1 impose `opentelemetry-semantic-conventions <0.61`, alors que
l'`opentelemetry-sdk` 1.43.0 tiré par Langfuse exige `==0.64b0` : `pip check`
échoue. Langfuse accepte `opentelemetry-sdk >=1.33.1,<2`, donc la pile OTel est
épinglée sur une version commune aux deux.

Ajouts : `mistralai==2.9.1`, `opentelemetry-api==1.39.1`,
`opentelemetry-sdk==1.39.1`, `opentelemetry-semantic-conventions==0.60b1`,
`opentelemetry-exporter-otlp-proto-http==1.39.1`. `openai==2.6.0` est conservé.

### `api/rag/adapters/llm/mistral_llm.py` (nouveau)

`MistralLLMProvider(LLMProvider)`, dataclass portant `mistral_client: Mistral`
et `model: str`, implémentant les trois méthodes du port :

- `generate` — appel unique, réponse complète, footer des sources citées
- `generate_stream` — un `TextDeltaEvent` par fragment, puis un `SourcesEvent`
  unique construit à partir des `[N]` trouvés dans le texte complet
- `generate_followup_questions` — court-circuitée quand la réponse contient
  `NO_INFO_SENTINEL` ; toute erreur (API, JSON invalide) est journalisée et
  renvoie un événement vide

Helpers dupliqués depuis `openai_llm.py` : `_build_user_message`,
`_extract_cited_indices`, `_extract_cited_sources`, `_format_page_range`,
`_build_cited_sources`, `_deduplicate`. Ajout d'un `_as_text` propre à Mistral
pour normaliser le contenu `str | list[ContentChunk]`.

Chaque méthode ouvre une observation Langfuse `as_type="generation"`
renseignant `model`, `input`, `output` et `usage_details`. En streaming,
l'usage est lu sur le dernier chunk.

### `api/rag/adapters/embedding/mistral_embedder.py`

Les imports sont absolus (`from domain.models import ...`) là où le package
utilise du relatif : `from ...domain.models import ...`. En l'état le module ne
s'importe pas. Ajout aussi d'un docstring de module, comme les autres adapters.

### `api/rag/config.py`

Nouveaux champs : `mistral_api_key`, `mistral_embedding_model` (défaut
`mistral-embed`), `mistral_llm_model` (défaut `mistral-medium-latest`).
`openai_api_key`, `openai_embedding_model` et `llm_model` deviennent
optionnels, les adapters OpenAI n'étant plus construits au démarrage.

### `api/rag/container.py`

Construit un client `Mistral` et câble `MistralDenseEmbedder` et
`MistralLLMProvider`. L'import `langfuse.openai` disparaît. `build_llm_provider`
est annotée par le port `LLMProvider` plutôt que par une classe concrète.

### `api/index.py`

Le global `llm_provider` est typé `LLMProvider` au lieu de `OpenAILLMProvider`.

### `api/rag/adapters/vectorstore/qdrant_store.py`

Docstrings uniquement : les mentions « 1536-dim » et « OpenAI embeddings » ne
correspondent plus à la collection utilisée.

### `api/rag/ports/llm.py`

`generate_stream` était déclarée `async def -> AsyncIterator[StreamEvent]`, ce
qui se type comme une coroutine à awaiter, alors que les deux adapters sont des
générateurs asynchrones. L'incohérence était masquée tant qu'`index.py`
référençait la classe concrète ; typer le global par le port l'a révélée. La
méthode passe en `def` simple retournant l'itérateur, sans effet à l'exécution.

### `.env.local`

Ajout de `MISTRAL_LLM_MODEL`. `QDRANT_COLLECTION_NAME` pointe sur
`tepsia-mistral` (collection 1024-dim déjà réindexée).

## Hors périmètre

- Le pipeline d'ingestion et la création de la collection Qdrant (hors repo)
- Toute refonte de `openai_llm.py`
- Le choix entre providers à l'exécution (une variable `LLM_PROVIDER` a été
  écartée : pas de besoin réel, surface de test doublée)
- `api/utils/stream.py` et `api/utils/prompt.py`. Ces restes du template Vercel
  importent `openai` au niveau module et sont chargés au démarrage via
  `index.py`, alors que `stream_text` et `convert_to_openai_messages` n'ont
  aucun appelant. **`openai` reste donc une dépendance runtime obligatoire**,
  indépendamment des adapters conservés. Les supprimer toucherait au chemin de
  streaming qui fonctionne : écarté au profit d'un périmètre minimal.

## Vérification

Le repo n'a pas de suite de tests. Résultats obtenus le 2026-08-09 :

1. `pip check` — « No broken requirements found »
2. Import de `api.index` et construction des adapters — `MistralLLMProvider`,
   `mistral-medium-latest`, `mistral-embed`, collection `tepsia-mistral`
3. Pipeline complet en direct sur Qdrant et l'API Mistral — 8 passages
   retrouvés, 105 deltas streamés, citations `[N]` produites et reparsées en
   3 `CitedSource` avec `link_preview`, 3 questions de suivi
4. Garde-fou hors-sujet — une question sans rapport renvoie exactement la
   phrase de refus, sans sources ni questions de suivi
5. Route HTTP réelle (`POST /api/chat` sur uvicorn) — ordre des trames
   `start → text-start → text-delta ×76 → text-end → data-sources →
   data-questions → finish → [DONE]`, aucune trame `error`, en-têtes de
   streaming Vercel présents
6. Contrat UI — `data-sources` porte bien `index`, `file_name`, `page_start`,
   `page_end`, `link_preview` (ce que lit `components/message.tsx`) et
   `data-questions` une liste de chaînes ; flux valide en UTF-8, sans mojibake
7. Langfuse — `auth_check()` à `True`, observations de génération émises et
   flushées

Point non vérifié : le rendu visuel dans le navigateur. Le contrat de données
consommé par `components/message.tsx` est inchangé et validé trame par trame,
mais aucune session Next.js n'a été ouverte.

## Risques

- **Dimensions Qdrant** — si `QDRANT_COLLECTION_NAME` pointe encore sur la
  collection 1536-dim, toute requête échoue. Symptôme net et immédiat.
- **Pile OTel épinglée plus bas** — une mise à jour future de Langfuse exigeant
  un OTel plus récent rouvrira le conflit avec `mistralai`.
