# Fiabiliser les questions de suivi (seed chunks)

Date : 2026-09-11
Branche : `dev`

## Objectif

Corriger le cas où une question de suivi suggérée, une fois cliquée, reçoit
une réponse "Cette information n'est pas disponible dans les documents
fournis." alors que l'information existe bien dans le corpus.

## Root cause (confirmée par reproduction en direct)

`FOLLOWUP_QUESTIONS_PROMPT` exige que chaque question suggérée soit
répondable à partir des passages qui ont servi à générer la réponse
courante. Cette garantie tient au moment de la génération : le modèle voit
les 8 passages retrouvés pour la question initiale.

Mais au clic, `components/message.tsx` envoie la question de suivi comme un
message tout neuf (`sendMessage({ text: question })`), et
`api/index.py::_rag_stream` relance une recherche hybride *indépendante* sur
le seul texte de cette question (`retrieval_service.retrieve(query)`). Rien
ne relie cette nouvelle recherche aux passages d'origine.

Repro concrète :
- Q1 : "parle moi de l'objectif du projet tepsout" → le passage
  `agir_greenvoice_fr...petitions...pdf` chunk 16 ("Le lancement de la
  pétition a fait tellement qu'il a attiré l'attention de la presse
  nationale : Les Echos ont consacré un article...") est retrouvé en position
  5/8 et fonde la question de suivi générée "Quels médias nationaux ont
  relayé la mobilisation du collectif Teps'out ?".
- Au clic sur cette question, la recherche fraîche sur son texte ne fait
  *pas* remonter ce chunk 16 dans son propre top-8 (il est évincé par des
  chunks voisins parlant de mobilisation en général) → le LLM ne reçoit pas
  le passage contenant "Les Echos" → refus.

Ce n'est pas une régression de la migration Mistral (le flux "nouvelle
requête, nouvelle recherche indépendante" existait déjà avant) ; la
migration a pu changer la fréquence du symptôme (embeddings différents →
classement différent) mais pas sa cause.

## Décisions

| Sujet | Décision |
|---|---|
| Transport du contexte | Via les metadata du message (mécanisme natif AI SDK v5), pas d'état serveur |
| Fusion | Seeds d'origine + nouvelle recherche fraîche, dédupliqués — pas seeds seuls |
| Portée des seeds | Tous les passages uniques utilisés pour générer la réponse/les questions de suivi (pas seulement les `[N]` cités dans le texte) |
| Cap de taille | Aucun cap supplémentaire après fusion (union dédupliquée, ~16 passages au pire) |
| Erreur sur fetch des seeds | Non bloquante : log et repli sur la recherche fraîche seule |

## Flux de données

1. `_rag_stream` récupère `sources = await retrieval_service.retrieve(query, seed_chunk_ids)` où `seed_chunk_ids` vient du dernier message utilisateur.
2. La réponse est générée et streamée comme aujourd'hui à partir de `sources`.
3. `generate_followup_questions` tourne sur le même `sources`, comme aujourd'hui.
4. La frame `finish` porte désormais `messageMetadata: {finishReason, seedChunkIds}`, où `seedChunkIds` est la liste dédupliquée (ordre préservé) des `chunk.id` de `sources`.
5. Côté client, `useChat` attache automatiquement ce `messageMetadata` au message assistant (`message.metadata.seedChunkIds`).
6. Au clic sur une question de suivi, `message.tsx` appelle `sendMessage({ text: question, metadata: { seedChunkIds: message.metadata?.seedChunkIds } })`.
7. Le body JSON de la requête suivante contient ces `seedChunkIds` sur le nouveau message utilisateur ; `api/index.py` les extrait et reboucle à l'étape 1.

Un message tapé librement (pas de clic sur une suggestion) n'a pas de
`seedChunkIds` → comportement inchangé (recherche fraîche seule).

Les questions de suivi générées à partir de cette nouvelle réponse portent
leurs propres `seedChunkIds` (= `sources` de *cette* réponse), donc la
chaîne se propage correctement sur plusieurs clics successifs sans
accumulation incontrôlée : chaque saut ne porte que le contexte de son
parent immédiat.

## Composants modifiés

### `api/rag/ports/vector_store.py`
Nouvelle méthode abstraite :
```python
async def get_by_ids(self, ids: list[str]) -> list[SearchResult]
```

### `api/rag/adapters/vectorstore/qdrant_store.py`
Implémentation via `async_qdrant_client.retrieve(collection_name=..., ids=ids, with_payload=True)`.
Les `Record` retournés n'ont pas de score de similarité ; `SearchResult.score`
est fixé à `1.0` (constante, documentée comme "score synthétique — fetch par
ID, pas de recherche"). IDs introuvables (doc ré-ingéré, supprimé) : ignorés
silencieusement, pas d'exception.

### `api/rag/services/retrieval_service.py`
```python
async def retrieve(self, query: str, seed_chunk_ids: list[str] | None = None) -> list[SearchResult]
```
- Lance `hybrid_search` comme aujourd'hui.
- Si `seed_chunk_ids` est non vide, appelle `vector_store.get_by_ids(seed_chunk_ids)`.
- Fusionne : résultats frais d'abord, puis seeds non déjà présents (dédup par `chunk.id`), sans tri par score ni troncature.
- Toute exception sur `get_by_ids` est loggée et ignorée (repli sur la recherche fraîche seule) — cohérent avec le traitement des erreurs de `generate_followup_questions` déjà présent dans le codebase.

### `api/utils/prompt.py`
`ClientMessage` gagne :
```python
metadata: Optional[dict] = None
```

### `api/index.py`
- Nouvelle fonction `_extract_seed_chunk_ids(messages) -> list[str]`, symétrique de `_extract_last_user_query` : lit `metadata.get("seedChunkIds")` sur le dernier message utilisateur, valide que c'est une liste de strings, sinon `[]`.
- `_rag_stream` appelle `retrieval_service.retrieve(query, seed_chunk_ids)`.
- La frame `finish` devient :
  ```python
  yield sse({
      "type": "finish",
      "messageMetadata": {
          "finishReason": "stop",
          "seedChunkIds": list(dict.fromkeys(r.chunk.id for r in sources)),
      },
  })
  ```
  (dédup en préservant l'ordre, `dict.fromkeys` plutôt que `set` pour la stabilité)

### `components/message.tsx`
Le handler du bouton de question de suivi devient :
```tsx
onClick={() =>
  sendMessage({
    text: question,
    metadata: { seedChunkIds: message.metadata?.seedChunkIds ?? [] },
  })
}
```

## Hors périmètre

- Pas de changement au contrat `data-sources` (citations affichées) : reste basé sur `cited_sources`, indépendant de ce mécanisme.
- Pas de mémoire serveur / session : l'API reste stateless, tout le contexte transite par le message.
- Pas de changement au prompt `FOLLOWUP_QUESTIONS_PROMPT` lui-même.
- Pas de ré-ingestion ni de changement du pipeline d'embedding.

## Risques

- **Taille de prompt** : dans le pire cas (aucun recoupement entre seeds et recherche fraîche), jusqu'à 16 passages envoyés au LLM au lieu de 8. Impact latence/coût mineur avec `mistral-medium-latest`, pas de mitigation prévue dans ce chantier.
- **IDs de chunks obsolètes** : si le corpus est ré-ingéré entre deux messages d'une même conversation (fenêtre courte en pratique), `get_by_ids` ne retrouve rien pour ces IDs — dégradation silencieuse vers le comportement actuel, pas de régression.
