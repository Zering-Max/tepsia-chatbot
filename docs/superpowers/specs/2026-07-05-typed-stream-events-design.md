# Spec : Événements de stream typés de bout en bout

## Contexte

Le spec `2026-07-01-personalised-stream-response-design.md` a introduit les
modèles `CitedSource`, `TextDeltaEvent`, `SourcesEvent`, `QuestionsEvent` et
l'union `StreamEvent` dans `api/rag/domain/models.py`. L'implémentation qui a
suivi s'en est écartée sur deux points :

- les questions de relance sont générées par un **second appel LLM**
  (`generate_followup_questions`, JSON simple) au lieu d'un appel unique en
  `json_schema` strict — choix conservé, plus simple et déjà fonctionnel ;
- seuls `QuestionsEvent` est branché. `generate_stream` yield toujours des
  `str` bruts et les sources sont concaténées en **markdown** à la fin du flux
  texte (`_build_cited_sources`), ce qui fait fuir de la présentation dans le
  domaine et empêche le frontend de rendre `link_preview` cliquable.

Ce spec branche les trois modèles restants (approche « A — typé de bout en
bout », validée), et corrige au passage les défauts de robustesse relevés :
pas de frame `error` SSE en cas d'exception, `text-end` retardé par l'appel
questions, accès `chunk.choices[0]` sans garde, formatage de pages dupliqué.

## 1. Port (`api/rag/ports/llm.py`)

```python
@abstractmethod
async def generate_stream(
    self, query: str, sources: list[SearchResult]
) -> AsyncIterator[StreamEvent]:
    ...
```

Seul le type de retour change (`str` → `StreamEvent`). `generate` et
`generate_followup_questions` sont inchangés.

## 2. Adaptateur (`api/rag/adapters/llm/openai_llm.py`)

- `generate_stream` yield un `TextDeltaEvent(delta=...)` par fragment reçu,
  avec une garde `if not chunk.choices: continue` avant l'accès à
  `choices[0]`.
- Une fois le stream texte terminé, il construit `list[CitedSource]` à partir
  des `[N]` effectivement cités dans le texte assemblé et yield un
  `SourcesEvent`. **Le bloc markdown `**Sources :**` disparaît du flux.**
- Extraction des citations : la regex actuelle `\[(\d+)\]` est étendue pour
  couvrir les citations groupées `[1, 2]` :
  `\[(\d+(?:\s*,\s*\d+)*)\]`, chaque numéro du groupe étant traité comme une
  citation. La garde de bornes `1 <= idx <= len(sources)` est conservée
  (index hors bornes ignoré silencieusement).
- Nouveau helper `_extract_cited_sources(text, sources) -> list[CitedSource]`
  qui mappe chaque index cité vers `CitedSource(index, file_name,
  link_preview, page_start, page_end)` depuis `TextChunkMetadata`
  (`file_name="document inconnu"` si métadonnées absentes, autres champs à
  `None`).
- Le formatage « p. X–Y » est factorisé dans un helper unique
  `_format_page_range(meta) -> str` utilisé par `_build_user_message` et
  `_build_cited_sources`.
- La méthode non-streamée `generate` (inutilisée par l'API) garde son
  comportement actuel — appendice markdown inclus.

## 3. API (`api/index.py`, `_rag_stream`)

Dispatch par type d'événement :

| Événement domaine   | Frame SSE                                              |
|---------------------|--------------------------------------------------------|
| `TextDeltaEvent`    | `{"type":"text-delta","id":"text-1","delta":...}`      |
| `SourcesEvent`      | `{"type":"data-sources","data":[{...}, ...]}`          |
| `QuestionsEvent`    | `{"type":"data-questions","data":[...]}` (inchangé)    |

- Chaque `CitedSource` est sérialisé en dict avec ses noms de champs Python
  (`index`, `file_name`, `link_preview`, `page_start`, `page_end`) via
  `dataclasses.asdict`. Un `SourcesEvent` vide n'émet **pas** de frame
  `data-sources`.
- Le texte complet est accumulé à partir des `TextDeltaEvent` pour alimenter
  `generate_followup_questions`.
- **Ordre des frames** : `start` → `text-start` → `text-delta`×N →
  `text-end` → `data-sources` → `data-questions` → `finish` → `[DONE]`.
  `text-end` est émis dès la fin du flux de l'adaptateur, **avant** l'appel
  LLM des questions de relance.
- **Gestion d'erreur** : le corps de `_rag_stream` (récupération + génération)
  est enveloppé dans un `try/except` qui log l'exception
  (`logger.exception`) et émet un frame `{"type":"error","errorText":...}`
  (format Vercel AI SDK), suivi de `[DONE]`, au lieu de couper la connexion
  SSE silencieusement. `errorText` reste générique (pas de fuite de détails
  internes au client).

## 4. Frontend (`components/message.tsx`)

Nouveau rendu pour `part.type === "data-sources"` :

- Un bloc « Sources : » listant chaque entrée `[N] file_name, p. X–Y`
  (plage de pages omise si absente ; « p. X » si `page_start == page_end`).
- Chaque entrée est un lien (`<a target="_blank" rel="noopener noreferrer">`)
  quand `link_preview` est présent, du texte simple sinon.
- Affiché pour **tous** les messages assistants (contrairement à
  `data-questions`, limité au dernier message).
- Style cohérent avec le bloc questions existant (`text-sm
  text-muted-foreground` pour le titre).

## 5. Cas limites

- LLM ne cite rien → `SourcesEvent(sources=[])` → aucun frame
  `data-sources` → pas de bloc Sources côté front.
- Citation `[N]` hors bornes → ignorée (comportement actuel conservé).
- Échec de `generate_followup_questions` → déjà géré dans l'adaptateur
  (event vide, dégradation silencieuse).
- Exception Qdrant/OpenAI en cours de stream → frame `error` + fin propre du
  flux (pas de hang côté AI SDK).

## 6. Tests

Le projet n'a pas de suite de tests automatisés (décision existante,
inchangée). Vérification manuelle end-to-end après implémentation :

1. `curl -N` sur `http://127.0.0.1:8000/api/chat` avec timestamps : frames
   `text-delta` progressifs, puis `text-end`, `data-sources` (structuré,
   plus de markdown `**Sources :**`), `data-questions`, `finish`.
2. Contrôle du rendu navigateur : texte streamé en direct, bloc Sources
   cliquable, questions de relance fonctionnelles.
