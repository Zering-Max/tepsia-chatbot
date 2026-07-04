# Spec : Questions de relance ancrées dans le corpus

## Contexte

Les questions de relance sont générées par `generate_followup_questions` à
partir de la seule paire (question, réponse) : le LLM ne voit jamais les
passages récupérés, invente donc des questions plausibles auxquelles le
corpus ne sait souvent pas répondre, et le prompt exige toujours exactement
3 questions — même quand la réponse vient de dire que l'information n'est
pas disponible. Résultat : des relances en impasse.

## 1. Prompt (`api/rag/domain/prompts.py`)

- Nouvelle constante module `NO_INFO_SENTINEL = "Cette information n'est pas
  disponible dans les documents fournis."`, interpolée dans
  `RAG_SYSTEM_PROMPT` (version bump 1.2 → 1.3, texte inchangé par ailleurs)
  pour garantir que la phrase sentinelle et l'instruction restent synchrones.
- `FOLLOWUP_QUESTIONS_PROMPT` version 2.0 : reçoit désormais les passages du
  corpus en plus de la question et de la réponse. Règles : (1) ne proposer
  une question que si sa réponse figure explicitement dans les passages ;
  (2) renvoyer une liste vide si la réponse indique que l'information n'est
  pas disponible ou si aucune question pertinente n'a de réponse dans les
  passages ; (3) 0 à 3 questions, en français, périmètre Tepsa / Teps'out.

## 2. Port (`api/rag/ports/llm.py`)

```python
@abstractmethod
async def generate_followup_questions(
    self, query: str, answer: str, sources: list[SearchResult]
) -> QuestionsEvent:
    ...
```

## 3. Adaptateur (`api/rag/adapters/llm/openai_llm.py`)

- **Raccourci sentinelle** : si `NO_INFO_SENTINEL in answer`, retourner
  `QuestionsEvent(questions=[])` sans appel API (latence et coût nuls).
  Couverture partielle assumée : si le LLM reformule au lieu de citer la
  phrase verbatim, c'est la règle (2) du prompt qui prend le relais.
- Sinon : dédupliquer les sources (`_deduplicate`, même logique que la
  génération) et construire le message utilisateur en réutilisant
  `_build_user_message(query, unique_sources)` suivi de `Réponse : {answer}`.
- La gestion d'erreur existante (log + event vide) est conservée.

## 4. API (`api/index.py`)

`_rag_stream` passe `sources` (déjà en scope) au call :
`generate_followup_questions(query, full_answer, sources)`.

## 5. Frontend

Aucun changement : une liste vide n'émet déjà pas de frame `data-questions`.

## 6. Vérification (manuelle, pas de suite de tests — décision existante)

1. Question couverte par le corpus → frame `data-questions` avec 1-3
   questions, chacune trouvant une réponse dans le corpus.
2. Question dans le périmètre mais non couverte → réponse sentinelle et
   **aucune** frame `data-questions` (et logs uvicorn : pas d'appel
   questions).
