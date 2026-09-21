# App chatbot du projet Teps'IA

Le projet part d'une base template de code de ce Github : https://github.com/vercel-labs/ai-sdk-preview-python-streaming
Il s'agit d'un front-end constitué de briques NextJS & ReactJS & Tailwind et d'une partie requêtes API (dossier `/api`) fait en Python. Pour l'instant, l'app est déployée sur la branche main sur un compte Vercel (Zering-Max).

## Comment lancer l'app en local

Il y a deux lancements en parallèle à faire :

- Tout d'abord, la partie app front-end NextJS, , lancer `yarn install` pour installer les dépendances front-end puis lancer `yarn next-dev`, à ce moment l'app sera déployé sur localhost:3000.
Vous pourrez alors interagir dans la page web créée mais dès que vous lancerez une question, vous n'aurez aucune réponse car la partie "fastAPI Python" (dossier `api`) qui reçoit la question pour interroger le modèle IA n'a pas été lancé.

- C'est ainsi qu'on lance dans un terminal parallèle la partie "API Python" avec la commande suivante : `yarn fastapi-dev`. Grâce à cela, vous pourrez en même temps installer les dépendances python nécessaires et interroger le modèle IA pour avoir une vraie interaction en Chat.

## Variables d'environnement

L'app a besoin d'un fichier `.env.local` à la racine avec les variables suivantes :

- `MISTRAL_API_KEY` — clé API Mistral (embeddings et génération).
- `QDRANT_URL` — URL de l'instance Qdrant.
- `QDRANT_API_KEY` — clé API Qdrant.
- `QDRANT_COLLECTION_NAME` — nom de la collection Qdrant contenant les documents (chunks).
- `QDRANT_CACHE_COLLECTION_NAME` — nom de la collection Qdrant utilisée par le cache sémantique.

### Cache sémantique : création de la collection Qdrant

La collection du cache sémantique (`QDRANT_CACHE_COLLECTION_NAME`) n'est pas créée automatiquement par le code — c'est une étape manuelle à faire une fois, avant le premier déploiement/lancement :

```python
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

async def main():
    client = AsyncQdrantClient(url="<QDRANT_URL>", api_key="<QDRANT_API_KEY>")
    await client.create_collection(
        collection_name="<QDRANT_CACHE_COLLECTION_NAME>",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

asyncio.run(main())
```

La dimension (1024) correspond au modèle d'embedding `mistral-embed`. La distance doit être `Distance.COSINE` — le seuil de similarité du cache (`SemanticCache.SIMILARITY_THRESHOLD`) suppose un score "plus haut = plus similaire", ce qui n'est vrai qu'avec cette métrique.
