# Tepsia — AddDocuments

Ingestion de documents PDF depuis **kDrive** vers une base vectorielle **Qdrant**.

```
kDrive ──► kdrive_downloader.py ──► PDF + manifest.json ──► main.py ──► Qdrant
                                                              │
                                     Gemini (parsing) ► chunking ► Mistral (embeddings)
```

1. `scripts/kdrive_downloader.py` télécharge un dossier kDrive et écrit un `manifest.json` (chemin local + lien de partage de chaque fichier).
2. `main.py` lit le manifest et, pour chaque PDF : parse en Markdown (Gemini), découpe en chunks, calcule les embeddings denses (Mistral), puis envoie le tout dans Qdrant. Les vecteurs BM25 (sparse) sont calculés côté serveur par Qdrant Cloud.

Un fichier est identifié par le SHA-256 de son contenu : **relancer l'ingestion ne réindexe pas ce qui est déjà en base**.

## Prérequis

- Python 3.13 et [uv](https://docs.astral.sh/uv/)
- Un cluster [Qdrant Cloud](https://cloud.qdrant.io/) (l'inférence cloud doit être disponible, elle sert au BM25)
- Des clés API Gemini et Mistral, un token kDrive (Infomaniak)

## Installation

```bash
uv sync                       # crée .venv et installe les dépendances
cp .env.example .env          # puis renseigner les valeurs (voir ci-dessous)
```

### Variables d'environnement (`.env`)

| Variable | Rôle |
|---|---|
| `GEMINI_API_KEY`, `GEMINI_PARSING_MODEL` | Parsing des PDF (ex. `gemini-2.5-flash`) |
| `MISTRAL_API_KEY`, `MISTRAL_EMBEDDING_MODEL` | Embeddings denses (`mistral-embed`, 1024 dimensions) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Accès au cluster Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | Collection cible **par défaut** |
| `KDRIVE_TOKEN`, `KDRIVE_DRIVE_ID`, `KDRIVE_SHARE_UUID` | Accès à kDrive et construction des liens de prévisualisation |

> `.env` contient des secrets : il est ignoré par git, ne le commitez pas.

### Créer la collection Qdrant

`main.py` refuse de tourner si la collection cible n'existe pas. Pour la créer avec le bon schéma (vecteur dense 1024 dims en cosinus, vecteur sparse BM25, index de payload) :

```bash
uv run python main.py --create-collection nom-de-la-collection
```

La commande échoue si la collection existe déjà : elle n'écrase jamais rien.

## Utilisation

Toutes les commandes se lancent depuis la racine du projet.

### 1. Télécharger un dossier kDrive

```bash
uv run python scripts/kdrive_downloader.py --folder-id 30609 --dest ./kdrive_downloads
```

| Option | Description |
|---|---|
| `--folder-id` | ID numérique du dossier kDrive (obligatoire) |
| `--dest` | Dossier local de destination (défaut : `kdrive_downloads`) |
| `--manifest` | Chemin du manifest (défaut : `<dest>/manifest.json`) |
| `--no-recursive` | Ne pas descendre dans les sous-dossiers |

Les fichiers déjà présents localement ne sont pas retéléchargés. L'API kDrive est limitée à 60 requêtes/minute, le script temporise en conséquence.

### 2. Ingérer dans Qdrant

```bash
# Tous les fichiers d'un manifest
uv run python main.py --ingest-manifest kdrive_downloads/manifest.json

# Un seul PDF
uv run python main.py --ingest chemin/vers/fichier.pdf --link-preview "https://kdrive.infomaniak.com/..."
```

| Option | Description |
|---|---|
| `--collection NOM` | Collection cible, à la place de `QDRANT_COLLECTION_NAME` |
| `--link-preview URL` | Lien de prévisualisation du fichier (avec `--ingest` ; le manifest le fournit déjà) |
| `--debug` | Écrit le Markdown parsé et les chunks dans `debug/<horodatage>/` pour inspection |

Comportement à connaître :

- Seuls les **PDF** sont ingérés ; les autres fichiers du manifest sont ignorés (comptés comme « skipped »).
- Un fichier en échec n'arrête pas le lot. Un récapitulatif (`ingested` / `skipped` / `failed`) est affiché à la fin, avec l'étape en cause (`parsing`, `embedding`, `upsert`…). Le code de sortie est `1` si au moins un fichier a échoué.
- Les logs sont affichés dans la console et écrits dans `logs/ingestion.log`.

### Tester sans toucher à la collection principale

Créer une collection de test et cibler celle-ci avec `--collection` :

```bash
uv run python main.py --create-collection tepsia-test
uv run python main.py --ingest-manifest kdrive_downloads/manifest.json --collection tepsia-test
```

Les points sont visibles dans l'interface Qdrant Cloud. Pour repartir de zéro, supprimer la collection depuis cette interface puis la recréer.

## Structure du projet

```
main.py                     CLI (--create-collection, --ingest, --ingest-manifest)
config.py                   Settings (.env) et configuration des logs
container.py                Câblage des adapters aux ports
scripts/kdrive_downloader.py  Téléchargement kDrive + manifest
services/ingestion_service.py Pipeline d'ingestion (existence → parsing → chunking → embedding → upsert)
ports/                      Interfaces (parser, chunker, embedder, vector store, ingestion)
adapters/
  parsing/                  Gemini (PDF → Markdown paginé)
  chunking/                 Découpage Markdown : titres → tableaux → taille max
  embedding/                Mistral (embeddings denses)
  vectorstore/              Qdrant (dense + BM25 sparse, création de collection)
domain/                     Modèles, identifiants déterministes, exceptions, prompt de parsing
```

## Dépannage

- **`Qdrant collection '...' does not exist`** : créer la collection (voir plus haut) ou corriger `QDRANT_COLLECTION_NAME`.
- **Erreurs Gemini 503 / `UNAVAILABLE`** : réessayées automatiquement (jusqu'à 5 fois avec attente croissante).
- **`finish_reason=MAX_TOKENS` au parsing** : la sortie du modèle est tronquée, réduire `max_pages_per_batch` dans `GeminiDocumentParser`.
- **Changer de modèle d'embedding** : les vecteurs doivent avoir la même dimension que la collection (1024 pour `mistral-embed`). Adapter `DENSE_VECTOR_SIZE` dans `container.py` et créer une nouvelle collection en conséquence.
