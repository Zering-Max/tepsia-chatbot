# App chatbot du projet Teps'IA

Le projet part d'une base template de code de ce Github : https://github.com/vercel-labs/ai-sdk-preview-python-streaming
Il s'agit d'un front-end constitué de briques NextJS & ReactJS & Tailwind et d'une partie requêtes API (dossier `/api`) fait en Python. Pour l'instant, l'app est déployée sur la branche main sur un compte Vercel (Zering-Max).

## Comment lancer l'app en local

Il y a deux lancements en parallèle à faire :

- Tout d'abord, la partie app front-end NextJS, , lancer `yarn install` pour installer les dépendances front-end puis lancer `yarn next-dev`, à ce moment l'app sera déployé sur localhost:3000.
Vous pourrez alors interagir dans la page web créée mais dès que vous lancerez une question, vous n'aurez aucune réponse car la partie "fastAPI Python" (dossier `api`) qui reçoit la question pour interroger le modèle IA n'a pas été lancé.

- C'est ainsi qu'on lance dans un terminal parallèle la partie "API Python" avec la commande suivante : `yarn fastapi-dev`. Grâce à cela, vous pourrez en même temps installer les dépendances python nécessaires et interroger le modèle IA pour avoir une vraie interaction en Chat.


