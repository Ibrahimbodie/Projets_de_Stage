# Assistant académique Master SIM

Assistant RAG (Retrieval-Augmented Generation) pour répondre aux questions des étudiants à partir d'un document officiel PDF du Master SIM.

Le mode principal actuel est **terminal (CLI)**.

## Fonctionnement

1. Le PDF est chargé depuis `data/Contrat_apprentissage_SIM_P27.pdf`.
2. Le texte est découpé en segments (`chunk_size=800`, `chunk_overlap=150`).
3. Les embeddings sont générés via `qwen3-embedding` (Ollama).
4. Les segments sont indexés dans FAISS (`vectorstore/`).
5. À chaque question:
   - récupération des passages les plus proches (`top_k`, seuil de similarité),
   - génération de réponse avec `qwen3.5:27b` (Ollama),
   - affichage des sources (pages).

Si aucune information pertinente n'est trouvée, la réponse retourne:
`Information non trouvée dans les règlements officiels.`

## Prérequis

- Linux/macOS (scripts bash)
- Python 3.12 (testé dans ce projet)
- [Ollama](https://ollama.com/download) installé
- Accès réseau pour télécharger les modèles Ollama au premier lancement

## Installation

Depuis la racine du projet:

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install langchain langchain-community langchain-core langchain-ollama langchain-text-splitters faiss-cpu pypdf streamlit
```

> `streamlit` reste utile si vous voulez lancer l'UI web legacy (`src/main.py`), même si le mode actif est terminal.

## Lancement rapide (recommandé)

```bash
bash run.sh
```

Le script:
- vérifie l'environnement virtuel,
- démarre `ollama serve` si nécessaire,
- télécharge les modèles requis si absents (`qwen3-embedding`, `qwen3.5:27b`),
- reconstruit l'index vectoriel,
- lance l'interface terminal.

## Utilisation CLI

### Mode interactif

```bash
env/bin/python src/cli.py
```

Commandes de sortie: `exit`, `quit`, `q`

### Mode question unique

```bash
env/bin/python src/cli.py --question "Quelle est la durée de l'apprentissage ?"
```

### Options disponibles

```bash
env/bin/python src/cli.py --help
```

Options utiles:
- `--top-k` (défaut: `4`)
- `--min-similarity` (défaut: `0.3`)
- `--llm-model` (défaut: `qwen3.5:27b`)

## Recréer uniquement l'index vectoriel

```bash
env/bin/python src/ingest.py
```

## Lancer l'ancienne interface Streamlit (optionnel)

```bash
env/bin/streamlit run src/main.py
```

## Structure du projet

```text
.
├── data/
│   └── Contrat_apprentissage_SIM_P27.pdf
├── src/
│   ├── ingest.py       # Chargement PDF + indexation FAISS
│   ├── retriever.py    # Recherche vectorielle + filtrage
│   ├── llm_chain.py    # Prompt + génération de réponse
│   ├── cli.py          # Interface terminal
│   └── main.py         # Interface Streamlit (legacy)
├── vectorstore/        # Index FAISS généré
└── run.sh              # Script d'exécution principal
```

## Dépannage

- `Index vectoriel introuvable ... Lance d'abord src/ingest.py.`
  - Exécuter: `env/bin/python src/ingest.py`

- `Connexion Ollama impossible...`
  - Vérifier que `ollama serve` tourne.
  - Vérifier que les modèles sont présents:
    - `ollama list`

- `PDF introuvable ...`
  - Vérifier le chemin: `data/Contrat_apprentissage_SIM_P27.pdf`

