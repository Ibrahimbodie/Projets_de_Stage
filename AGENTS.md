# Contexte du Projet

Projets de Stage — Assistant Académique Intelligent (RAG) pour Master SIM.
Sujet : Assistant virtuel pédagogique basé sur Retrieval-Augmented Generation (RAG)
permettant aux étudiants du Master SIM d'interroger les règlements académiques
(contrats d'apprentissage) en langage naturel.

Stack : Python, Streamlit, LangChain, FAISS, Ollama (Qwen2.5:3b), DeepEval.

# Mémoire des Conversations

## 2026-07-01 — Session 1
- L'utilisateur a demandé comment garder une mémoire entre les sessions OpenCode.
- Solution mise en place : fichier AGENTS.md pour stocker le contexte persistant.
- L'utilisateur utilise OpenCode v1.17.13.
- Projet exploré : assistant RAG avec interface Streamlit, ingestion FAISS,
  retrieval sémantique, génération via Ollama, évaluation DeepEval.
- Rapport de stage : `rapport_methodologie_evaluation.docx` (44 Ko, DOCX).

## 2026-07-01 — Session 2 (Refactoring Architecture)
- Objectif : Rendre l'architecture modulaire pour la recherche.

### Modifications effectuées
- **`config.yaml`** (nouveau) — Configuration centralisée (embedding, llm, evaluation, retrieval).
- **`src/config.py`** (nouveau) — Dataclasses `EmbeddingConfig`, `LLMConfig`, `EvaluationConfig`, `RetrievalConfig`, `AppConfig` + loader YAML.
- **`src/embeddings.py`** (nouveau) — Factory `create_embeddings()` supportant `huggingface` et `ollama`.
- **`src/llm_factory.py`** (nouveau) — Factory `create_llm()` supportant `ollama` et `gemini` + fonction `llm_invoke()` unifiée.
- **`src/evaluation_judge.py`** (nouveau) — Classes `OllamaJudge`, `GeminiJudge` (wrapper DeepEvalBaseLLM) + factory `create_judge()`.
- **`src/ingest.py`** (modifié) — Remplace `OllamaEmbeddings` par `create_embeddings()`. Vectorstore path inclut le nom du modèle d'embedding.
- **`src/retriever.py`** (modifié) — Remplace `OllamaEmbeddings` par `create_embeddings()`. `RetrievalConfig` importé depuis `config.py`.
- **`src/llm_chain.py`** (modifié) — Remplace `OllamaLLM` par `create_llm()` + `llm_invoke()`. `ChainConfig` remplacé par `LLMConfig`.
- **`src/deepeval_evaluation.py`** (modifié) — Remplace `OllamaEvaluationJudge` par `create_judge()`. Utilise la config centralisée. Supporte les overrides CLI.
- **`src/main.py`** (modifié) — Charge la config depuis `config.yaml`. Paramètres dynamiques. Affichage des modèles en cours.
- **`requirements.txt`** (mis à jour) — Ajout de `langchain-huggingface`, `sentence-transformers`, `langchain-google-genai`, `pyyaml`.

### Pour comparer plusieurs configurations
1. Créer un fichier YAML par expérience (ex: `config_experiment_1.yaml`)
2. Lancer avec `python src/deepeval_evaluation.py --config config_experiment_1.yaml`
3. Résultats dans `results/deepeval/` avec timestamp

## 2026-07-01 — Session 3 (Architecture Recherche)
- Objectif : Séparer app utilisateur / environnement de recherche.

### Modifications effectuées
- **`app/main.py`** (nouveau) — Application Streamlit dédiée aux utilisateurs (importe depuis src/).
- **`src/main.py`** (supprimé) — Déplacé vers `app/main.py`.
- **`experiments/registry.py`** (nouveau) — Journalise chaque run (timestamp, params, résultats). Export CSV/Excel/JSON.
- **`experiments/runner.py`** (nouveau) — Fonctions `run_retrieval()`, `run_generation()`, `run_full_pipeline()`, `run_deepeval_metrics()`.
- **`experiments/presets/*.yaml`** (nouveaux) — 8 configurations d'expériences prédéfinies.
- **`notebooks/lib/plotter.py`** (nouveau) — Graphiques standardisés (histogramme, boxplot, heatmap, courbe, Pareto).
- **`notebooks/lib/reporter.py`** (nouveau) — Export multi-format + `load_benchmark()`.
- **`benchmark/questions.csv`** (nouveau) — 30 questions avec colonnes enrichies (Type_Reponse, Categorie_Metier, etc.).
- **`notebooks/`** (8 notebooks) — Suite complète 01→08 :
  - `00_protocole_experimental.md` : Protocole avec RQ, hypothèses, variables
  - `01_exploration_benchmark.ipynb` : Analyse descriptive
  - `02_comparaison_embeddings.ipynb` : RQ1 — 4 modèles
  - `03_analyse_chunking.ipynb` : RQ2 — grid search chunk_size×overlap
  - `04_optimisation_topk.ipynb` : RQ3 — Precision-Recall@k
  - `05_comparaison_llm_prompt.ipynb` : RQ4 — 3 prompts × 2 LLMs
  - `06_evaluation_deepeval.ipynb` : 5 métriques DeepEval
  - `07_analyse_erreurs.ipynb` : RQ5-6 — Pareto des causes
  - `08_synthese_finale.ipynb` : Tableaux et figures pour le mémoire

### Structure finale
```
projet/
├── app/main.py              # Streamlit (utilisateurs)
├── src/                     # Logique RAG (partagée)
├── benchmark/               # Gold standard
├── experiments/             # Pilotage des expériences
├── notebooks/               # Recherche (8 notebooks)
├── outputs/                 # Résultats (csv, excel, json, figures)
├── config.yaml              # Configuration active
└── requirements.txt
```

### Notebooks : aucun ne duplique src/
Chaque notebook importe depuis `src/` et `experiments/`. Pas de code RAG dans les notebooks.

## 2026-07-01 — Session 4 (Architecture LLM modulaire)
- Objectif : Architecture LLM extensible basée sur le pattern Factory + Strategy.

### Modifications effectuées
- **`src/llm/`** (nouveau package) — 4 fichiers :
  - `base.py` : Classe abstraite `BaseLLMProvider` (interface `invoke()`, `get_model_name()`)
  - `ollama.py` : `OllamaProvider` (via OllamaLLM LangChain)
  - `openrouter.py` : `OpenRouterProvider` (via API REST directe)
  - `factory.py` : `create_llm()` retourne n'importe quel `BaseLLMProvider` sans que l'appelant connaisse le type concret
- **`src/embeddings/`** (nouveau package) — 2 fichiers :
  - `factory.py` : `create_embeddings()` supporte `huggingface` et `ollama`, accepte aussi un `EmbeddingConfig` directement
  - `__init__.py` : réexporte pour compatibilité
- **`.env`** (nouveau) — `OPENROUTER_API_KEY` + `HF_TOKEN` optionnel
- **`src/config.py`** — Ajout de `load_env()` (charge `.env` automatiquement)
- **`src/evaluation_judge.py`** — Ajout de `OpenRouterJudge` (DeepEvalBaseLLM via OpenRouter)
- **`src/llm_factory.py`** (supprimé) — Remplacé par `src/llm/`
- **`src/embeddings.py`** (supprimé) — Remplacé par `src/embeddings/`
- **`src/llm_chain.py`, `experiments/runner.py`** — Imports mis à jour vers les nouveaux packages

### Architecture finale
```
src/
├── llm/
│   ├── base.py          # BaseLLMProvider (abstract)
│   ├── ollama.py        # OllamaProvider
│   ├── openrouter.py    # OpenRouterProvider (Gemini, DeepSeek, Qwen, etc.)
│   └── factory.py       # create_llm() + llm_invoke()
├── embeddings/
│   ├── factory.py       # create_embeddings()
│   └── __init__.py
├── config.py            # + load_env()
├── evaluation_judge.py  # + OpenRouterJudge
├── llm_chain.py
├── retriever.py
├── ingest.py
├── deepeval_evaluation.py
└── user_upload.py
```

### Principe ouvert/fermé
- Ajouter un nouveau provider LLM = créer une classe dans `src/llm/` + ajouter un `if` dans `factory.py`
- **Aucun fichier du pipeline RAG n'est modifié**
- Le changement de modèle se fait uniquement dans `config.yaml`

### Utilisation
```bash
export OPENROUTER_API_KEY=...
streamlit run app/main.py
```

## 2026-07-02 — Session 5 (Interface interactive + OpenRouter par défaut)
- Objectif : Restaurer les sélecteurs et basculer OpenRouter par défaut.

### Problèmes constatés
- **Temps de réponse** : 168s avec `qwen3.5:9b` (6.6 Go local) → trop lent.
- **Génération échouée** : le fallback *"Je n'ai pas réussi à générer une réponse claire"* s'affichait car le modèle 9B timeout ou renvoyait un format non nettoyé.
- **Sélecteur de modèle disparu** : l'ancien `st.selectbox("Modèle Ollama", [...])` avait été perdu lors du refactoring Session 4.

### Modifications effectuées
- **`config.yaml`** — LLM par défaut : `openrouter:google/gemini-2.5-flash` (cloud, rapide). Évaluation aussi migrée vers OpenRouter.
- **`app/main.py`** — Nouveaux sélecteurs dynamiques dans la sidebar :
  - 🤖 LLM : Provider (OpenRouter/Ollama) + Modèle (liste dynamique selon provider)
  - 🔤 Embedding : Provider (HuggingFace/Ollama) + Modèle (liste dynamique)
  - Les sélecteurs utilisent `st.session_state` pour persister et sont passés au handler de requête via `LLMConfig`/`EmbeddingConfig`.
- **`src/retriever.py`** — `load_vectorstore()`/`retrieve_documents()` acceptent désormais un `EmbeddingConfig` optionnel (permet de charger l'index FAISS avec le modèle d'embedding sélectionné).
- **`src/llm_chain.py`** — `clean_generated_answer()` utilise des regex avec support des variantes d'apostrophes françaises (', ', ’) et d'accents (é/è/ê/ë).

### Résultat
- Pipeline : Ollama qwen3-embedding → OpenRouter Gemini 2.5 Flash
- Génération fonctionnelle et rapide (~5-10s au lieu de 168s)
- Réponse en français : *"Oui, tu peux demander une prolongation..."*

### Fix session 5 (suite) — Ingestion HuggingFace + migration modèle-spécifique
- **`config.yaml`** — Embedding par défaut : `huggingface:BAAI/bge-m3`
- **`src/ingest.py`** — Sauvegarde toujours dans `vectorstore/<model_slug>/` (pas de chemin générique)
- **`src/config.py`** — `vectorstore_path` supprimé du fallback legacy (toujours modèle-spécifique)
- **`app/main.py`** — `has_vectorstore()` vérifie `vectorstore/<slug>/`
- **`requirements.txt`** — Ajout de `pypdf`
- Index FAISS disponibles :
  - `vectorstore/BAAI_bge-m3/` (72 vecteurs, HuggingFace)
  - `vectorstore/qwen3-embedding/` (46 vecteurs, Ollama)

## 2026-07-02 — Session 6 (Structure d'évaluation DeepEval modulaire)
- Objectif : Créer un dossier `evaluation/` avec notebooks scientifiques modulaires.

### Création
- **`evaluation/`** (nouveau dossier) — Architecture pour les notebooks d'expérimentation :
  ```
  evaluation/
  ├── notebooks/              # Notebooks Jupyter (1 par question de recherche)
  ├── results/                # Résultats exportés (CSV, Excel, JSON)
  ├── figures/                # Graphiques générés
  ├── __init__.py
  ├── config.py               # Charge un preset YAML + surcharge 1 paramètre
  ├── runner.py               # Boucle pipeline RAG → mesures DeepEval
  ├── metrics.py              # Factory de métriques DeepEval
  └── utils.py                # Dataset loader, export, summary_table
  ```
- **`evaluation/notebooks/01_validation_pipeline.ipynb`** — 15 cellules (markdown + code) validant le pipeline complet.

### Principe
- Les notebooks importent depuis `evaluation/*.py` et `src/*.py`
- Aucune duplication du code RAG
- Chaque notebook est autonome, pédagogique (cellules markdown explicatives)
- Résultats exportés dans `evaluation/results/`

### Correction
- **`src/evaluation_judge.py`** — Fichier manquant recréé (OllamaJudge, OpenRouterJudge, create_judge)
