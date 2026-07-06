# 🎓 Assistant Virtuel Académique - Master SIM

## 🇫🇷 Version Française

### 📌 Description

Assistant virtuel pédagogique basé sur **Retrieval-Augmented Generation (RAG)** permettant aux étudiants du Master SIM d'interroger les règlements académiques en langage naturel.

Le système permet de :
- interroger des documents académiques en langage naturel,
- rechercher les informations pertinentes via **FAISS**,
- générer des réponses pédagogiques via **Ollama** (modèles locaux) ou **OpenRouter** (modèles cloud),
- évaluer la qualité du pipeline avec **DeepEval**.

---

## 🚀 Fonctionnalités

- 📄 Lecture de PDF, TXT, DOCX
- 🔍 Recherche sémantique vectorielle (FAISS)
- 🤖 Génération LLM locale (Ollama) ou cloud (OpenRouter : Gemini, DeepSeek, Qwen...)
- 💬 Interface Streamlit avec sélecteurs dynamiques (modèle LLM, embedding)
- 📚 Affichage des sources
- 📂 Upload de documents utilisateur
- 🧪 Évaluation modulaire avec DeepEval

---

## 🏗️ Architecture

```
├── app/main.py              # Interface Streamlit (utilisateurs)
├── src/                     # Logique RAG partagée
│   ├── llm/                 # Providers LLM (Ollama, OpenRouter)
│   ├── embeddings/          # Providers d'embedding (HuggingFace, Ollama)
│   ├── config.py            # Configuration centralisée
│   ├── ingest.py            # Ingestion vectorielle FAISS
│   ├── retriever.py         # Recherche sémantique
│   ├── llm_chain.py         # Génération de réponses
│   └── evaluation_judge.py  # Juges DeepEval
├── experiments/             # Pilotage d'expériences
├── evaluation/              # Notebooks d'évaluation scientifique
├── benchmark/               # Questions de test (gold standard)
├── notebooks/               # Notebooks de recherche
├── config.yaml              # Configuration active
└── requirements.txt
```

---

## 🛠️ Technologies

- Python, Streamlit, LangChain
- **Ollama** (modèles locaux : Qwen2.5)
- **OpenRouter** (modèles cloud : Gemini, DeepSeek, etc.)
- FAISS, Sentence Transformers
- DeepEval, HuggingFace

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <repo_url>
cd projet
```

### 2. Environnement virtuel

```bash
python -m venv env
source env/bin/activate
```

### 3. Dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copier et éditer le fichier `config.yaml` (provider LLM, embedding, température, top_k...).

Créer un fichier `.env` à la racine :

```bash
echo "OPENROUTER_API_KEY=votre_clé_ici" > .env
```

### 5. Lancer Ollama (optionnel, pour modèles locaux)

```bash
ollama pull qwen2.5:3b
ollama serve
```

### 6. Ingérer les documents

```bash
python src/ingest.py
```

### 7. Lancer l'application

```bash
streamlit run app/main.py
```

---

## 🔀 Changer de modèle

Tout se configure dans `config.yaml` :

```yaml
llm:
  provider: openrouter       # ou "ollama"
  model: google/gemini-2.5-flash  # ou "qwen2.5:3b"

embedding:
  provider: huggingface      # ou "ollama"
  model: BAAI/bge-m3
```

Pas de modification de code — architecture **Factory + Strategy**.

---

## 🧪 Évaluation DeepEval

```bash
python src/deepeval_evaluation.py \
  --input benchmark/questions.csv \
  --metrics all
```

Métriques : Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Contextual Relevancy.

Résultats dans `results/deepeval/`.

---

## 📸 Interface

![Interface](image.png)

---

## 🎯 Objectif

Projet de stage Master en IA — développer un assistant académique intelligent pour améliorer l'accès aux règlements universitaires via le dialogue en langage naturel.

---

# 🇬🇧 English Version

## 📌 Description

A **Retrieval-Augmented Generation (RAG)** academic assistant for Master SIM students to query academic regulations in natural language.

The system supports **local LLMs (Ollama)** and **cloud models (OpenRouter)** with a modular architecture.

## 🏗️ Architecture

```
├── app/main.py              # Streamlit UI
├── src/                     # Core RAG logic
│   ├── llm/                 # LLM providers (Ollama, OpenRouter)
│   ├── embeddings/          # Embedding providers (HuggingFace, Ollama)
│   ├── config.py            # Centralized config
│   ├── ingest.py            # FAISS ingestion
│   ├── retriever.py         # Semantic search
│   ├── llm_chain.py         # Answer generation
│   └── evaluation_judge.py  # DeepEval judges
├── experiments/             # Experiment runner
├── evaluation/              # Scientific evaluation notebooks
├── benchmark/               # Test questions (gold standard)
├── notebooks/               # Research notebooks
├── config.yaml              # Active config
└── requirements.txt
```

## 🛠️ Tech Stack

Python, Streamlit, LangChain, Ollama, OpenRouter, FAISS, Sentence Transformers, DeepEval

## ⚙️ Quick Start

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=your_key" > .env
python src/ingest.py
streamlit run app/main.py
```

Configure models in `config.yaml`.

## 🎯 Objective

Master's internship project — build an intelligent academic assistant to improve access to university regulations through natural language conversation.
