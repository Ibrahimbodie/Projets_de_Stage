# 🎓 Assistant Virtuel Académique - Master SIM

## 🇫🇷 Version Française

### 📌 Description

Ce projet consiste en la conception d’un assistant virtuel intelligent basé sur les technologies RAG (Retrieval-Augmented Generation) pour aider les étudiants du Master SIM à comprendre les règlements académiques de manière simple et interactive.

Le système permet de :
- interroger des documents académiques en langage naturel,
- rechercher les informations pertinentes,
- générer des réponses pédagogiques en français simple,
- afficher les sources utilisées.

L’assistant utilise des modèles LLM locaux via Ollama ainsi qu’une base vectorielle FAISS.

---

## 🚀 Fonctionnalités

- 📄 Lecture de documents PDF, TXT et DOCX
- 🔍 Recherche sémantique avec FAISS
- 🤖 Génération de réponses avec Ollama
- 💬 Interface conversationnelle avec Streamlit
- 📚 Affichage des sources utilisées
- 📂 Upload de documents utilisateur
- 🧠 Assistant pédagogique en français simple

---

## 🏗️ Architecture du Projet

```text
src/
│
├── main.py            # Interface Streamlit
├── ingest.py          # Ingestion des documents
├── retriever.py       # Recherche sémantique
├── llm_chain.py       # Génération des réponses
├── user_upload.py     # Gestion des documents utilisateur
```

---

## 🛠️ Technologies Utilisées

- Python
- Streamlit
- LangChain
- Ollama
- FAISS
- Qwen2.5
- Sentence Transformers

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <repo_url>
cd project
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama

https://ollama.com

### 5. Télécharger le modèle

```bash
ollama pull qwen2.5:3b
```

### 6. Lancer Ollama

```bash
ollama serve
```

### 7. Créer la base vectorielle

```bash
python src/ingest.py
```

### 8. Lancer l’application

```bash
streamlit run src/main.py
```

---

## 📸 Interface

![alt text](image.png)

---

## 🎯 Objectif du Projet

Ce projet a été réalisé dans le cadre d’un stage de Master en Intelligence Artificielle afin de développer un assistant académique intelligent capable d’améliorer l’accès aux règlements universitaires.

---

# 🇬🇧 English Version

## 📌 Description

This project consists of building an intelligent virtual assistant based on Retrieval-Augmented Generation (RAG) technologies to help Master SIM students better understand academic regulations through natural language interaction.

The system is able to:
- answer questions from academic documents,
- retrieve relevant information,
- generate pedagogical answers in simple French,
- display the sources used.

The assistant uses local LLMs through Ollama and a FAISS vector database.

---

## 🚀 Features

- 📄 PDF, TXT and DOCX document support
- 🔍 Semantic search using FAISS
- 🤖 Answer generation with Ollama
- 💬 Conversational interface with Streamlit
- 📚 Source citation display
- 📂 User document upload
- 🧠 Educational assistant in simple French

---

## 🏗️ Project Architecture

```text
src/
│
├── main.py            # Streamlit interface
├── ingest.py          # Document ingestion
├── retriever.py       # Semantic retrieval
├── llm_chain.py       # Answer generation
├── user_upload.py     # User document management
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Ollama
- FAISS
- Qwen2.5
- Sentence Transformers

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repo_url>
cd project
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

https://ollama.com

### 5. Download model

```bash
ollama pull qwen2.5:3b
```

### 6. Start Ollama

```bash
ollama serve
```

### 7. Build vector database

```bash
python src/ingest.py
```

### 8. Run application

```bash
streamlit run src/main.py
```

---

## 📸 Interface


![alt text](image.png)

---

## 🎯 Project Objective

This project was developed as part of a Master's internship in Artificial Intelligence to create an intelligent academic assistant capable of improving access to university regulations.