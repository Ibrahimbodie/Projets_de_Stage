# Protocole Expérimental — Assistant RAG pour Règlements Académiques

## 1. Questions de Recherche (RQ)

| ID | Question | Variable indépendante | Variable dépendante |
|---|---|---|---|
| RQ1 | Quel embedding maximise le rappel contextuel ? | embedding model | Contextual Recall |
| RQ2 | Quel impact a la taille des chunks sur la fidélité ? | chunk_size | Faithfulness |
| RQ3 | Quel top_k offre le meilleur compromis Precision/Recall ? | top_k | Contextual Precision, Recall |
| RQ4 | Quel prompt obtient les réponses les plus pertinentes ? | prompt template | Answer Relevancy |
| RQ5 | Quelles catégories de questions sont les plus difficiles ? | question category | Faithfulness par catégorie |
| RQ6 | Quelles sont les causes principales d'échec du RAG ? | — | analyse qualitative |

## 2. Hypothèses

- H1 : Les embeddings multilingues (bge-m3, e5-large) surpassent les modèles monolingues sur le français académique.
- H2 : chunks de 1000 caractères offrent le meilleur équilibre entre précision et complétude.
- H3 : top_k=5 est suffisant ; au-delà, le bruit dégrade la fidélité.
- H4 : Un prompt structuré améliore la fidélité mais réduit la pertinence.
- H5 : Les questions conditionnelles (Niveau 3) sont les plus sujettes à erreur.

## 3. Variables

**Indépendantes :**
- embedding ∈ {BAAI/bge-m3, intfloat/multilingual-e5-large, jinaai/jina-embeddings-v3, qwen3-embedding}
- chunk_size ∈ {500, 1000, 1500, 2000}
- overlap ∈ {0, 100, 200}
- top_k ∈ {3, 5, 8, 10}
- prompt ∈ {direct, pédagogique, structuré}
- llm ∈ {qwen2.5:3b, mistral:7b}

**Dépendantes :**
- Faithfulness [0,1]
- Answer Relevancy [0,1]
- Contextual Precision [0,1]
- Contextual Recall [0,1]
- Answer Correctness [0,1]
- Response time (s)

**Contrôlées :**
- Temperature = 0.2
- Benchmark = 30 questions
- Embedding device = cpu

## 4. Benchmark

30 questions réparties en 4 niveaux de complexité et 6 catégories métier.

## 5. Métriques (DeepEval)

- Faithfulness : absence d'hallucinations
- Answer Relevancy : pertinence de la réponse
- Contextual Precision : précision du contexte retrieval
- Contextual Recall : exhaustivité du contexte retrieval
- Answer Correctness : exactitude factuelle

Seuil de succès : 0.75

## 6. Plan d'analyse

1. Analyse descriptive du benchmark (01)
2. Comparaison des embeddings sur Contextual Recall (02)
3. Optimisation du chunking (03)
4. Optimisation de top_k (04)
5. Comparaison LLM + Prompts (05)
6. Évaluation complète DeepEval (06)
7. Classification et analyse des erreurs (07)
8. Synthèse et tableaux pour le mémoire (08)
