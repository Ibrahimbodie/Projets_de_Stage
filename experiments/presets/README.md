# Presets d'Expériences

Chaque fichier YAML est une configuration d'expérience complète.

## Utilisation
```python
from config import load_config
cfg = load_config("experiments/presets/01_embedding_bge-m3.yaml")
```

## Liste des presets
| Fichier | Variable testée | Valeur |
|---|---|---|
| `00_baseline.yaml` | Référence | bge-m3, chunk=1000, top_k=6 |
| `01_embedding_bge-m3.yaml` | Embedding | BAAI/bge-m3 |
| `02_embedding_e5.yaml` | Embedding | intfloat/multilingual-e5-large |
| `03_embedding_jina.yaml` | Embedding | jinaai/jina-embeddings-v3 |
| `04_chunk_500.yaml` | Chunk size | 500 |
| `05_chunk_1500.yaml` | Chunk size | 1500 |
| `06_topk_3.yaml` | Top_k | 3 |
| `07_topk_10.yaml` | Top_k | 10 |

Pour ajouter une expérience : créer un nouveau fichier YAML.
