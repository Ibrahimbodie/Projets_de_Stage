#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "env/bin/python" ]]; then
  echo "Erreur: environnement virtuel introuvable dans ./env"
  echo "Crée-le puis installe les dépendances avant de relancer."
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Erreur: ollama n'est pas installé."
  echo "Installe Ollama: https://ollama.com/download"
  exit 1
fi

if ! pgrep -x ollama >/dev/null 2>&1; then
  echo "Démarrage de ollama serve..."
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  sleep 2
fi

ensure_model() {
  local model="$1"
  if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
    echo "Téléchargement du modèle: $model"
    ollama pull "$model"
  else
    echo "Modèle déjà présent: $model"
  fi
}

ensure_model "qwen3-embedding"
ensure_model "qwen3.5:27b"

echo "Création / mise à jour de l'index vectoriel..."
"$ROOT_DIR/env/bin/python" "$ROOT_DIR/src/ingest.py"

echo "Lancement de l'application terminal..."
exec "$ROOT_DIR/env/bin/python" "$ROOT_DIR/src/cli.py"
