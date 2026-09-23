#!/usr/bin/env bash
# Bootstrap data, train models, build RAG index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "$ROOT"

python -m predictive_maintenance.data.generate --n-vehicles 120
python -m predictive_maintenance.ml.train
python -m predictive_maintenance.rag.index
echo "Bootstrap complete."
