# Predictive Maintenance Readiness Assistant

Predictive Maintenance App demo: traditional ML failure-risk scoring plus a GenAI maintenance advisor (RAG + simple agent tools), wrapped in a Streamlit fleet readiness dashboard.

## What the demo shows

1. **Fleet readiness dashboard** — vehicles banded GREEN / AMBER / RED by predicted failure risk  
2. **Vehicle detail** — component-level risk, recommended action, contributing signals  
3. **Explain risk (RAG)** — FAISS retrieval over synthetic maintenance docs  
4. **Recommend next action (agent)** — tools `get_vehicle_health`, `get_component_history`, `search_maintenance_manual`  
5. **Mock GenAI by default** — set `OPENAI_API_KEY` for live OpenAI; otherwise a grounded local fallback always works  

ML path: **Logistic Regression → Random Forest → XGBoost**, best model saved by ROC-AUC, scored with a **recall-biased operating threshold (0.35)**.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src

# Generate data, train models, build RAG index
bash scripts/bootstrap.sh

# Streamlit on uncommon port 8507
streamlit run app/streamlit_app.py --server.port=8507 --server.address=0.0.0.0
```

Optional API:

```bash
uvicorn predictive_maintenance.api:app --host 0.0.0.0 --port 8700
```

Optional live GenAI:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional
```

## Docker

```bash
docker compose up --build streamlit
# open http://localhost:8507

# optional API profile
docker compose --profile api up --build api
```

## Tests & lint

```bash
export PYTHONPATH=src
ruff check src tests app
pytest -q
```

## Project layout

```
app/streamlit_app.py          # Fleet readiness UI
src/predictive_maintenance/
  data/generate.py            # Synthetic fleet + docs
  ml/train.py                 # LogReg / RF / XGBoost
  ml/inference.py             # Score + contributing signals
  rag/index.py                # Chunk → embed → FAISS
  agent/tools.py              # Agent tool functions
  agent/advisor.py            # OpenAI or mock advisor
  api.py                      # Optional FastAPI
scripts/bootstrap.sh
docs_corpus/                  # Synthetic maintenance manuals
models/                       # best_model.joblib, metrics, RAG index
tests/
```
