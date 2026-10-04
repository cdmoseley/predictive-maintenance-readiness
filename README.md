# FRCE MV-22 PMI Delay Assistant

Notional concept prototype for a Virtualitics Solutions Architect take-home: retarget the Predictive Maintenance Readiness Assistant to an **MV-22 Planned Maintenance Interval (PMI)** pilot at **Fleet Readiness Center East (FRCE)**.

> Fictional scenario. FRCE is a real facility used for realism; all figures, systems, and data are invented. This demo does **not** claim real FRCE data or specific Virtualitics product features beyond a flexible AI platform pattern (e.g. Iris-like connectors, ML, RAG, agents).

## What the demo shows

1. **Planner / Production Controller view** — BUNO line board banded GREEN / AMBER / RED by **PMI overrun risk**, with **over-and-above / AWP / engineering** delay drivers  
2. **Aircraft detail** — plan vs projected days, contributing signals, recommended mitigation  
3. **Recommend next action (agent)** — tools `get_aircraft_health`, `get_parts_awp_status`, `search_tech_data`  
4. **Engineer view** — RAG over notional tech data / disposition archive / PMI guidance  
5. **Mock GenAI by default** — set `OPENAI_API_KEY` for live OpenAI; grounded local fallback otherwise  

ML path: **Logistic Regression → Random Forest → XGBoost**, best model saved by ROC-AUC, scored with a **recall-biased operating threshold (0.35)** on target `pmi_overrun`.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src

# Generate notional data, train models, build RAG index
bash scripts/bootstrap.sh

# Streamlit on uncommon port 8513
streamlit run app/streamlit_app.py --server.port=8513 --server.address=0.0.0.0
```

Optional API:

```bash
uvicorn predictive_maintenance.api:app --host 0.0.0.0 --port 8713
```

Optional live GenAI:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional
```

## Docker

```bash
docker compose up --build streamlit
# open http://localhost:8513

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
app/streamlit_app.py          # Planner/PC + Engineer role views
src/predictive_maintenance/
  data/generate.py            # Synthetic BUNO PMI line + docs + parts/AWP
  ml/train.py                 # LogReg / RF / XGBoost → pmi_overrun
  ml/inference.py             # Score + contributing signals
  rag/index.py                # Chunk → embed → FAISS
  agent/tools.py              # Aircraft health, AWP, tech-data search
  agent/advisor.py            # OpenAI or mock advisor
  api.py                      # Optional FastAPI
scripts/bootstrap.sh
docs_corpus/                  # Notional tech data / dispositions / PMI guidance
models/                       # best_model.joblib, metrics, RAG index
tests/
```

## Interview / panel tip

Lead with the CO problem (cut overrun in half), show the line board for the production lead, switch to Engineer RAG for the eng lead, and close with 30/60/90 KPIs + read-only stovepipe fusion for the CIO. Written W1/W2 live in the take-home Context docs, not this repo.
