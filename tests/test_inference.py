"""Tests for inference and critical demo paths."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from predictive_maintenance.agent.advisor import explain_risk, recommend_next_action
from predictive_maintenance.agent.tools import (
    get_component_history,
    get_vehicle_health,
    search_maintenance_manual,
)
from predictive_maintenance.config import FEATURE_COLUMNS, OPERATING_THRESHOLD
from predictive_maintenance.data.generate import generate_fleet_data, write_synthetic_docs
from predictive_maintenance.ml.inference import FailureRiskModel, contributing_signals, risk_band
from predictive_maintenance.ml.train import train
from predictive_maintenance.rag.index import MaintenanceRAG, build_index, chunk_text


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory):
    """Train a small model once for the test module."""
    root = tmp_path_factory.mktemp("pm")
    models = root / "models"
    raw = root / "raw"
    processed = root / "processed"
    docs = root / "docs"
    raw.mkdir()
    processed.mkdir()
    models.mkdir()
    docs.mkdir()

    fleet = generate_fleet_data(n_vehicles=60, seed=7)
    fleet.to_csv(raw / "fleet_maintenance.csv", index=False)
    write_synthetic_docs(docs)

    # Patch config paths used by train/inference via monkeypatch-like locals
    import predictive_maintenance.agent.tools as tools_mod
    import predictive_maintenance.config as cfg
    import predictive_maintenance.ml.inference as infer_mod
    import predictive_maintenance.ml.train as train_mod
    import predictive_maintenance.rag.index as rag_mod

    originals = {
        "MODELS_DIR": cfg.MODELS_DIR,
        "RAW_DIR": cfg.RAW_DIR,
        "PROCESSED_DIR": cfg.PROCESSED_DIR,
        "DOCS_DIR": cfg.DOCS_DIR,
        "RAG_INDEX_DIR": cfg.RAG_INDEX_DIR,
    }
    cfg.MODELS_DIR = models
    cfg.RAW_DIR = raw
    cfg.PROCESSED_DIR = processed
    cfg.DOCS_DIR = docs
    cfg.RAG_INDEX_DIR = models / "rag_index"
    train_mod.MODELS_DIR = models
    train_mod.RAW_DIR = raw
    train_mod.PROCESSED_DIR = processed
    infer_mod.MODELS_DIR = models
    infer_mod.PROCESSED_DIR = processed
    rag_mod.DOCS_DIR = docs
    rag_mod.RAG_INDEX_DIR = models / "rag_index"
    tools_mod.RAW_DIR = raw

    result = train(fleet=fleet, seed=7, models_dir=models)
    build_index(docs_dir=docs, index_dir=models / "rag_index")

    # Clear tool caches so they see patched paths
    tools_mod._history.cache_clear()
    tools_mod._model.cache_clear()
    tools_mod._rag.cache_clear()

    yield {
        "fleet": fleet,
        "models": models,
        "raw": raw,
        "processed": processed,
        "docs": docs,
        "train_result": result,
        "vehicle_id": fleet.iloc[0]["vehicle_id"],
        "component": fleet.iloc[0]["component"],
    }

    for key, val in originals.items():
        setattr(cfg, key, val)
    train_mod.MODELS_DIR = originals["MODELS_DIR"]
    train_mod.RAW_DIR = originals["RAW_DIR"]
    train_mod.PROCESSED_DIR = originals["PROCESSED_DIR"]
    infer_mod.MODELS_DIR = originals["MODELS_DIR"]
    infer_mod.PROCESSED_DIR = originals["PROCESSED_DIR"]
    rag_mod.DOCS_DIR = originals["DOCS_DIR"]
    rag_mod.RAG_INDEX_DIR = originals["RAG_INDEX_DIR"]
    tools_mod.RAW_DIR = originals["RAW_DIR"]
    tools_mod._history.cache_clear()
    tools_mod._model.cache_clear()
    tools_mod._rag.cache_clear()


def test_generate_fleet_has_required_columns():
    df = generate_fleet_data(n_vehicles=10, seed=1)
    for col in FEATURE_COLUMNS + ["vehicle_id", "component", "failure_next_30_days"]:
        assert col in df.columns
    assert df["failure_next_30_days"].isin([0, 1]).all()


def test_train_selects_best_model(trained_env):
    metrics_path = trained_env["models"] / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["best_model"] in {"logistic_regression", "random_forest", "xgboost"}
    assert (trained_env["models"] / "best_model.joblib").exists()
    assert (trained_env["processed"] / "fleet_scored.csv").exists()


def test_inference_predict_shape(trained_env):
    model = FailureRiskModel(trained_env["models"] / "best_model.joblib")
    row = trained_env["fleet"].iloc[0]
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    result = model.predict(features)
    assert 0.0 <= result["failure_risk"] <= 1.0
    assert result["risk_band"] in {"GREEN", "AMBER", "RED"}
    assert result["threshold"] == OPERATING_THRESHOLD
    assert len(result["contributing_signals"]) == len(FEATURE_COLUMNS)


def test_inference_missing_feature_raises(trained_env):
    model = FailureRiskModel(trained_env["models"] / "best_model.joblib")
    with pytest.raises(ValueError, match="Missing features"):
        model.predict({"component_age_days": 100})


def test_risk_band_edges():
    assert risk_band(0.10) == "GREEN"
    assert risk_band(0.20) == "AMBER"
    assert risk_band(0.49) == "AMBER"
    assert risk_band(0.50) == "RED"


def test_contributing_signals_sorted():
    row = pd.Series(
        {
            "component_age_days": 2000,
            "operating_hours": 100,
            "failures_last_90d": 5,
            "fleet_same_component_failures_90d": 1,
            "system_failures_last_30d": 0,
            "maintenance_actions_last_90d": 0,
        }
    )
    signals = contributing_signals(row)
    scores = [s["concern_score"] for s in signals]
    assert scores == sorted(scores, reverse=True)


def test_rag_search_returns_hits(trained_env):
    rag = MaintenanceRAG(trained_env["models"] / "rag_index")
    hits = rag.search("engine oil analysis failure risk", k=3)
    assert len(hits) >= 1
    assert "text" in hits[0]


def test_chunk_text_respects_size():
    text = "Sentence one. " * 80
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_agent_tools(trained_env):
    # Write history for tools
    from predictive_maintenance.data.generate import generate_component_history

    hist = generate_component_history(trained_env["fleet"], seed=7)
    hist.to_csv(trained_env["raw"] / "component_history.csv", index=False)

    vid = trained_env["vehicle_id"]
    health = get_vehicle_health(vid)
    assert health["vehicle_id"] == vid
    assert health["components"]

    history = get_component_history(vid, trained_env["component"])
    assert "events" in history

    manual = search_maintenance_manual("brake pad inspection hours")
    assert manual["hits"]


def test_advisor_mock_path(trained_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from predictive_maintenance.data.generate import generate_component_history

    hist = generate_component_history(trained_env["fleet"], seed=7)
    hist.to_csv(trained_env["raw"] / "component_history.csv", index=False)

    vid = trained_env["vehicle_id"]
    rec = recommend_next_action(vid)
    assert rec["mode"] == "mock"
    assert "recommendation" in rec

    exp = explain_risk(vid)
    assert exp["mode"] == "mock"
    assert "explanation" in exp
