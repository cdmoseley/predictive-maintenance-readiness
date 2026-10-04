"""Tests for inference and critical FRCE PMI demo paths."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from predictive_maintenance.agent.advisor import explain_risk, recommend_next_action
from predictive_maintenance.agent.tools import (
    get_aircraft_health,
    get_component_history,
    get_parts_awp_status,
    get_vehicle_health,
    search_tech_data,
)
from predictive_maintenance.config import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    OPERATING_THRESHOLD,
    TARGET_COLUMN,
    feature_label,
)
from predictive_maintenance.data.generate import (
    generate_component_history,
    generate_fleet_data,
    generate_parts_awp,
    write_synthetic_docs,
)
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
    generate_component_history(fleet, seed=7).to_csv(raw / "component_history.csv", index=False)
    generate_parts_awp(fleet, seed=7).to_csv(raw / "parts_awp.csv", index=False)
    write_synthetic_docs(docs)

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

    tools_mod._history.cache_clear()
    tools_mod._parts.cache_clear()
    tools_mod._model.cache_clear()
    tools_mod._rag.cache_clear()

    yield {
        "fleet": fleet,
        "models": models,
        "raw": raw,
        "processed": processed,
        "docs": docs,
        "train_result": result,
        "buno": str(fleet.iloc[0]["buno"]),
        "component": fleet.iloc[0]["focus_system"],
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
    tools_mod._parts.cache_clear()
    tools_mod._model.cache_clear()
    tools_mod._rag.cache_clear()


def test_feature_labels_cover_ml_columns():
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_LABELS
        label = feature_label(col)
        assert " " in label or label.isupper()
        assert "_" not in label


def test_generate_fleet_has_required_columns():
    df = generate_fleet_data(n_vehicles=10, seed=1)
    for col in FEATURE_COLUMNS + ["buno", TARGET_COLUMN, "primary_delay_driver"]:
        assert col in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


def test_train_selects_best_model(trained_env):
    metrics_path = trained_env["models"] / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["best_model"] in {"logistic_regression", "random_forest", "xgboost"}
    assert payload["target"] == TARGET_COLUMN
    assert (trained_env["models"] / "best_model.joblib").exists()
    assert (trained_env["processed"] / "fleet_scored.csv").exists()


def test_inference_predict_shape(trained_env):
    model = FailureRiskModel(trained_env["models"] / "best_model.joblib")
    row = trained_env["fleet"].iloc[0]
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    result = model.predict(features)
    assert 0.0 <= result["overrun_risk"] <= 1.0
    assert result["risk_band"] in {"GREEN", "AMBER", "RED"}
    assert result["threshold"] == OPERATING_THRESHOLD
    assert len(result["contributing_signals"]) == len(FEATURE_COLUMNS)


def test_inference_missing_feature_raises(trained_env):
    model = FailureRiskModel(trained_env["models"] / "best_model.joblib")
    with pytest.raises(ValueError, match="Missing features"):
        model.predict({"prior_oa_findings": 1})


def test_risk_band_edges():
    assert risk_band(0.10) == "GREEN"
    assert risk_band(0.20) == "AMBER"
    assert risk_band(0.49) == "AMBER"
    assert risk_band(0.50) == "RED"


def test_contributing_signals_sorted():
    row = pd.Series(
        {
            "prior_oa_findings": 5,
            "squadron_corr_wiring_score": 1.0,
            "awp_days_open": 20,
            "zero_balance_hits": 0,
            "eng_queue_age_days": 0,
            "prior_late_pmis": 0,
            "planned_turnaround_days": 100,
            "pct_work_complete": 0.1,
        }
    )
    signals = contributing_signals(row)
    scores = [s["concern_score"] for s in signals]
    assert scores == sorted(scores, reverse=True)


def test_rag_search_returns_hits(trained_env):
    rag = MaintenanceRAG(trained_env["models"] / "rag_index")
    hits = rag.search("engineering disposition wiring chafing", k=3)
    assert len(hits) >= 1
    assert "text" in hits[0]


def test_chunk_text_respects_size():
    text = "Sentence one. " * 80
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_agent_tools(trained_env):
    buno = trained_env["buno"]
    health = get_aircraft_health(buno)
    assert health["buno"] == buno
    assert "overrun_risk" in health

    parts = get_parts_awp_status(buno)
    assert "parts" in parts

    history = get_component_history(buno, trained_env["component"])
    assert "events" in history

    tech = search_tech_data("AWP zero balance hydraulic pump")
    assert tech["hits"]

    # back-compat alias
    vh = get_vehicle_health(buno)
    assert vh["vehicle_id"] == buno


def test_advisor_mock_path(trained_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    buno = trained_env["buno"]
    rec = recommend_next_action(buno)
    assert rec["mode"] == "mock"
    assert "recommendation" in rec
    assert "parts_awp" in rec["tool_results"]

    exp = explain_risk(buno)
    assert exp["mode"] == "mock"
    assert "explanation" in exp


def test_zip_without_strict():
    """Guard: rag search path must not use zip(..., strict=True) for Py3.9."""
    import inspect

    import predictive_maintenance.rag.index as rag_mod

    src = inspect.getsource(rag_mod.MaintenanceRAG.search)
    assert "strict=True" not in src


def test_tools_module_imports_without_loading_ml(monkeypatch):
    """Cloud failure mode: tools.py must bind search_tech_data without importing ML."""
    import importlib
    import sys

    # Drop cached modules so we can observe fresh imports
    for name in list(sys.modules):
        if name.startswith("predictive_maintenance.agent.tools") or name.startswith(
            "predictive_maintenance.ml"
        ):
            sys.modules.pop(name, None)

    import predictive_maintenance.agent.tools as tools_mod

    tools_mod = importlib.reload(tools_mod)
    assert hasattr(tools_mod, "search_tech_data")
    assert callable(tools_mod.search_tech_data)
    # Top-level tools import must not pull FailureRiskModel into sys.modules
    assert "predictive_maintenance.ml.inference" not in sys.modules
    assert "predictive_maintenance.ml.train" not in sys.modules


def test_search_tech_data_works_with_numpy_fallback(tmp_path, monkeypatch):
    """Engineer search must work even when FAISS is unavailable."""
    import predictive_maintenance.rag.index as rag_mod

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# AWP guidance\nZero-balance parts cause PMI delay. Engineering dispositions help.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "idx"

    monkeypatch.setattr(rag_mod, "_FAISS", None)
    monkeypatch.setattr(rag_mod, "_FAISS_TRIED", True)

    info = rag_mod.build_index(docs_dir=docs, index_dir=index_dir)
    assert info["faiss"] is False
    assert (index_dir / "vectors.npy").exists()

    rag = rag_mod.MaintenanceRAG(index_dir)
    hits = rag.search("AWP zero balance parts", k=2)
    assert hits
    assert "text" in hits[0]
