"""Smoke test for FastAPI endpoints."""

from fastapi.testclient import TestClient

from predictive_maintenance.api import app
from predictive_maintenance.config import FEATURE_COLUMNS

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_features():
    resp = client.get("/features")
    assert resp.status_code == 200
    cols = resp.json()["feature_columns"]
    assert "prior_oa_findings" in cols
    assert cols == FEATURE_COLUMNS
