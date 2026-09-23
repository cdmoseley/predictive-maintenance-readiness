"""Smoke test for FastAPI endpoints."""

from fastapi.testclient import TestClient

from predictive_maintenance.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_features():
    resp = client.get("/features")
    assert resp.status_code == 200
    assert "component_age_days" in resp.json()["feature_columns"]
