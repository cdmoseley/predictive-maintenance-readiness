"""Optional FastAPI surface for inference + advisor (Streamlit remains primary UI)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predictive_maintenance.agent.advisor import explain_risk, recommend_next_action
from predictive_maintenance.config import FEATURE_COLUMNS
from predictive_maintenance.ml.inference import FailureRiskModel, load_scored_fleet


def _worst_band(bands):
    if (bands == "RED").any():
        return "RED"
    if (bands == "AMBER").any():
        return "AMBER"
    return "GREEN"

app = FastAPI(
    title="Predictive Maintenance API",
    description="Inference and advisor endpoints for the readiness demo",
    version="0.1.0",
)


class FeaturePayload(BaseModel):
    component_age_days: float
    operating_hours: float
    failures_last_90d: float
    fleet_same_component_failures_90d: float
    system_failures_last_30d: float
    maintenance_actions_last_90d: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/fleet")
def fleet_summary() -> dict[str, Any]:
    df = load_scored_fleet()
    return {
        "n_rows": int(len(df)),
        "n_vehicles": int(df["vehicle_id"].nunique()),
        "band_counts": df.groupby("vehicle_id")["risk_band"]
        .apply(_worst_band)
        .value_counts()
        .to_dict(),
    }


@app.post("/predict")
def predict(payload: FeaturePayload) -> dict[str, Any]:
    model = FailureRiskModel()
    return model.predict(payload.model_dump())


@app.get("/vehicles/{vehicle_id}/recommend")
def recommend(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    result = recommend_next_action(vehicle_id, component)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/vehicles/{vehicle_id}/explain")
def explain(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    result = explain_risk(vehicle_id, component)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"feature_columns": FEATURE_COLUMNS}
