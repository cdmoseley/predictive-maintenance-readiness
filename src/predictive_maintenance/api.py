"""Optional FastAPI surface for PMI overrun inference + advisor."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictive_maintenance.agent.advisor import explain_risk, recommend_next_action
from predictive_maintenance.config import FEATURE_COLUMNS
from predictive_maintenance.ml.inference import FailureRiskModel, load_scored_fleet

app = FastAPI(
    title="FRCE MV-22 PMI Delay API",
    description="Notional inference and advisor endpoints for the FRCE PMI pilot demo",
    version="0.2.0",
)


class FeaturePayload(BaseModel):
    prior_oa_findings: float = 0
    squadron_corr_wiring_score: float = 0
    awp_days_open: float = 0
    zero_balance_hits: float = 0
    eng_queue_age_days: float = 0
    prior_late_pmis: float = 0
    planned_turnaround_days: float = 110
    pct_work_complete: float = Field(0.5, ge=0, le=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/fleet")
def fleet_summary() -> dict[str, Any]:
    df = load_scored_fleet()
    id_col = "buno" if "buno" in df.columns else "vehicle_id"
    return {
        "n_rows": int(len(df)),
        "n_aircraft": int(df[id_col].nunique()),
        "band_counts": df["risk_band"].value_counts().to_dict(),
        "delay_drivers": df["primary_delay_driver"].value_counts().to_dict()
        if "primary_delay_driver" in df.columns
        else {},
    }


@app.post("/predict")
def predict(payload: FeaturePayload) -> dict[str, Any]:
    model = FailureRiskModel()
    return model.predict(payload.model_dump())


@app.get("/aircraft/{buno}/recommend")
@app.get("/vehicles/{buno}/recommend")
def recommend(buno: str, component: str | None = None) -> dict[str, Any]:
    result = recommend_next_action(buno, component)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/aircraft/{buno}/explain")
@app.get("/vehicles/{buno}/explain")
def explain(buno: str, component: str | None = None) -> dict[str, Any]:
    result = explain_risk(buno, component)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"feature_columns": FEATURE_COLUMNS}
