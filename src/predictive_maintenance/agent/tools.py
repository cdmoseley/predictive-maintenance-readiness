"""Agent tools for the FRCE MV-22 PMI advisor.

Heavy deps (ML model / FAISS) are imported lazily so Streamlit Cloud can import
`search_tech_data` even when xgboost/joblib model load fails at boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from predictive_maintenance.config import FEATURE_COLUMNS, RAW_DIR


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    path = RAW_DIR / "component_history.csv"
    if not path.exists():
        from predictive_maintenance.data.generate import main as gen_main

        gen_main()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _parts() -> pd.DataFrame:
    path = RAW_DIR / "parts_awp.csv"
    if not path.exists():
        from predictive_maintenance.data.generate import main as gen_main

        gen_main()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _model():
    from predictive_maintenance.ml.inference import FailureRiskModel

    return FailureRiskModel()


@lru_cache(maxsize=1)
def _rag():
    from predictive_maintenance.rag.index import MaintenanceRAG

    return MaintenanceRAG()


def _load_scored_fleet() -> pd.DataFrame:
    from predictive_maintenance.ml.inference import load_scored_fleet

    return load_scored_fleet()


def _signals_from_row(row: pd.Series) -> list[dict[str, Any]]:
    """Heuristic signals from the scored CSV row — no model artifact required."""
    try:
        from predictive_maintenance.ml.inference import contributing_signals

        return contributing_signals(row)[:5]
    except Exception:
        return []


def get_aircraft_health(buno: str) -> dict[str, Any]:
    """Return scored PMI health / overrun risk for a BUNO.

    Prefers live model prediction when available; falls back to scored CSV +
    heuristic signals so Cloud boot is not blocked by model load failures.
    """
    fleet = _load_scored_fleet()
    id_col = "buno" if "buno" in fleet.columns else "vehicle_id"
    rows = fleet[fleet[id_col].astype(str) == str(buno)]
    if rows.empty:
        return {"error": f"Unknown BUNO: {buno}", "buno": buno}
    row = rows.iloc[0]

    overrun = float(row.get("overrun_risk", row.get("failure_risk", 0.0)))
    band = str(row.get("risk_band", "AMBER"))
    action = str(row.get("recommended_action", ""))
    signals = _signals_from_row(row)

    try:
        model = _model()
        feats = {c: float(row[c]) for c in model.feature_columns if c in row}
        if len(feats) == len(model.feature_columns):
            pred = model.predict(feats)
            overrun = float(pred["overrun_risk"])
            band = str(pred["risk_band"])
            action = str(pred["recommended_action"])
            signals = pred["contributing_signals"][:5]
    except Exception:
        # Model artifact / xgboost unavailable — scored CSV is enough for demo
        pass

    return {
        "buno": str(row.get("buno", buno)),
        "aircraft_type": row.get("aircraft_type", row.get("vehicle_type", "MV-22")),
        "squadron": row.get("squadron", row.get("unit", "")),
        "status": row.get("status", ""),
        "primary_delay_driver": row.get("primary_delay_driver", ""),
        "focus_system": row.get("focus_system", row.get("component", "")),
        "overrun_risk": overrun,
        "risk_band": band,
        "recommended_action": action,
        "awp_days_open": float(row.get("awp_days_open", 0)),
        "eng_queue_age_days": float(row.get("eng_queue_age_days", 0)),
        "prior_oa_findings": int(row.get("prior_oa_findings", 0)),
        "open_eng_requests": int(row.get("open_eng_requests", 0)),
        "planned_turnaround_days": float(row.get("planned_turnaround_days", 0)),
        "projected_days": float(row.get("projected_days", 0)),
        "pct_work_complete": float(row.get("pct_work_complete", 0)),
        "contributing_signals": signals,
        "features": {c: float(row[c]) for c in FEATURE_COLUMNS if c in row},
    }


def get_parts_awp_status(buno: str) -> dict[str, Any]:
    """Return notional parts / AWP status for a BUNO."""
    parts = _parts()
    subset = parts[parts["buno"].astype(str) == str(buno)]
    if subset.empty:
        return {
            "buno": buno,
            "parts": [],
            "awp_count": 0,
            "message": "No parts rows on file",
        }
    records = subset.to_dict(orient="records")
    awp_count = int(subset["awp_flag"].sum()) if "awp_flag" in subset.columns else 0
    return {
        "buno": buno,
        "awp_count": awp_count,
        "zero_balance_count": int(subset["zero_balance"].sum())
        if "zero_balance" in subset.columns
        else 0,
        "parts": records,
    }


def search_tech_data(query: str, k: int = 4) -> dict[str, Any]:
    """RAG search over notional tech data / disposition archive / PMI guidance.

    Only needs the RAG stack (numpy; FAISS optional). Does not load the ML model.
    """
    hits = _rag().search(query, k=k)
    return {
        "query": query,
        "hits": [
            {
                "source": h["source"],
                "chunk_id": h["chunk_id"],
                "score": round(h["score"], 3),
                "text": h["text"],
            }
            for h in hits
        ],
    }


# Back-compat aliases used by older tests / API paths
def get_vehicle_health(vehicle_id: str) -> dict[str, Any]:
    result = get_aircraft_health(vehicle_id)
    if "error" not in result:
        result["vehicle_id"] = result["buno"]
        result["vehicle_type"] = result.get("aircraft_type")
        result["unit"] = result.get("squadron")
        result["worst_band"] = result.get("risk_band")
        result["components"] = [
            {
                "component": result.get("focus_system"),
                "failure_risk": result.get("overrun_risk"),
                "risk_band": result.get("risk_band"),
                "recommended_action": result.get("recommended_action"),
                "operating_hours": result.get("days_in_pmi", 0)
                if "days_in_pmi" in result
                else result.get("planned_turnaround_days", 0),
                "failures_last_90d": result.get("prior_oa_findings", 0),
                "contributing_signals": result.get("contributing_signals", []),
            }
        ]
    return result


def get_component_history(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    hist = _history()
    id_col = "buno" if "buno" in hist.columns else "vehicle_id"
    mask = hist[id_col].astype(str) == str(vehicle_id)
    if component and "component" in hist.columns:
        mask &= hist["component"] == component
    subset = hist.loc[mask].sort_values("days_ago")
    if subset.empty:
        return {
            "buno": vehicle_id,
            "vehicle_id": vehicle_id,
            "component": component,
            "events": [],
            "message": "No history found",
        }
    events = subset.head(12).to_dict(orient="records")
    return {
        "buno": vehicle_id,
        "vehicle_id": vehicle_id,
        "component": component,
        "events": events,
    }


def search_maintenance_manual(query: str, k: int = 4) -> dict[str, Any]:
    return search_tech_data(query, k=k)


TOOLS = {
    "get_aircraft_health": get_aircraft_health,
    "get_parts_awp_status": get_parts_awp_status,
    "search_tech_data": search_tech_data,
    "get_vehicle_health": get_vehicle_health,
    "get_component_history": get_component_history,
    "search_maintenance_manual": search_maintenance_manual,
}
