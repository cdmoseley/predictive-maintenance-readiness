"""Agent tools for the maintenance advisor."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from predictive_maintenance.config import RAW_DIR
from predictive_maintenance.ml.inference import FailureRiskModel, load_scored_fleet
from predictive_maintenance.rag.index import MaintenanceRAG


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    path = RAW_DIR / "component_history.csv"
    if not path.exists():
        from predictive_maintenance.data.generate import main as gen_main

        gen_main()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _model() -> FailureRiskModel:
    return FailureRiskModel()


@lru_cache(maxsize=1)
def _rag() -> MaintenanceRAG:
    return MaintenanceRAG()


def get_vehicle_health(vehicle_id: str) -> dict[str, Any]:
    """Return scored health for all components on a vehicle."""
    fleet = load_scored_fleet()
    rows = fleet[fleet["vehicle_id"] == vehicle_id]
    if rows.empty:
        return {"error": f"Unknown vehicle_id: {vehicle_id}", "vehicle_id": vehicle_id}
    model = _model()
    components = []
    for _, row in rows.iterrows():
        pred = model.predict(row[model.feature_columns].to_dict())
        components.append(
            {
                "component": row["component"],
                "failure_risk": float(row.get("failure_risk", pred["failure_risk"])),
                "risk_band": row.get("risk_band", pred["risk_band"]),
                "recommended_action": row.get("recommended_action", pred["recommended_action"]),
                "operating_hours": float(row["operating_hours"]),
                "failures_last_90d": int(row["failures_last_90d"]),
                "contributing_signals": pred["contributing_signals"][:3],
            }
        )
    worst = max(components, key=lambda c: c["failure_risk"])
    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": rows.iloc[0]["vehicle_type"],
        "unit": rows.iloc[0]["unit"],
        "worst_band": worst["risk_band"],
        "components": components,
    }


def get_component_history(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    """Return recent maintenance / fault events for a vehicle (optional component filter)."""
    hist = _history()
    mask = hist["vehicle_id"] == vehicle_id
    if component:
        mask &= hist["component"] == component
    subset = hist.loc[mask].sort_values("days_ago")
    if subset.empty:
        return {
            "vehicle_id": vehicle_id,
            "component": component,
            "events": [],
            "message": "No history found",
        }
    events = subset.head(12).to_dict(orient="records")
    return {"vehicle_id": vehicle_id, "component": component, "events": events}


def search_maintenance_manual(query: str, k: int = 4) -> dict[str, Any]:
    """RAG search over synthetic maintenance docs."""
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


TOOLS = {
    "get_vehicle_health": get_vehicle_health,
    "get_component_history": get_component_history,
    "search_maintenance_manual": search_maintenance_manual,
}
