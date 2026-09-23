"""Load the best saved model and score vehicles / feature contributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from predictive_maintenance.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    OPERATING_THRESHOLD,
    PROCESSED_DIR,
)


class FailureRiskModel:
    """Thin inference wrapper around the trained artifact."""

    def __init__(self, model_path: Path | None = None):
        path = model_path or (MODELS_DIR / "best_model.joblib")
        if not path.exists():
            from predictive_maintenance.ml.train import train

            train()
        self.artifact: dict[str, Any] = joblib.load(path)
        self.model = self.artifact["model"]
        self.model_name: str = self.artifact["model_name"]
        self.feature_columns: list[str] = list(self.artifact["feature_columns"])
        self.threshold: float = float(self.artifact.get("threshold", OPERATING_THRESHOLD))

    def predict_proba(self, features: pd.DataFrame | dict) -> np.ndarray:
        frame = self._as_frame(features)
        return self.model.predict_proba(frame[self.feature_columns])[:, 1]

    def predict(self, features: pd.DataFrame | dict) -> dict[str, Any]:
        frame = self._as_frame(features)
        proba = float(self.predict_proba(frame)[0])
        band = risk_band(proba)
        return {
            "failure_risk": round(proba, 4),
            "risk_band": band,
            "threshold": self.threshold,
            "predicted_failure": proba >= self.threshold,
            "recommended_action": recommended_action(band),
            "model_name": self.model_name,
            "contributing_signals": contributing_signals(frame.iloc[0]),
        }

    def _as_frame(self, features: pd.DataFrame | dict) -> pd.DataFrame:
        if isinstance(features, dict):
            missing = [c for c in self.feature_columns if c not in features]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            return pd.DataFrame([{c: features[c] for c in self.feature_columns}])
        return features


def risk_band(p: float) -> str:
    if p >= 0.50:
        return "RED"
    if p >= 0.20:
        return "AMBER"
    return "GREEN"


def recommended_action(band: str) -> str:
    return {
        "GREEN": "Continue scheduled PM",
        "AMBER": "Inspect within 7 days; brief commander",
        "RED": "Deadline vehicle — priority 1 maintenance",
    }[band]


def contributing_signals(row: pd.Series) -> list[dict[str, Any]]:
    """Heuristic feature contributions for demo explainability (not SHAP).

    Ranks features by how far they sit above a healthy baseline so maintainers
    can see *why* the model is concerned without a black-box explainer dependency.
    """
    baselines = {
        "component_age_days": 365.0,
        "operating_hours": 1500.0,
        "failures_last_90d": 0.5,
        "fleet_same_component_failures_90d": 2.0,
        "system_failures_last_30d": 0.3,
        "maintenance_actions_last_90d": 2.0,  # more maintenance is protective
    }
    directions = {
        "component_age_days": 1,
        "operating_hours": 1,
        "failures_last_90d": 1,
        "fleet_same_component_failures_90d": 1,
        "system_failures_last_30d": 1,
        "maintenance_actions_last_90d": -1,
    }
    signals = []
    for col in FEATURE_COLUMNS:
        value = float(row[col])
        base = baselines[col]
        direction = directions[col]
        if direction > 0:
            excess = max(0.0, (value - base) / max(base, 1.0))
        else:
            excess = max(0.0, (base - value) / max(base, 1.0))
        signals.append(
            {
                "feature": col,
                "value": value,
                "baseline": base,
                "concern_score": round(excess, 3),
                "direction": "elevates_risk" if direction > 0 else "protective",
            }
        )
    signals.sort(key=lambda s: s["concern_score"], reverse=True)
    return signals


def load_scored_fleet(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (PROCESSED_DIR / "fleet_scored.csv")
    if not csv_path.exists():
        from predictive_maintenance.ml.train import train

        train()
    return pd.read_csv(csv_path)
