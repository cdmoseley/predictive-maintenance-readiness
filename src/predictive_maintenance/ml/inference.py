"""Load the best saved model and score BUNO PMI overrun risk."""

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
    """Inference wrapper — name retained for compatibility; scores PMI overrun risk."""

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
            "overrun_risk": round(proba, 4),
            "failure_risk": round(proba, 4),  # alias
            "risk_band": band,
            "threshold": self.threshold,
            "predicted_overrun": proba >= self.threshold,
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


# Alias for clarity in FRCE framing
PMIOverrunModel = FailureRiskModel


def risk_band(p: float) -> str:
    if p >= 0.50:
        return "RED"
    if p >= 0.20:
        return "AMBER"
    return "GREEN"


def recommended_action(band: str) -> str:
    return {
        "GREEN": "Continue PMI plan; monitor drivers at stand-up",
        "AMBER": "Re-plan within 48h — review O&A / AWP / eng queue",
        "RED": "Production priority — mitigate primary delay driver now",
    }[band]


def contributing_signals(row: pd.Series) -> list[dict[str, Any]]:
    """Heuristic feature contributions for demo explainability (not SHAP)."""
    baselines = {
        "prior_oa_findings": 1.0,
        "squadron_corr_wiring_score": 2.5,
        "awp_days_open": 3.0,
        "zero_balance_hits": 0.5,
        "eng_queue_age_days": 2.0,
        "prior_late_pmis": 0.5,
        "planned_turnaround_days": 110.0,
        "pct_work_complete": 0.55,  # higher completion is protective when on-plan
    }
    directions = {
        "prior_oa_findings": 1,
        "squadron_corr_wiring_score": 1,
        "awp_days_open": 1,
        "zero_balance_hits": 1,
        "eng_queue_age_days": 1,
        "prior_late_pmis": 1,
        "planned_turnaround_days": 1,  # longer plans still overrun when drivers spike
        "pct_work_complete": -1,
    }
    signals = []
    for col in FEATURE_COLUMNS:
        value = float(row[col])
        base = baselines[col]
        direction = directions[col]
        if direction > 0:
            excess = max(0.0, (value - base) / max(base, 1.0))
        else:
            excess = max(0.0, (base - value) / max(base, 1e-3))
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
