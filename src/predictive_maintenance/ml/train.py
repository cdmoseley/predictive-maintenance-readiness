"""Train LogReg → Random Forest → XGBoost; persist the best model by ROC-AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predictive_maintenance.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    OPERATING_THRESHOLD,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_COLUMN,
)


def load_fleet(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (RAW_DIR / "fleet_maintenance.csv")
    if not csv_path.exists():
        from predictive_maintenance.data.generate import main as gen_main

        gen_main()
    return pd.read_csv(csv_path)


def build_models(seed: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": threshold,
        "confusion_matrix": cm,
    }


def train(
    fleet: pd.DataFrame | None = None,
    seed: int = 42,
    models_dir: Path | None = None,
) -> dict:
    df = fleet if fleet is not None else load_fleet()
    out = models_dir or MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Persist split for demo reproducibility / notebook-less inspection
    split_meta = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, model in build_models(seed).items():
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)[:, 1]
        else:
            proba = model.decision_function(x_test)
            proba = 1 / (1 + np.exp(-proba))
        metrics = evaluate_binary(y_test.to_numpy(), proba)
        results[name] = metrics
        fitted[name] = model
        auc = metrics["roc_auc"]
        rec = metrics["recall"]
        print(f"{name}: ROC-AUC={auc:.3f}  recall@{OPERATING_THRESHOLD}={rec:.3f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": OPERATING_THRESHOLD,
        "metrics": results[best_name],
        "all_metrics": results,
    }
    model_path = out / "best_model.joblib"
    metrics_path = out / "metrics.json"
    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "operating_threshold": OPERATING_THRESHOLD,
                "split": split_meta,
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Scored fleet for the dashboard
    if hasattr(best_model, "predict_proba"):
        full_proba = best_model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    else:
        full_proba = best_model.decision_function(df[FEATURE_COLUMNS])
        full_proba = 1 / (1 + np.exp(-full_proba))
    scored = df.copy()
    scored["failure_risk"] = full_proba
    scored["risk_band"] = scored["failure_risk"].apply(_risk_band)
    scored["recommended_action"] = scored["risk_band"].map(
        {
            "GREEN": "Continue scheduled PM",
            "AMBER": "Inspect within 7 days; brief commander",
            "RED": "Deadline vehicle — priority 1 maintenance",
        }
    )
    scored_path = PROCESSED_DIR / "fleet_scored.csv"
    scored.to_csv(scored_path, index=False)

    print(f"Best model: {best_name} → {model_path}")
    print(f"Scored fleet → {scored_path}")
    return {"best_model": best_name, "metrics": results, "model_path": str(model_path)}


def _risk_band(p: float) -> str:
    if p >= 0.50:
        return "RED"
    if p >= 0.20:
        return "AMBER"
    return "GREEN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train failure-risk models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(seed=args.seed)


if __name__ == "__main__":
    main()
