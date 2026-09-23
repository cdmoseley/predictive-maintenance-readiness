"""Shared paths and feature definitions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs_corpus"
RAG_INDEX_DIR = MODELS_DIR / "rag_index"

FEATURE_COLUMNS = [
    "component_age_days",
    "operating_hours",
    "failures_last_90d",
    "fleet_same_component_failures_90d",
    "system_failures_last_30d",
    "maintenance_actions_last_90d",
]

TARGET_COLUMN = "failure_next_30_days"

# Recall-biased operating threshold for fleet readiness decisions
OPERATING_THRESHOLD = 0.35

COMPONENTS = [
    "engine",
    "transmission",
    "brakes",
    "electrical",
    "suspension",
    "cooling",
    "hydraulics",
]

VEHICLE_TYPES = ["HMMWV", "FMTV", "MRAP", "JLTV", "LMTV"]
