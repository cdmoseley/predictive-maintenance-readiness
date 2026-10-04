"""Shared paths and feature definitions for the FRCE MV-22 PMI pilot demo."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs_corpus"
RAG_INDEX_DIR = MODELS_DIR / "rag_index"

# Tabular features for P(PMI overrun / late delivery)
FEATURE_COLUMNS = [
    "prior_oa_findings",
    "squadron_corr_wiring_score",
    "awp_days_open",
    "zero_balance_hits",
    "eng_queue_age_days",
    "prior_late_pmis",
    "planned_turnaround_days",
    "pct_work_complete",
]

# Human-readable labels for UI / advisor copy (keys stay snake_case in ML code)
FEATURE_LABELS = {
    "prior_oa_findings": "Prior over-and-above findings",
    "squadron_corr_wiring_score": "Squadron corrosion / wiring score",
    "awp_days_open": "AWP days open",
    "zero_balance_hits": "Zero-balance part hits",
    "eng_queue_age_days": "Engineering queue age (days)",
    "prior_late_pmis": "Prior late PMI events",
    "planned_turnaround_days": "Planned turnaround (days)",
    "pct_work_complete": "Work package complete (%)",
}

# Extra column labels for boards / tables shown to users
COLUMN_LABELS = {
    **FEATURE_LABELS,
    "buno": "BUNO",
    "squadron": "Squadron",
    "status": "Status",
    "risk_band": "Risk band",
    "overrun_risk": "Overrun risk",
    "primary_delay_driver": "Primary delay driver",
    "planned_turnaround_days": "Planned turnaround (days)",
    "projected_days": "Projected days",
    "recommended_action": "Recommended action",
    "focus_system": "Focus system",
    "open_eng_requests": "Open eng. requests",
    "feature": "Signal",
    "value": "Value",
    "baseline": "Baseline",
    "concern_score": "Concern score",
    "direction": "Direction",
}


def feature_label(key: str) -> str:
    """Return a plain-English label for a feature or column key."""
    if key in COLUMN_LABELS:
        return COLUMN_LABELS[key]
    return key.replace("_", " ").strip().capitalize()


TARGET_COLUMN = "pmi_overrun"

# Recall-biased operating threshold — catch late aircraft early
OPERATING_THRESHOLD = 0.35

AIRCRAFT_TYPE = "MV-22"
SQUADRONS = ["VMM-261", "VMM-263", "VMM-365", "VMM-162", "VMM-266", "VMM-764"]

DELAY_DRIVERS = [
    "Over-and-above",
    "AWP",
    "Engineering",
    "On-plan",
]

# Notional shops / systems that surface O&A findings
SYSTEMS = [
    "proprotor",
    "drive_system",
    "avionics",
    "hydraulics",
    "structure",
    "wiring",
    "corrosion",
]
