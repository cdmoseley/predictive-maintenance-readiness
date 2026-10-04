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
