"""Generate synthetic fleet maintenance records for the demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_maintenance.config import (
    COMPONENTS,
    FEATURE_COLUMNS,
    RAW_DIR,
    TARGET_COLUMN,
    VEHICLE_TYPES,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_fleet_data(
    n_vehicles: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Create one row per vehicle/component with a latent failure risk."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n_vehicles):
        vehicle_id = f"V-{1000 + i}"
        vehicle_type = VEHICLE_TYPES[i % len(VEHICLE_TYPES)]
        # Each vehicle has 1–3 monitored components
        n_components = int(rng.integers(1, 4))
        comps = rng.choice(COMPONENTS, size=n_components, replace=False)

        for component in comps:
            age = float(rng.integers(60, 1600))
            hours = float(rng.integers(80, 5500))
            # Prefer mostly healthy recent history
            failures_90 = int(rng.choice([0, 0, 0, 0, 1, 1, 2, 3]))
            fleet_same = int(rng.integers(0, 8))
            system_30 = int(rng.choice([0, 0, 0, 0, 1, 1, 2]))
            maint_90 = int(rng.integers(0, 5))

            # Strong linear signal; base rate ~18–22%
            logit = (
                -3.4
                + 0.0015 * (age - 400)
                + 0.0004 * (hours - 1200)
                + 0.95 * failures_90
                + 0.15 * fleet_same
                + 0.85 * system_30
                - 0.50 * maint_90
                + rng.normal(0, 0.2)
            )
            p = float(_sigmoid(np.array([logit]))[0])
            failed = int(rng.random() < p)

            readiness = max(0.05, min(0.98, 1.0 - p + rng.normal(0, 0.05)))
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_type": vehicle_type,
                    "component": component,
                    "component_age_days": age,
                    "operating_hours": hours,
                    "failures_last_90d": failures_90,
                    "fleet_same_component_failures_90d": fleet_same,
                    "system_failures_last_30d": system_30,
                    "maintenance_actions_last_90d": maint_90,
                    TARGET_COLUMN: failed,
                    "true_risk": round(p, 4),
                    "readiness_score": round(float(readiness), 4),
                    "unit": f"Unit-{(i % 8) + 1}",
                    "last_service_days_ago": int(rng.integers(5, 220)),
                }
            )

    return pd.DataFrame(rows)


def generate_component_history(
    fleet: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic event history for agent tool `get_component_history`."""
    rng = np.random.default_rng(seed + 7)
    events: list[dict] = []
    event_types = [
        "inspection",
        "corrective_repair",
        "preventive_service",
        "parts_replacement",
        "oil_analysis",
        "fault_code",
    ]
    for _, row in fleet.iterrows():
        n_events = int(rng.integers(2, 7))
        for j in range(n_events):
            days_ago = int(rng.integers(1, 180))
            events.append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "component": row["component"],
                    "event_type": rng.choice(event_types),
                    "days_ago": days_ago,
                    "notes": _event_note(row["component"], days_ago, rng),
                    "severity_hours": max(0, int(row["operating_hours"] - days_ago * 4)),
                }
            )
    return pd.DataFrame(events).sort_values(["vehicle_id", "days_ago"])


def _event_note(component: str, days_ago: int, rng: np.random.Generator) -> str:
    templates = [
        f"{component} inspection — wear within limits ({days_ago}d ago)",
        f"{component} fault cleared after service ({days_ago}d ago)",
        f"Deferred {component} work pending parts ({days_ago}d ago)",
        f"Elevated {component} temperature recorded ({days_ago}d ago)",
        f"Scheduled {component} PM completed ({days_ago}d ago)",
    ]
    return str(rng.choice(templates))


def write_synthetic_docs(docs_dir: Path) -> list[Path]:
    """Small corpus for RAG: manual, preventive guide, readiness policy."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "maintenance_manual.md": """# Fleet Maintenance Manual (Synthetic)

## Engine
Engine failure risk rises sharply after 4,000 operating hours without a major service.
Monitor oil analysis trends, coolant temperature, and fault codes EC-12 / EC-44.
If failures_last_90d >= 2 for the engine, schedule a teardown inspection within 14 days.

## Transmission
Transmission slip and delayed engagement often precede hard failures.
High fleet_same_component_failures_90d for transmissions signals a parts-quality or operating-condition issue —
cross-check unit training tempo and fluid change intervals.

## Brakes
Brake pad life is hours-driven. After 2,500 hours, inspect thickness every 30 days.
System failures in the last 30 days involving brakes require immediate deadline status until cleared.

## Electrical
Intermittent electrical faults are high false-positive drivers. Prefer trend confirmation
(multiple failures_last_90d) before deadline. Check battery SOC, ground straps, and CAN bus logs.

## Cooling
Cooling system failures cluster in high ambient temperature seasons.
If component_age_days > 900 and system_failures_last_30d > 0, flush and pressure-test before next mission.

## Hydraulics / Suspension
Hydraulic leaks reduce readiness faster than raw failure counts suggest.
Any active leak plus maintenance_actions_last_90d == 0 is a readiness red flag.
""",
        "preventive_maintenance_guide.md": """# Preventive Maintenance Guide (Synthetic)

## Cadence
- Daily: fluid levels, tire pressure, fault-code scan
- Weekly: brake/steering walkaround, coolant concentration
- 90-day: oil analysis, transmission fluid sample, electrical load test
- Annual: major component age review vs OEM service life

## Risk-driven escalation
When predicted failure probability exceeds the operating threshold (recall-biased, typically ~0.35),
do not wait for the calendar interval. Pull the vehicle for condition-based maintenance.

## Parts and supply
If fleet_same_component_failures_90d is elevated, pre-position spare parts for that component
across the battalion before the next field exercise.

## Documentation
Every corrective action must cite the triggering signal (hours, age, failures, or model score)
so readiness officers can audit decisions after the mission.
""",
        "readiness_policy.md": """# Unit Readiness Policy (Synthetic)

## Readiness states
- GREEN: risk < 0.20 and no open deadline faults — fully mission capable
- AMBER: 0.20 <= risk < 0.50 or deferred non-critical maintenance — limited mission capable
- RED: risk >= 0.50 or safety-critical open fault — not mission capable

## Recommended actions
- GREEN: continue scheduled PM; no special action
- AMBER: schedule inspection within 7 days; brief commander on contingency
- RED: deadline vehicle; assign maintenance priority 1; notify readiness officer same day

## Operating philosophy
Missed failures cost more than false alarms. Prefer recall over precision at the decision threshold.
The predictive model is a decision-support signal, not an automatic deadline authority —
maintainers retain final say after reviewing contributing features and maintenance history.
""",
    }
    paths = []
    for name, body in documents.items():
        path = docs_dir / name
        path.write_text(body.strip() + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(n_vehicles: int = 120, out_dir: Path | None = None) -> None:
    out = out_dir or RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    fleet = generate_fleet_data(n_vehicles=n_vehicles)
    history = generate_component_history(fleet)
    fleet_path = out / "fleet_maintenance.csv"
    history_path = out / "component_history.csv"
    fleet.to_csv(fleet_path, index=False)
    history.to_csv(history_path, index=False)

    from predictive_maintenance.config import DOCS_DIR

    doc_paths = write_synthetic_docs(DOCS_DIR)
    print(f"Wrote {len(fleet)} component rows → {fleet_path}")
    print(f"Wrote {len(history)} history events → {history_path}")
    print(f"Wrote {len(doc_paths)} docs → {DOCS_DIR}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Positive rate: {fleet[TARGET_COLUMN].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic maintenance data")
    parser.add_argument("--n-vehicles", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    main(n_vehicles=args.n_vehicles, out_dir=args.out_dir)
