"""Generate synthetic MV-22 PMI line data for the FRCE pilot demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_maintenance.config import (
    AIRCRAFT_TYPE,
    DELAY_DRIVERS,
    DOCS_DIR,
    FEATURE_COLUMNS,
    RAW_DIR,
    SQUADRONS,
    SYSTEMS,
    TARGET_COLUMN,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_pmi_line(
    n_aircraft: int = 48,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per BUNO currently (or recently) in MV-22 PMI."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n_aircraft):
        buno = f"16{8200 + i}"  # notional Bureau Numbers
        squadron = SQUADRONS[i % len(SQUADRONS)]
        planned = float(rng.choice([90, 100, 110, 120, 130]))
        days_in = float(rng.integers(10, int(planned) + 25))
        pct_complete = float(np.clip(days_in / planned + rng.normal(0, 0.08), 0.05, 0.98))

        prior_oa = int(rng.choice([0, 0, 1, 1, 2, 3, 4, 5]))
        corr_wiring = float(np.clip(rng.normal(prior_oa * 0.8 + 1.5, 1.2), 0, 10))
        awp_days = float(rng.choice([0, 0, 0, 2, 5, 8, 12, 18, 25]))
        zero_bal = int(rng.choice([0, 0, 0, 1, 1, 2, 3]))
        eng_age = float(rng.choice([0, 0, 0, 1, 3, 5, 8, 12, 20]))
        prior_late = int(rng.choice([0, 0, 0, 1, 1, 2]))
        open_eng = int(0 if eng_age == 0 else rng.integers(1, 4))

        # Latent overrun risk — O&A history, AWP, eng queue dominate
        logit = (
            -2.8
            + 0.45 * prior_oa
            + 0.22 * corr_wiring
            + 0.09 * awp_days
            + 0.55 * zero_bal
            + 0.12 * eng_age
            + 0.70 * prior_late
            - 0.015 * (planned - 100)
            - 1.2 * pct_complete
            + rng.normal(0, 0.25)
        )
        p = float(_sigmoid(np.array([logit]))[0])
        overrun = int(rng.random() < p)

        # Primary delay driver from which signal is loudest
        driver_scores = {
            "Over-and-above": prior_oa * 1.2 + corr_wiring * 0.4,
            "AWP": awp_days * 0.5 + zero_bal * 2.0,
            "Engineering": eng_age * 0.8 + open_eng * 1.5,
            "On-plan": 1.5 if p < 0.25 else 0.2,
        }
        primary = max(driver_scores, key=driver_scores.get)
        if primary not in DELAY_DRIVERS:
            primary = "On-plan"

        projected = planned * (1.0 + max(0.0, p - 0.15))
        if overrun:
            projected = max(projected, planned * 1.15)

        status = "In work"
        if awp_days >= 8 and primary == "AWP":
            status = "AWP hold"
        elif eng_age >= 5 and primary == "Engineering":
            status = "Eng disposition hold"
        elif pct_complete > 0.9:
            status = "Final ops"

        focus_system = str(rng.choice(SYSTEMS))
        critical_nsn = f"1680-01-{rng.integers(10000, 99999)}" if zero_bal or awp_days else ""

        rows.append(
            {
                "buno": buno,
                "aircraft_type": AIRCRAFT_TYPE,
                "squadron": squadron,
                "pmi_event_id": f"PMI-MV22-{2026}-{100 + i}",
                "status": status,
                "primary_delay_driver": primary,
                "focus_system": focus_system,
                "critical_nsn": critical_nsn,
                "days_in_pmi": round(days_in, 1),
                "projected_days": round(float(projected), 1),
                "open_eng_requests": open_eng,
                "prior_oa_findings": prior_oa,
                "squadron_corr_wiring_score": round(corr_wiring, 2),
                "awp_days_open": awp_days,
                "zero_balance_hits": zero_bal,
                "eng_queue_age_days": eng_age,
                "prior_late_pmis": prior_late,
                "planned_turnaround_days": planned,
                "pct_work_complete": round(pct_complete, 3),
                TARGET_COLUMN: overrun,
                "true_risk": round(p, 4),
                # legacy aliases so older helpers keep working during retarget
                "vehicle_id": buno,
                "vehicle_type": AIRCRAFT_TYPE,
                "unit": squadron,
                "component": focus_system,
            }
        )

    return pd.DataFrame(rows)


# Back-compat name used by tests / scripts
def generate_fleet_data(n_vehicles: int = 48, seed: int = 42) -> pd.DataFrame:
    return generate_pmi_line(n_aircraft=n_vehicles, seed=seed)


def generate_component_history(
    fleet: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic squadron / depot event history for agent tools."""
    rng = np.random.default_rng(seed + 7)
    events: list[dict] = []
    event_types = [
        "squadron_discrepancy",
        "oa_finding",
        "parts_requisition",
        "awp_stoppage",
        "eng_request",
        "disposition_closed",
        "pmi_inspection",
    ]
    for _, row in fleet.iterrows():
        n_events = int(rng.integers(3, 8))
        for _ in range(n_events):
            days_ago = int(rng.integers(1, 400))
            et = str(rng.choice(event_types))
            events.append(
                {
                    "buno": row["buno"],
                    "vehicle_id": row["buno"],
                    "component": row["focus_system"],
                    "system": row["focus_system"],
                    "event_type": et,
                    "days_ago": days_ago,
                    "notes": _event_note(row["buno"], row["focus_system"], et, days_ago, rng),
                    "nsn": row["critical_nsn"] or f"1680-01-{rng.integers(10000, 99999)}",
                }
            )
    return pd.DataFrame(events).sort_values(["buno", "days_ago"])


def generate_parts_awp(fleet: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Notional AWP / parts rows for get_parts_awp_status."""
    rng = np.random.default_rng(seed + 11)
    rows: list[dict] = []
    for _, row in fleet.iterrows():
        n_parts = int(rng.integers(1, 4))
        for j in range(n_parts):
            lead = int(rng.choice([3, 7, 14, 30, 60, 90]))
            on_hand = int(rng.choice([0, 0, 0, 1, 2, 5]))
            qty_req = int(rng.integers(1, 4))
            awp = on_hand < qty_req
            rows.append(
                {
                    "buno": row["buno"],
                    "nsn": row["critical_nsn"] or f"1680-01-{rng.integers(10000, 99999)}",
                    "nomenclature": str(
                        rng.choice(
                            [
                                "Proprotor hub seal kit",
                                "Wire harness assembly",
                                "Hydraulic pump",
                                "Avionics LRU mount",
                                "Corrosion repair patch kit",
                                "Drive shaft coupling",
                            ]
                        )
                    ),
                    "qty_required": qty_req,
                    "qty_on_hand": on_hand,
                    "lead_time_days": lead,
                    "zero_balance": int(on_hand == 0),
                    "awp_flag": int(awp),
                    "requisition_status": "AWP" if awp else str(rng.choice(["Filled", "In transit"])),
                    "shop": str(rng.choice(["Structures", "Avionics", "Power Plants", "Hydraulics"])),
                }
            )
    return pd.DataFrame(rows)


def _event_note(
    buno: str,
    system: str,
    event_type: str,
    days_ago: int,
    rng: np.random.Generator,
) -> str:
    templates = {
        "squadron_discrepancy": f"{buno} squadron write-up on {system} ({days_ago}d ago)",
        "oa_finding": f"O&A finding during PMI teardown — {system} ({days_ago}d ago)",
        "parts_requisition": f"Requisition opened for {system} support parts ({days_ago}d ago)",
        "awp_stoppage": f"Work stoppage AWP — {system} ({days_ago}d ago)",
        "eng_request": f"Engineering request submitted — condition not in tech data ({days_ago}d ago)",
        "disposition_closed": f"Prior disposition closed for similar {system} condition ({days_ago}d ago)",
        "pmi_inspection": f"PMI inspection complete on {system} ({days_ago}d ago)",
    }
    return templates.get(event_type, f"{event_type} on {system} ({days_ago}d ago)")


def write_synthetic_docs(docs_dir: Path) -> list[Path]:
    """Notional NAMP-aligned tech data, disposition archive, PMI guidance for RAG."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "mv22_pmi_guidance.md": """# MV-22 PMI Production Guidance (Notional — Demo Only)

## Purpose
This document is synthetic guidance for a Fleet Readiness Center East (FRCE) MV-22
Planned Maintenance Interval (PMI) pilot demonstration. It is not official NAVAIR tech data.

## Work-package build
Before induction, planners should review the BUNO's squadron maintenance history for
corrosion, wiring, and structural discrepancies. Prior over-and-above (O&A) density on
sister aircraft in the same squadron is a leading indicator of mid-event re-planning.

## Delay drivers
1. Over-and-above — discrepancies found after disassembly that were not in the planned package.
2. Awaiting parts (AWP) — long-lead or zero-balance NSNs discovered at the wrench.
3. Engineering disposition backlog — conditions not covered by tech data; work stops until disposition.

## Production control actions
- If predicted PMI overrun risk is AMBER or RED, brief the MV-22 production lead at the next stand-up.
- Pre-position long-lead parts when zero_balance_hits >= 1 or awp_days_open is trending up mid-event.
- Do not wait for calendar completion percentage alone; risk models supplement plan vs actual days.
""",
        "namp_tech_data_excerpts.md": """# NAMP / Tech Data Excerpts (Notional — Demo Only)

## Traceability
Under the Naval Aviation Maintenance Program (NAMP), all maintenance actions must be
traceable to authoritative technical data. Generative AI outputs in this pilot are
advisory decision support only and must cite retrieved sources. Maintainers and engineers
retain final authority.

## Wiring and corrosion
Intermittent wiring discrepancies and corrosion findings frequently drive MV-22 O&A growth
during PMI. When squadron_corr_wiring_score is elevated, expand inspection hours in the
affected zones during work-package build rather than discovering them post-teardown.

## Hydraulics and drive system
Hydraulic leaks and drive-system seal kits are common AWP drivers. If a critical NSN is
zero-balance, open or expedite the requisition at package build. Cannibalization decisions
require production controller and material expediter concurrence.

## Structure
Structural discrepancies not covered by existing repair limits require an engineering
request. Search the local disposition archive for similar zone/defect patterns before
opening a duplicate request.
""",
        "disposition_archive.md": """# Local Engineering Disposition Archive (Notional — Demo Only)

## DISP-MV22-1042 — Proprotor hub seal weepage
Condition: Light weepage at proprotor hub seal during PMI, within limited reuse criteria
after inspection. Disposition: Continue with increased inspection interval; replace seal
kit if weepage exceeds limit in tech data table. Keywords: proprotor, seal, weepage, PMI.

## DISP-MV22-1188 — Avionics wire chafing zone 3
Condition: Chafing on wire harness not explicitly illustrated in IETM figure. Disposition:
Repair per standard wiring practice; add protective grommet; document as local O&A lesson
learned for sister BUNOs. Keywords: wiring, chafing, avionics, harness.

## DISP-MV22-1201 — Skin corrosion under fairing
Condition: Corrosion under fairing beyond blend-out limits in referenced repair. Disposition:
Install approved patch kit; engineering sign-off required before close. Keywords: corrosion,
structure, fairing, patch.

## DISP-MV22-1266 — Hydraulic pump case porosity
Condition: Porosity indication on hydraulic pump case during NDI. Disposition: Remove and
replace pump; do not blend. Expedite NSN if zero-balance. Keywords: hydraulics, pump, NDI, AWP.

## Reuse policy
Engineers should search this archive for similar ATA/zone/defect patterns before submitting
a new request. Cite the prior disposition ID in the new request if requesting confirmation
or deviation. AI search results are not a substitute for engineer judgment.
""",
    }
    paths = []
    for name, body in documents.items():
        path = docs_dir / name
        path.write_text(body.strip() + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(n_vehicles: int = 48, out_dir: Path | None = None) -> None:
    out = out_dir or RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    fleet = generate_pmi_line(n_aircraft=n_vehicles)
    history = generate_component_history(fleet)
    parts = generate_parts_awp(fleet)

    fleet_path = out / "fleet_maintenance.csv"
    history_path = out / "component_history.csv"
    parts_path = out / "parts_awp.csv"
    fleet.to_csv(fleet_path, index=False)
    history.to_csv(history_path, index=False)
    parts.to_csv(parts_path, index=False)

    doc_paths = write_synthetic_docs(DOCS_DIR)
    print(f"Wrote {len(fleet)} BUNO PMI rows → {fleet_path}")
    print(f"Wrote {len(history)} history events → {history_path}")
    print(f"Wrote {len(parts)} parts/AWP rows → {parts_path}")
    print(f"Wrote {len(doc_paths)} docs → {DOCS_DIR}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Overrun rate: {fleet[TARGET_COLUMN].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MV-22 PMI data")
    parser.add_argument("--n-vehicles", type=int, default=48)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    main(n_vehicles=args.n_vehicles, out_dir=args.out_dir)
