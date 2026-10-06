"""Streamlit FRCE MV-22 PMI Delay Assistant (notional pilot concept)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Streamlit Cloud secrets → env bridge (retry again in main() — import-time may be early)
try:
    from predictive_maintenance.agent.advisor import apply_openai_secrets

    apply_openai_secrets()
except Exception:
    pass

# Imports after sys.path bootstrap so Cloud finds `src/predictive_maintenance`
from predictive_maintenance.agent.advisor import (
    apply_openai_secrets,
    explain_risk,
    recommend_next_action,
)
from predictive_maintenance.config import COLUMN_LABELS, FEATURE_COLUMNS, feature_label
from predictive_maintenance.ml.inference import contributing_signals, load_scored_fleet

st.set_page_config(
    page_title="FRCE MV-22 PMI Delay Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BAND_COLORS = {"GREEN": "#1f7a4c", "AMBER": "#c47a00", "RED": "#b42318"}
DRIVER_COLORS = {
    "Over-and-above": "#8B4513",
    "AWP": "#1f4e79",
    "Engineering": "#6b2d5c",
    "On-plan": "#3d7a4c",
}


def labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snake_case columns (and feature keys) for user-facing tables."""
    out = df.copy()
    if "feature" in out.columns:
        out["feature"] = out["feature"].map(lambda k: feature_label(str(k)))
    if "direction" in out.columns:
        out["direction"] = (
            out["direction"]
            .map(
                {
                    "elevates_risk": "Elevates risk",
                    "protective": "Protective",
                }
            )
            .fillna(out["direction"])
        )
    rename = {c: COLUMN_LABELS[c] for c in out.columns if c in COLUMN_LABELS}
    return out.rename(columns=rename)


@st.cache_resource
def get_model():
    """Load ML artifact when possible; return None so the app still boots on Cloud."""
    try:
        from predictive_maintenance.ml.inference import FailureRiskModel

        return FailureRiskModel()
    except Exception:
        return None


@st.cache_data
def get_line() -> pd.DataFrame:
    df = load_scored_fleet()
    if "buno" not in df.columns and "vehicle_id" in df.columns:
        df = df.copy()
        df["buno"] = df["vehicle_id"]
    if "overrun_risk" not in df.columns and "failure_risk" in df.columns:
        df = df.copy()
        df["overrun_risk"] = df["failure_risk"]
    return df


def _signals_for_row(row: pd.Series, model) -> list[dict]:
    if model is not None:
        try:
            feats = {c: float(row[c]) for c in model.feature_columns}
            return model.predict(feats)["contributing_signals"]
        except Exception:
            pass
    try:
        return contributing_signals(row[FEATURE_COLUMNS])
    except Exception:
        return []


def main() -> None:
    apply_openai_secrets()

    st.title("FRCE MV-22 PMI Delay Assistant")
    st.caption(
        "Notional MV-22 PMI line board for Fleet Readiness Center East — "
        "BUNO overrun risk and delay drivers (over-and-above, AWP, engineering). "
        "Not real FRCE data."
    )

    line = get_line()
    model = get_model()

    role = st.sidebar.radio(
        "Role view",
        ["Planner / Production Controller", "Engineer"],
        index=0,
    )

    if role == "Planner / Production Controller":
        render_planner(line, model)
    else:
        render_engineer(line, model)


def render_planner(line: pd.DataFrame, model) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BUNOs on line", line["buno"].nunique())
    c2.metric("RED (likely late)", int((line["risk_band"] == "RED").sum()))
    c3.metric("AMBER", int((line["risk_band"] == "AMBER").sum()))
    c4.metric("AWP holds", int((line.get("status", pd.Series(dtype=str)) == "AWP hold").sum()))

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Overrun risk bands")
        band_counts = (
            line["risk_band"]
            .value_counts()
            .reindex(["GREEN", "AMBER", "RED"])
            .fillna(0)
            .reset_index()
        )
        band_counts.columns = ["risk_band", "count"]
        fig = px.bar(
            band_counts,
            x="risk_band",
            y="count",
            color="risk_band",
            color_discrete_map=BAND_COLORS,
            labels={"risk_band": "Band", "count": "Aircraft"},
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Primary delay drivers")
        if "primary_delay_driver" in line.columns:
            drv = line["primary_delay_driver"].value_counts().reset_index()
            drv.columns = ["driver", "count"]
            fig2 = px.bar(
                drv,
                x="driver",
                y="count",
                color="driver",
                color_discrete_map=DRIVER_COLORS,
            )
            fig2.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Delay driver tags not present in scored data — re-run bootstrap.")

    st.subheader("PMI line board")
    board_cols = [
        c
        for c in [
            "buno",
            "squadron",
            "status",
            "risk_band",
            "overrun_risk",
            "primary_delay_driver",
            "awp_days_open",
            "eng_queue_age_days",
            "prior_oa_findings",
            "planned_turnaround_days",
            "projected_days",
            "recommended_action",
        ]
        if c in line.columns
    ]
    board = line.sort_values("overrun_risk", ascending=False)[board_cols].copy()
    board["overrun_risk"] = board["overrun_risk"].map(lambda x: f"{x:.0%}")
    st.dataframe(labeled_frame(board), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Aircraft detail")
    options = line.sort_values("overrun_risk", ascending=False)["buno"].astype(str).tolist()
    selected = st.selectbox("Select BUNO", options, index=0)
    row = line[line["buno"].astype(str) == selected].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk band", row["risk_band"])
    m2.metric("Overrun risk", f"{row['overrun_risk']:.0%}")
    m3.metric("Primary driver", row.get("primary_delay_driver", "—"))
    m4.metric(
        "Plan → projected",
        f"{row.get('planned_turnaround_days', 0):.0f} → {row.get('projected_days', 0):.0f}d",
    )

    st.markdown(
        f"**BUNO {selected}** · {row.get('aircraft_type', 'MV-22')} · "
        f"{row.get('squadron', '')} · {row.get('status', '')} · "
        f"focus system `{row.get('focus_system', row.get('component', ''))}`"
    )

    d1, d2, d3 = st.columns(3)
    d1.metric(feature_label("prior_oa_findings"), int(row.get("prior_oa_findings", 0)))
    d2.metric(feature_label("awp_days_open"), f"{row.get('awp_days_open', 0):.0f}")
    d3.metric(feature_label("eng_queue_age_days"), f"{row.get('eng_queue_age_days', 0):.0f}")

    prediction_signals = _signals_for_row(row, model)
    st.markdown("#### Contributing signals")
    st.dataframe(
        labeled_frame(pd.DataFrame(prediction_signals)),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Aircraft insight")
    a1, a2 = st.columns(2)
    focus = str(row.get("focus_system", row.get("component", "structure")))
    with a1:
        if st.button("Recommend next action", use_container_width=True):
            with st.spinner("Preparing recommendation…"):
                apply_openai_secrets()
                st.session_state["recommend"] = recommend_next_action(selected, focus)
    with a2:
        if st.button("Explain overrun risk", use_container_width=True):
            with st.spinner("Preparing risk explanation…"):
                apply_openai_secrets()
                st.session_state["explain"] = explain_risk(selected, focus)

    if "recommend" in st.session_state:
        rec = st.session_state["recommend"]
        st.markdown("##### Recommended next action")
        st.markdown(rec["recommendation"])

    if "explain" in st.session_state:
        exp = st.session_state["explain"]
        st.markdown("##### Overrun risk explanation")
        st.markdown(exp["explanation"])


def render_engineer(line: pd.DataFrame, model) -> None:
    st.subheader("Engineering disposition view")
    st.caption(
        "BUNOs with engineering pressure on the MV-22 PMI line. "
        "Use Explain for diagnostic context; Recommend for a concrete next step."
    )

    eng = line[line.get("primary_delay_driver", pd.Series(dtype=str)) == "Engineering"].copy()
    if eng.empty:
        eng = line.sort_values("eng_queue_age_days", ascending=False).head(12)
    else:
        eng = eng.sort_values("eng_queue_age_days", ascending=False)

    st.markdown("#### BUNOs with engineering pressure")
    show = eng[
        [
            c
            for c in [
                "buno",
                "squadron",
                "focus_system",
                "eng_queue_age_days",
                "open_eng_requests",
                "risk_band",
                "overrun_risk",
                "status",
            ]
            if c in eng.columns
        ]
    ].copy()
    if "overrun_risk" in show.columns:
        show["overrun_risk"] = show["overrun_risk"].map(lambda x: f"{float(x):.0%}")
    st.dataframe(labeled_frame(show), use_container_width=True, hide_index=True)

    options = line.sort_values("eng_queue_age_days", ascending=False)["buno"].astype(str).tolist()
    selected = st.selectbox("BUNO context", options, index=0, key="eng_buno")
    row = line[line["buno"].astype(str) == selected].iloc[0]
    focus = str(row.get("focus_system", "structure"))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Recommend next action", use_container_width=True, key="eng_rec"):
            with st.spinner("Preparing recommendation…"):
                apply_openai_secrets()
                st.session_state["eng_recommend"] = recommend_next_action(selected, focus)
    with col_b:
        if st.button("Explain overrun risk", use_container_width=True, key="eng_exp"):
            with st.spinner("Preparing risk explanation…"):
                apply_openai_secrets()
                st.session_state["eng_explain"] = explain_risk(selected, focus)

    if "eng_recommend" in st.session_state:
        st.markdown("##### Recommended next action")
        st.markdown(st.session_state["eng_recommend"]["recommendation"])

    if "eng_explain" in st.session_state:
        st.markdown("##### Overrun risk explanation")
        st.markdown(st.session_state["eng_explain"]["explanation"])


if __name__ == "__main__":
    main()
