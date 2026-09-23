"""Streamlit Fleet Readiness Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictive_maintenance.agent.advisor import explain_risk, recommend_next_action
from predictive_maintenance.ml.inference import FailureRiskModel, load_scored_fleet

st.set_page_config(
    page_title="Fleet Readiness Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BAND_COLORS = {"GREEN": "#1f7a4c", "AMBER": "#c47a00", "RED": "#b42318"}


@st.cache_resource
def get_model() -> FailureRiskModel:
    return FailureRiskModel()


@st.cache_data
def get_fleet() -> pd.DataFrame:
    return load_scored_fleet()


def vehicle_rollup(df: pd.DataFrame) -> pd.DataFrame:
    def worst_band(bands: pd.Series) -> str:
        if (bands == "RED").any():
            return "RED"
        if (bands == "AMBER").any():
            return "AMBER"
        return "GREEN"

    agg = (
        df.groupby(["vehicle_id", "vehicle_type", "unit"], as_index=False)
        .agg(
            max_risk=("failure_risk", "max"),
            n_components=("component", "count"),
            risk_band=("risk_band", worst_band),
            readiness_score=("readiness_score", "min"),
        )
        .sort_values("max_risk", ascending=False)
    )
    agg["recommended_action"] = agg["risk_band"].map(
        {
            "GREEN": "Continue scheduled PM",
            "AMBER": "Inspect within 7 days; brief commander",
            "RED": "Deadline vehicle — priority 1 maintenance",
        }
    )
    return agg


def main() -> None:
    st.title("Predictive Maintenance Readiness Assistant")
    st.caption(
        "Synthetic fleet demo — traditional ML risk scores + RAG/agent maintenance advisor. "
        "Set `OPENAI_API_KEY` for live GenAI; otherwise the mock advisor runs locally."
    )

    fleet = get_fleet()
    model = get_model()
    vehicles = vehicle_rollup(fleet)

    # ---- Fleet overview ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles", vehicles["vehicle_id"].nunique())
    c2.metric("RED (NMC)", int((vehicles["risk_band"] == "RED").sum()))
    c3.metric("AMBER (LMC)", int((vehicles["risk_band"] == "AMBER").sum()))
    c4.metric("Model", model.model_name.replace("_", " ").title())

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Fleet readiness")
        band_counts = (
            vehicles["risk_band"]
            .value_counts()
            .reindex(["GREEN", "AMBER", "RED"])
            .fillna(0)
            .reset_index()
        )
        fig = px.bar(
            band_counts,
            x="risk_band",
            y="count",
            color="risk_band",
            color_discrete_map=BAND_COLORS,
            labels={"risk_band": "Band", "count": "Vehicles"},
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Highest risk vehicles")
        top = vehicles.head(8)[
            ["vehicle_id", "vehicle_type", "unit", "risk_band", "max_risk", "recommended_action"]
        ].copy()
        top["max_risk"] = top["max_risk"].map(lambda x: f"{x:.0%}")
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Vehicle detail")
    options = vehicles["vehicle_id"].tolist()
    default_idx = 0
    selected = st.selectbox("Select vehicle", options, index=default_idx)
    vrow = vehicles[vehicles["vehicle_id"] == selected].iloc[0]
    comps = fleet[fleet["vehicle_id"] == selected].sort_values("failure_risk", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk band", vrow["risk_band"])
    m2.metric("Max failure risk", f"{vrow['max_risk']:.0%}")
    m3.metric("Readiness", f"{vrow['readiness_score']:.0%}")
    m4.metric("Action", vrow["recommended_action"].split("—")[0].strip())

    st.markdown(f"**{selected}** · {vrow['vehicle_type']} · {vrow['unit']}")
    st.dataframe(
        comps[
            [
                "component",
                "failure_risk",
                "risk_band",
                "operating_hours",
                "component_age_days",
                "failures_last_90d",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    focus = st.selectbox("Focus component", comps["component"].tolist())
    crow = comps[comps["component"] == focus].iloc[0]
    prediction = model.predict({c: float(crow[c]) for c in model.feature_columns})

    st.markdown("#### Contributing signals")
    signal_df = pd.DataFrame(prediction["contributing_signals"])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Maintenance advisor (GenAI)")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Explain risk (RAG)", use_container_width=True):
            with st.spinner("Retrieving docs + explaining…"):
                result = explain_risk(selected, focus)
            st.session_state["explain"] = result
    with a2:
        if st.button("Recommend next action (agent)", use_container_width=True):
            with st.spinner("Running tools + advisor…"):
                result = recommend_next_action(selected, focus)
            st.session_state["recommend"] = result

    if "explain" in st.session_state:
        exp = st.session_state["explain"]
        st.info(f"Mode: `{exp.get('mode')}`")
        st.markdown(exp["explanation"])
        if exp.get("sources"):
            src_lbl = ", ".join(f"{s['source']}#{s['chunk_id']}" for s in exp["sources"])
            st.caption(f"Sources: {src_lbl}")

    if "recommend" in st.session_state:
        rec = st.session_state["recommend"]
        st.success(f"Mode: `{rec.get('mode')}`")
        st.markdown(rec["recommendation"])
        with st.expander("Tool traces"):
            st.json(
                {
                    "get_vehicle_health": rec["tool_results"]["vehicle_health"],
                    "get_component_history": rec["tool_results"]["component_history"],
                    "search_maintenance_manual": rec["tool_results"]["manual_hits"],
                }
            )

    with st.sidebar:
        st.header("About this demo")
        st.markdown(
            """
            **Story for the interview**
            1. Fleet dashboard shows at-risk vehicles
            2. Drill into ML risk + contributing signals
            3. RAG explains *why* using maintenance docs
            4. Agent tools recommend the next action

            **ML:** LogReg → Random Forest → XGBoost (best by ROC-AUC)
            **Threshold:** recall-biased operating point (0.35)
            **GenAI:** OpenAI if keyed, else local mock grounded in FAISS hits
            """
        )
        metrics_path = ROOT / "models" / "metrics.json"
        if metrics_path.exists():
            st.subheader("Model metrics")
            st.json(metrics_path.read_text(encoding="utf-8")[:2000])


if __name__ == "__main__":
    main()
