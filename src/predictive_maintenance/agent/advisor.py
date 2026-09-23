"""Maintenance advisor: OpenAI when keyed, otherwise grounded mock fallback."""

from __future__ import annotations

import json
import os
from typing import Any

from predictive_maintenance.agent.tools import (
    get_component_history,
    get_vehicle_health,
    search_maintenance_manual,
)


def recommend_next_action(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    """Run tools then produce a grounded recommendation (live or mock LLM)."""
    health = get_vehicle_health(vehicle_id)
    if "error" in health:
        return health

    focus = component
    if focus is None:
        focus = max(health["components"], key=lambda c: c["failure_risk"])["component"]

    history = get_component_history(vehicle_id, focus)
    manual = search_maintenance_manual(
        f"{focus} failure risk readiness maintenance {health['worst_band']}",
        k=3,
    )

    tool_payload = {
        "vehicle_health": health,
        "component_history": history,
        "manual_hits": manual["hits"],
    }

    if os.getenv("OPENAI_API_KEY"):
        recommendation = _openai_recommend(vehicle_id, focus, tool_payload)
        mode = "openai"
    else:
        recommendation = _mock_recommend(vehicle_id, focus, tool_payload)
        mode = "mock"

    return {
        "vehicle_id": vehicle_id,
        "component": focus,
        "mode": mode,
        "recommendation": recommendation,
        "tool_results": tool_payload,
    }


def explain_risk(vehicle_id: str, component: str | None = None) -> dict[str, Any]:
    """RAG-grounded explanation of why a vehicle/component is at risk."""
    health = get_vehicle_health(vehicle_id)
    if "error" in health:
        return health

    focus_row = (
        next((c for c in health["components"] if c["component"] == component), None)
        if component
        else max(health["components"], key=lambda c: c["failure_risk"])
    )
    assert focus_row is not None
    focus = focus_row["component"]
    signals = ", ".join(
        f"{s['feature']}={s['value']}" for s in focus_row["contributing_signals"]
    )
    query = (
        f"Explain failure risk for {focus} on {health['vehicle_type']}. "
        f"Signals: {signals}. Risk band {focus_row['risk_band']}."
    )
    manual = search_maintenance_manual(query, k=4)

    if os.getenv("OPENAI_API_KEY"):
        text = _openai_explain(query, manual["hits"], focus_row)
        mode = "openai"
    else:
        text = _mock_explain(vehicle_id, focus, focus_row, manual["hits"])
        mode = "mock"

    return {
        "vehicle_id": vehicle_id,
        "component": focus,
        "mode": mode,
        "explanation": text,
        "sources": [{"source": h["source"], "chunk_id": h["chunk_id"]} for h in manual["hits"]],
        "contributing_signals": focus_row["contributing_signals"],
    }


def _mock_recommend(vehicle_id: str, component: str, payload: dict) -> str:
    health = payload["vehicle_health"]
    focus = next(c for c in health["components"] if c["component"] == component)
    events = payload["component_history"].get("events", [])
    latest = events[0]["notes"] if events else "no recent events on file"
    doc_snip = payload["manual_hits"][0]["text"][:220] if payload["manual_hits"] else ""
    return (
        f"**Recommendation for {vehicle_id} / {component}** (mock advisor — set "
        f"`OPENAI_API_KEY` for live GenAI)\n\n"
        f"- Current band: **{focus['risk_band']}** "
        f"(model risk {focus['failure_risk']:.0%})\n"
        f"- Action: {focus['recommended_action']}\n"
        f"- Recent history: {latest}\n"
        f"- Manual guidance: {doc_snip}...\n\n"
        f"Tools used: `get_vehicle_health`, `get_component_history`, "
        f"`search_maintenance_manual`."
    )


def _mock_explain(vehicle_id: str, component: str, focus: dict, hits: list[dict]) -> str:
    top = focus["contributing_signals"][:3]
    signal_lines = "\n".join(
        f"- `{s['feature']}` = {s['value']} (baseline {s['baseline']}, "
        f"concern {s['concern_score']})"
        for s in top
    )
    citations = "\n".join(
        f"- [{h['source']}#{h['chunk_id']}] {h['text'][:160]}..." for h in hits[:3]
    )
    return (
        f"**Why {vehicle_id} / {component} is {focus['risk_band']}** "
        f"(mock RAG explanation)\n\n"
        f"Predicted failure risk is **{focus['failure_risk']:.0%}**. "
        f"Top contributing signals:\n{signal_lines}\n\n"
        f"Grounded in maintenance docs:\n{citations}\n\n"
        f"Policy reminder: missed failures cost more than false alarms — "
        f"prefer recall at the operating threshold."
    )


def _openai_recommend(vehicle_id: str, component: str, payload: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "You are a military fleet readiness advisor. Using ONLY the tool JSON, "
        "recommend the next maintenance action in <=120 words. Cite signals and docs.\n\n"
        f"Vehicle: {vehicle_id}\nComponent: {component}\n"
        f"TOOL_JSON:\n{json.dumps(payload, default=str)[:6000]}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _openai_explain(query: str, hits: list[dict], focus: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    prompt = (
        "Explain the vehicle failure risk using the context and signals. "
        "Be concise (<=150 words) and cite sources by filename.\n\n"
        f"Query: {query}\nSignals: {json.dumps(focus['contributing_signals'][:4])}\n"
        f"Context:\n{context}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
