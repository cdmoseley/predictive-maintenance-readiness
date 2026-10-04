"""PMI advisor: OpenAI when keyed, otherwise grounded mock fallback."""

from __future__ import annotations

import json
import os
from typing import Any

from predictive_maintenance.agent.tools import (
    get_aircraft_health,
    get_component_history,
    get_parts_awp_status,
    search_tech_data,
)
from predictive_maintenance.config import feature_label


def recommend_next_action(buno: str, component: str | None = None) -> dict[str, Any]:
    """Run tools then produce a grounded recommendation (live or mock LLM)."""
    health = get_aircraft_health(buno)
    if "error" in health:
        return health

    focus = component or health.get("focus_system") or "structure"
    history = get_component_history(buno, focus)
    parts = get_parts_awp_status(buno)
    tech = search_tech_data(
        f"MV-22 PMI {focus} {health.get('primary_delay_driver')} "
        f"overrun {health.get('risk_band')} disposition AWP",
        k=3,
    )

    tool_payload = {
        "aircraft_health": health,
        "vehicle_health": health,  # alias for older UI
        "component_history": history,
        "parts_awp": parts,
        "manual_hits": tech["hits"],
        "tech_hits": tech["hits"],
    }

    if os.getenv("OPENAI_API_KEY"):
        recommendation = _openai_recommend(buno, focus, tool_payload)
        mode = "openai"
    else:
        recommendation = _mock_recommend(buno, focus, tool_payload)
        mode = "mock"

    return {
        "buno": buno,
        "vehicle_id": buno,
        "component": focus,
        "mode": mode,
        "recommendation": recommendation,
        "tool_results": tool_payload,
    }


def explain_risk(buno: str, component: str | None = None) -> dict[str, Any]:
    """RAG-grounded explanation of why a BUNO is at overrun risk."""
    health = get_aircraft_health(buno)
    if "error" in health:
        return health

    focus = component or health.get("focus_system") or "structure"
    signals = ", ".join(
        f"{feature_label(s['feature'])}={s['value']}"
        for s in health.get("contributing_signals", [])[:4]
    )
    query = (
        f"Explain PMI overrun risk for MV-22 BUNO {buno} system {focus}. "
        f"Primary delay driver {health.get('primary_delay_driver')}. "
        f"Signals: {signals}. Risk band {health.get('risk_band')}."
    )
    tech = search_tech_data(query, k=4)

    focus_row = {
        "component": focus,
        "failure_risk": health.get("overrun_risk"),
        "risk_band": health.get("risk_band"),
        "contributing_signals": health.get("contributing_signals", []),
    }

    if os.getenv("OPENAI_API_KEY"):
        text = _openai_explain(query, tech["hits"], focus_row)
        mode = "openai"
    else:
        text = _mock_explain(buno, focus, health, tech["hits"])
        mode = "mock"

    return {
        "buno": buno,
        "vehicle_id": buno,
        "component": focus,
        "mode": mode,
        "explanation": text,
        "sources": [{"source": h["source"], "chunk_id": h["chunk_id"]} for h in tech["hits"]],
        "contributing_signals": health.get("contributing_signals", []),
    }


def _mock_recommend(buno: str, component: str, payload: dict) -> str:
    health = payload["aircraft_health"]
    parts = payload.get("parts_awp", {})
    events = payload["component_history"].get("events", [])
    latest = events[0]["notes"] if events else "no recent events on file"
    doc_snip = payload["manual_hits"][0]["text"][:220] if payload["manual_hits"] else ""
    awp_count = parts.get("awp_count", 0)
    return (
        f"**Recommendation for BUNO {buno} / {component}** "
        f"(mock advisor — set `OPENAI_API_KEY` for live GenAI)\n\n"
        f"- Risk band: **{health['risk_band']}** "
        f"(overrun risk {health['overrun_risk']:.0%})\n"
        f"- Primary delay driver: **{health.get('primary_delay_driver')}**\n"
        f"- Action: {health['recommended_action']}\n"
        f"- AWP lines open: {awp_count}; "
        f"{feature_label('eng_queue_age_days')}: {health.get('eng_queue_age_days')}; "
        f"{feature_label('prior_oa_findings')}: {health.get('prior_oa_findings')}\n"
        f"- Recent history: {latest}\n"
        f"- Tech data / disposition guidance: {doc_snip}...\n\n"
        f"_Advisory only — cite authoritative NAMP tech data before acting._\n\n"
        f"Tools used: `get_aircraft_health`, `get_parts_awp_status`, `search_tech_data`."
    )


def _mock_explain(buno: str, component: str, health: dict, hits: list[dict]) -> str:
    top = health.get("contributing_signals", [])[:3]
    signal_lines = "\n".join(
        f"- **{feature_label(s['feature'])}** = {s['value']} "
        f"(baseline {s['baseline']}, concern {s['concern_score']})"
        for s in top
    )
    citations = "\n".join(
        f"- [{h['source']}#{h['chunk_id']}] {h['text'][:160]}..." for h in hits[:3]
    )
    return (
        f"**Why BUNO {buno} / {component} is {health['risk_band']}** "
        f"(mock RAG explanation)\n\n"
        f"Predicted PMI overrun risk is **{health['overrun_risk']:.0%}**. "
        f"Primary delay driver: **{health.get('primary_delay_driver')}**.\n\n"
        f"Top contributing signals:\n{signal_lines}\n\n"
        f"Grounded in notional tech data / dispositions:\n{citations}\n\n"
        f"_NAMP reminder: AI guidance is advisory; maintenance actions require "
        f"authoritative technical data citation._"
    )


def _openai_recommend(buno: str, component: str, payload: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "You are an FRCE MV-22 PMI production advisor. Using ONLY the tool JSON, "
        "recommend the next action for planners/production controllers in <=120 words. "
        "Cite delay drivers (O&A, AWP, engineering), signals, and doc sources. "
        "Remind that guidance is advisory under NAMP.\n\n"
        f"BUNO: {buno}\nSystem: {component}\n"
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
        "Explain MV-22 PMI overrun risk using the context and signals. "
        "Be concise (<=150 words), cite sources by filename, and note NAMP advisory limits.\n\n"
        f"Query: {query}\nSignals: {json.dumps(focus.get('contributing_signals', [])[:4])}\n"
        f"Context:\n{context}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
