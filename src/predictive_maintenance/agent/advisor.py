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


def apply_openai_secrets() -> None:
    """Copy Streamlit secrets into os.environ (safe to call repeatedly).

    Import-time bridges often fail on Streamlit Cloud before secrets are ready.
    Call this at app start and immediately before any OpenAI path.
    """
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
            val = None
            try:
                # AttrDict supports both .get and __getitem__
                if hasattr(secrets, "get"):
                    val = secrets.get(key)
                if val is None:
                    val = secrets[key]
            except Exception:
                continue
            if val is None:
                continue
            text = str(val).strip()
            # Ignore empty / placeholder mistakes like literal "Key"
            if text and text.lower() not in {"key", "none", "null", "changeme"}:
                os.environ[key] = text
    except Exception:
        pass


def openai_api_key() -> str | None:
    """Return a usable OpenAI API key, or None if mock mode should run."""
    apply_openai_secrets()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key or key.lower() in {"key", "none", "null", "changeme"}:
        return None
    return key


def recommend_next_action(buno: str, component: str | None = None) -> dict[str, Any]:
    """Run tools then produce a grounded recommendation (live or mock LLM)."""
    apply_openai_secrets()
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

    api_key = openai_api_key()
    if api_key:
        # Do not catch auth/API errors — surface 401/etc. so Cloud secrets can be debugged
        recommendation = _openai_recommend(buno, focus, tool_payload, api_key=api_key)
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
    apply_openai_secrets()
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

    api_key = openai_api_key()
    if api_key:
        # Do not catch auth/API errors — surface 401/etc. so Cloud secrets can be debugged
        text = _openai_explain(query, tech["hits"], focus_row, api_key=api_key)
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
        f"(mock advisor — set Streamlit secret `OPENAI_API_KEY` for live GenAI)\n\n"
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


def _format_llm_markdown(text: str) -> str:
    """Coerce model output to panel-friendly markdown.

    Some models echo tool JSON or wrap answers in {"next_action": {...}}.
    Prefer plain prose; if JSON is returned, extract recommendation fields.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # Strip optional ```json fences
    fenced = raw
    if fenced.startswith("```"):
        lines = fenced.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()

    candidate = fenced
    if not (candidate.startswith("{") or candidate.startswith("[")):
        # Maybe JSON is embedded after a short preamble
        for opener in ("{", "["):
            idx = candidate.find(opener)
            if idx >= 0 and candidate.rstrip().endswith("}" if opener == "{" else "]"):
                candidate = candidate[idx:]
                break
        else:
            return raw

    try:
        data = json.loads(candidate)
    except Exception:
        return raw

    return _json_payload_to_markdown(data) or raw


def _json_payload_to_markdown(data: Any) -> str:
    """Render common structured advisor payloads as concise bullets."""
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, list):
        bullets = [f"- {item}" for item in data if item]
        return "\n".join(bullets)

    if not isinstance(data, dict):
        return str(data)

    # Unwrap common envelopes
    for key in ("next_action", "recommendation", "result", "answer", "explanation"):
        if key in data and isinstance(data[key], (dict, str, list)):
            nested = data[key]
            if isinstance(nested, str) and key in {"recommendation", "answer", "explanation"}:
                # Keep sibling fields if present
                break
            if isinstance(nested, dict):
                data = {**data, **nested}
                break
            if isinstance(nested, str):
                return nested.strip()

    lines: list[str] = []
    title = data.get("title") or data.get("summary") or data.get("headline")
    if title:
        lines.append(f"**{title}**")
        lines.append("")

    rec = (
        data.get("recommendation")
        or data.get("next_action")
        or data.get("action")
        or data.get("recommended_action")
    )
    if isinstance(rec, dict):
        rec = (
            rec.get("recommendation")
            or rec.get("action")
            or rec.get("text")
            or rec.get("summary")
        )
    if rec:
        lines.append(f"- **Next action:** {rec}")

    driver = (
        data.get("primary_delay_driver")
        or data.get("delay_driver")
        or data.get("primary_driver")
    )
    drivers = data.get("delay_drivers")
    if not driver and isinstance(drivers, dict):
        # Pick loudest / primary if marked
        driver = drivers.get("primary") or drivers.get("primary_delay_driver")
        if not driver:
            # e.g. {"AWP": true, "Engineering": false}
            true_keys = [k for k, v in drivers.items() if v]
            driver = ", ".join(true_keys) if true_keys else None
    if isinstance(drivers, list) and not driver:
        driver = ", ".join(str(x) for x in drivers)
    if driver:
        lines.append(f"- **Primary delay driver:** {driver}")

    cites = (
        data.get("citations")
        or data.get("sources")
        or data.get("doc_filenames")
        or data.get("documents")
    )
    if isinstance(cites, list) and cites:
        cite_txt = ", ".join(
            (c.get("source") if isinstance(c, dict) else str(c)) for c in cites
        )
        lines.append(f"- **Sources:** {cite_txt}")
    elif isinstance(cites, str) and cites.strip():
        lines.append(f"- **Sources:** {cites.strip()}")

    namp = data.get("namp_note") or data.get("advisory") or data.get("disclaimer")
    if namp:
        lines.append(f"- _{namp}_")
    else:
        lines.append(
            "- _Advisory only under NAMP — cite authoritative tech data before acting._"
        )

    # If we only got opaque keys, fall back to compact bullets of remaining scalars
    if len(lines) <= 1:
        for k, v in data.items():
            if isinstance(v, (str, int, float)) and str(v).strip():
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    return "\n".join(lines).strip()


def _openai_recommend(
    buno: str, component: str, payload: dict, api_key: str | None = None
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or openai_api_key())
    prompt = (
        "You are briefing an FRCE MV-22 production lead / Commanding Officer.\n"
        "Using ONLY the tool data below, write a concise plain-English markdown brief "
        "(<=120 words). Use short bullets. Do NOT return JSON, YAML, or code fences.\n\n"
        "Must include:\n"
        "1) Primary delay driver (Over-and-above, AWP, or Engineering)\n"
        "2) One concrete next action for the production team\n"
        "3) Cite supporting doc filenames from the tool hits (e.g. disposition_archive.md)\n"
        "4) End with a one-line NAMP advisory note (AI guidance is not authoritative tech data)\n\n"
        f"BUNO: {buno}\nSystem focus: {component}\n"
        f"TOOL_DATA:\n{json.dumps(payload, default=str)[:6000]}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Respond in plain-English markdown only. Never output a JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _format_llm_markdown(resp.choices[0].message.content or "")


def _openai_explain(
    query: str, hits: list[dict], focus: dict, api_key: str | None = None
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or openai_api_key())
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    prompt = (
        "You are briefing an FRCE depot engineer and production lead.\n"
        "Explain MV-22 PMI overrun risk in concise plain-English markdown (<=150 words). "
        "Use short bullets. Do NOT return JSON, YAML, or code fences.\n\n"
        "Must include:\n"
        "1) Primary delay driver\n"
        "2) Why the aircraft is at risk (top signals in plain language)\n"
        "3) Cite source filenames from the context\n"
        "4) One-line NAMP advisory note\n\n"
        f"Query: {query}\n"
        f"Signals: {json.dumps(focus.get('contributing_signals', [])[:4])}\n"
        f"Context:\n{context}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Respond in plain-English markdown only. Never output a JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _format_llm_markdown(resp.choices[0].message.content or "")
