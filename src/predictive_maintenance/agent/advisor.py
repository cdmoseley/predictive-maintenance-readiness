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
    """Prescriptive fallback narrative — concrete next steps, not a diagnosis essay."""
    health = payload["aircraft_health"]
    parts = payload.get("parts_awp", {})
    awp_count = parts.get("awp_count", 0)
    doc = ""
    if payload.get("manual_hits"):
        doc = payload["manual_hits"][0].get("source", "local PMI guidance")
    driver = health.get("primary_delay_driver", "the leading delay driver")
    action = health.get("recommended_action", "re-plan the work package within 48 hours")
    return (
        f"For BUNO {buno} ({component}), prioritize mitigation now while the aircraft "
        f"sits in {health.get('status', 'PMI')} at {health.get('overrun_risk', 0):.0%} "
        f"overrun risk. The priority driver to attack is {driver}; treat that as the "
        f"first production action rather than waiting for calendar completion.\n\n"
        f"Next step: {action}. With {awp_count} AWP line(s) and "
        f"{health.get('eng_queue_age_days', 0):.0f} days on the engineering queue, "
        f"align material expediting and disposition chase in the same stand-up. "
        f"Supporting process language is available in {doc or 'mv22_pmi_guidance.md'}. "
        f"This guidance is advisory under NAMP and is not a substitute for "
        f"authoritative technical data."
    )


def _mock_explain(buno: str, component: str, health: dict, hits: list[dict]) -> str:
    """Diagnostic fallback narrative — why at risk; no action plan."""
    top = health.get("contributing_signals", [])[:3]
    signal_bits = []
    for s in top:
        signal_bits.append(
            f"{feature_label(s['feature'])} at {s['value']} "
            f"(baseline {s['baseline']})"
        )
    signal_txt = "; ".join(signal_bits) if signal_bits else "elevated delay-driver signals"
    src = hits[0]["source"] if hits else "local PMI guidance"
    return (
        f"BUNO {buno} is in risk band {health['risk_band']} with a predicted PMI "
        f"overrun probability of {health['overrun_risk']:.0%} on the {component} focus "
        f"area. The primary delay driver showing up in the fused line picture is "
        f"{health.get('primary_delay_driver')}, consistent with the aircraft's current "
        f"status of {health.get('status', 'in work')}.\n\n"
        f"What is driving that assessment is {signal_txt}. Similar patterns are "
        f"described in {src}. This explanation is diagnostic only — it does not "
        f"prescribe the next production move. Guidance remains advisory under NAMP "
        f"and must be confirmed against authoritative technical data."
    )


def _format_llm_markdown(text: str) -> str:
    """Coerce model output to panel-friendly narrative prose.

    Some models echo tool JSON or wrap answers in {"next_action": {...}}.
    Prefer plain paragraphs; if JSON is returned, unwrap into connected prose.
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

    return _json_payload_to_prose(data) or raw


def _json_payload_to_prose(data: Any) -> str:
    """Render common structured advisor payloads as short narrative paragraphs."""
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, list):
        parts = [str(item).strip() for item in data if str(item).strip()]
        return " ".join(parts)

    if not isinstance(data, dict):
        return str(data)

    # Unwrap common envelopes while keeping sibling fields
    for key in ("next_action", "recommendation", "result", "answer", "explanation"):
        if key in data and isinstance(data[key], dict):
            data = {**data, **data[key]}
            break
        if key in data and isinstance(data[key], str) and key in {
            "recommendation",
            "answer",
            "explanation",
        }:
            # Keep going so we can still pull driver/sources from siblings
            pass

    rec = (
        data.get("recommendation")
        or data.get("next_action")
        or data.get("action")
        or data.get("recommended_action")
        or data.get("summary")
        or data.get("headline")
    )
    if isinstance(rec, dict):
        rec = (
            rec.get("recommendation")
            or rec.get("action")
            or rec.get("text")
            or rec.get("summary")
        )
    if isinstance(rec, str):
        rec = rec.strip()
    else:
        rec = ""

    driver = (
        data.get("primary_delay_driver")
        or data.get("delay_driver")
        or data.get("primary_driver")
    )
    drivers = data.get("delay_drivers")
    if not driver and isinstance(drivers, dict):
        driver = drivers.get("primary") or drivers.get("primary_delay_driver")
        if not driver:
            true_keys = [k for k, v in drivers.items() if v]
            driver = ", ".join(true_keys) if true_keys else None
    if isinstance(drivers, list) and not driver:
        driver = ", ".join(str(x) for x in drivers)

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
    elif isinstance(cites, str) and cites.strip():
        cite_txt = cites.strip()
    else:
        cite_txt = ""

    namp = data.get("namp_note") or data.get("advisory") or data.get("disclaimer")
    if not namp:
        namp = (
            "This guidance is advisory under NAMP and is not a substitute for "
            "authoritative technical data."
        )

    paras: list[str] = []
    situ = data.get("situation") or data.get("context") or data.get("status")
    if situ:
        paras.append(str(situ).strip())

    if driver and rec:
        paras.append(
            f"The primary delay driver is {driver}. The concrete next action is {rec}."
        )
    elif driver:
        paras.append(f"The primary delay driver is {driver}.")
    elif rec:
        paras.append(str(rec))

    if cite_txt:
        paras.append(
            f"Supporting guidance is drawn from {cite_txt}."
        )

    paras.append(str(namp).strip())

    # Opaque payload fallback: join scalar fields into one paragraph
    if len(paras) <= 1 and not rec and not driver:
        bits = []
        for k, v in data.items():
            if isinstance(v, (str, int, float)) and str(v).strip():
                bits.append(f"{k.replace('_', ' ')}: {v}")
        if bits:
            paras = [". ".join(bits) + ".", str(namp).strip()]

    return "\n\n".join(p for p in paras if p).strip()


def _openai_recommend(
    buno: str, component: str, payload: dict, api_key: str | None = None
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or openai_api_key())
    prompt = (
        "You are writing a PRESCRIPTIVE next-action brief for an FRCE MV-22 "
        "production lead.\n"
        "Using ONLY the tool data below, write 1–3 short narrative paragraphs "
        "(about 90–130 words).\n\n"
        "This button is ONLY for recommending the next action. Lead with the "
        "concrete next maintenance, supply, or engineering step and who should "
        "own it. Mention the primary delay driver in one clause as justification, "
        "cite a doc filename inline, and close with a NAMP advisory note.\n\n"
        "Do NOT explain the full risk picture. Do NOT list contributing signals. "
        "Do NOT write a diagnostic 'why late' essay. Do NOT say 'the aircraft is "
        "at risk because…' for more than one short clause.\n\n"
        "Hard rules: NO JSON, NO YAML, NO code fences, NO bullet lists, NO numbered lists. "
        "Write connected sentences only.\n\n"
        f"BUNO: {buno}\nSystem focus: {component}\n"
        f"TOOL_DATA:\n{json.dumps(payload, default=str)[:6000]}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce prescriptive narrative briefs only: concrete next "
                    "actions for production. Never diagnose without an action. "
                    "Never output JSON, bullets, or numbered lists."
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
        "You are writing a DIAGNOSTIC risk explanation for an FRCE MV-22 "
        "production lead and depot engineer.\n"
        "Using ONLY the context and signals below, write 1–3 short narrative "
        "paragraphs (about 90–130 words) explaining WHY this BUNO/component is "
        "at overrun risk.\n\n"
        "This button is ONLY for explaining overrun risk. Cover: current risk "
        "picture → primary delay driver and what in the history/features is "
        "elevating risk → cite source filenames inline → NAMP advisory close.\n\n"
        "Do NOT recommend next steps, priorities, expedites, shop tasking, or "
        "'what to do next'. Do NOT include an action plan. Stay explanatory.\n\n"
        "Hard rules: NO JSON, NO YAML, NO code fences, NO bullet lists, NO numbered lists. "
        "Write connected sentences only.\n\n"
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
                    "You produce diagnostic narrative explanations only: why risk is "
                    "elevated. Never prescribe next actions. "
                    "Never output JSON, bullets, or numbered lists."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _format_llm_markdown(resp.choices[0].message.content or "")
