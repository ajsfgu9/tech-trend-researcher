"""
Stage 4a: LLM Dynamic Refinement Factor (Phi_LLM)
Evaluates Technical Substance (S), Industry Impact (I), and Lifecycle Phase (L):
    Phi_LLM = (0.4 * S + 0.6 * I) * L
    Final Score(i) = M_i * Phi_LLM
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from src.models import NormalizedCandidate

log = logging.getLogger(__name__)

LIFECYCLE_MULTIPLIERS = {
    "emerging": 1.30,
    "peak_viral": 0.80,
    "trailing": 0.50,
}


def evaluate_with_llm(cand: NormalizedCandidate, provider: str = "none") -> None:
    if provider == "none":
        s = min(1.0, 0.45 + 0.12 * len(cand.verified_entities) + 0.20 * cand.source_authority_ws)
        i = min(1.5, 0.55 + 0.15 * cand.dispersion_k + 0.10 * len(cand.verified_entities))
        phase = "emerging" if cand.dispersion_k <= 2 else "peak_viral"
        l_mult = 1.20 if phase == "emerging" else 0.85
        phi = (0.4 * s + 0.6 * i) * l_mult
        cand.llm_audit = {
            "technical_substance": round(s, 2),
            "industry_impact": round(i, 2),
            "lifecycle_phase": phase,
            "lifecycle_multiplier": l_mult,
            "contrarian_flag": False,
            "rationale": f"Deterministic audit across {cand.dispersion_k} platform(s) with entities {', '.join(cand.verified_entities) or cand.vertical}.",
            "critical_implication": f"Impacts {cand.vertical} supply-chain and technical roadmap execution.",
        }
        cand.llm_multiplier_phi = round(phi, 3)
        cand.final_score = round(cand.base_score_mi * phi, 2)
        return

    prompt = (
        "Evaluate this tech/semiconductor candidate in strict JSON.\n"
        f"Title: {cand.title}\n"
        f"Source: {cand.source_name} ({cand.source_platform})\n"
        f"Vertical: {cand.vertical}\n"
        f"Entities: {cand.verified_entities}\n"
        f"Summary: {cand.body[:500]}\n\n"
        "Return JSON with exact keys:\n"
        '  "technical_substance": float in [0.2, 1.0],\n'
        '  "industry_impact": float in [0.3, 1.5],\n'
        '  "lifecycle_phase": "emerging" | "peak_viral" | "trailing",\n'
        '  "lifecycle_multiplier": float (1.3 for emerging, 0.8 for peak_viral, 0.5 for trailing),\n'
        '  "contrarian_flag": bool,\n'
        '  "rationale": 1-2 sentence technical/supply-chain summary,\n'
        '  "critical_implication": 1 sentence on downstream value-chain impact.'
    )

    try:
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=25,
            )
            resp.raise_for_status()
            data: dict[str, Any] = json.loads(resp.json()["choices"][0]["message"]["content"])
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
                timeout=25,
            )
            resp.raise_for_status()
            data = json.loads(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
        else:
            evaluate_with_llm(cand, "none")
            return

        s = max(0.2, min(1.0, float(data.get("technical_substance", 0.6))))
        i = max(0.3, min(1.5, float(data.get("industry_impact", 0.8))))
        phase = str(data.get("lifecycle_phase", "emerging"))
        l_mult = max(0.5, min(1.3, float(data.get("lifecycle_multiplier", LIFECYCLE_MULTIPLIERS.get(phase, 1.0)))))
        phi = (0.4 * s + 0.6 * i) * l_mult

        cand.llm_audit = data
        cand.llm_multiplier_phi = round(phi, 3)
        cand.final_score = round(cand.base_score_mi * phi, 2)
    except Exception as exc:
        log.warning("LLM audit fallback for %s: %s", cand.event_id, exc)
        evaluate_with_llm(cand, "none")
