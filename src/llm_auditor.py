"""
Stage 4a: "Executive Editor" LLM Evaluator (Phi_editorial)
Evaluates Narrative Tension (T), Counter-Intuitive Factor (C), and Visual/Exploratory Depth (E):
    Phi_editorial = (0.35 * T + 0.40 * C + 0.25 * E) * Hook_Multiplier (1.0 to 1.5)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from src.models import NormalizedCandidate

log = logging.getLogger(__name__)


def evaluate_as_executive_editor(cand: NormalizedCandidate, provider: str = "none") -> None:
    if provider == "none":
        t = min(1.0, 0.45 + 0.15 * (cand.dispersion_k - 1) + (0.20 if cand.comments_count >= 50 else 0.05))
        c = min(1.0, 0.40 + 0.50 * cand.information_asymmetry_ia)
        e = min(1.0, 0.45 + 0.12 * len(cand.verified_entities))
        hook_mult = 1.35 if cand.information_asymmetry_ia >= 0.55 else 1.15
        phi = (0.35 * t + 0.40 * c + 0.25 * e) * hook_mult
        raw_score = cand.m_viral * phi
        ed_score = round(min(99.5, max(25.0, raw_score * 38.0)), 2)

        verdict = "PUBLISH_IMMEDIATELY" if ed_score >= 70.0 else ("MONITOR" if ed_score >= 50.0 else "DISCARD")
        entities_str = ", ".join(cand.verified_entities[:3]) or cand.vertical
        cand.phi_editorial = round(phi, 3)
        cand.editorial_score = ed_score
        cand.editorial_eval = {
            "editorial_score": ed_score,
            "story_angles": [
                {"type": "Contrarian / Debunk", "hook": f"Everyone is focused on headline specs, but {cand.title} exposes the real constraint in {entities_str}."},
                {"type": "David vs. Goliath", "hook": f"How {entities_str} supply-chain bottlenecks shift leverage across the {cand.vertical} stack."},
                {"type": "Under-the-Hood Teardown", "hook": f"The physical and unit-economic mechanics behind {cand.title}."},
            ],
            "anchor_visual": f"Unit-economics & bottleneck schematic comparing {entities_str} throughput vs. yield/cost limits.",
            "key_villain_or_bottleneck": f"{entities_str} engineering & supply-chain constraint",
            "editorial_verdict": verdict,
            "narrative_framework": {
                "the_hook": f"Public commentary around {cand.title} misses the underlying {cand.vertical} bottleneck.",
                "the_misconception": f"Mainstream coverage treats {entities_str} scaling as purely a demand or headline-node story.",
                "the_engineering_truth": f"Primary signals ({cand.source_name}) point to physical yield, packaging, or architecture constraints.",
                "the_conflict": f"Incumbent roadmap assumptions vs. real-world {entities_str} execution limits.",
            },
        }
        return

    prompt = (
        "You are the Executive Editor of a premier deep-tech publication (in the vein of SemiAnalysis and Stratechery).\n"
        "Evaluate the following trend candidate for its editorial value:\n\n"
        "TREND PAYLOAD:\n"
        f"- Cluster Title: {cand.title}\n"
        f"- Source: {cand.source_name} ({cand.editorial_layer})\n"
        f"- Summary: {cand.body[:600]}\n"
        f"- Extracted Entities: {cand.verified_entities}\n\n"
        "Evaluate across three editorial criteria (0.0 to 1.0):\n"
        "1. Narrative Tension (T): Active conflict, rivalry, or corporate blunder.\n"
        "2. Counter-Intuitive Factor (C): Debunks a common assumption or reveals an unexpected bottleneck.\n"
        "3. Visual/Exploratory Depth (E): Can be anchored by a concrete schematic, unit-economic chart, or flow diagram.\n"
        "Calculate Phi_editorial = (0.35 * T + 0.40 * C + 0.25 * E) * Hook_Multiplier (1.0 to 1.5).\n\n"
        "OUTPUT FORMAT (JSON ONLY):\n"
        "{\n"
        '  "editorial_score": float (0-100),\n'
        '  "phi_editorial": float,\n'
        '  "story_angles": [\n'
        '    {"type": "Contrarian / Debunk", "hook": "string"},\n'
        '    {"type": "David vs. Goliath", "hook": "string"},\n'
        '    {"type": "Under-the-Hood Teardown", "hook": "string"}\n'
        "  ],\n"
        '  "anchor_visual": "Description of the essential chart or architectural diagram needed",\n'
        '  "key_villain_or_bottleneck": "string",\n'
        '  "editorial_verdict": "PUBLISH_IMMEDIATELY" | "MONITOR" | "DISCARD",\n'
        '  "narrative_framework": {\n'
        '    "the_hook": "string",\n'
        '    "the_misconception": "string",\n'
        '    "the_engineering_truth": "string",\n'
        '    "the_conflict": "string"\n'
        "  }\n"
        "}"
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
            evaluate_as_executive_editor(cand, "none")
            return

        phi = float(data.get("phi_editorial", 1.15))
        ed_score = float(data.get("editorial_score", min(99.5, cand.m_viral * phi * 38.0)))
        cand.phi_editorial = round(phi, 3)
        cand.editorial_score = round(max(0.0, min(100.0, ed_score)), 2)
        cand.editorial_eval = data
    except Exception as exc:
        log.warning("Executive Editor LLM fallback for %s: %s", cand.event_id, exc)
        evaluate_as_executive_editor(cand, "none")
