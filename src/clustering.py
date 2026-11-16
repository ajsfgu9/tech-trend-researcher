"""
Stage 4b: Automated Content Briefs & Editorial Decision Matrix
Synthesizes LongitudinalTrendSignal objects (burst trajectory + causal propagation +
Executive Editor evaluation) into actionable content briefs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.models import LongitudinalTrendSignal


def _decision_tier(score: float) -> tuple[str, str]:
    if score >= 85.0:
        return "Lead Story / Deep-Dive Essay", "Dedicate a full 1,500-2,500 word post. High information asymmetry and strong conflict."
    if score >= 70.0:
        return "Fast-Turnaround Thread / Commentary", "Publish a 5-8 post breakdown on X or LinkedIn within 4 hours while debate is active."
    if score >= 50.0:
        return "Curated Newsletter Segment", "Add as a 200-word analyzed link in a weekly roundup."
    return "Archive to Knowledge Graph", "Background technical context. Save for entity cross-referencing; do not publish standalone."


def format_longitudinal_briefs(
    signals: list[LongitudinalTrendSignal],
    top_n: int = 12,
) -> tuple[list[dict[str, Any]], str]:
    now = datetime.now(timezone.utc)
    date_tag = now.strftime("%Y%m%d")

    briefs: list[dict[str, Any]] = []
    for idx, sig in enumerate(signals[:top_n], start=1):
        lead = max(sig.candidates, key=lambda c: c.editorial_score or c.m_viral)
        slug = re.sub(r"[^a-z0-9]+", "_", sig.bottleneck_concept.lower()).strip("_")[:28]

        final_editorial_score = round(
            min(99.5, 0.75 * sig.unified_trend_index + 0.25 * (lead.editorial_score or sig.unified_trend_index)),
            2,
        )
        action_required, strategy = _decision_tier(final_editorial_score)

        hotspots = [
            f"{c.source_name} ({int(c.comments_count)} comments)" if c.comments_count > 0 else c.source_name
            for c in sig.candidates
        ]
        ev = lead.editorial_eval
        nf = ev.get("narrative_framework", {})

        briefs.append(
            {
                "editorial_id": f"brief_{date_tag}_{slug or idx}",
                "headline_recommendation": lead.title,
                "bottleneck_concept": sig.bottleneck_concept,
                "primary_entity": sig.primary_entity,
                "vertical": sig.vertical,
                "editorial_score": final_editorial_score,
                "decision_tier": action_required,
                "editorial_verdict": ev.get("editorial_verdict", "PUBLISH_IMMEDIATELY" if final_editorial_score >= 70 else "MONITOR"),
                "longitudinal_trend_proof": {
                    "unified_trend_index": sig.unified_trend_index,
                    "longitudinal_structural_score_80pct": sig.longitudinal_score,
                    "reserved_item_subscore_20pct": sig.reserved_item_subscore,
                    "burst_regime": sig.burst_regime,
                    "ewma_velocity_z": sig.ewma_velocity_z,
                    "acceleration_delta": sig.acceleration_delta,
                    "cusum_regime_stat": sig.cusum_stat,
                    "origin_layer": sig.origin_layer,
                    "causal_propagation_chain": sig.active_layers,
                    "causal_propagation_score": sig.causal_propagation_score,
                    "mainstream_saturation_ratio": sig.mainstream_saturation_ratio,
                    "co_occurrence_lift": sig.co_occurrence_lift,
                    "is_novel_coupling": sig.is_novel_coupling,
                },
                "virality_metrics": {
                    "total_engagement_velocity_z": sig.ewma_velocity_z,
                    "information_asymmetry_ia": lead.information_asymmetry_ia,
                    "cross_platform_dispersion": len(sig.active_layers),
                    "discussion_hotspots": hotspots,
                },
                "verified_entities": sorted({e for c in sig.candidates for e in c.verified_entities}),
                "narrative_framework": {
                    "the_hook": nf.get("the_hook", ""),
                    "the_misconception": nf.get("the_misconception", ""),
                    "the_engineering_truth": nf.get("the_engineering_truth", ""),
                    "the_conflict": nf.get("the_conflict", ""),
                },
                "story_angles": ev.get("story_angles", []),
                "anchor_visual": ev.get("anchor_visual", ""),
                "key_villain_or_bottleneck": ev.get("key_villain_or_bottleneck", sig.bottleneck_concept),
                "content_structure": [
                    f"1. The Setup: Why {sig.bottleneck_concept} moved from {sig.origin_layer} into {', '.join(sig.active_layers)}.",
                    f"2. The Bottleneck: {ev.get('key_villain_or_bottleneck', sig.bottleneck_concept)}.",
                    "3. The Economics: Unit-economics & supply-chain trade-offs across vendors.",
                    "4. The Takeaway: Implications for deployment timelines and margins.",
                ],
                "recommended_formats": {
                    "long_form_essay": "1,800 words on Substack / technical blog" if final_editorial_score >= 85 else "Optional deep-dive",
                    "x_thread": "5-8 part visual thread anchored by schematic/table",
                    "linkedin_post": "Condensed summary targeting hardware investors and infrastructure buyers",
                },
                "content_strategy_note": strategy,
                "source_urls": [c.url for c in sig.candidates if c.url],
            }
        )

    lines = [
        "| Rank | Editorial Score | Regime | Causal Chain | Bottleneck Concept | Headline Recommendation | Action Required |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, b in enumerate(briefs, start=1):
        lp = b["longitudinal_trend_proof"]
        chain = " -> ".join(lp["causal_propagation_chain"])
        lines.append(
            f"| **{rank:02d}** | **{b['editorial_score']:.1f}** | `{lp['burst_regime']}` (CUSUM {lp['cusum_regime_stat']:.2f}) | "
            f"{chain} | **{b['bottleneck_concept']}** | {b['headline_recommendation'][:50]} | {b['decision_tier']} |"
        )

    return briefs, "\n".join(lines)
