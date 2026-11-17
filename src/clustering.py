"""
Stage 4b: Automated Content Briefs & Editorial Decision Matrix
Synthesizes LongitudinalTrendSignal objects + X/Twitter Breakout Playbooks
so every discovered trend is engineered to become a trend on X/Twitter.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.models import LongitudinalTrendSignal
from src.x_breakout_engine import compute_x_breakout_potential


def _decision_tier(score: float) -> tuple[str, str]:
    if score >= 85.0:
        return "Lead Story + Flagship X Thread", "Launch an 8-part visual X thread + 1,800-word deep dive during the First-Mover X Window."
    if score >= 70.0:
        return "Fast-Turnaround X Breakout Thread", "Publish a 5-7 post X thread with spec/margin card within 2 hours to capture reply/bookmark velocity."
    if score >= 50.0:
        return "Single Contrarian X Post + Chart", "Post a single high-density X chart post with a polarizing technical question."
    return "Archive to Knowledge Graph", "Background technical context. Save for entity cross-referencing; do not post standalone."


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

        all_entities = sorted({e for c in sig.candidates for e in c.verified_entities})
        x_mentions = sum(1 for c in sig.candidates if c.source_platform == "x_twitter")
        x_breakout = compute_x_breakout_potential(
            text=" ".join(f"{c.title} {c.body}" for c in sig.candidates),
            entities=all_entities,
            bottleneck_concept=sig.bottleneck_concept,
            corroboration_count=len(sig.candidates),
            has_primary_source=("primary_engineering" in sig.active_layers),
            x_mentions_count=x_mentions,
        )

        final_editorial_score = round(
            min(
                99.5,
                0.55 * sig.unified_trend_index
                + 0.25 * x_breakout["x_breakout_score"]
                + 0.20 * (lead.editorial_score or sig.unified_trend_index),
            ),
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
                "x_twitter_breakout_engine": x_breakout,
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
                "verified_entities": all_entities,
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
                    "x_thread": x_breakout["x_catalyst_playbook"]["optimal_format"],
                    "linkedin_post": "Condensed summary targeting hardware investors and infrastructure buyers",
                },
                "content_strategy_note": strategy,
                "source_urls": [c.url for c in sig.candidates if c.url],
            }
        )

    lines = [
        "| Rank | Editorial Score | X Breakout Score | X Window | Regime | Bottleneck Concept | Cashtags | Action Required |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, b in enumerate(briefs, start=1):
        lp = b["longitudinal_trend_proof"]
        xb = b["x_twitter_breakout_engine"]
        tags = " ".join(xb["target_x_cashtags"]) or "-"
        lines.append(
            f"| **{rank:02d}** | **{b['editorial_score']:.1f}** | **{xb['x_breakout_score']:.1f}** (`{xb['will_trend_on_x']}`) | "
            f"`{xb['x_arbitrage_window']}` | `{lp['burst_regime']}` | **{b['bottleneck_concept']}** | {tags} | {b['decision_tier']} |"
        )

    return briefs, "\n".join(lines)
