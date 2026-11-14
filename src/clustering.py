"""
Stage 4b: Automated Content Briefs & Editorial Decision Matrix
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.models import NormalizedCandidate


def _tokens(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9\-\.]{3,}", text.lower())}


def _decision_tier(score: float) -> tuple[str, str]:
    if score >= 85.0:
        return "Lead Story / Deep-Dive Essay", "Dedicate a full 1,500-2,500 word post. High information asymmetry and strong conflict."
    if score >= 70.0:
        return "Fast-Turnaround Thread / Commentary", "Publish a 5-8 post breakdown on X or LinkedIn within 4 hours while debate is active."
    if score >= 50.0:
        return "Curated Newsletter Segment", "Add as a 200-word analyzed link in a weekly roundup."
    return "Archive to Knowledge Graph", "Background technical context. Save for entity cross-referencing; do not publish standalone."


def cluster_and_format_briefs(
    candidates: list[NormalizedCandidate],
    top_n: int = 12,
) -> tuple[list[dict[str, Any]], str]:
    ranked = sorted(candidates, key=lambda c: c.editorial_score, reverse=True)
    clusters: list[list[NormalizedCandidate]] = []
    cluster_tokens: list[set[str]] = []

    for cand in ranked:
        toks = _tokens(cand.title)
        matched = -1
        for idx, c_toks in enumerate(cluster_tokens):
            union = toks | c_toks
            if union and (len(toks & c_toks) / len(union)) >= 0.16:
                matched = idx
                break
        if matched >= 0:
            clusters[matched].append(cand)
            cluster_tokens[matched].update(toks)
        else:
            clusters.append([cand])
            cluster_tokens.append(set(toks))

    now = datetime.now(timezone.utc)
    date_tag = now.strftime("%Y%m%d")

    briefs: list[dict[str, Any]] = []
    for idx, group in enumerate(clusters[:top_n], start=1):
        lead = group[0]
        slug = re.sub(r"[^a-z0-9]+", "_", lead.title.lower()).strip("_")[:28]
        hotspots = [
            f"{c.source_name} ({int(c.comments_count)} comments)" if c.comments_count > 0 else c.source_name
            for c in group
        ]
        action_required, strategy = _decision_tier(lead.editorial_score)
        ev = lead.editorial_eval
        nf = ev.get("narrative_framework", {})

        briefs.append(
            {
                "editorial_id": f"brief_{date_tag}_{slug or idx}",
                "headline_recommendation": lead.title,
                "vertical": lead.vertical,
                "editorial_score": lead.editorial_score,
                "decision_tier": action_required,
                "editorial_verdict": ev.get("editorial_verdict", "MONITOR"),
                "virality_metrics": {
                    "total_engagement_velocity_z": lead.velocity_vi,
                    "information_asymmetry_ia": lead.information_asymmetry_ia,
                    "cross_platform_dispersion": max(lead.dispersion_k, len({c.source_platform for c in group})),
                    "discussion_hotspots": hotspots,
                },
                "verified_entities": sorted({e for c in group for e in c.verified_entities}),
                "narrative_framework": {
                    "the_hook": nf.get("the_hook", ""),
                    "the_misconception": nf.get("the_misconception", ""),
                    "the_engineering_truth": nf.get("the_engineering_truth", ""),
                    "the_conflict": nf.get("the_conflict", ""),
                },
                "story_angles": ev.get("story_angles", []),
                "anchor_visual": ev.get("anchor_visual", ""),
                "key_villain_or_bottleneck": ev.get("key_villain_or_bottleneck", ""),
                "content_structure": [
                    "1. The Setup: Public narrative vs. what technical insiders are actually debating.",
                    f"2. The Bottleneck: {ev.get('key_villain_or_bottleneck', 'Core physical and architectural constraint')}.",
                    "3. The Economics: Unit-economics & supply-chain trade-offs across vendors.",
                    "4. The Takeaway: Implications for deployment timelines and margins.",
                ],
                "recommended_formats": {
                    "long_form_essay": "1,800 words on Substack / technical blog" if lead.editorial_score >= 85 else "Optional deep-dive",
                    "x_thread": "5-8 part visual thread anchored by schematic/table",
                    "linkedin_post": "Condensed summary targeting hardware investors and infrastructure buyers",
                },
                "content_strategy_note": strategy,
                "source_urls": [c.url for c in group if c.url],
            }
        )

    lines = [
        "| Rank | Editorial Score | Action Required | Headline Recommendation | Vertical | Hotspots | Key Bottleneck |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, b in enumerate(briefs, start=1):
        hs = ", ".join(b["virality_metrics"]["discussion_hotspots"][:2])
        lines.append(
            f"| **{rank:02d}** | **{b['editorial_score']:.1f}** | {b['decision_tier']} | "
            f"**{b['headline_recommendation'][:55]}** | {b['vertical']} | {hs} | {b['key_villain_or_bottleneck'][:55]} |"
        )

    return briefs, "\n".join(lines)
