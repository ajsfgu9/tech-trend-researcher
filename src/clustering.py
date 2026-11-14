"""
Stage 4b: Semantic Clustering, Exact JSON Schema Output & Formatted Delivery Dashboard
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.models import NormalizedCandidate


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\-\.]{3,}", text.lower()))


def cluster_and_format(
    candidates: list[NormalizedCandidate],
    top_n: int = 12,
) -> tuple[list[dict[str, Any]], str]:
    """Group related candidates into trend clusters and emit exact JSON schema + Markdown dashboard."""
    ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    clusters: list[list[NormalizedCandidate]] = []
    cluster_tokens: list[set[str]] = []

    for cand in ranked:
        toks = _tokens(cand.title)
        matched = -1
        for idx, c_toks in enumerate(cluster_tokens):
            union = toks | c_toks
            if union and (len(toks & c_toks) / len(union)) >= 0.25:
                matched = idx
                break
        if matched >= 0:
            clusters[matched].append(cand)
            cluster_tokens[matched].update(toks)
        else:
            clusters.append([cand])
            cluster_tokens.append(set(toks))

    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_tag = now.strftime("%Y%m%d")

    output_clusters: list[dict[str, Any]] = []
    for idx, group in enumerate(clusters[:top_n], start=1):
        lead = group[0]
        all_entities = sorted({e for c in group for e in c.verified_entities})
        source_dist: dict[str, Any] = {}
        press_list: list[str] = []

        for c in group:
            if c.source_platform == "hacker_news":
                source_dist["hacker_news"] = {
                    "mentions": source_dist.get("hacker_news", {}).get("mentions", 0) + 1,
                    "engagement": round(c.engagement_value, 1),
                    "url": c.url,
                }
            elif c.source_platform == "reddit":
                key = f"reddit_{c.source_name.replace('r/', '').lower()}"
                source_dist[key] = {
                    "mentions": source_dist.get(key, {}).get("mentions", 0) + 1,
                    "engagement": round(c.engagement_value, 1),
                    "url": c.url,
                }
            elif c.source_platform == "x_twitter":
                source_dist["x_tech_nodes"] = {
                    "mentions": source_dist.get("x_tech_nodes", {}).get("mentions", 0) + 1,
                    "engagement": round(c.engagement_value, 1),
                }
            else:
                if c.source_name not in press_list:
                    press_list.append(c.source_name)

        if press_list:
            source_dist["industry_and_primary_press"] = press_list

        scaled_final = round(min(99.9, lead.final_score * 25.0), 2)
        scaled_base = round(min(99.9, lead.base_score_mi * 25.0), 2)

        output_clusters.append(
            {
                "timestamp_utc": now_iso,
                "cluster_id": f"trend_sc_{date_tag}_{idx:03d}",
                "topic": lead.title,
                "vertical": lead.vertical,
                "final_score": scaled_final,
                "metrics": {
                    "algorithmic_base_score": scaled_base,
                    "llm_multiplier": lead.llm_multiplier_phi,
                    "velocity_z_score": lead.velocity_vi,
                    "cross_platform_dispersion_count": max(lead.dispersion_k, len({c.source_platform for c in group})),
                },
                "verified_entities": all_entities,
                "source_distribution": source_dist,
                "source_urls": [c.url for c in group if c.url],
                "llm_synthesis": {
                    "summary": lead.llm_audit.get("rationale", ""),
                    "critical_implication": lead.llm_audit.get("critical_implication", ""),
                },
            }
        )

    lines = [
        "| Rank | Trend Cluster | Vertical | Final Score | Cross-Source Confirmation | Key Entities | Velocity ($z$) | Industry Takeaway |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, cl in enumerate(output_clusters, start=1):
        sources_str = ", ".join(cl["source_distribution"].keys()) or "direct"
        entities_str = ", ".join(cl["verified_entities"][:4]) or "-"
        takeaway = cl["llm_synthesis"]["critical_implication"] or cl["llm_synthesis"]["summary"]
        lines.append(
            f"| **{rank:02d}** | **{cl['topic'][:60]}** | {cl['vertical']} | **{cl['final_score']:.1f}** | "
            f"{sources_str} | {entities_str} | +{cl['metrics']['velocity_z_score']:.2f} | {takeaway[:110]} |"
        )

    return output_clusters, "\n".join(lines)
