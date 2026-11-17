"""
Main Two-Part Trend & Database Viral Matcher Orchestrator (`src/researcher.py`)

Distinct Roles:
- `ai-hardware-bot`: Broad internet fetcher & LLM relevance sorter (`agi_hardware_trends.db`).
- `tech-trend-researcher`: Broad social trend detector & viral feature comparator.

Final Output (`content_briefs.json` & `dashboard.md`) contains two explicit sections:
1. `output_1_trending_now`: What is trending right now across the research pipeline (EWMA/CUSUM
   burst trajectory, cross-layer propagation, and Executive Editor content briefs).
2. `output_2_database_viral_matches`: Cross-comparison against `ai-hardware-bot`'s sorted database
   items (`has_viral_db_matches`: `"YES"` with the exact matching database items and their shared
   viral features, or `"NO"` if none match).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.clustering import format_longitudinal_briefs
from src.llm_auditor import evaluate_as_executive_editor
from src.metadata_repo import MetadataRepository
from src.models import RawEvent
from src.raw_layer import ImmutableRawLayer
from src.relevance_gate import normalize_raw_event, run_relevance_gate
from src.scoring_engine import compute_editorial_base_scores
from src.trend_analyzer import discover_longitudinal_trends
from src.viral_matcher import match_sorted_db_items_against_trends

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tech-trend-researcher")


def run_pipeline(
    aggregator_db_path: str,
    raw_db_path: str,
    metadata_db_path: str,
    output_json_path: str,
    dashboard_md_path: str,
    lookback_hours: int = 336,
    top_n: int = 12,
    ingestion_threshold_tau: float = 0.35,
    llm_provider: str = "none",
    social_events_override: list[RawEvent] | None = None,
) -> tuple[dict[str, Any], str]:
    raw_layer = ImmutableRawLayer(raw_db_path=raw_db_path, aggregator_db_path=aggregator_db_path)
    metadata_repo = MetadataRepository(db_path=metadata_db_path)

    if social_events_override is not None:
        raw_events = social_events_override
        raw_layer.persist_events(raw_events)
    else:
        raw_events = raw_layer.collect_broad_social_streams()
    log.info("Stage 1a (Broad Social Streams): collected %d social/community events.", len(raw_events))

    sorted_db_items = raw_layer.load_sorted_db_items(lookback_hours=lookback_hours * 2)
    log.info("Stage 1b (ai-hardware-bot Sorted DB): loaded %d sorted items from %s.", len(sorted_db_items), aggregator_db_path)

    for db_it in sorted_db_items:
        raw_events.append(
            RawEvent(
                event_id=f"db_{db_it.id}",
                source_platform="arxiv" if "arxiv" in db_it.source.lower() else (
                    "sec" if "sec" in db_it.source.lower() else "technical_news"
                ),
                source_name=db_it.source or "ai-hardware-bot",
                ingested_at_utc=datetime.now(timezone.utc).isoformat(),
                published_at_utc=db_it.published_at or datetime.now(timezone.utc).isoformat(),
                payload={
                    "url": db_it.url,
                    "title": db_it.title,
                    "summary": f"{db_it.why_it_matters} {db_it.raw_summary}",
                    "points": db_it.relevance_score * 10.0,
                    "num_comments": 0.0,
                },
            )
        )

    normalized = [normalize_raw_event(ev) for ev in raw_events]
    gated = run_relevance_gate(normalized, sim_threshold=0.62)
    log.info("Stage 2 (Relevance Gate & Bottleneck Extraction): %d/%d candidates passed.", len(gated), len(normalized))

    scored_items = compute_editorial_base_scores(gated, metadata_repo)
    above_tau = [c for c in scored_items if c.m_viral >= ingestion_threshold_tau]

    signals = discover_longitudinal_trends(above_tau, metadata_repo)
    log.info("Stage 3 (Longitudinal Trend Engine): identified %d structural trend trajectories.", len(signals))

    for sig in signals[:top_n]:
        lead = max(sig.candidates, key=lambda c: c.m_viral)
        evaluate_as_executive_editor(lead, provider=llm_provider)

    briefs, trend_table_md = format_longitudinal_briefs(signals, top_n=top_n)
    db_viral_comparison = match_sorted_db_items_against_trends(signals[:top_n], sorted_db_items)

    final_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_1_trending_now": briefs,
        "output_2_database_viral_matches": db_viral_comparison,
    }

    md_lines = [
        "# Tech Trend Researcher — Two-Part Intelligence Report",
        "",
        "## Output 1: What Is Trending Now (Longitudinal Research Pipeline)",
        "",
        trend_table_md,
        "",
        "## Output 2: Previously Sorted Database Items With Identical Viral Features (`ai-hardware-bot` Cross-Comparison)",
        "",
        f"- **Viral Feature Match Status**: **{db_viral_comparison['has_viral_db_matches']}**",
        f"- **Summary**: {db_viral_comparison['summary']}",
        "",
    ]
    if db_viral_comparison["has_viral_db_matches"] == "YES":
        md_lines.extend(
            [
                "| DB ID | Sorted DB Item Title | Source | DB Relevance | Viral Match Score | Matched Social Trend | Identical Features (`bottleneck_addressed` / `specific_tech`) |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for m in db_viral_comparison["matched_database_items"]:
            feats = m["identical_viral_features"]
            feat_str = (
                f"Bottleneck: `{feats.get('db_bottleneck_addressed') or ', '.join(feats.get('identical_bottleneck_concepts', []))}` | "
                f"Tech: `{feats.get('db_specific_tech') or ', '.join(feats.get('shared_entities', []))}`"
            )
            md_lines.append(
                f"| `{m['db_item_id']}` | [{m['title'][:55]}]({m['url']}) | {m['source']} | "
                f"{m['db_relevance_score']:.1f} | **{m['viral_match_score']:.1f}%** | "
                f"**{m['matched_social_trend']}** (`{m['matched_trend_regime']}`) | {feat_str} |"
            )

    full_dashboard_md = "\n".join(md_lines)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    with open(dashboard_md_path, "w", encoding="utf-8") as f:
        f.write(full_dashboard_md + "\n")

    log.info(
        "Stage 4 Complete: %d trends in Output 1; Output 2 DB Viral Match = %s (%d items).",
        len(briefs),
        db_viral_comparison["has_viral_db_matches"],
        db_viral_comparison["matched_count"],
    )
    return final_payload, full_dashboard_md


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Longitudinal Tech Trend Researcher & DB Viral Matcher")
    parser.add_argument("--hours", type=int, default=int(os.getenv("LOOKBACK_HOURS", "336")))
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP_N_TRENDS", "12")))
    parser.add_argument("--tau", type=float, default=float(os.getenv("INGESTION_THRESHOLD_TAU", "0.35")))
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    llm_provider = "none" if args.no_llm else os.getenv("LLM_PROVIDER", "none")

    run_pipeline(
        aggregator_db_path=os.getenv("AI_HARDWARE_DB_PATH", "/data/ai_hardware_bot/agi_hardware_trends.db"),
        raw_db_path=os.getenv("RAW_DB_PATH", "./data/raw_events.db"),
        metadata_db_path=os.getenv("METADATA_DB_PATH", "./data/metadata_repo.db"),
        output_json_path=os.getenv("OUTPUT_JSON_PATH", "./data/content_briefs.json"),
        dashboard_md_path=os.getenv("DASHBOARD_MD_PATH", "./data/dashboard.md"),
        lookback_hours=args.hours,
        top_n=args.top,
        ingestion_threshold_tau=args.tau,
        llm_provider=llm_provider,
    )


if __name__ == "__main__":
    main()
