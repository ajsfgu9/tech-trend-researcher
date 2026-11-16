"""
Main Longitudinal Tech & Semiconductor Trend Researcher Pipeline

Role Separation:
- Reads exclusively from `ai-hardware-bot`'s SQLite DB (`agi_hardware_trends.db`).
- Reserves item-level scoring (`scoring_engine.py`) as a 20% secondary sub-component.
- Discovers real structural trends via `trend_analyzer.py` (EWMA/CUSUM burst detection,
  cross-layer lead-lag causal propagation, and novel bottleneck co-occurrence shift).
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from dotenv import load_dotenv

from src.clustering import format_longitudinal_briefs
from src.llm_auditor import evaluate_as_executive_editor
from src.metadata_repo import MetadataRepository
from src.raw_layer import ImmutableRawLayer
from src.relevance_gate import normalize_raw_event, run_relevance_gate
from src.scoring_engine import compute_editorial_base_scores
from src.trend_analyzer import discover_longitudinal_trends

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
    ingestion_threshold_tau: float = 0.40,
    llm_provider: str = "none",
    offline_db_only: bool = True,
) -> tuple[list[dict], str]:
    raw_layer = ImmutableRawLayer(raw_db_path=raw_db_path, aggregator_db_path=aggregator_db_path)
    metadata_repo = MetadataRepository(db_path=metadata_db_path)

    raw_events = raw_layer.ingest_from_aggregator_db(lookback_hours=lookback_hours)
    log.info("Stage 1 (Aggregator DB Bridge): loaded %d historical/recent events.", len(raw_events))

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

    briefs, dashboard_md = format_longitudinal_briefs(signals, top_n=top_n)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(briefs, f, indent=2, ensure_ascii=False)

    with open(dashboard_md_path, "w", encoding="utf-8") as f:
        f.write(dashboard_md + "\n")

    log.info("Stage 4 (Content Briefs): exported %d longitudinal briefs to %s.", len(briefs), output_json_path)
    return briefs, dashboard_md


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Longitudinal Tech & Semiconductor Trend Researcher")
    parser.add_argument("--hours", type=int, default=int(os.getenv("LOOKBACK_HOURS", "336")))
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP_N_TRENDS", "12")))
    parser.add_argument("--tau", type=float, default=float(os.getenv("INGESTION_THRESHOLD_TAU", "0.40")))
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
