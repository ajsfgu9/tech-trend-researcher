"""
Main Editorial Tech & Semiconductor Trend Researcher Pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from dotenv import load_dotenv

from src.clustering import cluster_and_format_briefs
from src.llm_auditor import evaluate_as_executive_editor
from src.metadata_repo import MetadataRepository
from src.raw_layer import ImmutableRawLayer
from src.relevance_gate import normalize_raw_event, run_relevance_gate
from src.scoring_engine import compute_editorial_base_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tech-trend-researcher")


def run_pipeline(
    aggregator_db_path: str,
    raw_db_path: str,
    metadata_db_path: str,
    output_json_path: str,
    dashboard_md_path: str,
    lookback_hours: int = 72,
    top_n: int = 12,
    ingestion_threshold_tau: float = 0.45,
    llm_provider: str = "none",
    offline_db_only: bool = False,
) -> tuple[list[dict], str]:
    raw_layer = ImmutableRawLayer(raw_db_path=raw_db_path, aggregator_db_path=aggregator_db_path)
    metadata_repo = MetadataRepository(db_path=metadata_db_path)

    raw_events = raw_layer.ingest_from_aggregator_db(lookback_hours=lookback_hours)
    if not offline_db_only:
        raw_events.extend(raw_layer.ingest_live_feeds())
    log.info("Stage 1 (Immutable Raw Layer): ingested %d raw events.", len(raw_events))

    normalized = [normalize_raw_event(ev) for ev in raw_events]
    gated = run_relevance_gate(normalized, sim_threshold=0.62)
    log.info("Stage 2 (Relevance Gate & IA): %d/%d candidates passed.", len(gated), len(normalized))

    scored = compute_editorial_base_scores(gated, metadata_repo)
    above_tau = [c for c in scored if c.m_viral >= ingestion_threshold_tau]
    log.info("Stage 3 (Editorial Base Scoring): %d candidates above tau=%.2f.", len(above_tau), ingestion_threshold_tau)

    for cand in above_tau:
        evaluate_as_executive_editor(cand, provider=llm_provider)

    briefs, dashboard_md = cluster_and_format_briefs(above_tau, top_n=top_n)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(briefs, f, indent=2, ensure_ascii=False)

    with open(dashboard_md_path, "w", encoding="utf-8") as f:
        f.write(dashboard_md + "\n")

    log.info("Stage 4 (Content Briefs): exported %d editorial briefs to %s and %s.", len(briefs), output_json_path, dashboard_md_path)
    return briefs, dashboard_md


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Editorial Tech & Semiconductor Trend Researcher")
    parser.add_argument("--hours", type=int, default=int(os.getenv("LOOKBACK_HOURS", "72")))
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP_N_TRENDS", "12")))
    parser.add_argument("--tau", type=float, default=float(os.getenv("INGESTION_THRESHOLD_TAU", "0.45")))
    parser.add_argument("--offline-db-only", action="store_true")
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
        offline_db_only=args.offline_db_only,
    )


if __name__ == "__main__":
    main()
