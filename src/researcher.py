"""
Main Two-Part Trend & Database-to-X Breakout Orchestrator (`src/researcher.py`)

Features:
1. Produces the two-part output (`output_1_trending_now` and `output_2_database_viral_matches`).
2. Duplicates the complete output (formatted Telegram report + `dashboard.md` + `content_briefs.json`)
   to a Telegram Bot via `src/telegram_notifier.py`.
3. Operates autonomously in dual-trigger daemon mode (`src/autonomous_daemon.py`):
   - Trigger A (Time-based): Runs periodically every `SCHEDULE_INTERVAL_MINUTES` (default 30m).
   - Trigger B (Database-change): Watches `ai-hardware-bot`'s `agi_hardware_trends.db` (`MAX(id)`,
     `COUNT(*)`, `MAX(created_at)`) every `DB_WATCH_POLL_SECONDS` (default 15s) and immediately
     triggers a fresh run whenever new items are sorted into the database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.autonomous_daemon import DBStateWatcher, run_autonomous_daemon, touch_heartbeat
from src.clustering import format_longitudinal_briefs
from src.llm_auditor import evaluate_as_executive_editor
from src.metadata_repo import MetadataRepository
from src.models import RawEvent
from src.raw_layer import ImmutableRawLayer
from src.relevance_gate import normalize_raw_event, run_relevance_gate
from src.scoring_engine import compute_editorial_base_scores
from src.telegram_notifier import duplicate_output_to_telegram
from src.trend_analyzer import discover_longitudinal_trends
from src.viral_matcher import match_sorted_db_items_against_trends
from src.x_breakout_engine import fetch_x_and_social_pulse

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
    trigger_reason: str = "manual",
    telegram_session: Any | None = None,
) -> tuple[dict[str, Any], str]:
    raw_layer = ImmutableRawLayer(raw_db_path=raw_db_path, aggregator_db_path=aggregator_db_path)
    metadata_repo = MetadataRepository(db_path=metadata_db_path)

    if social_events_override is not None:
        raw_events = list(social_events_override)
        raw_layer.persist_events(raw_events)
    else:
        raw_events = raw_layer.collect_broad_social_streams()
        x_pulse = fetch_x_and_social_pulse()
        raw_layer.persist_events(x_pulse)
        raw_events.extend(x_pulse)
    log.info("Stage 1a (Broad Social & X Pulse): collected %d social/community events.", len(raw_events))

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
        "trigger_reason": trigger_reason,
        "output_1_trending_now": briefs,
        "output_2_database_viral_matches": db_viral_comparison,
    }

    md_lines = [
        "# Tech Trend Researcher — Two-Part X/Twitter Intelligence Report",
        "",
        f"- **Trigger**: `{trigger_reason}`",
        "",
        "## Output 1: What Is Trending Now & Poised to Break Out on X/Twitter",
        "",
        trend_table_md,
        "",
        "## Output 2: Sorted Database Items Primed to Become X/Twitter Trends (`ai-hardware-bot` Cross-Comparison)",
        "",
        f"- **Viral Feature Match Status**: **{db_viral_comparison['has_viral_db_matches']}**",
        f"- **Summary**: {db_viral_comparison['summary']}",
        "",
    ]
    if db_viral_comparison["has_viral_db_matches"] == "YES":
        md_lines.extend(
            [
                "| DB ID | Sorted DB Item Title | DB Relevance | Viral Match | X Breakout Score | X Window | Ready-to-Post X Hook Catalyst |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for m in db_viral_comparison["matched_database_items"]:
            hook_preview = m["x_catalyst_playbook"]["hook_tweet_draft"].replace("\n", " ")[:95]
            md_lines.append(
                f"| `{m['db_item_id']}` | [{m['title'][:48]}]({m['url']}) | "
                f"{m['db_relevance_score']:.1f} | **{m['viral_match_score']:.1f}%** | "
                f"**{m['x_breakout_score']:.1f}** (`{m['will_become_x_trend']}`) | `{m['x_arbitrage_window']}` | {hook_preview}... |"
            )

    full_dashboard_md = "\n".join(md_lines)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    with open(dashboard_md_path, "w", encoding="utf-8") as f:
        f.write(full_dashboard_md + "\n")

    duplicate_output_to_telegram(
        payload=final_payload,
        dashboard_md=full_dashboard_md,
        trigger_reason=trigger_reason,
        session=telegram_session,
    )

    log.info(
        "Stage 4 Complete (%s): %d trends in Output 1; Output 2 DB-to-X Viral Match = %s (%d items).",
        trigger_reason,
        len(briefs),
        db_viral_comparison["has_viral_db_matches"],
        db_viral_comparison["matched_count"],
    )
    return final_payload, full_dashboard_md


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Autonomous Tech Trend Researcher & Telegram Duplicator")
    parser.add_argument("--hours", type=int, default=int(os.getenv("LOOKBACK_HOURS", "336")))
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP_N_TRENDS", "12")))
    parser.add_argument("--tau", type=float, default=float(os.getenv("INGESTION_THRESHOLD_TAU", "0.35")))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit instead of autonomous daemon mode")
    args = parser.parse_args()

    llm_provider = "none" if args.no_llm else os.getenv("LLM_PROVIDER", "none")
    agg_db = os.getenv("AI_HARDWARE_DB_PATH", "/data/ai_hardware_bot/agi_hardware_trends.db")
    raw_db = os.getenv("RAW_DB_PATH", "./data/raw_events.db")
    meta_db = os.getenv("METADATA_DB_PATH", "./data/metadata_repo.db")
    out_json = os.getenv("OUTPUT_JSON_PATH", "./data/content_briefs.json")
    dash_md = os.getenv("DASHBOARD_MD_PATH", "./data/dashboard.md")
    hb_file = os.getenv("HEALTHCHECK_FILE", "./data/heartbeat.txt")

    daemon_enabled = (
        not args.once
        and os.getenv("DAEMON_MODE", "true").strip().lower() in ("1", "true", "yes")
    )

    def _execute(reason: str) -> None:
        run_pipeline(
            aggregator_db_path=agg_db,
            raw_db_path=raw_db,
            metadata_db_path=meta_db,
            output_json_path=out_json,
            dashboard_md_path=dash_md,
            lookback_hours=args.hours,
            top_n=args.top,
            ingestion_threshold_tau=args.tau,
            llm_provider=llm_provider,
            trigger_reason=reason,
        )

    if not daemon_enabled:
        _execute("single_run")
        touch_heartbeat(hb_file)
        return

    watcher = DBStateWatcher(aggregator_db_path=agg_db, state_db_path=meta_db)
    run_autonomous_daemon(
        pipeline_callback=_execute,
        watcher=watcher,
        schedule_interval_minutes=int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "30")),
        db_watch_poll_seconds=int(os.getenv("DB_WATCH_POLL_SECONDS", "15")),
        heartbeat_path=hb_file,
    )


if __name__ == "__main__":
    main()
