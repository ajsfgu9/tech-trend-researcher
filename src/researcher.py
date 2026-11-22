"""Main Tech Trend Researcher orchestrator (multi-DB + external trend discovery + Telegram delivery)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from src.autonomous_daemon import run_autonomous_daemon, touch_heartbeat
from src.db_bridge import MultiDBStateWatcher, load_all_bot_databases, parse_source_db_specs
from src.external_collectors import collect_all_external_signals
from src.llm_client import DualLLMClient
from src.matcher import match_db_items_to_external_trends
from src.models import CollectorStatus, ExternalSignal, SourceDBConfig
from src.similarity import EmbeddingEngine
from src.state_store import StateStore
from src.telegram_notifier import duplicate_output_to_telegram
from src.trend_detector import detect_external_trends

log = logging.getLogger("tech-trend-researcher")


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _render_dashboard_markdown(payload: dict[str, Any]) -> str:
    out1 = payload["output_1_external_trends"]
    out2 = payload["output_2_database_viral_matches"]
    health = payload["source_health"]

    lines = [
        "# Tech Trend Researcher — External Trends & Multi-DB Match Report",
        "",
        f"- **Generated (UTC)**: `{payload['generated_at_utc']}`",
        f"- **Trigger**: `{payload['trigger_reason']}`",
        f"- **LLM Chain**: `{payload['llm_chain']}`",
        "",
        "## Source Health",
        "",
        "| Layer | Source | Status | Items | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for st in health.get("external_collectors", []):
        lines.append(
            f"| External | `{st['collector']}` | `{st['status']}` | {st['items_collected']} | {st.get('detail') or '-'} |"
        )
    for name, info in (health.get("bot_databases") or {}).items():
        lines.append(
            f"| Bot DB | `{name}` | `{info['status']}` | {info['loaded_items']} | `{info['configured_path']}` |"
        )

    lines.extend(["", "## Output 1: Live External Trends", ""])
    if not out1:
        lines.append("_No live external trends detected in this window._")
    else:
        lines.extend(
            [
                "| Rank | Trend Score | Rate (/hr) | Velocity z | Platforms | Signals | Topic |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for rank, tr in enumerate(out1, start=1):
            z_str = f"{tr['avg_velocity_z']:+.2f}" if tr.get("avg_velocity_z") is not None else "n/a"
            plats = ", ".join(tr.get("platforms", []))
            lines.append(
                f"| **{rank:02d}** | **{tr['trend_score']:.1f}** | {tr['max_engagement_rate_per_hour']:.1f} | "
                f"`{z_str}` | `{plats}` | {tr['signal_count']} | {tr['label'][:85]} |"
            )

    lines.extend(
        [
            "",
            "## Output 2: Upstream Bot DB Items Matching Live External Trends",
            "",
            f"- **Match Status**: **{out2['has_viral_db_matches']}**",
            f"- **Similarity Backend**: `{out2['similarity_backend']}`",
            f"- **Summary**: {out2['summary']}",
            "",
        ]
    )

    if out2["has_viral_db_matches"] == "YES":
        lines.extend(
            [
                "| Bot DB | ID | Title | Domain | DB Rel | Match % | LLM Verified | Matched External Trend |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for m in out2["matched_database_items"]:
            ver = (
                "YES" if m["llm_verified"] is True else ("NO" if m["llm_verified"] is False else "n/a")
            )
            lines.append(
                f"| `{m['bot_name']}` | `{m['db_item_id']}` | [{m['title'][:52]}]({m['url']}) | "
                f"{m['domain'][:28]} | {m['db_relevance_score']:.1f} | **{m['combined_match_score']:.1f}%** | "
                f"`{ver}` | {m['matched_trend_label'][:48]} |"
            )

    return "\n".join(lines)


def run_pipeline(
    db_configs: list[SourceDBConfig],
    state_db_path: str,
    output_json_path: str,
    dashboard_md_path: str,
    external_lookback_hours: int = 48,
    db_lookback_hours: int = 336,
    top_n: int = 12,
    cluster_similarity_threshold: float = 0.28,
    db_match_threshold: float = 0.32,
    min_db_relevance: float = 7.0,
    max_llm_trend_analyses: int = 8,
    max_llm_match_verifications: int = 15,
    llm_client: DualLLMClient | None = None,
    embedding_engine: EmbeddingEngine | None = None,
    external_signals_override: list[ExternalSignal] | None = None,
    auto_detect_single_db: bool = True,
    trigger_reason: str = "manual",
    telegram_session: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute one full, strictly separated trend discovery and multi-DB matching cycle."""
    state_store = StateStore(state_db_path)
    llm = llm_client or DualLLMClient()
    embedder = embedding_engine or EmbeddingEngine(state_store=state_store)

    # Step 1: Strictly external live trend collection
    if external_signals_override is not None:
        external_signals = list(external_signals_override)
        collector_statuses = [
            CollectorStatus("override", True, "ok", len(external_signals), "Injected test/override signals")
        ]
    else:
        external_signals, collector_statuses = collect_all_external_signals(
            lookback_hours=external_lookback_hours
        )
    log.info("External collection complete: %d live signals.", len(external_signals))

    # Step 2: Read-only multi-DB load from all upstream bots
    db_items, db_health = load_all_bot_databases(
        configs=db_configs,
        lookback_hours=db_lookback_hours,
        min_relevance=min_db_relevance,
        auto_detect_single_db=auto_detect_single_db,
    )
    log.info(
        "Loaded %d classified DB items across %d bot database(s).",
        len(db_items),
        len(db_configs),
    )

    # Step 3: Output 1 — Strictly external trend clustering & velocity ranking
    external_trends = detect_external_trends(
        signals=external_signals,
        state_store=state_store,
        embedding_engine=embedder,
        similarity_threshold=cluster_similarity_threshold,
        top_n=top_n,
    )
    for cluster in external_trends[:max_llm_trend_analyses]:
        cluster.llm_analysis = llm.analyze_external_trend(cluster)

    # Step 4: Output 2 — Cross-compare bot DB items strictly against Output 1
    db_matches = match_db_items_to_external_trends(
        external_trends=external_trends,
        db_items=db_items,
        embedding_engine=embedder,
        llm_client=llm,
        match_threshold=db_match_threshold,
        max_llm_verifications=max_llm_match_verifications,
    )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_reason": trigger_reason,
        "llm_chain": llm.active_label(),
        "source_health": {
            "external_collectors": [st.to_dict() for st in collector_statuses],
            "bot_databases": db_health,
        },
        "output_1_external_trends": [t.to_dict() for t in external_trends],
        "output_2_database_viral_matches": db_matches,
    }
    dashboard_md = _render_dashboard_markdown(payload)

    for target_path in (output_json_path, dashboard_md_path):
        parent = os.path.dirname(os.path.abspath(target_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    with open(dashboard_md_path, "w", encoding="utf-8") as fh:
        fh.write(dashboard_md + "\n")

    duplicate_output_to_telegram(
        payload=payload,
        dashboard_md=dashboard_md,
        trigger_reason=trigger_reason,
        session=telegram_session,
    )

    state_store.prune_old_snapshots(retain_days=30)

    log.info(
        "Run complete (%s): Output 1 = %d external trend(s); Output 2 = %s (%d match(es)).",
        trigger_reason,
        len(external_trends),
        db_matches["has_viral_db_matches"],
        db_matches["matched_count"],
    )
    return payload, dashboard_md


def main() -> None:
    load_dotenv()
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Tech Trend Researcher — Multi-DB External Trend Matcher & Telegram Notifier"
    )
    parser.add_argument(
        "--external-hours",
        type=int,
        default=int(os.getenv("EXTERNAL_LOOKBACK_HOURS", "48")),
    )
    parser.add_argument(
        "--db-hours",
        type=int,
        default=int(os.getenv("DB_LOOKBACK_HOURS", "336")),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=int(os.getenv("TOP_N_TRENDS", "12")),
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls for this run")
    parser.add_argument("--once", action="store_true", help="Run once and exit instead of daemon mode")
    args = parser.parse_args()

    db_configs = parse_source_db_specs(
        specs_raw=os.getenv("SOURCE_DB_SPECS"),
        legacy_single_path=os.getenv("AI_HARDWARE_DB_PATH"),
    )
    state_db = os.getenv("STATE_DB_PATH", "./data/researcher_state.db")
    out_json = os.getenv("OUTPUT_JSON_PATH", "./data/trend_report.json")
    dash_md = os.getenv("DASHBOARD_MD_PATH", "./data/dashboard.md")
    hb_file = os.getenv("HEALTHCHECK_FILE", "./data/heartbeat.txt")
    auto_detect = os.getenv("AUTO_DETECT_SINGLE_DB", "true").strip().lower() in ("1", "true", "yes", "on")

    llm_client = DualLLMClient(provider="none") if args.no_llm else DualLLMClient()

    log.info("Configured LLM chain: %s", llm_client.active_label())
    log.info("Configured source DBs: %s", [(c.name, c.db_path) for c in db_configs])

    daemon_enabled = (
        not args.once
        and os.getenv("DAEMON_MODE", "true").strip().lower() in ("1", "true", "yes", "on")
    )

    def _execute(reason: str) -> None:
        run_pipeline(
            db_configs=db_configs,
            state_db_path=state_db,
            output_json_path=out_json,
            dashboard_md_path=dash_md,
            external_lookback_hours=args.external_hours,
            db_lookback_hours=args.db_hours,
            top_n=args.top,
            cluster_similarity_threshold=float(os.getenv("CLUSTER_SIMILARITY_THRESHOLD", "0.28")),
            db_match_threshold=float(os.getenv("DB_MATCH_THRESHOLD", "0.32")),
            min_db_relevance=float(os.getenv("MIN_DB_RELEVANCE_SCORE", "7.0")),
            max_llm_trend_analyses=int(os.getenv("MAX_LLM_TREND_ANALYSES", "8")),
            max_llm_match_verifications=int(os.getenv("MAX_LLM_MATCH_VERIFICATIONS", "15")),
            llm_client=llm_client,
            auto_detect_single_db=auto_detect,
            trigger_reason=reason,
        )

    if not daemon_enabled:
        _execute("single_run")
        touch_heartbeat(hb_file)
        return

    watcher = MultiDBStateWatcher(
        db_configs=db_configs,
        state_store=StateStore(state_db),
        auto_detect_single_db=auto_detect,
    )
    run_autonomous_daemon(
        pipeline_callback=_execute,
        watcher=watcher,
        schedule_interval_minutes=int(os.getenv("SCHEDULE_INTERVAL_MINUTES", "30")),
        db_watch_poll_seconds=int(os.getenv("DB_WATCH_POLL_SECONDS", "15")),
        heartbeat_path=hb_file,
    )


if __name__ == "__main__":
    main()
