"""
Main Tech Trend Researcher Orchestrator

1. Reads recent items from ai-hardware-bot's SQLite DB (read-only WAL mode).
2. Collects live multi-vertical RSS/Atom feeds (semiconductor, financing, biomedical, robotics, quantum).
3. Clusters & scores trends with fairness_engine + reverse-engineered x_algo_scorer.
4. Optionally enriches top trends via OpenRouter or Gemini API.
5. Persists results to SQLite (WAL mode) and exports trends_output.json for the News Writer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

from src.fairness_engine import cluster_items, compute_fairness_score, select_balanced_trends
from src.models import ALL_VERTICALS, TrendCluster
from src.sources import AggregatorDBReader, MultiVerticalCollector
from src.x_algo_scorer import score_trend_for_x

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tech-trend-researcher")


def enrich_with_llm(cluster: TrendCluster, provider: str) -> None:
    """Add fair technical context, caveat, and X hook angle via OpenRouter or Gemini."""
    if provider == "none":
        cluster.why_tech_fanatics_care = (
            f"Combines {cluster.corroboration_count} source(s) in {cluster.vertical} "
            f"with key metrics: {', '.join(cluster.metrics_extracted) or 'qualitative architecture shift'}."
        )
        cluster.fair_caveat = (
            "Verify primary benchmark conditions and production timeline before extrapolating."
            if cluster.primary_source_count == 0
            else "Backed by primary research/filing; check peer replication or commercial scalability."
        )
        return

    sources_text = "\n".join(
        f"- [{i.source}] {i.title} ({i.url}): {i.summary[:220]}"
        for i in cluster.items[:5]
    )
    prompt = (
        "You are a rigorous semiconductor & deep-tech trend researcher. "
        "Analyze this trend cluster fairly for tech-savvy X/Twitter readers.\n"
        f"Vertical: {cluster.vertical}\n"
        f"Topic: {cluster.topic}\n"
        f"Extracted metrics: {cluster.metrics_extracted}\n"
        f"Sources:\n{sources_text}\n\n"
        "Return strict JSON with keys:\n"
        '  "why_tech_fanatics_care": (2 concise sentences with concrete technical/financial stakes),\n'
        '  "fair_caveat": (1 sentence noting limitations, unverified claims, or counter-arguments),\n'
        '  "hook_angle": (1 punchy, non-clickbait opening line under 200 chars optimized for X replies/bookmarks).'
    )

    try:
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=25,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
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
            content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return

        data: dict[str, Any] = json.loads(content)
        cluster.why_tech_fanatics_care = str(data.get("why_tech_fanatics_care", "")).strip()
        cluster.fair_caveat = str(data.get("fair_caveat", "")).strip()
        if data.get("hook_angle"):
            cluster.x_strategy["hook_angle"] = str(data["hook_angle"]).strip()
    except Exception as exc:
        log.warning("LLM enrichment skipped for %s: %s", cluster.cluster_id, exc)
        enrich_with_llm(cluster, "none")


def save_to_sqlite(db_path: str, trends: list[TrendCluster]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trends (
                cluster_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                vertical TEXT NOT NULL,
                combined_score REAL NOT NULL,
                fairness_score REAL NOT NULL,
                x_algo_score REAL NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        for t in trends:
            conn.execute(
                """
                INSERT INTO trends (cluster_id, topic, vertical, combined_score, fairness_score, x_algo_score, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    topic=excluded.topic,
                    combined_score=excluded.combined_score,
                    fairness_score=excluded.fairness_score,
                    x_algo_score=excluded.x_algo_score,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    t.cluster_id,
                    t.topic,
                    t.vertical,
                    t.combined_score,
                    t.fairness_score,
                    t.x_algo_score,
                    json.dumps(t.to_dict()),
                    now_iso,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def run_pipeline(
    aggregator_db_path: str,
    trends_db_path: str,
    output_json_path: str,
    lookback_hours: int = 72,
    top_n: int = 12,
    verticals: list[str] | None = None,
    exclude_sources: tuple[str, ...] = ("Hacker News",),
    llm_provider: str = "none",
    offline_db_only: bool = False,
) -> list[TrendCluster]:
    items = AggregatorDBReader(aggregator_db_path, exclude_sources=exclude_sources).fetch_recent_items(lookback_hours)
    log.info("Loaded %d items from ai-hardware-bot DB.", len(items))

    if not offline_db_only:
        live_items = MultiVerticalCollector().fetch_all(verticals=verticals, lookback_hours=lookback_hours)
        log.info("Fetched %d items from live multi-vertical feeds.", len(live_items))
        items.extend(live_items)

    if verticals:
        allowed = set(verticals)
        items = [i for i in items if i.vertical in allowed]

    clusters = cluster_items(items)
    for cluster in clusters:
        cluster.fairness_score = compute_fairness_score(cluster)
        cluster.x_algo_score, cluster.x_strategy = score_trend_for_x(cluster)
        # Balanced blend: 55% fairness/substance + 45% X algorithmic attractiveness
        cluster.combined_score = 0.55 * cluster.fairness_score + 0.45 * cluster.x_algo_score

    top_trends = select_balanced_trends(clusters, top_n=top_n, max_per_vertical=max(2, top_n // 4))
    for trend in top_trends:
        enrich_with_llm(trend, llm_provider)

    save_to_sqlite(trends_db_path, top_trends)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_raw_items": len(items),
        "total_clusters": len(clusters),
        "selected_count": len(top_trends),
        "trends": [t.to_dict() for t in top_trends],
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Exported %d balanced trends to %s", len(top_trends), output_json_path)
    return top_trends


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multi-vertical Tech Trend Researcher")
    parser.add_argument("--hours", type=int, default=int(os.getenv("LOOKBACK_HOURS", "72")))
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP_N_TRENDS", "12")))
    parser.add_argument("--verticals", type=str, default=",".join(ALL_VERTICALS))
    parser.add_argument("--offline-db-only", action="store_true", help="Only read from ai-hardware-bot SQLite DB")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enrichment")
    args = parser.parse_args()

    llm_provider = "none" if args.no_llm else os.getenv("LLM_PROVIDER", "none")
    excluded = tuple(os.getenv("EXCLUDE_SOURCES", "Hacker News").split(","))
    vertical_list = [v.strip() for v in args.verticals.split(",") if v.strip()]

    run_pipeline(
        aggregator_db_path=os.getenv("AI_HARDWARE_DB_PATH", "/data/ai_hardware_bot/agi_hardware_trends.db"),
        trends_db_path=os.getenv("TRENDS_DB_PATH", "./data/tech_trends.db"),
        output_json_path=os.getenv("OUTPUT_JSON_PATH", "./data/trends_output.json"),
        lookback_hours=args.hours,
        top_n=args.top,
        verticals=vertical_list,
        exclude_sources=excluded,
        llm_provider=llm_provider,
        offline_db_only=args.offline_db_only,
    )


if __name__ == "__main__":
    main()
