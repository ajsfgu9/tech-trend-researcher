from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.models import RawEvent
from src.researcher import run_pipeline


def _create_sorted_ai_hardware_bot_db(db_path: str, include_matching_item: bool = True) -> None:
    """Create a realistic ai-hardware-bot SQLite DB with its 19 LLM-sorted columns."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            content_fingerprint TEXT,
            title TEXT,
            source TEXT,
            source_type TEXT,
            published_at TEXT,
            category TEXT,
            hardware_domain TEXT,
            bottleneck_addressed TEXT,
            specific_tech TEXT,
            relevance_score REAL,
            why_it_matters TEXT,
            ai_revolution_thesis TEXT,
            telegram_summary TEXT,
            raw_summary TEXT,
            raw_llm_json TEXT,
            sent_alert INTEGER DEFAULT 0,
            digest_sent INTEGER DEFAULT 0,
            created_at TEXT
        )
        """
    )
    now = datetime.now(timezone.utc)
    d2 = (now - timedelta(days=2)).isoformat()

    if include_matching_item:
        rows = [
            (
                "https://semiengineering.com/cowos-interposer-warpage-analysis",
                "fp_db_1",
                "Thermomechanical Warpage in 3.5x Reticle TSMC CoWoS Interposers",
                "Semiconductor Engineering",
                "specialist_rss",
                d2,
                "packaging",
                "Advanced Packaging",
                "CoWoS interposer warpage and substrate yield limits",
                "TSMC CoWoS-L 3.5x reticle silicon interposer",
                9.2,
                "Interposer warpage above 3.5x reticle directly caps AI GPU packaging yields.",
                "Packaging yield rather than wafer lithography is the binding constraint on 2026-2027 AI cluster delivery.",
                "TSMC CoWoS warpage analysis",
                "Detailed finite-element analysis of CoWoS interposer warpage and yield bottlenecks.",
                "{}",
                1,
                1,
                d2,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO items (
                url, content_fingerprint, title, source, source_type, published_at, category,
                hardware_domain, bottleneck_addressed, specific_tech,
                relevance_score, why_it_matters, ai_revolution_thesis,
                telegram_summary, raw_summary, raw_llm_json,
                sent_alert, digest_sent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    conn.close()


def test_two_part_output_yes_and_no(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    d1 = (now - timedelta(days=1)).isoformat()
    d0 = now.isoformat()

    social_events = [
        RawEvent(
            event_id="soc_1",
            source_platform="reddit",
            source_name="r/semiconductors",
            ingested_at_utc=d1,
            published_at_utc=d1,
            payload={
                "url": "https://reddit.com/r/semiconductors/cowos_debate",
                "title": "TSMC CoWoS interposer warpage yield bottleneck is worse than expected",
                "summary": "Debating 3.5x reticle CoWoS warpage and packaging scrap rates.",
                "score": 180.0,
                "num_comments": 75.0,
            },
        ),
        RawEvent(
            event_id="soc_2",
            source_platform="hacker_news",
            source_name="Hacker News",
            ingested_at_utc=d0,
            published_at_utc=d0,
            payload={
                "url": "https://news.ycombinator.com/item?id=301",
                "title": "The CoWoS Squeeze: Why Advanced Packaging Yields Are Quietly Capping AI Margins",
                "summary": "TSMC CoWoS interposer warpage and substrate shortage discussion.",
                "points": 320.0,
                "num_comments": 145.0,
            },
        ),
    ]

    agg_db_yes = str(tmp_path / "agi_yes.db")
    _create_sorted_ai_hardware_bot_db(agg_db_yes, include_matching_item=True)

    payload_yes, md_yes = run_pipeline(
        aggregator_db_path=agg_db_yes,
        raw_db_path=str(tmp_path / "raw_yes.db"),
        metadata_db_path=str(tmp_path / "meta_yes.db"),
        output_json_path=str(tmp_path / "out_yes.json"),
        dashboard_md_path=str(tmp_path / "dash_yes.md"),
        lookback_hours=336,
        top_n=5,
        ingestion_threshold_tau=0.35,
        llm_provider="none",
        social_events_override=list(social_events),
    )

    assert len(payload_yes["output_1_trending_now"]) >= 1
    out2_yes = payload_yes["output_2_database_viral_matches"]
    assert out2_yes["has_viral_db_matches"] == "YES"
    assert out2_yes["matched_count"] == 1
    assert "Thermomechanical Warpage" in out2_yes["matched_database_items"][0]["title"]
    assert "**YES**" in md_yes

    agg_db_no = str(tmp_path / "agi_no.db")
    _create_sorted_ai_hardware_bot_db(agg_db_no, include_matching_item=False)

    payload_no, md_no = run_pipeline(
        aggregator_db_path=agg_db_no,
        raw_db_path=str(tmp_path / "raw_no.db"),
        metadata_db_path=str(tmp_path / "meta_no.db"),
        output_json_path=str(tmp_path / "out_no.json"),
        dashboard_md_path=str(tmp_path / "dash_no.md"),
        lookback_hours=336,
        top_n=5,
        ingestion_threshold_tau=0.35,
        llm_provider="none",
        social_events_override=list(social_events),
    )

    out2_no = payload_no["output_2_database_viral_matches"]
    assert out2_no["has_viral_db_matches"] == "NO"
    assert out2_no["matched_count"] == 0
    assert out2_no["matched_database_items"] == []
    assert "**NO**" in md_no
