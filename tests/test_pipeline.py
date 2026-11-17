from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from src.autonomous_daemon import DBStateWatcher
from src.models import RawEvent
from src.researcher import run_pipeline
from src.telegram_notifier import duplicate_output_to_telegram


class DummyTelegramSession:
    """Mock HTTP session capturing Telegram sendMessage and sendDocument calls."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.documents: list[str] = []

    def post(self, url: str, json: dict[str, Any] | None = None, data: dict[str, Any] | None = None, files: dict[str, Any] | None = None, timeout: int = 20) -> Any:
        if "sendMessage" in url and json:
            self.messages.append(json)
        elif "sendDocument" in url and files:
            doc_tuple = files.get("document", ("unknown",))
            self.documents.append(doc_tuple[0])

        class _Resp:
            ok = True
            status_code = 200
            text = '{"ok": true}'

        return _Resp()


def _create_sorted_ai_hardware_bot_db(db_path: str, include_matching_item: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
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
        conn.execute(
            """
            INSERT INTO items (
                url, content_fingerprint, title, source, source_type, published_at, category,
                hardware_domain, bottleneck_addressed, specific_tech,
                relevance_score, why_it_matters, ai_revolution_thesis,
                telegram_summary, raw_summary, raw_llm_json,
                sent_alert, digest_sent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
        )
    conn.commit()
    conn.close()


def test_two_part_output_telegram_and_db_watcher(tmp_path) -> None:
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

    agg_db = str(tmp_path / "agi_watch.db")
    state_db = str(tmp_path / "meta_watch.db")
    _create_sorted_ai_hardware_bot_db(agg_db, include_matching_item=False)

    watcher = DBStateWatcher(aggregator_db_path=agg_db, state_db_path=state_db)
    changed_init, sig_init = watcher.check_and_mark_changed()
    assert changed_init is True
    assert sig_init == "0:0:"

    changed_again, _ = watcher.check_and_mark_changed()
    assert changed_again is False

    payload_no, md_no = run_pipeline(
        aggregator_db_path=agg_db,
        raw_db_path=str(tmp_path / "raw_no.db"),
        metadata_db_path=state_db,
        output_json_path=str(tmp_path / "out_no.json"),
        dashboard_md_path=str(tmp_path / "dash_no.md"),
        lookback_hours=336,
        top_n=5,
        ingestion_threshold_tau=0.35,
        llm_provider="none",
        social_events_override=list(social_events),
        trigger_reason="scheduled_timer",
    )
    assert payload_no["output_2_database_viral_matches"]["has_viral_db_matches"] == "NO"

    _create_sorted_ai_hardware_bot_db(agg_db, include_matching_item=True)
    changed_after_insert, sig_new = watcher.check_and_mark_changed()
    assert changed_after_insert is True
    assert sig_new.startswith("1:1:")

    tg_session = DummyTelegramSession()
    payload_yes, md_yes = run_pipeline(
        aggregator_db_path=agg_db,
        raw_db_path=str(tmp_path / "raw_yes.db"),
        metadata_db_path=state_db,
        output_json_path=str(tmp_path / "out_yes.json"),
        dashboard_md_path=str(tmp_path / "dash_yes.md"),
        lookback_hours=336,
        top_n=5,
        ingestion_threshold_tau=0.35,
        llm_provider="none",
        social_events_override=list(social_events),
        trigger_reason=f"db_change:{sig_new}",
    )

    ok = duplicate_output_to_telegram(
        payload=payload_yes,
        dashboard_md=md_yes,
        trigger_reason=f"db_change:{sig_new}",
        bot_token="123456:TEST_TOKEN",
        chat_id="987654321",
        send_documents=True,
        session=tg_session,
    )
    assert ok is True
    assert len(tg_session.messages) >= 1
    assert "PART 1: WHAT IS TRENDING NOW" in tg_session.messages[0]["text"]
    assert "PART 2: DATABASE VIRAL FEATURE MATCHES" in tg_session.messages[0]["text"]
    assert len(tg_session.documents) == 2
