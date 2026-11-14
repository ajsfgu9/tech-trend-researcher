"""
Stage 1: Immutable Raw Layer
Pulls unstructured streams from the 7 source platforms + ai-hardware-bot SQLite DB
and persists untouched JSON payloads with ingestion timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import requests

from src.models import RawEvent

log = logging.getLogger(__name__)


class ImmutableRawLayer:
    def __init__(self, raw_db_path: str, aggregator_db_path: str | None = None) -> None:
        self.raw_db_path = raw_db_path
        self.aggregator_db_path = aggregator_db_path
        os.makedirs(os.path.dirname(os.path.abspath(raw_db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.raw_db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_events (
                    event_id TEXT PRIMARY KEY,
                    source_platform TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    ingested_at_utc TEXT NOT NULL,
                    published_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def persist_events(self, events: list[RawEvent]) -> int:
        if not events:
            return 0
        inserted = 0
        with sqlite3.connect(self.raw_db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            for ev in events:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO raw_events
                    (event_id, source_platform, source_name, ingested_at_utc, published_at_utc, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.event_id,
                        ev.source_platform,
                        ev.source_name,
                        ev.ingested_at_utc,
                        ev.published_at_utc,
                        json.dumps(ev.payload, ensure_ascii=False),
                    ),
                )
                inserted += cur.rowcount
        return inserted

    def ingest_from_aggregator_db(self, lookback_hours: int = 72) -> list[RawEvent]:
        """Bridge from jijiji88888/ai-hardware-bot (agi_hardware_trends.db) in read-only URI mode."""
        if not self.aggregator_db_path or not os.path.exists(self.aggregator_db_path):
            return []

        uri = f"file:{os.path.abspath(self.aggregator_db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        now_iso = datetime.now(timezone.utc).isoformat()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        events: list[RawEvent] = []
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            if not cols:
                return []
            select_cols = [c for c in ("url", "title", "source", "source_type", "published_at", "category", "summary") if c in cols]
            order_col = "published_at" if "published_at" in cols else "id"
            rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM items ORDER BY {order_col} DESC LIMIT 500").fetchall()
            for r in rows:
                d = dict(r)
                pub_str = d.get("published_at") or now_iso
                try:
                    dt = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                    pub_iso = dt.isoformat()
                except ValueError:
                    pub_iso = now_iso

                src = str(d.get("source") or "ai-hardware-bot")
                platform = "hacker_news" if "hacker news" in src.lower() else (
                    "arxiv" if "arxiv" in src.lower() else "technical_news"
                )
                eid = hashlib.sha1(f"{platform}:{d.get('url')}".encode("utf-8")).hexdigest()[:16]
                events.append(
                    RawEvent(
                        event_id=eid,
                        source_platform=platform,
                        source_name=src,
                        ingested_at_utc=now_iso,
                        published_at_utc=pub_iso,
                        payload=d,
                    )
                )
        finally:
            conn.close()
        self.persist_events(events)
        return events

    def ingest_live_feeds(self) -> list[RawEvent]:
        """Pull live streams from HN Algolia, Reddit JSON, GitHub, Technical RSS, and arXiv."""
        now_iso = datetime.now(timezone.utc).isoformat()
        events: list[RawEvent] = []

        # 1. Hacker News Algolia API
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=30",
                timeout=10,
            )
            if resp.ok:
                for hit in resp.json().get("hits", []):
                    url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                    eid = hashlib.sha1(f"hacker_news:{url}".encode("utf-8")).hexdigest()[:16]
                    events.append(
                        RawEvent(
                            event_id=eid,
                            source_platform="hacker_news",
                            source_name="Hacker News",
                            ingested_at_utc=now_iso,
                            published_at_utc=hit.get("created_at") or now_iso,
                            payload=hit,
                        )
                    )
        except Exception as exc:
            log.warning("HN Algolia ingestion failed: %s", exc)

        # 2. Specialized Subreddits
        for sub in ("semiconductors", "hardware", "LocalLLaMA", "biotech", "robotics", "QuantumComputing"):
            try:
                resp = requests.get(
                    f"https://www.reddit.com/r/{sub}/hot.json?limit=15",
                    headers={"User-Agent": "TechTrendResearcher/2.0"},
                    timeout=10,
                )
                if resp.ok:
                    for child in resp.json().get("data", {}).get("children", []):
                        post = child.get("data", {})
                        url = post.get("url") or f"https://reddit.com{post.get('permalink', '')}"
                        eid = hashlib.sha1(f"reddit:{url}".encode("utf-8")).hexdigest()[:16]
                        pub_ts = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc).isoformat()
                        events.append(
                            RawEvent(
                                event_id=eid,
                                source_platform="reddit",
                                source_name=f"r/{sub}",
                                ingested_at_utc=now_iso,
                                published_at_utc=pub_ts,
                                payload=post,
                            )
                        )
            except Exception as exc:
                log.warning("Reddit r/%s ingestion failed: %s", sub, exc)

        # 3. Technical News & Academic Preprints (RSS/Atom)
        rss_targets: list[tuple[str, str, str]] = [
            ("technical_news", "TrendForce", "https://www.trendforce.com/news/feed_v2/"),
            ("technical_news", "Semiconductor Engineering", "https://semiengineering.com/feed/"),
            ("technical_news", "FierceBiotech", "https://www.fiercebiotech.com/rss/xml"),
            ("technical_news", "The Robot Report", "https://www.therobotreport.com/feed/"),
            ("arxiv", "arXiv cs.AR", "https://export.arxiv.org/rss/cs.AR"),
            ("arxiv", "arXiv cs.RO", "https://export.arxiv.org/rss/cs.RO"),
            ("arxiv", "arXiv quant-ph", "https://export.arxiv.org/rss/quant-ph"),
        ]
        for platform, name, url in rss_targets:
            try:
                import feedparser
                parsed = feedparser.parse(url, request_headers={"User-Agent": "TechTrendResearcher/2.0"})
                for entry in parsed.entries[:20]:
                    link = str(getattr(entry, "link", "") or "")
                    if not link:
                        continue
                    eid = hashlib.sha1(f"{platform}:{link}".encode("utf-8")).hexdigest()[:16]
                    events.append(
                        RawEvent(
                            event_id=eid,
                            source_platform=platform,
                            source_name=name,
                            ingested_at_utc=now_iso,
                            published_at_utc=now_iso,
                            payload={
                                "title": getattr(entry, "title", ""),
                                "url": link,
                                "summary": getattr(entry, "summary", "")[:800],
                            },
                        )
                    )
            except Exception as exc:
                log.warning("RSS %s failed: %s", name, exc)

        self.persist_events(events)
        return events
