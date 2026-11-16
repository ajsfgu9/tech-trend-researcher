"""
Stage 1: Read-Only Aggregator Bridge (`src/raw_layer.py`)

Strict Separation of Concerns:
- All external HTTP/RSS/API scraping lives exclusively in `jijiji88888/ai-hardware-bot`.
- `tech-trend-researcher` performs ZERO duplicate network scraping; it reads ingested
  items and historical time-series observations from `agi_hardware_trends.db` in
  read-only SQLite WAL mode (`?mode=ro`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from src.models import RawEvent

log = logging.getLogger(__name__)


def _infer_platform(source_name: str, source_type: str, url: str) -> str:
    s_low = f"{source_name} {source_type} {url}".lower()
    if "hacker news" in s_low or "ycombinator" in s_low or source_type == "hn":
        return "hacker_news"
    if "reddit" in s_low or "r/" in s_low:
        return "reddit"
    if "arxiv" in s_low:
        return "arxiv"
    if "biorxiv" in s_low or "medrxiv" in s_low:
        return "biorxiv"
    if "github.com" in s_low:
        return "github"
    if "sec.gov" in s_low or "edgar" in s_low or "8-k" in s_low or "s-1" in s_low:
        return "sec"
    if any(m in s_low for m in ("verge", "techcrunch", "tom's hardware", "ars technica", "reuters", "bloomberg")):
        return "mainstream_media"
    return "technical_news"


class ImmutableRawLayer:
    """Reads ingested records from ai-hardware-bot's SQLite DB without duplicate scraping."""

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

    def ingest_from_aggregator_db(self, lookback_hours: int = 336) -> list[RawEvent]:
        """
        Read historical and recent items from jijiji88888/ai-hardware-bot (`agi_hardware_trends.db`)
        over a multi-day window (default 14 days / 336h) for longitudinal trend analysis.
        """
        if not self.aggregator_db_path or not os.path.exists(self.aggregator_db_path):
            log.warning("Aggregator DB not found at %s", self.aggregator_db_path)
            return []

        uri = f"file:{os.path.abspath(self.aggregator_db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        now_iso = datetime.now(timezone.utc).isoformat()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        events: list[RawEvent] = []
        try:
            tables = {
                r["name"]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "items" not in tables:
                return []

            cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            select_cols = [
                c for c in (
                    "url", "title", "source", "source_type", "published_at",
                    "category", "summary", "engagement_points", "comments_count",
                )
                if c in cols
            ]
            order_col = "published_at" if "published_at" in cols else "id"
            rows = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM items ORDER BY {order_col} DESC LIMIT 2500"
            ).fetchall()

            obs_by_url: dict[str, dict[str, float]] = {}
            if "item_observations" in tables:
                for obs in conn.execute(
                    "SELECT url, MAX(engagement_points) AS max_pts, MAX(comments_count) AS max_com "
                    "FROM item_observations GROUP BY url"
                ).fetchall():
                    obs_by_url[str(obs["url"])] = {
                        "points": float(obs["max_pts"] or 0.0),
                        "num_comments": float(obs["max_com"] or 0.0),
                    }

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

                url = str(d.get("url") or "")
                src = str(d.get("source") or "ai-hardware-bot")
                stype = str(d.get("source_type") or "")
                platform = _infer_platform(src, stype, url)

                if url in obs_by_url:
                    d["points"] = max(float(d.get("engagement_points") or 0.0), obs_by_url[url]["points"])
                    d["num_comments"] = max(float(d.get("comments_count") or 0.0), obs_by_url[url]["num_comments"])
                else:
                    d.setdefault("points", float(d.get("engagement_points") or 0.0))
                    d.setdefault("num_comments", float(d.get("comments_count") or 0.0))

                eid = hashlib.sha1(f"{platform}:{url}:{pub_iso}".encode("utf-8")).hexdigest()[:16]
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
