"""
Stage 1: Broad Social Trend Collectors & `ai-hardware-bot` Sorted DB Reader (`src/raw_layer.py`)

Distinct Roles:
1. `collect_broad_social_streams()`: Polls broad social & developer discussion channels
   (Specialized Subreddits `r/semiconductors`, `r/hardware`, `r/LocalLLaMA`, `r/biotech`,
   `r/robotics`, `r/QuantumComputing`, Hacker News Algolia, GitHub active repos/discussions)
   to discover what is actively trending and debating online.
2. `load_sorted_db_items()`: Reads the LLM-sorted `items` table from `jijiji88888/ai-hardware-bot`
   (`agi_hardware_trends.db`, columns `hardware_domain`, `bottleneck_addressed`, `specific_tech`,
   `relevance_score`, `why_it_matters`, `ai_revolution_thesis`, `raw_summary`) so the researcher
   can cross-compare active social trends against previously sorted database items.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import requests

from src.models import RawEvent, SortedDBItem

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

    def collect_broad_social_streams(self) -> list[RawEvent]:
        """
        Fetch live discussion & velocity streams from broad social media and developer hubs
        (Reddit, Hacker News Algolia, GitHub) to identify what is currently trending.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        events: list[RawEvent] = []

        for sub in ("semiconductors", "hardware", "LocalLLaMA", "biotech", "robotics", "QuantumComputing"):
            try:
                resp = requests.get(
                    f"https://www.reddit.com/r/{sub}/hot.json?limit=20",
                    headers={"User-Agent": "TechTrendResearcher/3.0"},
                    timeout=10,
                )
                if resp.ok:
                    for child in resp.json().get("data", {}).get("children", []):
                        post = child.get("data", {})
                        url = post.get("url") or f"https://www.reddit.com{post.get('permalink', '')}"
                        title = str(post.get("title") or "").strip()
                        if not url or not title:
                            continue
                        pub_ts = datetime.fromtimestamp(float(post.get("created_utc") or 0), tz=timezone.utc).isoformat()
                        eid = hashlib.sha1(f"reddit:{url}".encode("utf-8")).hexdigest()[:16]
                        events.append(
                            RawEvent(
                                event_id=eid,
                                source_platform="reddit",
                                source_name=f"r/{sub}",
                                ingested_at_utc=now_iso,
                                published_at_utc=pub_ts,
                                payload={
                                    "url": url,
                                    "title": title,
                                    "summary": str(post.get("selftext") or "")[:1200],
                                    "score": float(post.get("score") or post.get("ups") or 0.0),
                                    "num_comments": float(post.get("num_comments") or 0.0),
                                },
                            )
                        )
            except Exception as exc:
                log.warning("Reddit r/%s collection skipped: %s", sub, exc)

        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search?tags=story&numericFilters=points>20&hitsPerPage=40",
                timeout=10,
            )
            if resp.ok:
                for hit in resp.json().get("hits", []):
                    url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
                    title = str(hit.get("title") or "").strip()
                    if not url or not title:
                        continue
                    eid = hashlib.sha1(f"hacker_news:{url}".encode("utf-8")).hexdigest()[:16]
                    events.append(
                        RawEvent(
                            event_id=eid,
                            source_platform="hacker_news",
                            source_name="Hacker News",
                            ingested_at_utc=now_iso,
                            published_at_utc=str(hit.get("created_at") or now_iso),
                            payload={
                                "url": url,
                                "title": title,
                                "summary": str(hit.get("story_text") or "")[:1000],
                                "points": float(hit.get("points") or 0.0),
                                "num_comments": float(hit.get("num_comments") or 0.0),
                            },
                        )
                    )
        except Exception as exc:
            log.warning("HN Algolia collection skipped: %s", exc)

        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": "topic:cuda OR topic:rocm OR topic:riscv OR topic:quantum OR topic:robotics",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 20,
                },
                headers={"User-Agent": "TechTrendResearcher/3.0", "Accept": "application/vnd.github+json"},
                timeout=10,
            )
            if resp.ok:
                for repo in resp.json().get("items", []):
                    url = str(repo.get("html_url") or "").strip()
                    name = str(repo.get("full_name") or "").strip()
                    desc = str(repo.get("description") or "").strip()
                    if not url or not name:
                        continue
                    eid = hashlib.sha1(f"github:{url}".encode("utf-8")).hexdigest()[:16]
                    events.append(
                        RawEvent(
                            event_id=eid,
                            source_platform="github",
                            source_name="GitHub",
                            ingested_at_utc=now_iso,
                            published_at_utc=str(repo.get("pushed_at") or now_iso),
                            payload={
                                "url": url,
                                "title": f"{name}: {desc}"[:220],
                                "summary": desc,
                                "points": float(repo.get("stargazers_count") or 0.0),
                                "num_comments": float(repo.get("open_issues_count") or 0.0),
                            },
                        )
                    )
        except Exception as exc:
            log.warning("GitHub collection skipped: %s", exc)

        self.persist_events(events)
        return events

    def load_sorted_db_items(self, lookback_hours: int = 720, min_relevance: float = 0.0) -> list[SortedDBItem]:
        """
        Read previously sorted, LLM-classified items from jijiji88888/ai-hardware-bot
        (`agi_hardware_trends.db`) in read-only SQLite WAL mode.
        """
        if not self.aggregator_db_path or not os.path.exists(self.aggregator_db_path):
            log.warning("ai-hardware-bot DB not found at %s", self.aggregator_db_path)
            return []

        uri = f"file:{os.path.abspath(self.aggregator_db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        sorted_items: list[SortedDBItem] = []
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            if not cols:
                return []

            rows = conn.execute("SELECT * FROM items ORDER BY id DESC LIMIT 2000").fetchall()
            for r in rows:
                d = dict(r)
                pub_str = str(d.get("published_at") or d.get("created_at") or "")
                if pub_str:
                    try:
                        dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt < cutoff:
                            continue
                    except ValueError:
                        pass

                rel = float(d.get("relevance_score") or 0.0)
                if rel < min_relevance:
                    continue

                sorted_items.append(
                    SortedDBItem(
                        id=int(d.get("id") or 0),
                        url=str(d.get("url") or ""),
                        title=str(d.get("title") or ""),
                        source=str(d.get("source") or ""),
                        source_type=str(d.get("source_type") or ""),
                        published_at=pub_str,
                        category=str(d.get("category") or ""),
                        hardware_domain=str(d.get("hardware_domain") or ""),
                        bottleneck_addressed=str(d.get("bottleneck_addressed") or ""),
                        specific_tech=str(d.get("specific_tech") or ""),
                        relevance_score=rel,
                        why_it_matters=str(d.get("why_it_matters") or ""),
                        ai_revolution_thesis=str(d.get("ai_revolution_thesis") or ""),
                        raw_summary=str(d.get("raw_summary") or d.get("summary") or ""),
                    )
                )
        finally:
            conn.close()

        return sorted_items
