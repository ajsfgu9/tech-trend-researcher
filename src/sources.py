"""
Multi-Vertical Source Collectors:
1. AggregatorDBReader: Safely reads ai-hardware-bot's SQLite DB (WAL mode, read-only URI).
2. MultiVerticalCollector: Fetches live RSS/Atom feeds across semiconductors, financing,
   biomedical, robotics, and quantum computing.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable

from src.models import RawItem, Vertical

log = logging.getLogger(__name__)

PRIMARY_SOURCE_DOMAINS = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "sec.gov",
    "nature.com",
    "science.org",
    "ieee.org",
    "patents.google.com",
)

DEFAULT_RSS_FEEDS: dict[str, list[tuple[str, str]]] = {
    Vertical.SEMICONDUCTOR.value: [
        ("TrendForce", "https://www.trendforce.com/news/feed_v2/"),
        ("Semiconductor Engineering", "https://semiengineering.com/feed/"),
        ("ServeTheHome", "https://www.servethehome.com/feed/"),
        ("arXiv cs.AR", "https://export.arxiv.org/rss/cs.AR"),
    ],
    Vertical.FINANCING.value: [
        ("TechCrunch Venture", "https://techcrunch.com/category/venture/feed/"),
        ("Crunchbase News", "https://news.crunchbase.com/feed/"),
        ("SEC EDGAR 8-K", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom"),
    ],
    Vertical.BIOMEDICAL.value: [
        ("bioRxiv Bioinformatics", "https://connect.biorxiv.org/biorxiv_xml.php?subject=bioinformatics"),
        ("FierceBiotech", "https://www.fiercebiotech.com/rss/xml"),
        ("Nature Biotechnology", "https://www.nature.com/nbt.rss"),
    ],
    Vertical.ROBOTICS.value: [
        ("arXiv cs.RO", "https://export.arxiv.org/rss/cs.RO"),
        ("IEEE Spectrum Robotics", "https://spectrum.ieee.org/feeds/topic/robotics.rss"),
        ("The Robot Report", "https://www.therobotreport.com/feed/"),
    ],
    Vertical.QUANTUM.value: [
        ("arXiv quant-ph", "https://export.arxiv.org/rss/quant-ph"),
        ("Quantum Computing Report", "https://quantumcomputingreport.com/feed/"),
        ("Physics World Quantum", "https://physicsworld.com/c/quantum/feed/"),
    ],
}

VERTICAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    Vertical.SEMICONDUCTOR.value: (
        "tsmc", "asml", "nvidia", "amd", "intel", "hbm", "dram", "nand", "wafer",
        "lithography", "euv", "chiplet", "cowos", "silicon photonics", "cpo", "gpu",
        "npu", "asic", "foundry", "fab", "semiconductor", "gaa", "2nm", "1.4nm",
    ),
    Vertical.FINANCING.value: (
        "series a", "series b", "series c", "series d", "funding", "raised", "valuation",
        "ipo", "s-1", "capex", "acquisition", "m&a", "venture", "private equity",
        "debt financing", "credit facility", "sec filing", "8-k", "10-q", "billion",
    ),
    Vertical.BIOMEDICAL.value: (
        "crispr", "mrna", "protein", "alphafold", "fda", "clinical trial", "phase 1",
        "phase 2", "phase 3", "biotech", "genomics", "gene therapy", "antibody",
        "biomedical", "cell therapy", "bci", "neuralink", "biorxiv", "sequencing",
    ),
    Vertical.ROBOTICS.value: (
        "robot", "robotics", "humanoid", "actuator", "dexterous", "embodied ai",
        "vla", "teleoperation", "autonomous", "drone", "lidar", "slam", "manipulator",
        "figure", "optimus", "unitree", "boston dynamics",
    ),
    Vertical.QUANTUM.value: (
        "quantum", "qubit", "qec", "error correction", "logical qubit", "superconducting",
        "trapped ion", "photonic quantum", "neutral atom", "cryogenic", "dilution",
        "quant-ph", "surface code", "quantum supremacy", "quantum advantage",
    ),
}


def is_primary_source(url: str) -> bool:
    lowered = (url or "").lower()
    return any(domain in lowered for domain in PRIMARY_SOURCE_DOMAINS)


def classify_vertical(text: str, default_vertical: str = Vertical.SEMICONDUCTOR.value) -> str:
    lowered = (text or "").lower()
    scores: dict[str, int] = {}
    for vertical, keywords in VERTICAL_KEYWORDS.items():
        scores[vertical] = sum(1 for kw in keywords if kw in lowered)
    best_vertical, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_vertical if best_score > 0 else default_vertical


def _parse_datetime(raw_dt: str | None) -> datetime:
    if not raw_dt:
        return datetime.now(timezone.utc)
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw_dt.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


class AggregatorDBReader:
    """Reads recent articles from ai-hardware-bot's SQLite DB safely in read-only URI mode."""

    def __init__(self, db_path: str, exclude_sources: Iterable[str] = ("Hacker News",)) -> None:
        self.db_path = db_path
        self.exclude_sources = [s.strip() for s in exclude_sources if s.strip()]

    def fetch_recent_items(self, lookback_hours: int = 72) -> list[RawItem]:
        if not self.db_path or not os.path.exists(self.db_path):
            log.warning("Aggregator DB not found at %s; skipping local DB bridge.", self.db_path)
            return []

        uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(items)").fetchall()
            }
            if not cols:
                return []

            select_cols = ["url", "title", "source"]
            for opt in ("source_type", "published_at", "category", "summary", "description"):
                if opt in cols:
                    select_cols.append(opt)

            where_clauses = []
            params: list[str] = []
            for excluded in self.exclude_sources:
                where_clauses.append("source NOT LIKE ?")
                params.append(f"%{excluded}%")

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            order_col = "published_at" if "published_at" in cols else "id"
            query = f"SELECT {', '.join(select_cols)} FROM items {where_sql} ORDER BY {order_col} DESC LIMIT 500"

            cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            items: list[RawItem] = []
            for row in conn.execute(query, params).fetchall():
                pub_str = row["published_at"] if "published_at" in row.keys() else ""
                pub_dt = _parse_datetime(pub_str)
                if pub_dt < cutoff:
                    continue
                summary = ""
                if "summary" in row.keys() and row["summary"]:
                    summary = str(row["summary"])
                elif "description" in row.keys() and row["description"]:
                    summary = str(row["description"])

                title = str(row["title"] or "").strip()
                url = str(row["url"] or "").strip()
                if not title or not url:
                    continue

                vertical = classify_vertical(f"{title} {summary}", default_vertical=Vertical.SEMICONDUCTOR.value)
                items.append(
                    RawItem(
                        url=url,
                        title=title,
                        source=str(row["source"] or "ai-hardware-bot"),
                        published_at=pub_dt.isoformat(),
                        vertical=vertical,
                        summary=summary[:600],
                        source_type=str(row["source_type"]) if "source_type" in row.keys() else "aggregator_db",
                        is_primary_source=is_primary_source(url),
                    )
                )
            return items
        finally:
            conn.close()


class MultiVerticalCollector:
    """Collects live RSS/Atom feeds across semiconductor, financing, biomedical, robotics, and quantum."""

    def __init__(self, feeds: dict[str, list[tuple[str, str]]] | None = None, timeout: int = 12) -> None:
        self.feeds = feeds or DEFAULT_RSS_FEEDS
        self.timeout = timeout

    def fetch_all(self, verticals: list[str] | None = None, lookback_hours: int = 72) -> list[RawItem]:
        target_verticals = set(verticals) if verticals else set(self.feeds.keys())
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        collected: list[RawItem] = []

        for vertical, source_list in self.feeds.items():
            if vertical not in target_verticals:
                continue
            for source_name, feed_url in source_list:
                try:
                    import feedparser
                    parsed = feedparser.parse(feed_url, request_headers={"User-Agent": "TechTrendResearcher/1.0"})
                    for entry in parsed.entries[:30]:
                        title = str(getattr(entry, "title", "") or "").strip()
                        link = str(getattr(entry, "link", "") or "").strip()
                        if not title or not link:
                            continue
                        pub_raw = getattr(entry, "published", "") or getattr(entry, "updated", "")
                        pub_dt = _parse_datetime(pub_raw)
                        if pub_dt < cutoff:
                            continue
                        summary = str(getattr(entry, "summary", "") or "")[:600]
                        detected_vertical = classify_vertical(f"{title} {summary}", default_vertical=vertical)
                        collected.append(
                            RawItem(
                                url=link,
                                title=title,
                                source=source_name,
                                published_at=pub_dt.isoformat(),
                                vertical=detected_vertical,
                                summary=summary,
                                source_type="rss",
                                is_primary_source=is_primary_source(link),
                            )
                        )
                except Exception as exc:
                    log.warning("Failed fetching %s (%s): %s", source_name, feed_url, exc)
        return collected
