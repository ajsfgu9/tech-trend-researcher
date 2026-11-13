from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from src.models import Vertical
from src.researcher import run_pipeline


def _create_mock_aggregator_db(db_path: str) -> None:
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
            category TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        ("https://arxiv.org/abs/2611.0001", "fp1", "Silicon Photonics CPO vs Copper at 1.6Tbps: 3.2pJ/bit Benchmark", "arXiv cs.AR", "arxiv", now, "semiconductor"),
        ("https://semiengineering.com/cpo-vs-copper-2026", "fp2", "Co-Packaged Optics vs Copper 1.6Tbps 3.2pJ/bit Tradeoffs", "Semiconductor Engineering", "rss", now, "semiconductor"),
        ("https://www.sec.gov/Archives/edgar/data/123/s1.htm", "fp3", "AI Infrastructure Neocloud Files S-1 for $2.4B GPU Cluster Capex", "SEC EDGAR 8-K", "sec", now, "financing"),
        ("https://www.biorxiv.org/content/10.1101/2026.11.01", "fp4", "CRISPR Prime Editing Achieves 94% Fidelity in Phase 2 Trial", "bioRxiv Bioinformatics", "biorxiv", now, "biomedical"),
        ("https://arxiv.org/abs/2611.0002", "fp5", "28-DOF Humanoid Dexterous Hand with Tactile VLA Control", "arXiv cs.RO", "arxiv", now, "robotics"),
        ("https://arxiv.org/abs/2611.0003", "fp6", "Fault-Tolerant Surface Code Reaches 99.92% Logical Qubit Fidelity", "arXiv quant-ph", "arxiv", now, "quantum"),
        ("https://news.ycombinator.com/item?id=999", "fp7", "Random Hacker News Post That Should Be Excluded", "Hacker News", "rss", now, "semiconductor"),
    ]
    conn.executemany(
        "INSERT INTO items (url, content_fingerprint, title, source, source_type, published_at, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_end_to_end_offline_pipeline(tmp_path) -> None:
    agg_db = str(tmp_path / "agi_hardware_trends.db")
    trends_db = str(tmp_path / "tech_trends.db")
    out_json = str(tmp_path / "trends_output.json")

    _create_mock_aggregator_db(agg_db)

    trends = run_pipeline(
        aggregator_db_path=agg_db,
        trends_db_path=trends_db,
        output_json_path=out_json,
        lookback_hours=72,
        top_n=10,
        verticals=[v.value for v in Vertical],
        exclude_sources=("Hacker News",),
        llm_provider="none",
        offline_db_only=True,
    )

    assert len(trends) == 5  # 2 CPO items clustered into 1; HN item excluded
    verticals_found = {t.vertical for t in trends}
    assert verticals_found == {"semiconductor", "financing", "biomedical", "robotics", "quantum"}

    cpo_trend = next(t for t in trends if t.vertical == "semiconductor")
    assert cpo_trend.corroboration_count == 2
    assert cpo_trend.primary_source_count == 1
    assert cpo_trend.x_strategy["link_placement"] == "first_reply_only"
    assert cpo_trend.x_algo_score > 40.0

    with open(out_json, "r", encoding="utf-8") as f:
        exported = json.load(f)
    assert exported["selected_count"] == 5
