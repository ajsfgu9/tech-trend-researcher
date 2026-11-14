from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

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
        ("https://semianalysis.com/cowos-squeeze", "fp1", "The CoWoS Squeeze: Why Advanced Packaging Yields Are Quietly Capping AI Margins", "SemiAnalysis", "rss", now, "semiconductor"),
        ("https://news.ycombinator.com/item?id=101", "fp2", "CoWoS Interposer Warpage and TSMC Packaging Yield Bottlenecks", "Hacker News", "api", now, "semiconductor"),
        ("https://www.sec.gov/Archives/edgar/8k.htm", "fp3", "AI Datacenter Capex S-1 Filing for HBM4 & CoWoS GPU Cluster Financing", "SEC EDGAR", "sec", now, "financing"),
        ("https://www.biorxiv.org/content/crispr", "fp4", "CRISPR Prime Editing Phase 2 Trial Reaches 94% Fidelity in Genomics", "bioRxiv", "biorxiv", now, "biomedical"),
        ("https://arxiv.org/abs/2611.0002", "fp5", "Humanoid Robot Tactile Actuator with VLA Control", "arXiv cs.RO", "arxiv", now, "robotics"),
        ("https://arxiv.org/abs/2611.0003", "fp6", "Fault-Tolerant Quantum Surface Code Logical Qubit Error Correction", "arXiv quant-ph", "arxiv", now, "quantum"),
        ("https://example.com/iphone-battery", "fp7", "Best iPhone battery life tips for Black Friday", "GenericBlog", "rss", now, "consumer"),
    ]
    conn.executemany(
        "INSERT INTO items (url, content_fingerprint, title, source, source_type, published_at, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_editorial_content_brief_pipeline(tmp_path) -> None:
    agg_db = str(tmp_path / "agi_hardware_trends.db")
    raw_db = str(tmp_path / "raw_events.db")
    meta_db = str(tmp_path / "metadata_repo.db")
    out_json = str(tmp_path / "content_briefs.json")
    dash_md = str(tmp_path / "dashboard.md")

    _create_mock_aggregator_db(agg_db)

    briefs, md = run_pipeline(
        aggregator_db_path=agg_db,
        raw_db_path=raw_db,
        metadata_db_path=meta_db,
        output_json_path=out_json,
        dashboard_md_path=dash_md,
        lookback_hours=72,
        top_n=10,
        ingestion_threshold_tau=0.40,
        llm_provider="none",
        offline_db_only=True,
    )

    headlines = " ".join(b["headline_recommendation"] for b in briefs)
    assert "iPhone" not in headlines
    assert len(briefs) >= 4

    cowos = next(b for b in briefs if "CoWoS" in b["headline_recommendation"])
    assert cowos["editorial_id"].startswith("brief_")
    assert "narrative_framework" in cowos
    assert "recommended_formats" in cowos
    assert cowos["virality_metrics"]["cross_platform_dispersion"] >= 2

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    assert "| Rank | Editorial Score | Action Required |" in md
