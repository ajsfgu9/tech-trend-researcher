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
        ("https://semianalysis.com/high-na-euv-exe5000", "fp1", "High-NA EUV EXE:5000 Insertion Acceleration at TSMC A16 and Intel Foundry", "SemiAnalysis", "rss", now, "semiconductor"),
        ("https://news.ycombinator.com/item?id=101", "fp2", "ASML High-NA EUV EXE:5000 Pilot Wafer Yield and TSMC A16 Mask Reduction", "Hacker News", "api", now, "semiconductor"),
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


def test_four_stage_pipeline(tmp_path) -> None:
    agg_db = str(tmp_path / "agi_hardware_trends.db")
    raw_db = str(tmp_path / "raw_events.db")
    meta_db = str(tmp_path / "metadata_repo.db")
    out_json = str(tmp_path / "trends_output.json")
    dash_md = str(tmp_path / "dashboard.md")

    _create_mock_aggregator_db(agg_db)

    clusters, md = run_pipeline(
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

    topics = " ".join(c["topic"] for c in clusters)
    assert "iPhone" not in topics
    assert len(clusters) >= 4

    euv = next(c for c in clusters if "High-NA EUV" in c["topic"])
    assert euv["metrics"]["cross_platform_dispersion_count"] >= 2
    assert "ASML" in euv["verified_entities"]
    assert "llm_synthesis" in euv

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    assert "| Rank | Trend Cluster |" in md
