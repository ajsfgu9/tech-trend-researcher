from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.researcher import run_pipeline


def _create_mock_longitudinal_aggregator_db(db_path: str) -> None:
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
            engagement_points REAL DEFAULT 0.0,
            comments_count REAL DEFAULT 0.0
        )
        """
    )
    now = datetime.now(timezone.utc)
    d3 = (now - timedelta(days=3)).isoformat()
    d2 = (now - timedelta(days=2)).isoformat()
    d1 = (now - timedelta(days=1)).isoformat()
    d0 = now.isoformat()

    rows = [
        ("https://arxiv.org/abs/2611.0101", "fp1", "Thermomechanical Interposer Warpage in 3.5x Reticle CoWoS Packaging", "arXiv cs.AR", "arxiv", d3, "semiconductor", 15.0, 4.0),
        ("https://reddit.com/r/semiconductors/cowos_warpage", "fp2", "TSMC CoWoS interposer warpage yield bottleneck discussion", "r/semiconductors", "reddit", d2, "semiconductor", 85.0, 42.0),
        ("https://news.ycombinator.com/item?id=202", "fp3", "Why CoWoS Packaging Yields Are Quietly Capping AI Margins", "Hacker News", "hn", d1, "semiconductor", 240.0, 110.0),
        ("https://semianalysis.com/cowos-squeeze-2026", "fp4", "The CoWoS Squeeze: TSMC CoWoS Interposer Warpage & Packaging Yields", "SemiAnalysis", "rss", d0, "semiconductor", 310.0, 95.0),
        ("https://www.sec.gov/Archives/edgar/8k.htm", "fp5", "AI Datacenter Capex S-1 Filing for HBM4 & GPU Cluster Debt Financing", "SEC EDGAR", "sec", d0, "financing", 40.0, 12.0),
        ("https://www.biorxiv.org/content/crispr", "fp6", "CRISPR Prime Editing Phase 2 Trial Reaches 94% Fidelity in Genomics", "bioRxiv", "biorxiv", d1, "biomedical", 55.0, 18.0),
        ("https://arxiv.org/abs/2611.0002", "fp7", "Humanoid Robot Tactile Actuator with VLA Dexterous Control", "arXiv cs.RO", "arxiv", d1, "robotics", 45.0, 10.0),
        ("https://arxiv.org/abs/2611.0003", "fp8", "Fault-Tolerant Quantum Surface Code Logical Qubit Error Correction", "arXiv quant-ph", "arxiv", d0, "quantum", 50.0, 14.0),
        ("https://example.com/iphone-battery", "fp9", "Best iPhone battery life tips for Black Friday", "GenericBlog", "rss", d0, "consumer", 500.0, 200.0),
    ]
    conn.executemany(
        """
        INSERT INTO items
        (url, content_fingerprint, title, source, source_type, published_at, category, engagement_points, comments_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def test_longitudinal_trend_pipeline(tmp_path) -> None:
    agg_db = str(tmp_path / "agi_hardware_trends.db")
    raw_db = str(tmp_path / "raw_events.db")
    meta_db = str(tmp_path / "metadata_repo.db")
    out_json = str(tmp_path / "content_briefs.json")
    dash_md = str(tmp_path / "dashboard.md")

    _create_mock_longitudinal_aggregator_db(agg_db)

    briefs, md = run_pipeline(
        aggregator_db_path=agg_db,
        raw_db_path=raw_db,
        metadata_db_path=meta_db,
        output_json_path=out_json,
        dashboard_md_path=dash_md,
        lookback_hours=336,
        top_n=10,
        ingestion_threshold_tau=0.35,
        llm_provider="none",
    )

    headlines = " ".join(b["headline_recommendation"] for b in briefs)
    assert "iPhone" not in headlines
    assert len(briefs) == 5

    cowos = next(b for b in briefs if "CoWoS" in b["bottleneck_concept"])
    proof = cowos["longitudinal_trend_proof"]
    assert proof["origin_layer"] == "primary_engineering"
    assert len(proof["causal_propagation_chain"]) == 3
    assert proof["burst_regime"] in ("breakout", "accelerating")
    assert proof["longitudinal_structural_score_80pct"] > 0
    assert proof["reserved_item_subscore_20pct"] > 0

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    assert "| Rank | Editorial Score | Regime | Causal Chain |" in md
