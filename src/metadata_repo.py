"""
Metadata Repository (SQLite WAL mode)
Stores historical entity/bottleneck daily time-series counts, co-occurrence baselines,
and source engagement distributions so longitudinal trends can be detected reliably.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone


class MetadataRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_baselines (
                    source_platform TEXT PRIMARY KEY,
                    mu_engagement REAL NOT NULL,
                    sigma_engagement REAL NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    false_positive_rate REAL NOT NULL DEFAULT 0.0,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entity_graph (
                    entity_a TEXT NOT NULL,
                    entity_b TEXT NOT NULL,
                    co_occurrence_count INTEGER NOT NULL DEFAULT 1,
                    last_seen_utc TEXT NOT NULL,
                    PRIMARY KEY (entity_a, entity_b)
                );
                CREATE TABLE IF NOT EXISTS concept_timeseries (
                    trend_key TEXT NOT NULL,
                    bucket_date TEXT NOT NULL,
                    mention_count INTEGER NOT NULL DEFAULT 0,
                    weighted_intensity REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (trend_key, bucket_date)
                );
                """
            )

    def get_source_baseline(self, source_platform: str) -> tuple[float, float]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT mu_engagement, sigma_engagement FROM source_baselines WHERE source_platform = ?",
                (source_platform,),
            ).fetchone()
            if row:
                return float(row["mu_engagement"]), max(1.0, float(row["sigma_engagement"]))
        defaults = {
            "hacker_news": (45.0, 60.0),
            "reddit": (35.0, 50.0),
            "x_twitter": (1500.0, 3000.0),
            "github": (80.0, 120.0),
            "technical_news": (10.0, 8.0),
            "sec": (12.0, 8.0),
            "arxiv": (5.0, 4.0),
            "biorxiv": (5.0, 4.0),
            "mainstream_media": (100.0, 150.0),
        }
        return defaults.get(source_platform, (20.0, 25.0))

    def update_source_baseline(self, source_platform: str, observations: list[float]) -> None:
        if not observations:
            return
        n = len(observations)
        mu = sum(observations) / n
        var = sum((x - mu) ** 2 for x in observations) / max(1, n)
        sigma = max(1.0, var ** 0.5)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_baselines (source_platform, mu_engagement, sigma_engagement, sample_count, updated_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_platform) DO UPDATE SET
                    mu_engagement = 0.7 * source_baselines.mu_engagement + 0.3 * excluded.mu_engagement,
                    sigma_engagement = 0.7 * source_baselines.sigma_engagement + 0.3 * excluded.sigma_engagement,
                    sample_count = source_baselines.sample_count + excluded.sample_count,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (source_platform, mu, sigma, n, now),
            )

    def record_entity_links(self, entities: list[str]) -> None:
        if len(entities) < 2:
            return
        now = datetime.now(timezone.utc).isoformat()
        sorted_ents = sorted(set(entities))
        with self._connect() as conn:
            for i in range(len(sorted_ents)):
                for j in range(i + 1, len(sorted_ents)):
                    conn.execute(
                        """
                        INSERT INTO entity_graph (entity_a, entity_b, co_occurrence_count, last_seen_utc)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(entity_a, entity_b) DO UPDATE SET
                            co_occurrence_count = co_occurrence_count + 1,
                            last_seen_utc = excluded.last_seen_utc
                        """,
                        (sorted_ents[i], sorted_ents[j], now),
                    )

    def get_co_occurrence_count(self, entity_a: str, entity_b: str) -> int:
        a, b = sorted((entity_a, entity_b))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT co_occurrence_count FROM entity_graph WHERE entity_a = ? AND entity_b = ?",
                (a, b),
            ).fetchone()
            return int(row["co_occurrence_count"]) if row else 0

    def record_concept_bucket(self, trend_key: str, bucket_date: str, count: int, intensity: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO concept_timeseries (trend_key, bucket_date, mention_count, weighted_intensity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(trend_key, bucket_date) DO UPDATE SET
                    mention_count = excluded.mention_count,
                    weighted_intensity = excluded.weighted_intensity
                """,
                (trend_key, bucket_date, int(count), float(intensity)),
            )

    def get_concept_history(self, trend_key: str) -> list[tuple[str, int, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT bucket_date, mention_count, weighted_intensity
                FROM concept_timeseries
                WHERE trend_key = ?
                ORDER BY bucket_date ASC
                """,
                (trend_key,),
            ).fetchall()
            return [(str(r["bucket_date"]), int(r["mention_count"]), float(r["weighted_intensity"])) for r in rows]
