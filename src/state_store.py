"""SQLite WAL state store for real time-series snapshots, baselines, watermarks, and embedding cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone


class StateStore:
    """Persistent SQLite WAL store backing velocity deltas, real baselines, and caches."""

    MIN_BASELINE_SAMPLES = 10

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signal_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    observed_at_utc TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    comments REAL NOT NULL DEFAULT 0.0,
                    shares REAL NOT NULL DEFAULT 0.0,
                    engagement_total REAL NOT NULL DEFAULT 0.0,
                    engagement_rate_per_hour REAL NOT NULL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_signal_time
                    ON signal_snapshots (signal_id, observed_at_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_snapshots_platform_time
                    ON signal_snapshots (platform, observed_at_utc DESC);

                CREATE TABLE IF NOT EXISTS pipeline_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )

    def get_state(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_value FROM pipeline_state WHERE state_key = ?",
                (key,),
            ).fetchone()
            return str(row["state_value"]) if row else None

    def set_state(self, key: str, value: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_state (state_key, state_value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (key, value, now_iso),
            )

    def get_previous_snapshot(self, signal_id: str) -> dict[str, float | str] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT observed_at_utc, score, comments, shares, engagement_total, engagement_rate_per_hour
                FROM signal_snapshots
                WHERE signal_id = ?
                ORDER BY observed_at_utc DESC
                LIMIT 1
                """,
                (signal_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "observed_at_utc": str(row["observed_at_utc"]),
                "score": float(row["score"]),
                "comments": float(row["comments"]),
                "shares": float(row["shares"]),
                "engagement_total": float(row["engagement_total"]),
                "engagement_rate_per_hour": float(row["engagement_rate_per_hour"]),
            }

    def record_snapshot(
        self,
        signal_id: str,
        platform: str,
        observed_at_utc: str,
        score: float,
        comments: float,
        shares: float,
        engagement_total: float,
        engagement_rate_per_hour: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_snapshots (
                    signal_id, platform, observed_at_utc,
                    score, comments, shares, engagement_total, engagement_rate_per_hour
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    platform,
                    observed_at_utc,
                    float(score),
                    float(comments),
                    float(shares),
                    float(engagement_total),
                    float(engagement_rate_per_hour),
                ),
            )

    def get_platform_rate_baseline(
        self,
        platform: str,
        lookback_days: int = 14,
    ) -> tuple[float, float, int] | None:
        """Return (mean, std, sample_count) from real historical snapshots only.

        Returns None when fewer than MIN_BASELINE_SAMPLES real observations exist
        so callers never fabricate z-scores from hardcoded guesses.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT engagement_rate_per_hour
                FROM signal_snapshots
                WHERE platform = ? AND observed_at_utc >= ?
                ORDER BY observed_at_utc DESC
                LIMIT 1000
                """,
                (platform, cutoff),
            ).fetchall()

        values = [float(r["engagement_rate_per_hour"]) for r in rows]
        n = len(values)
        if n < self.MIN_BASELINE_SAMPLES:
            return None

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
        std = math.sqrt(variance)
        return mean, max(std, 1e-6), n

    def prune_old_snapshots(self, retain_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retain_days)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM signal_snapshots WHERE observed_at_utc < ?",
                (cutoff,),
            )
            return cur.rowcount

    @staticmethod
    def make_embedding_cache_key(model_name: str, dimensions: int, text: str) -> str:
        digest = hashlib.sha256(f"{model_name}:{dimensions}:{text}".encode("utf-8")).hexdigest()
        return digest

    def get_cached_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        found: dict[str, list[float]] = {}
        with self._connect() as conn:
            for i in range(0, len(keys), 200):
                chunk = keys[i : i + 200]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT cache_key, vector_json FROM embedding_cache WHERE cache_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    try:
                        found[str(row["cache_key"])] = json.loads(str(row["vector_json"]))
                    except Exception:
                        continue
        return found

    def save_cached_embeddings(
        self,
        model_name: str,
        vectors_by_key: dict[str, list[float]],
    ) -> None:
        if not vectors_by_key:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (cache_key, model_name, vector_json, created_at_utc)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (k, model_name, json.dumps(vec), now_iso)
                    for k, vec in vectors_by_key.items()
                ],
            )
