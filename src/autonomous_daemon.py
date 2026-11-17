"""
Autonomous Dual-Trigger Daemon & State Watcher (`src/autonomous_daemon.py`)

Ensures `tech-trend-researcher` runs 100% autonomously under two independent triggers
with zero external scheduler dependencies (uses Python stdlib `time.monotonic()` +
`sqlite3` state watermarks):
1. Database-Change Trigger (`DBStateWatcher`):
   - Polls `ai-hardware-bot`'s SQLite database (`agi_hardware_trends.db`, read-only WAL mode)
     every `DB_WATCH_POLL_SECONDS` (default 15s).
   - Inspects `MAX(id)`, `COUNT(*)`, and `MAX(created_at)` on `items`.
   - Persists the last processed database watermark in `MetadataRepository` (`pipeline_state` table)
     so even across container restarts, any new or updated items sorted by `ai-hardware-bot`
     immediately trigger a fresh trend research & Telegram duplication run.
2. Time-Based Scheduled Trigger (`SCHEDULE_INTERVAL_MINUTES`, default 30m):
   - Tracks elapsed monotonic time and triggers a full scheduled sweep every
     `SCHEDULE_INTERVAL_MINUTES` even if `agi_hardware_trends.db` had no new inserts,
     ensuring live social/X velocity shifts are captured on schedule.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class DBStateWatcher:
    """Detects new or updated records in ai-hardware-bot's SQLite database (`agi_hardware_trends.db`)."""

    def __init__(self, aggregator_db_path: str, state_db_path: str) -> None:
        self.aggregator_db_path = aggregator_db_path
        self.state_db_path = state_db_path
        os.makedirs(os.path.dirname(os.path.abspath(state_db_path)), exist_ok=True)
        self._init_state_table()

    def _init_state_table(self) -> None:
        with sqlite3.connect(self.state_db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )

    def _get_saved_signature(self) -> str | None:
        with sqlite3.connect(self.state_db_path, timeout=10.0) as conn:
            row = conn.execute(
                "SELECT state_value FROM pipeline_state WHERE state_key = 'aggregator_db_signature'"
            ).fetchone()
            return str(row[0]) if row else None

    def _save_signature(self, signature: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.state_db_path, timeout=10.0) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_state (state_key, state_value, updated_at_utc)
                VALUES ('aggregator_db_signature', ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (signature, now_iso),
            )

    def compute_current_db_signature(self) -> str | None:
        if not self.aggregator_db_path or not os.path.exists(self.aggregator_db_path):
            return None

        uri = f"file:{os.path.abspath(self.aggregator_db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            tables = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "items" not in tables:
                return None

            cols = {r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()}
            ts_col = "created_at" if "created_at" in cols else (
                "published_at" if "published_at" in cols else "id"
            )
            row = conn.execute(
                f"SELECT COALESCE(MAX(id), 0), COUNT(*), COALESCE(MAX({ts_col}), '') FROM items"
            ).fetchone()
            max_id, count, max_ts = int(row[0] or 0), int(row[1] or 0), str(row[2] or "")
            return f"{max_id}:{count}:{max_ts}"
        except Exception as exc:
            log.warning("Could not inspect aggregator DB state at %s: %s", self.aggregator_db_path, exc)
            return None
        finally:
            conn.close()

    def check_and_mark_changed(self) -> tuple[bool, str | None]:
        current_sig = self.compute_current_db_signature()
        if current_sig is None:
            return False, None

        saved_sig = self._get_saved_signature()
        if saved_sig != current_sig:
            self._save_signature(current_sig)
            return True, current_sig
        return False, current_sig


def touch_heartbeat(heartbeat_path: str) -> None:
    if not heartbeat_path:
        return
    try:
        p = Path(heartbeat_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write heartbeat file %s: %s", heartbeat_path, exc)


def run_autonomous_daemon(
    pipeline_callback: Callable[[str], None],
    watcher: DBStateWatcher,
    schedule_interval_minutes: int = 30,
    db_watch_poll_seconds: int = 15,
    heartbeat_path: str = "/app/data/heartbeat.txt",
    max_cycles: int | None = None,
) -> None:
    interval_sec = max(10.0, float(schedule_interval_minutes) * 60.0)
    poll_sec = max(1.0, float(db_watch_poll_seconds))

    log.info(
        "Starting autonomous dual-trigger daemon (time_interval=%dm, db_poll=%ds)",
        schedule_interval_minutes,
        db_watch_poll_seconds,
    )

    watcher.check_and_mark_changed()
    pipeline_callback("startup")
    touch_heartbeat(heartbeat_path)
    last_scheduled_ts = time.monotonic()

    cycles = 0
    try:
        while max_cycles is None or cycles < max_cycles:
            time.sleep(poll_sec)
            cycles += 1
            now_mono = time.monotonic()

            changed, sig = watcher.check_and_mark_changed()
            if changed:
                log.info("Database change detected in ai-hardware-bot (%s); triggering pipeline.", sig)
                pipeline_callback(f"db_change:{sig}")
                last_scheduled_ts = time.monotonic()
            elif (now_mono - last_scheduled_ts) >= interval_sec:
                log.info("Time-based scheduled trigger fired (interval=%dm).", schedule_interval_minutes)
                pipeline_callback("scheduled_timer")
                last_scheduled_ts = time.monotonic()

            touch_heartbeat(heartbeat_path)
    except (KeyboardInterrupt, SystemExit):
        log.info("Autonomous daemon shutting down cleanly.")
