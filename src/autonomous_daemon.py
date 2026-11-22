"""Fault-tolerant dual-trigger autonomous daemon with graceful shutdown."""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.db_bridge import MultiDBStateWatcher

log = logging.getLogger(__name__)


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
    watcher: MultiDBStateWatcher,
    schedule_interval_minutes: int = 30,
    db_watch_poll_seconds: int = 15,
    heartbeat_path: str = "/app/data/heartbeat.txt",
    max_cycles: int | None = None,
) -> None:
    """Run pipeline on multi-DB changes and fixed time intervals without crashing on transient errors."""
    interval_sec = max(10.0, float(schedule_interval_minutes) * 60.0)
    poll_sec = max(1.0, float(db_watch_poll_seconds))
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        log.info("Received signal %s; shutting down daemon gracefully.", signum)
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                pass

    def _safe_run(reason: str) -> bool:
        try:
            pipeline_callback(reason)
            return True
        except Exception:
            log.exception("Pipeline run (%s) failed; daemon remains alive for next cycle.", reason)
            return False

    log.info(
        "Starting autonomous dual-trigger daemon (time_interval=%dm, db_poll=%ds, dbs=%d)",
        schedule_interval_minutes,
        db_watch_poll_seconds,
        len(watcher.db_configs),
    )

    watcher.check_and_mark_changed(force_inspect=True)
    _safe_run("startup")
    touch_heartbeat(heartbeat_path)
    last_scheduled_ts = time.monotonic()

    cycles = 0
    while not stop_event.is_set() and (max_cycles is None or cycles < max_cycles):
        if stop_event.wait(timeout=poll_sec):
            break
        cycles += 1
        now_mono = time.monotonic()

        changed, sig = watcher.check_and_mark_changed()
        if changed:
            log.info("Upstream DB change detected (%s); triggering pipeline.", sig)
            _safe_run(f"db_change:{sig}")
            last_scheduled_ts = time.monotonic()
        elif (now_mono - last_scheduled_ts) >= interval_sec:
            log.info("Scheduled timer fired (interval=%dm).", schedule_interval_minutes)
            _safe_run("scheduled_timer")
            last_scheduled_ts = time.monotonic()

        touch_heartbeat(heartbeat_path)

    log.info("Autonomous daemon stopped cleanly.")
