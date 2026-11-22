"""Read-only multi-database bridge and watermark watcher for the 3 upstream intelligence bots.

Supports:
1. ai-hardware-bot (`hardware_domain`, `ai_revolution_thesis`)
2. frontier-deeptech-bot (`hardware_domain`, `ai_revolution_thesis`)
3. ai-software-stack-quantum-bot (`stack_layer`, `ai_stack_thesis`)

Safely reads WAL-mode SQLite databases mounted `:ro` in Docker (or owned by a
different container UID) by falling back to a private temporary snapshot of
`.db` + `.db-wal` when direct `?mode=ro` cannot open/create `-shm` in place.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

from src.models import BotDBItem, SourceDBConfig
from src.state_store import StateStore

log = logging.getLogger(__name__)

DEFAULT_SOURCE_DB_SPECS = (
    "ai_hardware=/data/ai_hardware_bot/agi_hardware_trends.db,"
    "frontier_deeptech=/data/frontier_deeptech_bot/frontier_deeptech_trends.db,"
    "ai_software_stack=/data/ai_software_stack_quantum_bot/ai_stack_quantum_trends.db"
)


def parse_source_db_specs(
    specs_raw: str | None = None,
    legacy_single_path: str | None = None,
) -> list[SourceDBConfig]:
    """Parse SOURCE_DB_SPECS (`name=/path/file.db,...`) with backwards-compatible fallback."""
    raw = (specs_raw or "").strip()
    if not raw and legacy_single_path:
        return [SourceDBConfig(name="ai_hardware", db_path=legacy_single_path.strip())]
    if not raw:
        raw = DEFAULT_SOURCE_DB_SPECS

    configs: list[SourceDBConfig] = []
    seen_names: set[str] = set()
    for entry in raw.split(","):
        segment = entry.strip()
        if not segment:
            continue
        if "=" in segment:
            name, path = segment.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = segment
            name = os.path.splitext(os.path.basename(path))[0] or "bot_db"

        if not path or name in seen_names:
            continue
        seen_names.add(name)
        configs.append(SourceDBConfig(name=name, db_path=path))
    return configs


def resolve_db_path(db_path: str, auto_detect_single_db: bool = True) -> str | None:
    """Resolve an existing SQLite file path, with optional single-.db auto-detection in the mount dir."""
    if not db_path:
        return None
    if os.path.isfile(db_path):
        return db_path

    parent = os.path.dirname(os.path.abspath(db_path))
    if not os.path.isdir(parent):
        return None

    candidates = sorted(
        os.path.join(parent, fname)
        for fname in os.listdir(parent)
        if fname.endswith(".db") and os.path.isfile(os.path.join(parent, fname))
    )
    if not candidates:
        return None

    if auto_detect_single_db and len(candidates) == 1:
        log.info(
            "Configured DB '%s' not found; auto-detected sole SQLite DB in directory: '%s'.",
            db_path,
            candidates[0],
        )
        return candidates[0]

    log.warning(
        "Configured DB '%s' not found. Available .db files in %s: %s",
        db_path,
        parent,
        ", ".join(os.path.basename(c) for c in candidates),
    )
    return None


def _parse_iso_utc(raw_ts: str) -> datetime | None:
    if not raw_ts:
        return None
    text = raw_ts.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


@contextlib.contextmanager
def open_readonly_sqlite(db_path: str, retries: int = 3) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database strictly read-only, handling :ro WAL mounts safely.

    Strategy:
    1. If the parent directory is writable by the current user, attempt direct
       `file:<path>?mode=ro` + `PRAGMA query_only = ON`.
    2. If the mount is read-only (`:ro` Docker volume) or owned by another UID
       (preventing WAL `-shm` mapping), copy `<db>` and `<db>-wal` (if present)
       into a private temporary directory and open that snapshot cleanly.
    """
    abs_path = os.path.abspath(db_path)
    parent_dir = os.path.dirname(abs_path)

    if os.access(parent_dir, os.W_OK):
        conn: sqlite3.Connection | None = None
        try:
            uri = f"file:{abs_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            # Touch sqlite_master to confirm WAL/SHM access actually works
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            yield conn
            return
        except sqlite3.OperationalError as exc:
            log.debug("Direct RO SQLite open failed for %s (%s); using WAL snapshot fallback.", abs_path, exc)
            if conn is not None:
                conn.close()
                conn = None
        finally:
            if conn is not None:
                conn.close()

    last_err: Exception | None = None
    for attempt in range(retries):
        tmpdir = tempfile.mkdtemp(prefix="ttr_db_snap_")
        conn_snap: sqlite3.Connection | None = None
        try:
            snap_db = os.path.join(tmpdir, os.path.basename(abs_path))
            shutil.copy2(abs_path, snap_db)
            wal_src = f"{abs_path}-wal"
            if os.path.isfile(wal_src):
                try:
                    shutil.copy2(wal_src, f"{snap_db}-wal")
                except FileNotFoundError:
                    # Writer may have checkpointed and removed -wal between checks
                    pass

            conn_snap = sqlite3.connect(snap_db, timeout=5.0)
            conn_snap.row_factory = sqlite3.Row
            # Trigger WAL recovery on the private copy before setting query_only
            conn_snap.execute("SELECT count(*) FROM sqlite_master").fetchone()
            conn_snap.execute("PRAGMA query_only = ON;")
            try:
                yield conn_snap
            finally:
                conn_snap.close()
                conn_snap = None
                shutil.rmtree(tmpdir, ignore_errors=True)
            return
        except (sqlite3.DatabaseError, OSError) as exc:
            last_err = exc
            if conn_snap is not None:
                conn_snap.close()
            shutil.rmtree(tmpdir, ignore_errors=True)
            if attempt < retries - 1:
                time.sleep(0.2 * (attempt + 1))

    raise sqlite3.OperationalError(
        f"Failed to read SQLite DB snapshot at {abs_path} after {retries} attempts: {last_err}"
    )


def load_bot_db_items(
    config: SourceDBConfig,
    lookback_hours: int = 336,
    min_relevance: float = 0.0,
    auto_detect_single_db: bool = True,
    max_rows: int = 2000,
) -> tuple[list[BotDBItem], str]:
    """Load and schema-normalize classified items from one upstream bot SQLite database."""
    resolved = resolve_db_path(config.db_path, auto_detect_single_db=auto_detect_single_db)
    if not resolved:
        log.warning("Bot DB [%s] not found at %s; skipping.", config.name, config.db_path)
        return [], "missing"

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    items: list[BotDBItem] = []

    try:
        with open_readonly_sqlite(resolved) as conn:
            tables = {
                str(r["name"])
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "items" not in tables:
                log.warning("Bot DB [%s] at %s has no 'items' table.", config.name, resolved)
                return [], "no_items_table"

            rows = conn.execute(
                "SELECT * FROM items ORDER BY id DESC LIMIT ?",
                (int(max_rows),),
            ).fetchall()

            for row in rows:
                d = dict(row)
                pub_str = str(d.get("published_at") or d.get("created_at") or "").strip()
                pub_dt = _parse_iso_utc(pub_str)
                if pub_dt is not None and pub_dt < cutoff:
                    continue

                rel = float(d.get("relevance_score") or 0.0)
                if rel < min_relevance:
                    continue

                item_id = int(d.get("id") or 0)
                domain = str(
                    d.get("hardware_domain")
                    or d.get("stack_layer")
                    or d.get("category")
                    or ""
                ).strip()
                thesis = str(
                    d.get("ai_revolution_thesis")
                    or d.get("ai_stack_thesis")
                    or ""
                ).strip()

                items.append(
                    BotDBItem(
                        bot_name=config.name,
                        item_id=item_id,
                        composite_id=f"{config.name}:{item_id}",
                        url=str(d.get("url") or "").strip(),
                        title=str(d.get("title") or "").strip(),
                        source=str(d.get("source") or config.name).strip(),
                        source_type=str(d.get("source_type") or "").strip(),
                        published_at=pub_str,
                        domain=domain,
                        bottleneck_addressed=str(d.get("bottleneck_addressed") or "").strip(),
                        specific_tech=str(d.get("specific_tech") or "").strip(),
                        relevance_score=rel,
                        why_it_matters=str(d.get("why_it_matters") or "").strip(),
                        thesis=thesis,
                        raw_summary=str(d.get("raw_summary") or d.get("summary") or "").strip(),
                    )
                )
        return items, "ok"
    except Exception as exc:
        log.error("Failed reading bot DB [%s] at %s: %s", config.name, resolved, exc)
        return [], f"error:{type(exc).__name__}"


def load_all_bot_databases(
    configs: list[SourceDBConfig],
    lookback_hours: int = 336,
    min_relevance: float = 0.0,
    auto_detect_single_db: bool = True,
) -> tuple[list[BotDBItem], dict[str, dict[str, int | str]]]:
    """Load items from all configured bot DBs and return items + per-DB health summary."""
    all_items: list[BotDBItem] = []
    db_health: dict[str, dict[str, int | str]] = {}

    for cfg in configs:
        items, status = load_bot_db_items(
            config=cfg,
            lookback_hours=lookback_hours,
            min_relevance=min_relevance,
            auto_detect_single_db=auto_detect_single_db,
        )
        all_items.extend(items)
        db_health[cfg.name] = {
            "status": status,
            "loaded_items": len(items),
            "configured_path": cfg.db_path,
        }

    return all_items, db_health


class MultiDBStateWatcher:
    """Watches all configured upstream bot SQLite databases for new/updated rows.

    Uses lightweight `os.stat()` pre-checks on `.db` and `.db-wal` so idle polls
    incur zero SQLite/snapshot I/O.
    """

    STATE_KEY = "multi_db_watermark"

    def __init__(
        self,
        db_configs: list[SourceDBConfig],
        state_store: StateStore,
        auto_detect_single_db: bool = True,
    ) -> None:
        self.db_configs = db_configs
        self.state_store = state_store
        self.auto_detect_single_db = auto_detect_single_db
        self._last_stat_fingerprint: str | None = None

    def _compute_stat_fingerprint(self) -> str:
        parts: list[str] = []
        for cfg in self.db_configs:
            resolved = resolve_db_path(cfg.db_path, auto_detect_single_db=self.auto_detect_single_db)
            if not resolved:
                parts.append(f"{cfg.name}:missing")
                continue
            try:
                st_db = os.stat(resolved)
                wal_path = f"{resolved}-wal"
                if os.path.isfile(wal_path):
                    st_wal = os.stat(wal_path)
                    parts.append(
                        f"{cfg.name}:{st_db.st_mtime_ns}:{st_db.st_size}:"
                        f"{st_wal.st_mtime_ns}:{st_wal.st_size}"
                    )
                else:
                    parts.append(f"{cfg.name}:{st_db.st_mtime_ns}:{st_db.st_size}:nowal")
            except OSError:
                parts.append(f"{cfg.name}:stat_err")
        return "|".join(parts)

    def compute_current_watermark(self) -> str | None:
        parts: list[str] = []
        any_found = False

        for cfg in self.db_configs:
            resolved = resolve_db_path(cfg.db_path, auto_detect_single_db=self.auto_detect_single_db)
            if not resolved:
                parts.append(f"{cfg.name}:missing")
                continue
            try:
                with open_readonly_sqlite(resolved) as conn:
                    tables = {
                        str(r["name"])
                        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    }
                    if "items" not in tables:
                        parts.append(f"{cfg.name}:notable")
                        continue
                    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(items)").fetchall()}
                    ts_col = (
                        "created_at"
                        if "created_at" in cols
                        else ("published_at" if "published_at" in cols else "id")
                    )
                    row = conn.execute(
                        f"SELECT COALESCE(MAX(id), 0), COUNT(*), COALESCE(MAX({ts_col}), '') FROM items"
                    ).fetchone()
                    max_id, cnt, max_ts = int(row[0] or 0), int(row[1] or 0), str(row[2] or "")
                    parts.append(f"{cfg.name}:{max_id}:{cnt}:{max_ts}")
                    any_found = True
            except Exception as exc:
                log.warning("Could not inspect DB [%s] watermark: %s", cfg.name, exc)
                parts.append(f"{cfg.name}:err")

        return "|".join(parts) if any_found else None

    def check_and_mark_changed(self, force_inspect: bool = False) -> tuple[bool, str | None]:
        stat_fp = self._compute_stat_fingerprint()
        if not force_inspect and self._last_stat_fingerprint == stat_fp:
            return False, self.state_store.get_state(self.STATE_KEY)

        self._last_stat_fingerprint = stat_fp
        current = self.compute_current_watermark()
        if current is None:
            return False, None

        saved = self.state_store.get_state(self.STATE_KEY)
        if saved != current:
            self.state_store.set_state(self.STATE_KEY, current)
            return True, current
        return False, current
