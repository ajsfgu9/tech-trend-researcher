from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from src.autonomous_daemon import run_autonomous_daemon
from src.db_bridge import MultiDBStateWatcher, load_all_bot_databases, open_readonly_sqlite
from src.external_collectors import collect_hacker_news
from src.llm_client import DualLLMClient
from src.models import ExternalSignal, SourceDBConfig
from src.researcher import run_pipeline
from src.similarity import EmbeddingEngine
from src.state_store import StateStore
from src.telegram_notifier import duplicate_output_to_telegram


class DummyResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        return self._payload


class DummyTelegramSession:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.documents: list[str] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: int = 20,
    ) -> DummyResponse:
        if "sendMessage" in url and json:
            self.messages.append(json)
        elif "sendDocument" in url and files:
            self.documents.append(files["document"][0])
        return DummyResponse(200, {"ok": True})


def _create_hw_or_deeptech_db(db_path: str, rows: list[tuple[Any, ...]]) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            source TEXT,
            source_type TEXT,
            published_at TEXT,
            category TEXT,
            hardware_domain TEXT,
            bottleneck_addressed TEXT,
            specific_tech TEXT,
            relevance_score REAL,
            why_it_matters TEXT,
            ai_revolution_thesis TEXT,
            raw_summary TEXT,
            created_at TEXT
        )
        """
    )
    if rows:
        conn.executemany(
            """
            INSERT INTO items (
                url, title, source, source_type, published_at, category,
                hardware_domain, bottleneck_addressed, specific_tech,
                relevance_score, why_it_matters, ai_revolution_thesis, raw_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    conn.close()


def _create_software_stack_db(db_path: str, rows: list[tuple[Any, ...]]) -> None:
    """Matches ai-software-stack-quantum-bot's schema (`stack_layer`, `ai_stack_thesis`)."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            source TEXT,
            source_type TEXT,
            published_at TEXT,
            category TEXT,
            stack_layer TEXT,
            bottleneck_addressed TEXT,
            specific_tech TEXT,
            relevance_score REAL,
            why_it_matters TEXT,
            ai_stack_thesis TEXT,
            raw_summary TEXT,
            created_at TEXT
        )
        """
    )
    if rows:
        conn.executemany(
            """
            INSERT INTO items (
                url, title, source, source_type, published_at, category,
                stack_layer, bottleneck_addressed, specific_tech,
                relevance_score, why_it_matters, ai_stack_thesis, raw_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    conn.close()


def _build_three_bot_dbs(tmp_path) -> list[SourceDBConfig]:
    now_iso = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()

    # Deliberately name hardware DB `agi_hardware_trends2.db` while configuring `agi_hardware_trends.db`
    # to test single-DB auto-detection.
    hw_actual = str(tmp_path / "hw" / "agi_hardware_trends2.db")
    hw_configured = str(tmp_path / "hw" / "agi_hardware_trends.db")
    _create_hw_or_deeptech_db(
        hw_actual,
        [
            (
                "https://semiengineering.com/cowos-warpage-2026",
                "Thermomechanical Warpage in 3.5x Reticle TSMC CoWoS-L Interposers",
                "Semiconductor Engineering",
                "specialist_rss",
                now_iso,
                "packaging",
                "Advanced Packaging",
                "CoWoS-L interposer warpage and packaging yield ceiling",
                "TSMC CoWoS-L 3.5x reticle interposer",
                9.1,
                "Interposer warpage limits AI accelerator packaging yields.",
                "Packaging yield caps 2026 GPU cluster delivery.",
                "Finite-element analysis of CoWoS-L interposer warpage.",
                now_iso,
            ),
        ],
    )

    dt_db = str(tmp_path / "dt" / "frontier_deeptech_trends.db")
    _create_hw_or_deeptech_db(
        dt_db,
        [
            (
                "https://world-nuclear-news.org/smr-molten-salt-2026",
                "Molten Salt Small Modular Reactor Achieves 99.4% Thermal Loop Stability",
                "World Nuclear News",
                "specialist_rss",
                now_iso,
                "energy",
                "Advanced Energy & Planetary Systems",
                "Baseload industrial heat and grid reliability for off-grid facilities",
                "Thorium Molten Salt SMR",
                8.4,
                "Demonstrates passive decay-heat removal under load-following swings.",
                "Compact nuclear heat unlocks high-density industrial compute.",
                "Detailed loop stability benchmark for molten salt SMR.",
                now_iso,
            ),
        ],
    )

    sw_db = str(tmp_path / "sw" / "ai_stack_quantum_trends.db")
    _create_software_stack_db(
        sw_db,
        [
            (
                "https://github.com/sgl-project/sglang/releases/v0.6",
                "SGLang Disaggregated Prefill-Decode KV-Cache Engine Cuts FP4 Latency by 38%",
                "GitHub Releases",
                "rss",
                now_iso,
                "inference",
                "Inference Serving & KV-Cache Systems",
                "Prefill-decode interference and KV-cache HBM bandwidth saturation",
                "SGLang RadixAttention FP4 disaggregated serving",
                9.3,
                "Separating prefill and decode nodes raises tokens/sec per GPU.",
                "Serving orchestration unlocks more throughput than raw silicon alone.",
                "Benchmarking SGLang FP4 disaggregated KV-cache serving on MoE models.",
                now_iso,
            ),
        ],
    )

    return [
        SourceDBConfig("ai_hardware", hw_configured),
        SourceDBConfig("frontier_deeptech", dt_db),
        SourceDBConfig("ai_software_stack", sw_db),
    ]


def test_three_bot_schema_normalization_and_autodetect(tmp_path) -> None:
    configs = _build_three_bot_dbs(tmp_path)
    items, health = load_all_bot_databases(configs, lookback_hours=336, min_relevance=7.0)

    assert len(items) == 3
    assert {i.bot_name for i in items} == {"ai_hardware", "frontier_deeptech", "ai_software_stack"}
    assert len({i.composite_id for i in items}) == 3  # No ID collision across DBs

    sw_item = next(i for i in items if i.bot_name == "ai_software_stack")
    assert sw_item.domain == "Inference Serving & KV-Cache Systems"
    assert "Serving orchestration" in sw_item.thesis
    assert all(info["status"] == "ok" for info in health.values())


def test_no_circular_self_matching_when_external_empty(tmp_path) -> None:
    """Critical regression test: empty external signals MUST yield 0 trends and NO DB matches."""
    configs = _build_three_bot_dbs(tmp_path)
    payload, md = run_pipeline(
        db_configs=configs,
        state_db_path=str(tmp_path / "state.db"),
        output_json_path=str(tmp_path / "out.json"),
        dashboard_md_path=str(tmp_path / "dash.md"),
        llm_client=DualLLMClient(provider="none"),
        embedding_engine=EmbeddingEngine(enable_embeddings=False),
        external_signals_override=[],
        trigger_reason="test_empty_external",
    )

    assert payload["output_1_external_trends"] == []
    assert payload["output_2_database_viral_matches"]["has_viral_db_matches"] == "NO"
    assert payload["output_2_database_viral_matches"]["matched_count"] == 0
    assert "**NO**" in md


def test_end_to_end_multi_db_match_telegram_html_and_watcher(tmp_path) -> None:
    configs = _build_three_bot_dbs(tmp_path)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    pub_iso = (now - timedelta(hours=3)).isoformat()

    external_signals = [
        ExternalSignal(
            signal_id="hn_sglang_1",
            platform="hacker_news",
            source_name="Hacker News",
            url="https://news.ycombinator.com/item?id=55501",
            title="SGLang disaggregated prefill decode KV-cache FP4 inference benchmarks",
            summary="Engineers debating SGLang RadixAttention KV-cache serving throughput and FP4 latency.",
            published_at_utc=pub_iso,
            fetched_at_utc=now_iso,
            score=280.0,
            comments=125.0,
            age_hours=3.0,
        ),
        ExternalSignal(
            signal_id="gh_sglang_2",
            platform="github",
            source_name="GitHub Velocity",
            url="https://github.com/sgl-project/sglang-fp4-bench",
            title="sglang-fp4-bench: Disaggregated KV-cache serving benchmark for MoE models",
            summary="Reproducing SGLang FP4 prefill-decode latency gains on multi-node clusters.",
            published_at_utc=pub_iso,
            fetched_at_utc=now_iso,
            score=410.0,
            comments=18.0,
            shares=52.0,
            tags=["sglang", "kv-cache", "fp4"],
            age_hours=3.0,
        ),
        ExternalSignal(
            signal_id="lob_unrelated",
            platform="lobsters",
            source_name="Lobste.rs",
            url="https://lobste.rs/s/abc123",
            title="Writing a minimalist PostgreSQL B-tree inspector in Rust",
            summary="Deep dive into PostgreSQL page layout and heap tuples.",
            published_at_utc=pub_iso,
            fetched_at_utc=now_iso,
            score=95.0,
            comments=22.0,
            tags=["rust", "databases"],
            age_hours=3.0,
        ),
    ]

    state_db = str(tmp_path / "state.db")
    tg = DummyTelegramSession()
    payload, md = run_pipeline(
        db_configs=configs,
        state_db_path=state_db,
        output_json_path=str(tmp_path / "report.json"),
        dashboard_md_path=str(tmp_path / "dashboard.md"),
        llm_client=DualLLMClient(provider="none"),
        embedding_engine=EmbeddingEngine(enable_embeddings=False),
        external_signals_override=external_signals,
        trigger_reason="test_run",
        telegram_session=tg,
    )

    assert len(payload["output_1_external_trends"]) >= 2
    out2 = payload["output_2_database_viral_matches"]
    assert out2["has_viral_db_matches"] == "YES"
    matched_bots = {m["bot_name"] for m in out2["matched_database_items"]}
    assert "ai_software_stack" in matched_bots
    assert "frontier_deeptech" not in matched_bots  # SMR should not falsely match SGLang/Postgres

    # Verify Telegram HTML delivery
    ok = duplicate_output_to_telegram(
        payload=payload,
        dashboard_md=md,
        trigger_reason="test_run",
        bot_token="123:TEST",
        chat_id="999",
        send_documents=True,
        session=tg,
    )
    assert ok is True
    assert tg.messages[0]["parse_mode"] == "HTML"
    assert "PART 1: LIVE EXTERNAL TRENDS" in tg.messages[0]["text"]
    assert len(tg.documents) == 2

    # Verify MultiDBStateWatcher detects changes across any of the 3 DBs
    watcher = MultiDBStateWatcher(configs, StateStore(state_db))
    changed1, _ = watcher.check_and_mark_changed(force_inspect=True)
    assert changed1 is True
    changed2, _ = watcher.check_and_mark_changed()
    assert changed2 is False


def test_dual_llm_provider_failover_and_header_auth() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class FakeSession:
        def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: int = 60) -> DummyResponse:
            calls.append((url, headers))
            if "openrouter.ai" in url:
                return DummyResponse(503, {"error": "overloaded"})
            return DummyResponse(
                200,
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "internal chain of thought", "thought": True},
                                    {"text": '{"is_true_match": true, "rationale": "Exact SGLang FP4 match"}'},
                                ]
                            }
                        }
                    ]
                },
            )

    client = DualLLMClient(
        provider="openrouter",
        fallback_provider="gemini",
        openrouter_api_key="or_test",
        openrouter_model="deepseek/deepseek-v4-flash",
        gemini_api_key="gem_test_secret",
        gemini_model="gemini-3.5-flash-lite",
        session=FakeSession(),
        max_retries=1,
    )
    res = client.generate_json("sys", "usr")
    assert res == {"is_true_match": True, "rationale": "Exact SGLang FP4 match"}
    assert len(calls) == 2
    gem_url, gem_headers = calls[1]
    assert "gemini-3.5-flash-lite:generateContent" in gem_url
    assert "key=" not in gem_url  # Never leak API key in query string
    assert gem_headers["x-goog-api-key"] == "gem_test_secret"


def test_hn_collector_uses_search_by_date_and_reports_http_error() -> None:
    seen_urls: list[str] = []

    class FakeErrSession:
        def get(self, url: str, params: dict[str, Any] | None = None, timeout: int = 15) -> DummyResponse:
            seen_urls.append(url)
            return DummyResponse(403, text="Forbidden")

    sigs, st = collect_hacker_news(FakeErrSession())
    assert sigs == []
    assert st.status == "error"
    assert "403" in st.detail
    assert "search_by_date" in seen_urls[0]


def test_daemon_survives_pipeline_exception(tmp_path) -> None:
    configs = _build_three_bot_dbs(tmp_path)
    watcher = MultiDBStateWatcher(configs, StateStore(str(tmp_path / "state.db")))
    reasons: list[str] = []

    def flaky_callback(reason: str) -> None:
        reasons.append(reason)
        if len(reasons) == 1:
            raise RuntimeError("Simulated transient outage")

    hb = str(tmp_path / "heartbeat.txt")
    run_autonomous_daemon(
        pipeline_callback=flaky_callback,
        watcher=watcher,
        schedule_interval_minutes=1,
        db_watch_poll_seconds=1,
        heartbeat_path=hb,
        max_cycles=1,
    )
    assert reasons == ["startup"]
    assert os.path.isfile(hb)


def test_readonly_snapshot_reader_does_not_mutate_db(tmp_path) -> None:
    db_file = str(tmp_path / "ro_test.db")
    _create_hw_or_deeptech_db(db_file, [])
    with open_readonly_sqlite(db_file) as conn:
        try:
            conn.execute("DELETE FROM items")
            assert False, "Expected OperationalError on read-only connection"
        except sqlite3.OperationalError:
            pass
