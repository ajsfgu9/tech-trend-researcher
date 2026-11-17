"""
Telegram Output Duplicator (`src/telegram_notifier.py`)

Duplicates the complete output of `tech-trend-researcher` (`Output 1: Trending Now`
and `Output 2: Sorted Database Items With Identical Viral Features`) and delivers it
to a Telegram Bot:
1. Sends a structured, scannable Telegram digest message (chunked safely at 4,000 chars)
   covering both Output 1 (top trends + X Breakout Scores + cashtags) and Output 2
   (`YES` with matching database items and ready-to-post X hooks, or `NO`).
2. Optionally attaches the exact generated `dashboard.md` and `content_briefs.json`
   files via Telegram's `sendDocument` API so the operator or downstream automation
   receives the full verbatim artifacts on Telegram.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

TELEGRAM_MAX_MSG_LEN = 4000


def format_telegram_report(payload: dict[str, Any], trigger_reason: str = "scheduled") -> str:
    """Format the two-part pipeline result into a clean Telegram message."""
    gen_at = payload.get("generated_at_utc", datetime.now(timezone.utc).isoformat())
    out1 = payload.get("output_1_trending_now", [])
    out2 = payload.get("output_2_database_viral_matches", {})

    lines = [
        f"📡 *TECH TREND RESEARCHER REPORT* (`{trigger_reason}`)",
        f"🕒 Generated: `{gen_at}`",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "🔥 *PART 1: WHAT IS TRENDING NOW (X/TWITTER BREAKOUT)*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if not out1:
        lines.append("_No active trends above threshold in this window._")
    else:
        for idx, b in enumerate(out1[:8], start=1):
            xb = b.get("x_twitter_breakout_engine", {})
            lp = b.get("longitudinal_trend_proof", {})
            cashtags = " ".join(xb.get("target_x_cashtags", [])) or "#Semiconductors"
            lines.append(
                f"*{idx:02d}. {b.get('bottleneck_concept', 'Trend')}* ({b.get('vertical', '')})\n"
                f"• Headline: {b.get('headline_recommendation', '')[:110]}\n"
                f"• Editorial Score: `{b.get('editorial_score', 0):.1f}` | "
                f"X Breakout: `{xb.get('x_breakout_score', 0):.1f}` (`{xb.get('will_trend_on_x', 'NO')}`)\n"
                f"• X Window: `{xb.get('x_arbitrage_window', '')}` | Regime: `{lp.get('burst_regime', '')}`\n"
                f"• Cashtags: {cashtags} | Action: _{b.get('decision_tier', '')}_"
            )
            lines.append("")

    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━",
            "🧬 *PART 2: DATABASE VIRAL FEATURE MATCHES (`ai-hardware-bot`)*",
            "━━━━━━━━━━━━━━━━━━━━",
            f"• *Match Status*: *{out2.get('has_viral_db_matches', 'NO')}*",
            f"• *Summary*: {out2.get('summary', '')}",
            "",
        ]
    )

    if out2.get("has_viral_db_matches") == "YES":
        for m in out2.get("matched_database_items", [])[:6]:
            hook = (
                m.get("x_catalyst_playbook", {})
                .get("hook_tweet_draft", "")
                .replace("\n", " ")[:180]
            )
            lines.append(
                f"🎯 *[DB #{m.get('db_item_id')}] {m.get('title', '')[:90]}*\n"
                f"• URL: {m.get('url', '')}\n"
                f"• DB Relevance: `{m.get('db_relevance_score', 0):.1f}` | "
                f"Viral Match: `{m.get('viral_match_score', 0):.1f}%` | "
                f"X Breakout: `{m.get('x_breakout_score', 0):.1f}`\n"
                f"• Matched Trend: `{m.get('matched_social_trend', '')}` (`{m.get('x_arbitrage_window', '')}`)\n"
                f"• X Hook Draft: _{hook}_"
            )
            lines.append("")

    return "\n".join(lines).strip()


def _chunk_text(text: str, max_len: int = TELEGRAM_MAX_MSG_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > max_len and current:
            chunks.append("".join(current).rstrip())
            current = [line]
            cur_len = len(line)
        else:
            current.append(line)
            cur_len += len(line)
    if current:
        chunks.append("".join(current).rstrip())
    return chunks


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    session: Any | None = None,
) -> bool:
    if not bot_token or not chat_id:
        log.info("Telegram credentials not configured; skipping Telegram message.")
        return False

    http = session or requests
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    all_ok = True

    for chunk in _chunk_text(text):
        try:
            resp = http.post(
                api_url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )
            if not resp.ok:
                log.warning("Telegram sendMessage HTTP %s: %s", resp.status_code, resp.text[:160])
                all_ok = False
        except Exception as exc:
            log.warning("Telegram sendMessage failed: %s", exc)
            all_ok = False
    return all_ok


def send_telegram_document(
    bot_token: str,
    chat_id: str,
    filename: str,
    content_str: str,
    caption: str = "",
    session: Any | None = None,
) -> bool:
    if not bot_token or not chat_id:
        return False

    http = session or requests
    api_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    bio = io.BytesIO(content_str.encode("utf-8"))
    bio.name = filename
    try:
        resp = http.post(
            api_url,
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"document": (filename, bio, "application/octet-stream")},
            timeout=30,
        )
        if not resp.ok:
            log.warning("Telegram sendDocument HTTP %s: %s", resp.status_code, resp.text[:160])
            return False
        return True
    except Exception as exc:
        log.warning("Telegram sendDocument failed for %s: %s", filename, exc)
        return False


def duplicate_output_to_telegram(
    payload: dict[str, Any],
    dashboard_md: str,
    trigger_reason: str = "scheduled",
    bot_token: str | None = None,
    chat_id: str | None = None,
    send_documents: bool | None = None,
    session: Any | None = None,
) -> bool:
    enabled = os.getenv("ENABLE_TELEGRAM_NOTIFY", "true").strip().lower() in ("1", "true", "yes")
    if not enabled and bot_token is None:
        return False

    token = (bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    target_chat = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not target_chat:
        log.info("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping Telegram duplication.")
        return False

    if send_documents is None:
        send_documents = os.getenv("SEND_TELEGRAM_DOCUMENTS", "true").strip().lower() in ("1", "true", "yes")

    report_text = format_telegram_report(payload, trigger_reason=trigger_reason)
    msg_ok = send_telegram_message(token, target_chat, report_text, session=session)

    if send_documents:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        send_telegram_document(
            token,
            target_chat,
            f"dashboard_{ts}.md",
            dashboard_md,
            caption=f"Duplicate Dashboard ({trigger_reason})",
            session=session,
        )
        send_telegram_document(
            token,
            target_chat,
            f"content_briefs_{ts}.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
            caption=f"Duplicate JSON Output ({trigger_reason})",
            session=session,
        )

    return msg_ok
