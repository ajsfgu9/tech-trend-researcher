"""HTML-escaped Telegram notifier with chunking, retries, and artifact uploads."""

from __future__ import annotations

import html
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.http_client import build_http_session

log = logging.getLogger(__name__)

TELEGRAM_MAX_MSG_LEN = 3900


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=False)


def format_telegram_report_html(payload: dict[str, Any], trigger_reason: str = "scheduled") -> str:
    """Format the two-part trend & multi-DB match report as valid Telegram HTML."""
    gen_at = _esc(payload.get("generated_at_utc", datetime.now(timezone.utc).isoformat()))
    out1 = payload.get("output_1_external_trends", [])
    out2 = payload.get("output_2_database_viral_matches", {})
    health = payload.get("source_health", {})

    ext_summary = ", ".join(
        f"{c.get('collector')}={c.get('items_collected')}({c.get('status')})"
        for c in health.get("external_collectors", [])
        if c.get("enabled")
    )
    db_summary = ", ".join(
        f"{name}={info.get('loaded_items')}({info.get('status')})"
        for name, info in (health.get("bot_databases") or {}).items()
    )

    lines = [
        f"📡 <b>TECH TREND RESEARCHER</b> (<code>{_esc(trigger_reason)}</code>)",
        f"🕒 <code>{gen_at}</code>",
        f"📊 <b>External:</b> <code>{_esc(ext_summary or 'none')}</code>",
        f"🗄 <b>Bot DBs:</b> <code>{_esc(db_summary or 'none')}</code>",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "🔥 <b>PART 1: LIVE EXTERNAL TRENDS</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if not out1:
        lines.append("<i>No live external trends collected in this window.</i>")
    else:
        for idx, tr in enumerate(out1[:8], start=1):
            platforms = ", ".join(tr.get("platforms", []))
            analysis = tr.get("llm_analysis") or {}
            summary = str(analysis.get("summary") or "")[:180]
            lead_url = ""
            sources = tr.get("sources") or []
            if sources and sources[0].get("url"):
                lead_url = str(sources[0]["url"])

            title_html = (
                f'<a href="{html.escape(lead_url, quote=True)}">{_esc(tr.get("label", "Trend")[:100])}</a>'
                if lead_url.startswith(("http://", "https://"))
                else _esc(tr.get("label", "Trend")[:100])
            )
            z_part = (
                f" | z=<code>{tr['avg_velocity_z']:+.2f}</code>"
                if tr.get("avg_velocity_z") is not None
                else ""
            )
            lines.append(
                f"<b>{idx:02d}. {title_html}</b>\n"
                f"• Score: <code>{tr.get('trend_score', 0):.1f}</code> | "
                f"Rate: <code>{tr.get('max_engagement_rate_per_hour', 0):.1f}/hr</code>{z_part}\n"
                f"• Platforms: <code>{_esc(platforms)}</code> ({tr.get('signal_count', 0)} signal(s))"
            )
            if summary:
                lines.append(f"• Note: {_esc(summary)}")
            lines.append("")

    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━",
            "🧬 <b>PART 2: BOT DATABASE MATCHES (3 DBs vs EXTERNAL)</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"• <b>Status:</b> <b>{_esc(out2.get('has_viral_db_matches', 'NO'))}</b>",
            f"• <b>Summary:</b> {_esc(out2.get('summary', ''))}",
            "",
        ]
    )

    if out2.get("has_viral_db_matches") == "YES":
        for m in out2.get("matched_database_items", [])[:6]:
            url = str(m.get("url") or "")
            title = _esc(str(m.get("title") or "")[:95])
            title_link = (
                f'<a href="{html.escape(url, quote=True)}">{title}</a>'
                if url.startswith(("http://", "https://"))
                else title
            )
            lines.append(
                f"🎯 <b>[{_esc(m.get('bot_name'))} #{m.get('db_item_id')}] {title_link}</b>\n"
                f"• Domain: <code>{_esc(m.get('domain') or '-')}</code> | "
                f"DB Rel: <code>{m.get('db_relevance_score', 0):.1f}</code> | "
                f"Match: <code>{m.get('combined_match_score', 0):.1f}%</code>\n"
                f"• Live Trend: <code>{_esc(str(m.get('matched_trend_label') or '')[:80])}</code>"
            )
            if m.get("llm_rationale"):
                lines.append(f"• Link: {_esc(str(m['llm_rationale'])[:180])}")
            if m.get("llm_x_hook"):
                lines.append(f"• Hook: <i>{_esc(str(m['llm_x_hook'])[:220])}</i>")
            lines.append("")

    return "\n".join(lines).strip()


def _chunk_html_lines(text: str, max_len: int = TELEGRAM_MAX_MSG_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > max_len and buf:
            chunks.append("".join(buf).rstrip())
            buf = [line]
            cur_len = len(line)
        else:
            buf.append(line)
            cur_len += len(line)
    if buf:
        chunks.append("".join(buf).rstrip())
    return chunks


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    html_text: str,
    session: Any | None = None,
) -> bool:
    if not bot_token or not chat_id:
        log.info("Telegram credentials not configured; skipping Telegram message.")
        return False

    http = session or build_http_session()
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    all_ok = True

    for chunk in _chunk_html_lines(html_text):
        try:
            resp = http.post(
                api_url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )
            if not resp.ok:
                log.warning("Telegram sendMessage HTTP %s: %s", resp.status_code, resp.text[:200])
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

    http = session or build_http_session()
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
            log.warning("Telegram sendDocument HTTP %s: %s", resp.status_code, resp.text[:200])
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
    enabled = os.getenv("ENABLE_TELEGRAM_NOTIFY", "true").strip().lower() in ("1", "true", "yes", "on")
    if not enabled and bot_token is None:
        return False

    token = (bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    target_chat = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not target_chat:
        log.info("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping Telegram delivery.")
        return False

    if send_documents is None:
        send_documents = os.getenv("SEND_TELEGRAM_DOCUMENTS", "true").strip().lower() in (
            "1", "true", "yes", "on"
        )

    report_html = format_telegram_report_html(payload, trigger_reason=trigger_reason)
    msg_ok = send_telegram_message(token, target_chat, report_html, session=session)

    if send_documents:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        send_telegram_document(
            token,
            target_chat,
            f"dashboard_{ts}.md",
            dashboard_md,
            caption=f"Tech Trend Dashboard ({trigger_reason})",
            session=session,
        )
        send_telegram_document(
            token,
            target_chat,
            f"trend_report_{ts}.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
            caption=f"Tech Trend JSON ({trigger_reason})",
            session=session,
        )

    return msg_ok
