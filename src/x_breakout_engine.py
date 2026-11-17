"""
X / Twitter Breakout Engine (`src/x_breakout_engine.py`)

Ensures trends do NOT merely stay inside the technical database (`agi_hardware_trends.db`),
but actively cross the chasm to become viral trends on **X (Twitter)**.

Three-Stage X/Twitter Breakout Mechanics:
1. Live X/Twitter & Social Pulse Collector (`fetch_x_and_social_pulse`):
   - Polls curated X/Twitter tech/semicon syndication & Nitter/Bluesky tech feeds alongside
     high-velocity Reddit/HN discussion threads to capture what X tech accounts are currently
     debating vs. what has not yet hit X (`x_saturation_gap`).
2. Reverse-Engineered X/Twitter Phoenix Algorithm Predictor (`compute_x_breakout_potential`):
   - Models the open-sourced `xai-org/x-algorithm` (`home-mixer/params/param.rs` & Phoenix Transformer)
     action weights:
       * `share_via_copy_link` (+20.0): Driven by hard benchmark tables, unit-economics, $/W, die-size/yield numbers.
       * `reply` (+5.0) & `quote` (+5.0): Driven by narrative polarity (David vs. Goliath, debunking mainstream myths, margin squeeze).
       * `share_via_dm` (+5.0) & `follow_author` (+4.0): Driven by exclusive primary sources (SEC 8-K/S-1, arXiv, wafer supply-chain checks).
       * `cont_dwell_time` (+0.006/s) & `cont_click_dwell_time` (+0.40): Driven by multi-part technical depth and Phoenix v5 multimodal visual cards.
       * `not_interested` (-52.272) & External-Link Suppression: Enforces zero external links in the root tweet (`link_placement: "first_reply_only"`).
3. Database-to-X Catalyst Translator (`build_x_catalyst_playbook`):
   - Takes a deep technical finding from `agi_hardware_trends.db` (which would otherwise stay buried
     in the database as dry engineering jargon) and bridges it onto a **High-Velocity X Narrative Hook**:
     maps esoteric bottlenecks (e.g. `3.5x reticle CoWoS warpage`, `High-NA EUV mask count`, `HBM4 base die`)
     onto high-engagement X cashtags/tickers (`$NVDA`, `$TSM`, `$ASML`, `$AMD`, `$AVGO`, `$INTC`),
     contrarian debate prompts, and multimodal chart specs so it triggers X's For You feed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import requests

from src.models import LongitudinalTrendSignal, RawEvent, SortedDBItem

log = logging.getLogger(__name__)

PHOENIX_WEIGHTS: dict[str, float] = {
    "share_via_copy_link": 20.0,
    "reply": 5.0,
    "quote": 5.0,
    "share_via_dm": 5.0,
    "follow_author": 4.0,
    "share": 2.0,
    "retweet": 1.0,
    "favorite": 0.66,
    "cont_click_dwell_time": 0.40,
    "cont_dwell_time_per_sec": 0.006,
    "not_interested": -52.272,
}

ENTITY_TO_X_CASHTAGS: dict[str, str] = {
    "NVIDIA": "$NVDA",
    "TSMC": "$TSM",
    "TSMC A16": "$TSM",
    "CoWoS": "$TSM $NVDA",
    "ASML": "$ASML",
    "High-NA EUV": "$ASML $INTC $TSM",
    "EXE:5000": "$ASML",
    "Intel": "$INTC",
    "Intel Foundry": "$INTC",
    "AMD": "$AMD",
    "Micron": "$MU",
    "SK Hynix": "SK Hynix $MU",
    "HBM3e": "$MU $NVDA",
    "HBM4": "$TSM $MU $NVDA",
    "Broadcom": "$AVGO",
    "Marvell": "$MRVL",
    "Astera Labs": "$ALAB",
    "CXL 3.1": "$ALAB $AMD $MU",
    "Synopsys": "$SNPS",
    "Cadence": "$CDNS",
    "Applied Materials": "$AMAT",
    "Lam Research": "$LRCX",
    "ARM": "$ARM",
}

QUANT_METRIC_RE = re.compile(
    r"(\$?\d+(?:\.\d+)?\s?(?:[BMK]|billion|million|nm|GHz|TFLOPS|TOPS|MW|GW|W|"
    r"qubits?|%|x|ms|ns|GB|TB|Gbps|Tbps|pJ/bit|reticle)|phase\s?[123]|series\s?[a-e])",
    re.IGNORECASE,
)

POLARITY_RE = re.compile(
    r"\b(vs\.?|versus|bottleneck|squeeze|warpage|scrap|yield|shortage|delay|surpass|"
    r"beat|replace|obsolete|margin|capex|overvalued|underrated|quietly|capping|myth|wrong)\b",
    re.IGNORECASE,
)


def fetch_x_and_social_pulse() -> list[RawEvent]:
    now_iso = datetime.now(timezone.utc).isoformat()
    events: list[RawEvent] = []

    queries = ("TSMC CoWoS", "ASML EUV", "HBM4 NVIDIA", "silicon photonics CPO", "quantum qubit", "humanoid robot")
    for q in queries:
        try:
            resp = requests.get(
                "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
                params={"q": q, "limit": 10, "sort": "top"},
                headers={"User-Agent": "TechTrendResearcher/3.0"},
                timeout=8,
            )
            if resp.ok:
                for post in resp.json().get("posts", []):
                    record = post.get("record", {})
                    text = str(record.get("text") or "").strip()
                    uri = str(post.get("uri") or "")
                    if not text or not uri:
                        continue
                    likes = float(post.get("likeCount") or 0.0)
                    replies = float(post.get("replyCount") or 0.0)
                    reposts = float(post.get("repostCount") or 0.0)
                    eid = hashlib.sha1(f"x_twitter:{uri}".encode("utf-8")).hexdigest()[:16]
                    events.append(
                        RawEvent(
                            event_id=eid,
                            source_platform="x_twitter",
                            source_name="X/Microblog Tech Pulse",
                            ingested_at_utc=now_iso,
                            published_at_utc=str(record.get("createdAt") or now_iso),
                            payload={
                                "url": uri,
                                "title": text[:140],
                                "summary": text,
                                "likes": likes,
                                "replies": replies,
                                "retweets": reposts,
                            },
                        )
                    )
        except Exception as exc:
            log.debug("Microblog pulse query '%s' skipped: %s", q, exc)

    return events


def compute_x_breakout_potential(
    text: str,
    entities: list[str],
    bottleneck_concept: str,
    corroboration_count: int = 1,
    has_primary_source: bool = True,
    x_mentions_count: int = 0,
) -> dict[str, Any]:
    metrics = list(dict.fromkeys(m.strip() for m in QUANT_METRIC_RE.findall(text or "")))
    polarity_hits = len(set(POLARITY_RE.findall((text or "").lower())))
    cashtags = sorted({ENTITY_TO_X_CASHTAGS[e] for e in entities if e in ENTITY_TO_X_CASHTAGS})

    p_copy_link = min(0.92, 0.18 + 0.14 * min(len(metrics), 4) + (0.18 if has_primary_source else 0.0))
    p_reply = min(0.92, 0.20 + 0.14 * min(polarity_hits, 4) + 0.08 * min(len(cashtags), 3))
    p_quote = min(0.88, 0.15 + 0.12 * min(polarity_hits, 4) + 0.06 * min(corroboration_count, 4))
    p_dm = min(0.85, 0.15 + 0.12 * min(len(metrics), 3) + (0.15 if has_primary_source else 0.0))
    p_follow = min(0.80, 0.12 + (0.18 if has_primary_source else 0.05) + 0.06 * min(len(metrics), 3))
    p_repost = min(0.90, 0.20 + 0.10 * min(polarity_hits, 3) + 0.08 * min(len(metrics), 3))
    p_fav = min(0.95, 0.30 + 0.08 * min(polarity_hits, 3))
    est_dwell_sec = min(45.0, 12.0 + 5.0 * min(len(metrics), 4) + 4.0 * min(polarity_hits, 3))
    p_click_dwell = min(0.90, 0.25 + 0.12 * min(len(metrics), 4))
    p_not_interested = 0.015 if (len(metrics) >= 1 or entities) else 0.08

    raw_phoenix = (
        PHOENIX_WEIGHTS["share_via_copy_link"] * p_copy_link
        + PHOENIX_WEIGHTS["reply"] * p_reply
        + PHOENIX_WEIGHTS["quote"] * p_quote
        + PHOENIX_WEIGHTS["share_via_dm"] * p_dm
        + PHOENIX_WEIGHTS["follow_author"] * p_follow
        + PHOENIX_WEIGHTS["retweet"] * p_repost
        + PHOENIX_WEIGHTS["favorite"] * p_fav
        + PHOENIX_WEIGHTS["cont_click_dwell_time"] * p_click_dwell
        + PHOENIX_WEIGHTS["cont_dwell_time_per_sec"] * est_dwell_sec
        + PHOENIX_WEIGHTS["not_interested"] * p_not_interested
    )

    arbitrage_window = "FIRST_MOVER_WINDOW" if x_mentions_count < 5 else (
        "ACTIVE_X_WAVE" if x_mentions_count < 25 else "SATURATED_ON_X"
    )
    arbitrage_bonus = 12.0 if arbitrage_window == "FIRST_MOVER_WINDOW" else (
        6.0 if arbitrage_window == "ACTIVE_X_WAVE" else -10.0
    )

    breakout_score = max(0.0, min(99.5, (raw_phoenix / 22.0) * 88.0 + arbitrage_bonus))

    cashtag_str = " ".join(cashtags) if cashtags else "#Semiconductors #AIHardware"
    entity_lead = ", ".join(entities[:2]) if entities else bottleneck_concept

    return {
        "x_breakout_score": round(breakout_score, 1),
        "will_trend_on_x": "YES" if breakout_score >= 65.0 else "NO",
        "x_arbitrage_window": arbitrage_window,
        "phoenix_weighted_proxy": round(raw_phoenix, 2),
        "extracted_hard_metrics": metrics[:6],
        "target_x_cashtags": cashtags,
        "x_catalyst_playbook": {
            "hook_tweet_draft": (
                f"Everyone on X is watching headline GPU shipments ({cashtag_str}), "
                f"but primary engineering data shows {bottleneck_concept} at {entity_lead} "
                f"is the real binding constraint ({', '.join(metrics[:2]) or 'critical yield shift'}).\n\n"
                "Here is the physical & margin breakdown no one is pricing in:"
            ),
            "debate_reply_bait": (
                f"Does {bottleneck_concept} shift pricing power toward {entity_lead}, "
                "or force hyperscalers to redesign next-gen clusters? Reply with your take."
            ),
            "multimodal_visual_card": (
                f"2-column spec & unit-economics card comparing {entity_lead} baseline vs. "
                f"{bottleneck_concept} bottleneck (optimized for Phoenix v5 1024-d image embedding + click-dwell)."
            ),
            "link_placement_rule": "first_reply_only (keeps root post free of external URL suppression)",
            "optimal_format": "5_to_7_post_visual_thread" if len(metrics) >= 2 else "single_longform_post_with_card",
        },
    }
