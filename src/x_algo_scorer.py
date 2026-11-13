"""
Reverse-Engineered X (Twitter) Algorithm Scorer

Grounded in xai-org/x-algorithm (Phoenix Transformer + home-mixer/params/param.rs + Grox classifiers):
- High-effort explicit actions dominate the positive weight mass (~96.9%):
  share_via_copy_link (+20.0), reply (+5.0), quote (+5.0), share_via_dm (+5.0),
  follow_author (+4.0), share (+2.0), retweet (+1.0).
- Passive / attention signals:
  favorite (+0.66), click (+0.30), open_link (+0.20), dwell (+0.05),
  cont_click_dwell_time (+0.40), cont_dwell_time (+0.006/s).
- Negative / suppression signals:
  not_interested (-52.272), not_dwelled (-0.02), Grox spam/quality filters,
  and indirect external-link suppression (url_click_score omitted from Phoenix's
  19-action prediction head; has_visible_link feature fed to scorer).
"""

from __future__ import annotations

import re
from typing import Any

from src.models import TrendCluster

# Published Phoenix / Home Mixer action weights (reference scale)
PHOENIX_ACTION_WEIGHTS: dict[str, float] = {
    "share_via_copy_link": 20.0,
    "reply": 5.0,
    "quote": 5.0,
    "share_via_dm": 5.0,
    "follow_author": 4.0,
    "share": 2.0,
    "retweet": 1.0,
    "favorite": 0.66,
    "click": 0.30,
    "open_link": 0.20,
    "dwell": 0.05,
    "cont_click_dwell_time": 0.40,
    "cont_dwell_time_per_sec": 0.006,
    "post_unexplored": 0.015,
    "not_dwelled": -0.02,
    "not_interested": -52.272,
}

# Signals that provoke replies & quote-tweets among tech fanatics
DEBATE_PATTERNS = re.compile(
    r"\b(vs\.?|versus|bottleneck|yield|trade-?off|surpass|beat|delay|cancel|shortage|"
    r"benchmark|latency|power wall|copper|optical|cpo|hbm\d*|tsmc|asml|nvidia|huawei|"
    r"breakthrough|skeptic|overvalued|valuation|bubble|fda|phase [123]|clinical|fault[- ]tolerant|"
    r"logical qubit|humanoid|actuator|funding|series [a-e]|ipo|s-1|capex)\b",
    re.IGNORECASE,
)

# Signals that drive shares/bookmarks (copy link / DM to colleagues)
UTILITY_PATTERNS = re.compile(
    r"\b(benchmark|spec|architecture|whitepaper|arxiv|biorxiv|sec filing|10-k|s-1|"
    r"dataset|open[- ]source|github|cost per|$/w|efficiency|roadmap|teardown|breakdown)\b",
    re.IGNORECASE,
)

# Visual-friendly keywords for Phoenix v5 multimodal (1024-d) parity
VISUAL_PATTERNS = re.compile(
    r"\b(chart|graph|die shot|wafer|architecture|topology|roadmap|benchmark|"
    r"comparison|table|capex|valuation|qubit|robot|humanoid|protein|molecule)\b",
    re.IGNORECASE,
)

# Clickbait / engagement-bait phrases penalized by Grox classifiers
SPAM_BAIT_PATTERNS = re.compile(
    r"\b(you won't believe|shocking|rip |dead forever|100x|guaranteed|secret trick|"
    r"like if you agree|repost this|thread 🧵👇)\b",
    re.IGNORECASE,
)


def score_trend_for_x(cluster: TrendCluster) -> tuple[float, dict[str, Any]]:
    """
    Estimate X/Twitter For You distribution potential (0-100) using
    Phoenix weighted-action probabilities and generate posting strategy
    for the downstream News Writer.
    """
    combined_text = " ".join(
        [cluster.topic] + [f"{item.title} {item.summary}" for item in cluster.items]
    )

    debate_hits = len(set(DEBATE_PATTERNS.findall(combined_text.lower())))
    utility_hits = len(set(UTILITY_PATTERNS.findall(combined_text.lower())))
    visual_hits = len(set(VISUAL_PATTERNS.findall(combined_text.lower())))
    metric_count = len(cluster.metrics_extracted)
    spam_hits = len(SPAM_BAIT_PATTERNS.findall(combined_text.lower()))

    # Estimated action probabilities [0.0, 1.0]
    p_reply = min(0.95, 0.15 + 0.14 * debate_hits + 0.05 * min(metric_count, 5))
    p_quote = min(0.90, 0.10 + 0.12 * debate_hits + 0.04 * cluster.corroboration_count)
    p_share_copy = min(0.90, 0.08 + 0.15 * utility_hits + 0.08 * min(metric_count, 4))
    p_share_dm = min(0.85, 0.08 + 0.12 * utility_hits + 0.05 * cluster.primary_source_count)
    p_follow = min(0.80, 0.10 + 0.10 * cluster.primary_source_count + 0.06 * utility_hits)
    p_retweet = min(0.90, 0.15 + 0.10 * debate_hits + 0.08 * min(metric_count, 4))
    p_favorite = min(0.95, 0.25 + 0.08 * debate_hits + 0.05 * visual_hits)

    # Dwell time estimate (seconds) & click dwell probability
    est_dwell_sec = min(45.0, 8.0 + 4.0 * min(metric_count, 5) + 3.0 * utility_hits)
    p_click_dwell = min(0.90, 0.15 + 0.10 * visual_hits + 0.08 * utility_hits)

    # Negative risk from hype/clickbait without substance
    p_not_interested = min(0.35, 0.02 + 0.10 * spam_hits + (0.08 if metric_count == 0 and cluster.corroboration_count == 1 else 0.0))

    raw_phoenix_score = (
        PHOENIX_ACTION_WEIGHTS["share_via_copy_link"] * p_share_copy
        + PHOENIX_ACTION_WEIGHTS["reply"] * p_reply
        + PHOENIX_ACTION_WEIGHTS["quote"] * p_quote
        + PHOENIX_ACTION_WEIGHTS["share_via_dm"] * p_share_dm
        + PHOENIX_ACTION_WEIGHTS["follow_author"] * p_follow
        + PHOENIX_ACTION_WEIGHTS["retweet"] * p_retweet
        + PHOENIX_ACTION_WEIGHTS["favorite"] * p_favorite
        + PHOENIX_ACTION_WEIGHTS["cont_click_dwell_time"] * p_click_dwell
        + PHOENIX_ACTION_WEIGHTS["cont_dwell_time_per_sec"] * est_dwell_sec
        + PHOENIX_ACTION_WEIGHTS["not_interested"] * p_not_interested
    )

    # Max theoretical benchmark ~ 36.0 -> normalize to 0-100
    normalized_score = max(0.0, min(100.0, (raw_phoenix_score / 36.0) * 100.0))

    # Determine optimal X format & strategy for the News Writer
    if metric_count >= 3 or cluster.primary_source_count >= 1:
        recommended_format = "mini_thread_3_to_4_posts"
    elif visual_hits >= 1:
        recommended_format = "single_post_with_chart_or_diagram"
    else:
        recommended_format = "single_analytical_post_with_question"

    if visual_hits >= 1 or metric_count >= 2:
        visual_suggestion = "Attach a clean comparison table, benchmark chart, or architecture diagram (leverages Phoenix v5 multimodal embeddings)."
    else:
        visual_suggestion = "Include a crisp screenshot of the key filing/abstract figure or a 2-column spec card."

    drivers = []
    if p_share_copy >= 0.40:
        drivers.append("high_bookmark_and_link_share_utility")
    if p_reply >= 0.45:
        drivers.append("high_reply_debate_potential")
    if est_dwell_sec >= 18.0:
        drivers.append("strong_dwell_time_depth")
    if visual_hits >= 1:
        drivers.append("multimodal_v5_visual_hook")

    strategy = {
        "recommended_format": recommended_format,
        "link_placement": "first_reply_only",
        "link_rationale": "Avoid placing external URLs in the parent post (url_click_score is omitted from Phoenix's 19-action head and has_visible_link risks indirect suppression).",
        "visual_suggestion": visual_suggestion,
        "author_engagement_rule": "Reply to top 3-5 substantive technical comments within the first 30-60 minutes to capture early engagement velocity and author-reply boost.",
        "predicted_drivers": drivers or ["baseline_tech_interest"],
        "estimated_dwell_seconds": round(est_dwell_sec, 1),
        "raw_phoenix_proxy": round(raw_phoenix_score, 3),
    }

    return normalized_score, strategy
