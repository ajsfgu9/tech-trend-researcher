"""
Viral Feature Matcher (`src/viral_matcher.py`)

Cross-compares:
- Output 1: Active social trends discovered by `tech-trend-researcher` (`LongitudinalTrendSignal`)
- Output 2: Previously sorted items stored in `ai-hardware-bot`'s SQLite database (`SortedDBItem`)

Crucially, for every database item, it also runs `compute_x_breakout_potential()` from
`src/x_breakout_engine.py` so a deep technical discovery in `agi_hardware_trends.db` does NOT
merely sit inside the database—it gets scored for **X/Twitter Breakout Potential** and packaged
with a ready-to-post **X Catalyst Playbook** (cashtag hooks, contrarian reply bait, Phoenix v5
visual card spec, and link-in-reply rule) to turn it into a real trend on X.
"""

from __future__ import annotations

import re
from typing import Any

from src.models import LongitudinalTrendSignal, SortedDBItem
from src.relevance_gate import TRACKED_ENTITIES, extract_bottleneck_concepts
from src.x_breakout_engine import compute_x_breakout_potential

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "by",
    "is", "are", "was", "were", "as", "at", "from", "new", "how", "why", "what",
}


def _tokenize(text: str) -> set[str]:
    return {
        w.rstrip("s")
        for w in re.findall(r"[a-z0-9\-\.]{3,}", (text or "").lower())
        if w not in STOPWORDS
    }


def extract_viral_feature_signature(sig: LongitudinalTrendSignal) -> dict[str, Any]:
    entities = sorted({e for c in sig.candidates for e in c.verified_entities})
    concepts = sorted({b for c in sig.candidates for b in c.bottleneck_concepts})
    all_titles = " ".join(c.title for c in sig.candidates)
    key_tokens = sorted(_tokenize(all_titles))[:25]
    x_mentions = sum(1 for c in sig.candidates if c.source_platform == "x_twitter")
    x_profile = compute_x_breakout_potential(
        text=" ".join(f"{c.title} {c.body}" for c in sig.candidates),
        entities=entities,
        bottleneck_concept=sig.bottleneck_concept,
        corroboration_count=len(sig.candidates),
        has_primary_source=("primary_engineering" in sig.active_layers),
        x_mentions_count=x_mentions,
    )
    return {
        "bottleneck_concept": sig.bottleneck_concept,
        "all_concepts": concepts,
        "verified_entities": entities,
        "vertical": sig.vertical,
        "burst_regime": sig.burst_regime,
        "ewma_velocity_z": sig.ewma_velocity_z,
        "key_tokens": key_tokens,
        "x_breakout_profile": x_profile,
    }


def match_sorted_db_items_against_trends(
    trends: list[LongitudinalTrendSignal],
    db_items: list[SortedDBItem],
    match_threshold: float = 0.55,
) -> dict[str, Any]:
    for sig in trends:
        sig.viral_feature_signature = extract_viral_feature_signature(sig)

    matched_records: list[dict[str, Any]] = []
    seen_db_urls: set[str] = set()

    for item in db_items:
        composite_text = " ".join(
            [
                item.title,
                item.hardware_domain,
                item.bottleneck_addressed,
                item.specific_tech,
                item.why_it_matters,
                item.ai_revolution_thesis,
                item.raw_summary[:500],
            ]
        )
        item_concepts = set(extract_bottleneck_concepts(composite_text, item.category or "semiconductor"))
        item_entities = sorted({ent for ent in TRACKED_ENTITIES if ent.lower() in composite_text.lower()})
        item_tokens = _tokenize(
            f"{item.title} {item.bottleneck_addressed} {item.specific_tech} {item.hardware_domain}"
        )

        best_match_score = 0.0
        best_trend: LongitudinalTrendSignal | None = None
        shared_features: dict[str, Any] = {}

        for sig in trends:
            sig_sig = sig.viral_feature_signature
            sig_concepts = set(sig_sig.get("all_concepts", [sig.bottleneck_concept]))
            sig_entities = set(sig_sig.get("verified_entities", []))
            sig_tokens = set(sig_sig.get("key_tokens", []))

            shared_concepts = sorted(item_concepts & sig_concepts)
            shared_ents = sorted(set(item_entities) & sig_entities)
            shared_toks = sorted(item_tokens & sig_tokens)

            concept_score = 0.45 if shared_concepts else 0.0
            entity_score = min(0.25, 0.15 * len(shared_ents))
            tok_union = len(item_tokens | sig_tokens)
            token_score = min(0.20, 0.80 * (len(shared_toks) / tok_union)) if tok_union else 0.0
            rel_bonus = 0.10 if item.relevance_score >= 7.0 else (0.05 if item.relevance_score > 0 else 0.05)

            total_sim = round(concept_score + entity_score + token_score + rel_bonus, 3)
            if total_sim > best_match_score:
                best_match_score = total_sim
                best_trend = sig
                shared_features = {
                    "identical_bottleneck_concepts": shared_concepts,
                    "shared_entities": shared_ents,
                    "shared_technical_tokens": shared_toks[:10],
                    "db_bottleneck_addressed": item.bottleneck_addressed,
                    "db_specific_tech": item.specific_tech,
                    "db_hardware_domain": item.hardware_domain,
                }

        db_x_breakout = compute_x_breakout_potential(
            text=composite_text,
            entities=item_entities,
            bottleneck_concept=item.bottleneck_addressed or (best_trend.bottleneck_concept if best_trend else "Hardware Bottleneck"),
            corroboration_count=2 if best_trend else 1,
            has_primary_source=True,
            x_mentions_count=0,
        )

        if (
            best_trend is not None
            and best_match_score >= match_threshold
            and db_x_breakout["will_trend_on_x"] == "YES"
            and item.url not in seen_db_urls
        ):
            seen_db_urls.add(item.url)
            matched_records.append(
                {
                    "db_item_id": item.id,
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "published_at": item.published_at,
                    "db_relevance_score": item.relevance_score,
                    "why_it_matters": item.why_it_matters,
                    "ai_revolution_thesis": item.ai_revolution_thesis,
                    "viral_match_score": round(best_match_score * 100.0, 1),
                    "x_breakout_score": db_x_breakout["x_breakout_score"],
                    "will_become_x_trend": db_x_breakout["will_trend_on_x"],
                    "x_arbitrage_window": db_x_breakout["x_arbitrage_window"],
                    "matched_social_trend": best_trend.bottleneck_concept,
                    "matched_trend_regime": best_trend.burst_regime,
                    "identical_viral_features": shared_features,
                    "x_catalyst_playbook": db_x_breakout["x_catalyst_playbook"],
                }
            )

    matched_records.sort(
        key=lambda r: (r["x_breakout_score"], r["viral_match_score"], r["db_relevance_score"]),
        reverse=True,
    )

    if matched_records:
        return {
            "has_viral_db_matches": "YES",
            "matched_count": len(matched_records),
            "summary": (
                f"YES — {len(matched_records)} previously sorted item(s) in ai-hardware-bot's database "
                "share identical viral features with live social momentum AND exceed the X/Twitter "
                "Phoenix breakout threshold to become a trend on X."
            ),
            "matched_database_items": matched_records,
        }

    return {
        "has_viral_db_matches": "NO",
        "matched_count": 0,
        "summary": "NO — None of the previously sorted items in ai-hardware-bot's database match the current social + X/Twitter breakout criteria.",
        "matched_database_items": [],
    }
