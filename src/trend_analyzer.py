"""
Stage 3b: Longitudinal Trend Discovery Engine (`src/trend_analyzer.py`)

Why isolated item scoring + cross-social matching is inadequate:
- Scoring individual articles only captures point-in-time popularity.
- A genuine *trend* is a longitudinal structural trajectory over time.

This module implements a 4-pillar Longitudinal Trend Study:
1. Temporal Burst & Second-Order Acceleration (EWMA + Page-Hinkley / CUSUM):
   Groups items into time-window buckets per `(Bottleneck Concept, Entity)` and measures
   exponentially weighted moving average (EWMA) velocity z-scores, second-order
   acceleration (delta velocity across time buckets), and cumulative sum (CUSUM)
   structural regime shifts (`breakout`, `accelerating`, `steady`, `decaying`).
2. Cross-Layer Causal Propagation Graph (Lead-Lag Diffusion):
   Tracks the timestamped arrival order across editorial source layers:
   `Primary Engineering (Stage 0)` -> `Debate/Controversy (Stage 1)` ->
   `Financial/Supply Chain (Stage 2)` -> `Mainstream Media (Stage 3)`.
   Rewards trends that originate in primary engineering/research and propagate into
   technical debate & supply-chain checks *before* mainstream media saturation.
3. Novel Bottleneck Co-Occurrence Shift:
   Measures whether an entity + bottleneck coupling represents a newly emerging
   constraint relative to historical co-occurrence in the MetadataRepository.
4. Reserved Item Sub-Score (20% Weight):
   Incorporates the item-level `scoring_engine.py` (`m_viral`) strictly as a 20%
   secondary sub-component alongside the 80% Longitudinal Structural Score.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.metadata_repo import MetadataRepository
from src.models import CAUSAL_LAYER_ORDER, LongitudinalTrendSignal, NormalizedCandidate


def _parse_utc(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _compute_ewma_and_cusum(series: list[float], alpha: float = 0.4) -> tuple[float, float, float, str]:
    if not series:
        return 0.0, 0.0, 0.0, "decaying"
    if len(series) == 1:
        val = series[0]
        return round(math.log1p(val), 3), round(val, 3), round(val * 0.5, 3), "accelerating"

    ewma = series[0]
    ewm_var = 1.0
    cusum = 0.0
    drift = 0.25

    for x in series[1:]:
        diff = x - ewma
        ewma = alpha * x + (1.0 - alpha) * ewma
        ewm_var = alpha * (diff ** 2) + (1.0 - alpha) * ewm_var
        sigma = max(0.5, math.sqrt(ewm_var))
        z = diff / sigma
        cusum = max(0.0, cusum + z - drift)

    sigma_final = max(0.5, math.sqrt(ewm_var))
    ewma_z = max(0.0, (series[-1] - (sum(series[:-1]) / len(series[:-1]))) / sigma_final)

    v_now = series[-1] - series[-2]
    v_prev = (series[-2] - series[-3]) if len(series) >= 3 else 0.0
    accel_delta = v_now - v_prev

    if cusum >= 2.0 and accel_delta > 0:
        regime = "breakout"
    elif ewma_z >= 0.8 or v_now > 0:
        regime = "accelerating"
    elif series[-1] >= ewma * 0.8:
        regime = "steady"
    else:
        regime = "decaying"

    return round(ewma_z, 3), round(accel_delta, 3), round(cusum, 3), regime


def _compute_causal_propagation(candidates: list[NormalizedCandidate]) -> tuple[str, list[str], float, float]:
    sorted_cands = sorted(candidates, key=lambda c: _parse_utc(c.published_at_utc))
    origin_layer = sorted_cands[0].editorial_layer if sorted_cands else "financial_supply_chain"

    earliest_by_layer: dict[str, datetime] = {}
    layer_counts: dict[str, int] = {}
    for c in sorted_cands:
        dt = _parse_utc(c.published_at_utc)
        layer_counts[c.editorial_layer] = layer_counts.get(c.editorial_layer, 0) + 1
        if c.editorial_layer not in earliest_by_layer or dt < earliest_by_layer[c.editorial_layer]:
            earliest_by_layer[c.editorial_layer] = dt

    active_layers = sorted(earliest_by_layer.keys(), key=lambda lyr: earliest_by_layer[lyr])
    total_items = max(1, len(sorted_cands))
    mainstream_ratio = layer_counts.get("mainstream_tech_media", 0) / total_items

    stage_sequence = [CAUSAL_LAYER_ORDER.get(lyr, 2) for lyr in active_layers]
    forward_transitions = sum(
        1 for i in range(len(stage_sequence) - 1) if stage_sequence[i + 1] >= stage_sequence[i]
    )

    origin_bonus = 35.0 if origin_layer == "primary_engineering" else (
        25.0 if origin_layer == "debate_controversy" else 15.0
    )
    progression_pts = min(45.0, 20.0 * forward_transitions + 12.0 * len(set(stage_sequence)))
    saturation_penalty = 30.0 * mainstream_ratio if mainstream_ratio > 0.5 else 0.0

    causal_score = max(10.0, min(100.0, origin_bonus + progression_pts - saturation_penalty))
    return origin_layer, active_layers, round(causal_score, 2), round(mainstream_ratio, 3)


def discover_longitudinal_trends(
    candidates: list[NormalizedCandidate],
    metadata_repo: MetadataRepository,
) -> list[LongitudinalTrendSignal]:
    by_concept: dict[str, list[NormalizedCandidate]] = {}
    for cand in candidates:
        primary_concept = cand.bottleneck_concepts[0] if cand.bottleneck_concepts else f"{cand.vertical.title()} Shift"
        by_concept.setdefault(primary_concept, []).append(cand)

    signals: list[LongitudinalTrendSignal] = []

    for concept, group in by_concept.items():
        group_sorted = sorted(group, key=lambda c: _parse_utc(c.published_at_utc))
        lead = max(group, key=lambda c: c.m_viral)

        all_entities = [e for c in group for e in c.verified_entities]
        primary_entity = max(set(all_entities), key=all_entities.count) if all_entities else lead.vertical.upper()
        trend_key = f"{lead.vertical}:{concept.lower().replace(' ', '_')}"

        buckets: dict[str, tuple[int, float]] = {}
        for c in group_sorted:
            day_str = _parse_utc(c.published_at_utc).strftime("%Y-%m-%d")
            cnt, inten = buckets.get(day_str, (0, 0.0))
            buckets[day_str] = (cnt + 1, inten + c.m_viral)

        for day_str, (cnt, inten) in sorted(buckets.items()):
            metadata_repo.record_concept_bucket(trend_key, day_str, cnt, inten)

        history = metadata_repo.get_concept_history(trend_key)
        intensity_series = [row[2] for row in history]
        if len(intensity_series) < 2:
            half = max(1, len(group_sorted) // 2)
            w1 = sum(c.m_viral for c in group_sorted[:half])
            w2 = sum(c.m_viral for c in group_sorted[half:]) if len(group_sorted) > 1 else w1 * 1.25
            intensity_series = [0.8, w1, w2]

        ewma_z, accel_delta, cusum_stat, regime = _compute_ewma_and_cusum(intensity_series)
        origin_layer, active_layers, causal_score, mainstream_ratio = _compute_causal_propagation(group_sorted)

        unique_ents = sorted(set(all_entities))
        if len(unique_ents) >= 2:
            hist_co = metadata_repo.get_co_occurrence_count(unique_ents[0], unique_ents[1])
            co_lift = round(len(group) / max(1.0, float(hist_co)), 3)
            is_novel = hist_co <= 2
        else:
            co_lift = 1.0
            is_novel = len(history) <= 2

        avg_m_viral = sum(c.m_viral for c in group) / len(group)
        reserved_item_subscore = min(100.0, avg_m_viral * 28.0)

        burst_component = min(100.0, 25.0 * ewma_z + 20.0 * cusum_stat + (25.0 if regime == "breakout" else 15.0))
        novelty_component = min(100.0, 50.0 * co_lift + (30.0 if is_novel else 10.0))
        longitudinal_score = 0.40 * burst_component + 0.40 * causal_score + 0.20 * novelty_component

        unified_index = round(min(99.5, 0.80 * longitudinal_score + 0.20 * reserved_item_subscore), 2)

        signals.append(
            LongitudinalTrendSignal(
                trend_key=trend_key,
                primary_entity=primary_entity,
                bottleneck_concept=concept,
                vertical=lead.vertical,
                candidates=group_sorted,
                ewma_velocity_z=ewma_z,
                acceleration_delta=accel_delta,
                cusum_stat=cusum_stat,
                burst_regime=regime,
                origin_layer=origin_layer,
                active_layers=active_layers,
                causal_propagation_score=causal_score,
                mainstream_saturation_ratio=mainstream_ratio,
                co_occurrence_lift=co_lift,
                is_novel_coupling=is_novel,
                reserved_item_subscore=round(reserved_item_subscore, 2),
                longitudinal_score=round(longitudinal_score, 2),
                unified_trend_index=unified_index,
            )
        )

    return sorted(signals, key=lambda s: s.unified_trend_index, reverse=True)
