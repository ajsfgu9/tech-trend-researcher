"""External trend detection using real snapshot velocity and dynamic similarity clustering.

Key guarantees:
- Operates STRICTLY on external signals (never on upstream bot DB rows).
- Computes real age-normalized engagement rates and real snapshot-to-snapshot
  engagement deltas from `StateStore`.
- Computes z-scores ONLY when `StateStore` has >= 10 real historical snapshots
  for that platform (no fabricated history arrays).
- Clusters related signals dynamically via cosine similarity (no hardcoded 10-bucket taxonomy).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.models import ExternalSignal, ExternalTrendCluster
from src.similarity import EmbeddingEngine, compute_tfidf_matrix, tokenize_technical
from src.state_store import StateStore


def _parse_iso(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def enrich_signal_velocities(
    signals: list[ExternalSignal],
    state_store: StateStore,
) -> list[ExternalSignal]:
    """Compute real engagement rates, observed snapshot deltas, and baseline z-scores."""
    for sig in signals:
        eng_total = sig.score + 2.0 * sig.comments + 1.5 * sig.shares
        eng_rate = eng_total / max(1.0, sig.age_hours)
        sig.engagement_total = round(eng_total, 2)
        sig.engagement_rate_per_hour = round(eng_rate, 3)

        prev = state_store.get_previous_snapshot(sig.signal_id)
        if prev is not None:
            dt_prev = _parse_iso(str(prev["observed_at_utc"]))
            dt_now = _parse_iso(sig.fetched_at_utc)
            elapsed_h = (dt_now - dt_prev).total_seconds() / 3600.0
            if elapsed_h >= 0.05:  # >= 3 minutes between observations
                delta_eng = max(0.0, eng_total - float(prev["engagement_total"]))
                sig.delta_engagement = round(delta_eng, 2)
                sig.delta_rate_per_hour = round(delta_eng / elapsed_h, 3)

        baseline = state_store.get_platform_rate_baseline(sig.platform)
        if baseline is not None:
            mean, std, _ = baseline
            sig.velocity_z_score = round((eng_rate - mean) / std, 3)
            sig.has_historical_baseline = True
        else:
            sig.velocity_z_score = None
            sig.has_historical_baseline = False

        state_store.record_snapshot(
            signal_id=sig.signal_id,
            platform=sig.platform,
            observed_at_utc=sig.fetched_at_utc,
            score=sig.score,
            comments=sig.comments,
            shares=sig.shares,
            engagement_total=eng_total,
            engagement_rate_per_hour=eng_rate,
        )

    return signals


def detect_external_trends(
    signals: list[ExternalSignal],
    state_store: StateStore,
    embedding_engine: EmbeddingEngine | None = None,
    similarity_threshold: float = 0.28,
    top_n: int = 12,
) -> list[ExternalTrendCluster]:
    """Enrich external signals with real velocity, cluster dynamically, and rank."""
    if not signals:
        return []

    enrich_signal_velocities(signals, state_store)

    texts = [s.semantic_text() for s in signals]
    tfidf_mat, idf, _ = compute_tfidf_matrix(texts)

    dense_mat: np.ndarray | None = None
    backend = "tfidf_bm25"
    if embedding_engine is not None:
        dense_mat, backend = embedding_engine.embed_texts(texts)

    # Dense cosine similarity lives on a higher baseline scale (~0.72) than sparse TF-IDF (~0.28)
    if dense_mat is not None and dense_mat.shape[0] == len(signals):
        sim_matrix = np.clip(dense_mat @ dense_mat.T, 0.0, 1.0)
        eff_threshold = max(0.72, similarity_threshold)
    else:
        sim_matrix = np.clip(tfidf_mat @ tfidf_mat.T, 0.0, 1.0) if tfidf_mat.size else np.eye(len(signals))
        eff_threshold = similarity_threshold
        backend = "tfidf_bm25"

    # Greedy single-pass clustering ordered by engagement velocity (avoids chaining unrelated nodes)
    order = sorted(
        range(len(signals)),
        key=lambda i: (
            signals[i].delta_rate_per_hour or 0.0,
            signals[i].engagement_rate_per_hour,
            signals[i].engagement_total,
        ),
        reverse=True,
    )

    cluster_indices: list[list[int]] = []
    cluster_seeds: list[int] = []

    for idx in order:
        best_c = -1
        best_sim = 0.0
        for c_i, seed_idx in enumerate(cluster_seeds):
            sim = float(sim_matrix[idx, seed_idx])
            if sim >= eff_threshold and sim > best_sim:
                best_sim = sim
                best_c = c_i
        if best_c >= 0:
            cluster_indices[best_c].append(idx)
        else:
            cluster_seeds.append(idx)
            cluster_indices.append([idx])

    clusters: list[ExternalTrendCluster] = []
    for members in cluster_indices:
        member_sigs = [signals[i] for i in members]
        member_sigs.sort(key=lambda s: s.engagement_rate_per_hour, reverse=True)
        lead = member_sigs[0]
        platforms = sorted({s.platform for s in member_sigs})

        total_eng = sum(s.engagement_total for s in member_sigs)
        max_rate = max(s.engagement_rate_per_hour for s in member_sigs)
        delta_rates = [s.delta_rate_per_hour for s in member_sigs if s.delta_rate_per_hour is not None]
        max_delta = max(delta_rates) if delta_rates else None
        z_scores = [s.velocity_z_score for s in member_sigs if s.velocity_z_score is not None]
        avg_z = (sum(z_scores) / len(z_scores)) if z_scores else None

        # Top distinctive corpus-weighted terms across cluster
        term_counts: Counter[str] = Counter()
        for s in member_sigs:
            for tok in set(tokenize_technical(f"{s.title} {' '.join(s.tags)}")):
                term_counts[tok] += 1
        top_terms = sorted(
            term_counts.keys(),
            key=lambda t: (term_counts[t], idf.get(t, 1.0)),
            reverse=True,
        )[:8]

        # Transparent, interpretable 0-100 ranking score:
        # 1. Age-normalized engagement velocity (log-scaled)
        # 2. Cross-platform corroboration bonus
        # 3. Cluster size corroboration bonus
        # 4. Observed delta acceleration / baseline z-score bonus (only when real data exists)
        velocity_pts = min(45.0, 12.0 * math.log1p(max_rate))
        platform_pts = min(25.0, 12.5 * (len(platforms) - 1))
        volume_pts = min(15.0, 5.0 * math.log1p(len(member_sigs)))
        accel_pts = 0.0
        if max_delta is not None and max_delta > 0:
            accel_pts += min(10.0, 3.0 * math.log1p(max_delta))
        if avg_z is not None and avg_z > 0:
            accel_pts += min(10.0, 4.0 * avg_z)
        trend_score = round(min(100.0, velocity_pts + platform_pts + volume_pts + accel_pts), 2)

        cid = hashlib.sha1(f"{lead.title}:{','.join(platforms)}".encode("utf-8")).hexdigest()[:12]
        clusters.append(
            ExternalTrendCluster(
                cluster_id=f"ext_{cid}",
                label=lead.title[:120],
                signals=member_sigs,
                platforms=platforms,
                total_engagement=round(total_eng, 2),
                max_engagement_rate_per_hour=round(max_rate, 3),
                max_delta_rate_per_hour=round(max_delta, 3) if max_delta is not None else None,
                avg_velocity_z=round(avg_z, 3) if avg_z is not None else None,
                has_historical_baseline=bool(z_scores),
                trend_score=trend_score,
                top_terms=top_terms,
                similarity_backend=backend,
            )
        )

    clusters.sort(
        key=lambda c: (c.trend_score, len(c.platforms), c.max_engagement_rate_per_hour),
        reverse=True,
    )
    return clusters[:top_n]
