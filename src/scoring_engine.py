"""
Stage 3: Editorial Scoring Engine (M_viral + Information Asymmetry)
Calculates:
- V: 4-hour comment/share velocity z-score (comments weighted 3x likes)
- D: Cross-platform spread multiplier (1.0 + 0.35 * (K - 1))
- M_viral = W_e * R_i * (1 + ln(1 + V)) * D * (1 + 0.5 * IA)
"""

from __future__ import annotations

import math
import re

from src.metadata_repo import MetadataRepository
from src.models import NormalizedCandidate

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "by",
    "is", "are", "was", "were", "as", "at", "from", "new", "how", "why", "what",
}


def _tokens(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9\-\.]{3,}", text.lower()) if w not in STOPWORDS}


def compute_editorial_base_scores(
    candidates: list[NormalizedCandidate],
    metadata_repo: MetadataRepository,
) -> list[NormalizedCandidate]:
    by_platform: dict[str, list[float]] = {}
    for c in candidates:
        by_platform.setdefault(c.source_platform, []).append(c.engagement_value)
    for platform, vals in by_platform.items():
        metadata_repo.update_source_baseline(platform, vals)

    cand_tokens = [_tokens(c.title) for c in candidates]

    for idx, c in enumerate(candidates):
        mu, sigma = metadata_repo.get_source_baseline(c.source_platform)
        vi = max(0.0, (c.engagement_value - mu) / sigma)

        platforms_seen = {c.source_platform}
        tok_i = cand_tokens[idx]
        for j, other in enumerate(candidates):
            if idx == j or other.source_platform in platforms_seen:
                continue
            tok_j = cand_tokens[j]
            union = tok_i | tok_j
            if union and (len(tok_i & tok_j) / len(union)) >= 0.16:
                platforms_seen.add(other.source_platform)

        k = min(5, len(platforms_seen))
        di = 1.0 + 0.35 * (k - 1)
        m_viral = c.source_weight_we * c.relevance_ri * (1.0 + math.log1p(vi)) * di * (1.0 + 0.5 * c.information_asymmetry_ia)

        c.velocity_vi = round(vi, 3)
        c.dispersion_k = k
        c.dispersion_di = round(di, 3)
        c.m_viral = round(m_viral, 3)
        metadata_repo.record_entity_links(c.verified_entities)

    return candidates
