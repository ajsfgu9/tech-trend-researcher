"""
Fairness, Anti-Hype & Cross-Vertical Balancing Engine

Ensures trends are fair, grounded in primary/corroborated sources, and balanced
across semiconductors, financing, biomedical, robotics, and quantum.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone

from src.models import RawItem, TrendCluster

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "by",
    "is", "are", "was", "were", "as", "at", "from", "new", "how", "why", "what",
    "that", "this", "its", "it", "into", "over", "after", "under", "more", "than",
}

METRIC_REGEX = re.compile(
    r"(\$?\d+(?:\.\d+)?\s?(?:[BMK]|billion|million|nm|GHz|MHz|TFLOPS|TOPS|MW|GW|kW|W|"
    r"qubits?|%|x|ms|ns|GB|TB|Gbps|Tbps|DOF)|phase\s?[123]|series\s?[a-e])",
    re.IGNORECASE,
)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9\-\.]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def extract_metrics(text: str) -> list[str]:
    matches = METRIC_REGEX.findall(text or "")
    seen: list[str] = []
    for m in matches:
        clean = m.strip()
        if clean and clean.lower() not in {x.lower() for x in seen}:
            seen.append(clean)
    return seen[:12]


def cluster_items(items: list[RawItem], similarity_threshold: float = 0.28) -> list[TrendCluster]:
    """Group related articles by Jaccard token overlap + shared URL deduplication."""
    seen_urls: set[str] = set()
    unique_items: list[RawItem] = []
    for item in items:
        if item.url not in seen_urls:
            seen_urls.add(item.url)
            unique_items.append(item)

    clusters: list[tuple[set[str], TrendCluster]] = []
    now = datetime.now(timezone.utc)

    for item in unique_items:
        tokens = _tokenize(f"{item.title} {item.summary[:180]}")
        matched_idx = -1
        best_sim = 0.0

        for idx, (c_tokens, cluster) in enumerate(clusters):
            if cluster.vertical != item.vertical:
                continue
            union = tokens | c_tokens
            if not union:
                continue
            sim = len(tokens & c_tokens) / len(union)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                matched_idx = idx

        if matched_idx >= 0:
            c_tokens, cluster = clusters[matched_idx]
            cluster.items.append(item)
            c_tokens.update(tokens)
        else:
            cid = hashlib.sha1(f"{item.vertical}:{item.title}".encode("utf-8")).hexdigest()[:12]
            cluster = TrendCluster(
                cluster_id=cid,
                topic=item.title,
                vertical=item.vertical,
                items=[item],
            )
            clusters.append((set(tokens), cluster))

    result: list[TrendCluster] = []
    for _, cluster in clusters:
        unique_sources = {i.source for i in cluster.items}
        cluster.corroboration_count = len(unique_sources)
        cluster.primary_source_count = sum(1 for i in cluster.items if i.is_primary_source)

        all_text = " ".join(f"{i.title} {i.summary}" for i in cluster.items)
        cluster.metrics_extracted = extract_metrics(all_text)

        # Pick most informative / primary title as topic
        primary_items = [i for i in cluster.items if i.is_primary_source]
        cluster.topic = (primary_items[0] if primary_items else cluster.items[0]).title

        # Compute recency in hours using newest item
        ages: list[float] = []
        for i in cluster.items:
            try:
                dt = datetime.fromisoformat(i.published_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ages.append(max(0.0, (now - dt).total_seconds() / 3600.0))
            except ValueError:
                ages.append(24.0)
        cluster.recency_hours = min(ages) if ages else 24.0
        result.append(cluster)

    return result


def compute_fairness_score(cluster: TrendCluster) -> float:
    """
    Score [0-100] rewarding primary sources, multi-source corroboration,
    hard quantitative metrics, and freshness while penalizing single-blog hype.
    """
    corroboration_pts = min(35.0, 15.0 * math.log1p(cluster.corroboration_count))
    primary_pts = min(25.0, 18.0 * cluster.primary_source_count)
    metric_pts = min(20.0, 5.0 * len(cluster.metrics_extracted))
    recency_pts = 20.0 * math.exp(-cluster.recency_hours / 48.0)

    # Anti-hype penalty: single uncorroborated secondary source with zero numbers
    hype_penalty = 12.0 if (
        cluster.corroboration_count == 1
        and cluster.primary_source_count == 0
        and len(cluster.metrics_extracted) == 0
    ) else 0.0

    return max(0.0, min(100.0, corroboration_pts + primary_pts + metric_pts + recency_pts - hype_penalty))


def select_balanced_trends(
    clusters: list[TrendCluster],
    top_n: int = 12,
    max_per_vertical: int = 3,
) -> list[TrendCluster]:
    """
    Select top_n trends while enforcing cross-vertical diversity so no single
    sector (e.g. semiconductors) crowds out financing, biomedical, robotics, or quantum.
    """
    ranked = sorted(clusters, key=lambda c: c.combined_score, reverse=True)
    selected: list[TrendCluster] = []
    vertical_counts: dict[str, int] = {}

    # Pass 1: Guarantee up to max_per_vertical per vertical
    for cluster in ranked:
        if len(selected) >= top_n:
            break
        count = vertical_counts.get(cluster.vertical, 0)
        if count < max_per_vertical:
            selected.append(cluster)
            vertical_counts[cluster.vertical] = count + 1

    # Pass 2: Fill remaining slots if fewer than top_n
    if len(selected) < top_n:
        selected_ids = {c.cluster_id for c in selected}
        for cluster in ranked:
            if len(selected) >= top_n:
                break
            if cluster.cluster_id not in selected_ids:
                selected.append(cluster)
                selected_ids.add(cluster.cluster_id)

    return selected
