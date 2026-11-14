"""
Stage 2: Staging Area & Two-Tier Tech/Semiconductor Relevance Gate

Tier 1: Fast Lexical Whitelist + Dense Vector Cosine Similarity (Sim >= 0.62 threshold).
Tier 2: Named Entity Recognition (NER) & Generic Consumer Noise Filter.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src.models import NormalizedCandidate, RawEvent, SOURCE_AUTHORITY_WEIGHTS

TIER1_WHITELIST = (
    "euv", "asml", "tsmc", "nvidia", "cowos", "hbm3e", "hbm4", "backside power delivery",
    "gaafet", "high-na", "substrate", "abf", "eda", "synopsys", "foundry", "packaging",
    "interconnect", "cxl", "pcie 6.0", "nvlink", "silicon photonics", "cpo", "glass core",
    "wafer", "lithography", "crispr", "phase 1", "phase 2", "phase 3", "humanoid", "actuator",
    "qubit", "surface code", "error correction", "capex", "s-1", "8-k", "series ",
)

DOMAIN_ANCHOR_CORPUS: dict[str, str] = {
    "semiconductor": (
        "semiconductor wafer fab lithography euv high-na asml tsmc intel foundry "
        "cowos packaging hbm3e hbm4 backside power delivery gaafet abf glass core substrate "
        "eda synopsys cadence interconnect cxl pcie nvlink silicon photonics cpo"
    ),
    "financing": (
        "semiconductor capex fab equipment spending venture funding series s-1 ipo "
        "8-k sec filing datacenter gpu cluster debt financing valuation acquisition"
    ),
    "biomedical": (
        "crispr gene editing mrna protein folding alphafold clinical trial phase 1 phase 2 "
        "phase 3 fda genomics biorxiv antibody sequencing biomedical biotech"
    ),
    "robotics": (
        "robotics humanoid dexterous hand actuator tactile vla embodied ai teleoperation "
        "autonomous manipulator lidar slam arxiv cs.ro"
    ),
    "quantum": (
        "quantum computing logical qubit error correction qec surface code superconducting "
        "trapped ion photonic neutral atom cryogenic dilution quant-ph"
    ),
}

TRACKED_ENTITIES = (
    "ASML", "TSMC", "Intel Foundry", "Intel", "NVIDIA", "AMD", "Micron", "SK Hynix",
    "Samsung", "Applied Materials", "Tokyo Electron", "Lam Research", "Synopsys",
    "Cadence", "Ibiden", "SEMCO", "Astera Labs", "Broadcom", "Marvell", "ARM",
    "EXE:5000", "TSMC A16", "PCIe 6.0", "CXL 3.1", "CXL", "NVLink", "HBM3e", "HBM4",
    "CoWoS", "High-NA EUV", "Glass Core",
)

CONSUMER_NOISE_PATTERNS = re.compile(
    r"\b(iphone battery life|best laptop|black friday|smartphone camera|earbuds review|"
    r"wallpaper|gaming chair|tiktok trend|celebrity)\b",
    re.IGNORECASE,
)


def _cosine_text_sim(text_a: str, text_b: str) -> float:
    """Lightweight character 3-gram + token vector cosine similarity (zero heavy deps fallback)."""
    def vec(t: str) -> Counter[str]:
        low = t.lower()
        tokens = re.findall(r"[a-z0-9\-\.]{2,}", low)
        grams = [low[i : i + 3] for i in range(max(0, len(low) - 2))]
        return Counter(tokens + grams)

    va, vb = vec(text_a), vec(text_b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[k] * vb[k] for k in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def normalize_raw_event(ev: RawEvent) -> NormalizedCandidate:
    p = ev.payload
    title = str(p.get("title") or "").strip()
    body = str(p.get("summary") or p.get("selftext") or p.get("text") or "").strip()
    url = str(p.get("url") or "").strip()

    if ev.source_platform == "hacker_news":
        eng = float(p.get("points") or 0) + 1.5 * float(p.get("num_comments") or 0)
    elif ev.source_platform == "reddit":
        eng = float(p.get("score") or p.get("ups") or 0) + 2.0 * float(p.get("num_comments") or 0)
    elif ev.source_platform == "x_twitter":
        eng = float(p.get("impressions") or 0) * 0.01 + float(p.get("replies") or 0) * 5.0
    elif ev.source_platform == "github":
        eng = float(p.get("stargazers_count") or 0)
    else:
        eng = 10.0

    return NormalizedCandidate(
        event_id=ev.event_id,
        source_platform=ev.source_platform,
        source_name=ev.source_name,
        url=url,
        title=title,
        body=body,
        published_at_utc=ev.published_at_utc,
        engagement_value=eng,
        source_authority_ws=SOURCE_AUTHORITY_WEIGHTS.get(ev.source_platform, 0.50),
    )


def run_relevance_gate(candidates: list[NormalizedCandidate], sim_threshold: float = 0.62) -> list[NormalizedCandidate]:
    """Execute Tier 1 (Whitelist + Dense Similarity >= 0.62) and Tier 2 (NER + Consumer Filter)."""
    passed: list[NormalizedCandidate] = []
    for cand in candidates:
        full_text = f"{cand.title} {cand.body}".strip()
        if len(full_text) < 15:
            continue

        lowered = full_text.lower()
        whitelist_hits = sum(1 for kw in TIER1_WHITELIST if kw in lowered)

        best_vertical = "semiconductor"
        best_sim = 0.0
        for vertical, corpus in DOMAIN_ANCHOR_CORPUS.items():
            raw_sim = _cosine_text_sim(full_text, corpus)
            calibrated = min(1.0, raw_sim * 2.2 + 0.18 * min(whitelist_hits, 3))
            if calibrated > best_sim:
                best_sim = calibrated
                best_vertical = vertical

        if best_sim < sim_threshold:
            continue

        entities = [ent for ent in TRACKED_ENTITIES if ent.lower() in lowered]
        if CONSUMER_NOISE_PATTERNS.search(lowered) and not entities:
            continue

        cand.vertical = best_vertical
        cand.relevance_ri = round(best_sim, 4)
        cand.verified_entities = entities
        passed.append(cand)

    return passed
