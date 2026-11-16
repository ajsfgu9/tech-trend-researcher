"""
Stage 2: Domain Relevance Gate & Bottleneck Concept Extraction
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src.models import (
    EDITORIAL_SOURCE_WEIGHTS,
    NormalizedCandidate,
    PLATFORM_TO_EDITORIAL_LAYER,
    RawEvent,
)

TIER1_WHITELIST = (
    "euv", "asml", "tsmc", "tsmc a16", "nvidia", "cowos", "hbm3e", "hbm4",
    "backside power delivery", "gaafet", "high-na", "hybrid bonding", "thermal throttling",
    "substrate", "abf", "eda", "synopsys", "foundry", "packaging", "interconnect",
    "cxl", "pcie 6.0", "nvlink", "silicon photonics", "cpo", "glass core",
    "wafer", "lithography", "yield", "warpage", "interposer", "reticle",
    "crispr", "phase 1", "phase 2", "phase 3", "humanoid", "actuator",
    "qubit", "surface code", "error correction", "capex", "s-1", "8-k",
)

BOTTLENECK_TAXONOMY: dict[str, tuple[str, ...]] = {
    "CoWoS & Interposer Warpage": ("cowos", "interposer", "warpage", "2.5d packaging", "reticle"),
    "High-NA EUV & Mask Yields": ("high-na", "exe:5000", "euv", "line-edge roughness", "mask"),
    "Backside Power & Sub-2nm Nodes": ("backside power", "bspdn", "a16", "2nm", "gaafet"),
    "HBM4 & Memory Wall": ("hbm3e", "hbm4", "cxl 3.1", "cxl", "memory pooling", "base die"),
    "Co-Packaged Optics & Interconnect": ("silicon photonics", "cpo", "co-packaged optics", "nvlink", "optical interconnect"),
    "Glass Core Substrates": ("glass core", "glass substrate", "abf substrate"),
    "AI Capex & GPU Financing": ("capex", "s-1", "8-k", "gpu cluster", "debt financing", "neocloud"),
    "Gene Editing & Clinical Trials": ("crispr", "prime editing", "phase 1", "phase 2", "phase 3", "mrna", "alphafold"),
    "Embodied AI & Dexterous Robotics": ("humanoid", "actuator", "tactile", "vla", "dexterous", "embodied ai"),
    "Fault-Tolerant Quantum QEC": ("logical qubit", "surface code", "error correction", "qec", "fault-tolerant"),
}

DOMAIN_ANCHOR_CORPUS: dict[str, str] = {
    "semiconductor": (
        "semiconductor wafer fab lithography euv high-na asml tsmc a16 intel foundry "
        "cowos packaging hybrid bonding hbm3e hbm4 backside power delivery gaafet abf glass core "
        "substrate interposer warpage eda synopsys interconnect cxl pcie nvlink silicon photonics cpo"
    ),
    "financing": (
        "semiconductor capex fab equipment margin erosion yield rates lead times venture funding "
        "series s-1 ipo 8-k sec filing datacenter gpu cluster debt financing valuation"
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
    "CoWoS", "High-NA EUV", "Glass Core", "Hybrid Bonding",
)

CONSUMER_NOISE_PATTERNS = re.compile(
    r"\b(iphone battery life|best laptop|black friday|smartphone camera|earbuds review|"
    r"wallpaper|gaming chair|tiktok trend|celebrity)\b",
    re.IGNORECASE,
)

CONTROVERSY_TOKENS = re.compile(
    r"\b(wrong|myth|misconception|actually|bottleneck|scrap|yield|overhyped|quietly|"
    r"capping|squeeze|margin|vs\.?|versus|flaw|throttling|warpage|delay|shortage)\b",
    re.IGNORECASE,
)


def _cosine_text_sim(text_a: str, text_b: str) -> float:
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


def extract_bottleneck_concepts(text: str, fallback_vertical: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for concept_name, kw_tuple in BOTTLENECK_TAXONOMY.items():
        if any(kw in low for kw in kw_tuple):
            found.append(concept_name)
    if not found:
        found.append(f"{fallback_vertical.title()} Architecture Shift")
    return found


def normalize_raw_event(ev: RawEvent) -> NormalizedCandidate:
    p = ev.payload
    title = str(p.get("title") or "").strip()
    body = str(p.get("summary") or p.get("selftext") or p.get("text") or "").strip()
    url = str(p.get("url") or "").strip()

    likes = float(p.get("points") or p.get("score") or p.get("ups") or p.get("likes") or p.get("engagement_points") or 0.0)
    comments = float(p.get("num_comments") or p.get("replies") or p.get("comments_count") or 0.0)
    eng = max(10.0, likes + 3.0 * comments)

    layer = PLATFORM_TO_EDITORIAL_LAYER.get(ev.source_platform, "financial_supply_chain")
    if any(m in ev.source_name.lower() for m in ("verge", "techcrunch", "tom's hardware", "ars technica")):
        layer = "mainstream_tech_media"

    return NormalizedCandidate(
        event_id=ev.event_id,
        source_platform=ev.source_platform,
        source_name=ev.source_name,
        editorial_layer=layer,
        url=url,
        title=title,
        body=body,
        published_at_utc=ev.published_at_utc,
        engagement_value=eng,
        comments_count=comments,
        source_weight_we=EDITORIAL_SOURCE_WEIGHTS.get(layer, 0.75),
    )


def run_relevance_gate(candidates: list[NormalizedCandidate], sim_threshold: float = 0.62) -> list[NormalizedCandidate]:
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

        controversy_hits = len(set(CONTROVERSY_TOKENS.findall(lowered)))
        ia_score = min(1.0, 0.25 + 0.12 * min(whitelist_hits, 4) + 0.10 * min(controversy_hits, 3))

        cand.vertical = best_vertical
        cand.relevance_ri = round(best_sim, 4)
        cand.information_asymmetry_ia = round(ia_score, 4)
        cand.verified_entities = entities
        cand.bottleneck_concepts = extract_bottleneck_concepts(full_text, best_vertical)
        passed.append(cand)

    return passed
