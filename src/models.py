from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SOURCE_AUTHORITY_WEIGHTS: dict[str, float] = {
    "technical_news": 0.95,      # SemiAnalysis, SemiEngineering, TrendForce, DigiTimes, FierceBiotech
    "hacker_news": 0.90,         # Firebase / Algolia API
    "reddit": 0.85,              # r/semiconductors, r/hardware, r/localllama
    "github": 0.85,              # GitHub Trending & Releases
    "arxiv": 0.80,               # arXiv cs.AR, physics.app-ph, cs.RO, quant-ph, bioRxiv
    "x_twitter": 0.70,           # Curated Tech/Semicon Lists
    "general_social": 0.40,      # Bluesky, Threads, YouTube Tech
}


@dataclass
class RawEvent:
    event_id: str
    source_platform: str
    source_name: str
    ingested_at_utc: str
    published_at_utc: str
    payload: dict[str, Any]


@dataclass
class NormalizedCandidate:
    event_id: str
    source_platform: str
    source_name: str
    url: str
    title: str
    body: str
    published_at_utc: str
    engagement_value: float
    source_authority_ws: float
    vertical: str = "semiconductor"
    relevance_ri: float = 0.0
    verified_entities: list[str] = field(default_factory=list)
    velocity_vi: float = 0.0
    dispersion_di: float = 1.0
    dispersion_k: int = 1
    base_score_mi: float = 0.0
    llm_audit: dict[str, Any] = field(default_factory=dict)
    llm_multiplier_phi: float = 1.0
    final_score: float = 0.0
