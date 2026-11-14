from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Re-Weighted Sources for Editorial Value (W_e)
EDITORIAL_SOURCE_WEIGHTS: dict[str, float] = {
    "primary_engineering": 0.95,     # GitHub Issues/Discussions, arXiv, bioRxiv, USPTO patents
    "debate_controversy": 0.90,      # Hacker News, Reddit (r/hardware, r/semiconductors, etc.)
    "financial_supply_chain": 0.85,  # Earnings transcripts, SEC, DigiTimes, TrendForce, SemiAnalysis
    "viral_amplification": 0.75,     # X/Twitter tech lists, Bluesky, Mastodon
    "mainstream_tech_media": 0.40,   # The Verge, TechCrunch, Tom's Hardware, Ars Technica
}

# Map ingestion platform IDs -> Editorial Source Layer
PLATFORM_TO_EDITORIAL_LAYER: dict[str, str] = {
    "arxiv": "primary_engineering",
    "github": "primary_engineering",
    "hacker_news": "debate_controversy",
    "reddit": "debate_controversy",
    "technical_news": "financial_supply_chain",
    "x_twitter": "viral_amplification",
    "general_social": "viral_amplification",
    "mainstream_media": "mainstream_tech_media",
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
    editorial_layer: str
    url: str
    title: str
    body: str
    published_at_utc: str
    engagement_value: float
    comments_count: float
    source_weight_we: float
    vertical: str = "semiconductor"
    relevance_ri: float = 0.0
    information_asymmetry_ia: float = 0.0
    verified_entities: list[str] = field(default_factory=list)
    velocity_vi: float = 0.0
    dispersion_di: float = 1.0
    dispersion_k: int = 1
    m_viral: float = 0.0
    editorial_eval: dict[str, Any] = field(default_factory=dict)
    phi_editorial: float = 1.0
    editorial_score: float = 0.0
