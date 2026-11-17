from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EDITORIAL_SOURCE_WEIGHTS: dict[str, float] = {
    "primary_engineering": 0.95,     # GitHub, arXiv, bioRxiv, patents
    "debate_controversy": 0.90,      # Reddit, Hacker News
    "financial_supply_chain": 0.85,  # SEC filings, DigiTimes, TrendForce, SemiAnalysis
    "viral_amplification": 0.75,     # X/Twitter tech lists, Bluesky, Mastodon
    "mainstream_tech_media": 0.40,   # The Verge, TechCrunch, Tom's Hardware, Ars Technica
}

CAUSAL_LAYER_ORDER: dict[str, int] = {
    "primary_engineering": 0,
    "debate_controversy": 1,
    "financial_supply_chain": 2,
    "viral_amplification": 2,
    "mainstream_tech_media": 3,
}

PLATFORM_TO_EDITORIAL_LAYER: dict[str, str] = {
    "arxiv": "primary_engineering",
    "biorxiv": "primary_engineering",
    "github": "primary_engineering",
    "hacker_news": "debate_controversy",
    "reddit": "debate_controversy",
    "technical_news": "financial_supply_chain",
    "sec": "financial_supply_chain",
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
class SortedDBItem:
    """Represents a previously sorted, LLM-classified item stored in ai-hardware-bot's SQLite DB."""
    id: int
    url: str
    title: str
    source: str
    source_type: str
    published_at: str
    category: str
    hardware_domain: str
    bottleneck_addressed: str
    specific_tech: str
    relevance_score: float
    why_it_matters: str
    ai_revolution_thesis: str
    raw_summary: str


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
    bottleneck_concepts: list[str] = field(default_factory=list)
    velocity_vi: float = 0.0
    dispersion_di: float = 1.0
    dispersion_k: int = 1
    m_viral: float = 0.0
    editorial_eval: dict[str, Any] = field(default_factory=dict)
    phi_editorial: float = 1.0
    editorial_score: float = 0.0


@dataclass
class LongitudinalTrendSignal:
    """Part 1: What is trending right now across broad social & developer channels."""
    trend_key: str
    primary_entity: str
    bottleneck_concept: str
    vertical: str
    candidates: list[NormalizedCandidate] = field(default_factory=list)
    ewma_velocity_z: float = 0.0
    acceleration_delta: float = 0.0
    cusum_stat: float = 0.0
    burst_regime: str = "baseline"
    origin_layer: str = "debate_controversy"
    active_layers: list[str] = field(default_factory=list)
    causal_propagation_score: float = 0.0
    mainstream_saturation_ratio: float = 0.0
    co_occurrence_lift: float = 0.0
    is_novel_coupling: bool = False
    reserved_item_subscore: float = 0.0
    longitudinal_score: float = 0.0
    unified_trend_index: float = 0.0
    viral_feature_signature: dict[str, Any] = field(default_factory=dict)
