"""Core typed data models for Tech Trend Researcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDBConfig:
    """Configuration pointing to one upstream intelligence bot SQLite database."""

    name: str
    db_path: str


@dataclass
class BotDBItem:
    """Normalized LLM-classified record loaded from any of the 3 upstream bot databases."""

    bot_name: str
    item_id: int
    composite_id: str
    url: str
    title: str
    source: str
    source_type: str
    published_at: str
    domain: str
    bottleneck_addressed: str
    specific_tech: str
    relevance_score: float
    why_it_matters: str
    thesis: str
    raw_summary: str

    def semantic_text(self) -> str:
        parts = [
            self.title,
            self.domain,
            self.bottleneck_addressed,
            self.specific_tech,
            self.why_it_matters,
            self.thesis,
            self.raw_summary[:500],
        ]
        return " ".join(p.strip() for p in parts if p and p.strip())


@dataclass
class ExternalSignal:
    """Single item collected strictly from an external live developer/social platform."""

    signal_id: str
    platform: str
    source_name: str
    url: str
    title: str
    summary: str
    published_at_utc: str
    fetched_at_utc: str
    score: float = 0.0
    comments: float = 0.0
    shares: float = 0.0
    tags: list[str] = field(default_factory=list)
    age_hours: float = 1.0
    engagement_total: float = 0.0
    engagement_rate_per_hour: float = 0.0
    delta_engagement: float = 0.0
    delta_rate_per_hour: float | None = None
    velocity_z_score: float | None = None
    has_historical_baseline: bool = False

    def semantic_text(self) -> str:
        tags_str = " ".join(self.tags)
        return f"{self.title} {tags_str} {self.summary[:500]}".strip()


@dataclass
class CollectorStatus:
    """Observability record for each external source collector."""

    collector: str
    enabled: bool
    status: str  # ok | error | skipped_no_credentials | disabled
    items_collected: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalTrendCluster:
    """Dynamically clustered external trend built strictly from live external signals."""

    cluster_id: str
    label: str
    signals: list[ExternalSignal] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    total_engagement: float = 0.0
    max_engagement_rate_per_hour: float = 0.0
    max_delta_rate_per_hour: float | None = None
    avg_velocity_z: float | None = None
    has_historical_baseline: bool = False
    trend_score: float = 0.0
    top_terms: list[str] = field(default_factory=list)
    similarity_backend: str = "tfidf_bm25"
    llm_analysis: dict[str, Any] = field(default_factory=dict)

    def semantic_text(self) -> str:
        parts = [self.label] + [s.semantic_text() for s in self.signals[:6]]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "trend_score": round(self.trend_score, 2),
            "signal_count": len(self.signals),
            "platforms": self.platforms,
            "cross_platform_count": len(self.platforms),
            "total_engagement": round(self.total_engagement, 1),
            "max_engagement_rate_per_hour": round(self.max_engagement_rate_per_hour, 2),
            "max_delta_rate_per_hour": (
                round(self.max_delta_rate_per_hour, 2)
                if self.max_delta_rate_per_hour is not None
                else None
            ),
            "avg_velocity_z": (
                round(self.avg_velocity_z, 2)
                if self.avg_velocity_z is not None
                else None
            ),
            "has_historical_baseline": self.has_historical_baseline,
            "similarity_backend": self.similarity_backend,
            "top_terms": self.top_terms,
            "llm_analysis": self.llm_analysis,
            "sources": [
                {
                    "platform": s.platform,
                    "source_name": s.source_name,
                    "title": s.title,
                    "url": s.url,
                    "published_at_utc": s.published_at_utc,
                    "score": s.score,
                    "comments": s.comments,
                    "engagement_rate_per_hour": round(s.engagement_rate_per_hour, 2),
                }
                for s in self.signals[:8]
            ],
        }


@dataclass
class DBViralMatch:
    """Verified match connecting an upstream bot DB item to a live external trend cluster."""

    bot_name: str
    db_item_id: int
    composite_id: str
    title: str
    url: str
    source: str
    domain: str
    published_at: str
    db_relevance_score: float
    bottleneck_addressed: str
    specific_tech: str
    why_it_matters: str
    thesis: str
    matched_cluster_id: str
    matched_trend_label: str
    matched_platforms: list[str]
    matched_trend_score: float
    combined_match_score: float
    semantic_similarity: float
    lexical_overlap: float
    similarity_backend: str
    shared_terms: list[str]
    llm_verified: bool | None = None
    llm_rationale: str = ""
    llm_x_hook: str = ""
    llm_reply_prompt: str = ""
    llm_visual_idea: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
