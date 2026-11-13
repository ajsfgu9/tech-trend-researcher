from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Vertical(str, Enum):
    SEMICONDUCTOR = "semiconductor"
    FINANCING = "financing"
    BIOMEDICAL = "biomedical"
    ROBOTICS = "robotics"
    QUANTUM = "quantum"


ALL_VERTICALS = [v.value for v in Vertical]


@dataclass
class RawItem:
    url: str
    title: str
    source: str
    published_at: str
    vertical: str
    summary: str = ""
    source_type: str = "rss"
    is_primary_source: bool = False


@dataclass
class TrendCluster:
    cluster_id: str
    topic: str
    vertical: str
    items: list[RawItem] = field(default_factory=list)
    metrics_extracted: list[str] = field(default_factory=list)
    corroboration_count: int = 1
    primary_source_count: int = 0
    recency_hours: float = 0.0
    fairness_score: float = 0.0
    x_algo_score: float = 0.0
    combined_score: float = 0.0
    x_strategy: dict[str, Any] = field(default_factory=dict)
    why_tech_fanatics_care: str = ""
    fair_caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "topic": self.topic,
            "vertical": self.vertical,
            "combined_score": round(self.combined_score, 2),
            "fairness_score": round(self.fairness_score, 2),
            "x_algo_score": round(self.x_algo_score, 2),
            "corroboration_count": self.corroboration_count,
            "primary_source_count": self.primary_source_count,
            "recency_hours": round(self.recency_hours, 1),
            "metrics_extracted": self.metrics_extracted,
            "why_tech_fanatics_care": self.why_tech_fanatics_care,
            "fair_caveat": self.fair_caveat,
            "x_strategy": self.x_strategy,
            "sources": [asdict(item) for item in self.items],
        }
