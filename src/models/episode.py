"""
Episode Data Model

An Episode represents a cluster of related TriggerEvents that together
describe "one thing that happened" - the atomic unit of the Weekly War-Room Memo.

Key design:
1. Episodes are built from TriggerEvents (detector output)
2. Each Episode has exactly one ReasonCode
3. Scoring is deterministic (severity * confidence * category_weight)
4. Impact is always a range (low/base/high) with assumptions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class EpisodeStatus(Enum):
    """Lifecycle status of an episode."""
    NEW = "new"                    # Just detected
    IN_PROGRESS = "in_progress"   # Being worked on
    RESOLVED = "resolved"         # Action completed
    DISMISSED = "dismissed"       # Marked as not actionable


@dataclass
class Episode:
    """
    A cluster of related market events representing "one thing happened."

    This is the atomic unit of the Weekly War-Room Memo.
    """
    # Identity
    episode_id: str                        # UUID
    episode_type: str                      # "PRICE_WAR", "COMPETITOR_OOS", etc.
    reason_code: str                       # From ReasonCode enum

    # Affected products
    primary_asins: List[str]               # Your ASINs affected
    competitor_asins: Optional[List[str]] = None  # Competitor ASINs if relevant

    # Time bounds
    start_week: str = ""                   # ISO week (2026-W05)
    end_week: str = ""                     # ISO week (same as start for point events)

    # Evidence chain
    evidence_refs: List[str] = field(default_factory=list)  # TriggerEvent IDs

    # Scoring (deterministic)
    severity_score: float = 0.0            # 0.0-1.0
    confidence_score: float = 0.0          # 0.0-1.0

    # Impact estimation (always a range with assumptions)
    impact_low: float = 0.0                # $ conservative
    impact_base: float = 0.0               # $ expected
    impact_high: float = 0.0               # $ aggressive
    impact_assumptions: List[str] = field(default_factory=list)  # Show to user

    # Action linkage
    action_template_id: str = ""           # Links to ActionTemplate

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    status: EpisodeStatus = EpisodeStatus.NEW
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    # Owner assignment (for Action Queue)
    assigned_to: Optional[str] = None      # Role or person
    due_date: Optional[datetime] = None

    @staticmethod
    def generate_id() -> str:
        """Generate a unique episode ID."""
        return str(uuid.uuid4())[:8]  # Short UUID for readability

    @property
    def composite_score(self) -> float:
        """
        Primary scoring for Top 10 selection.
        Higher = more important.
        """
        return self.severity_score * self.confidence_score

    def get_impact_range_str(self) -> str:
        """Format impact as human-readable range."""
        if self.impact_low == 0 and self.impact_high == 0:
            return "TBD"
        return f"${self.impact_low:,.0f} - ${self.impact_high:,.0f}"

    def get_status_emoji(self) -> str:
        """Get emoji for status."""
        return {
            EpisodeStatus.NEW: "🆕",
            EpisodeStatus.IN_PROGRESS: "🔄",
            EpisodeStatus.RESOLVED: "✅",
            EpisodeStatus.DISMISSED: "❌",
        }.get(self.status, "❓")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/Supabase storage."""
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type,
            "reason_code": self.reason_code,
            "primary_asins": self.primary_asins,
            "competitor_asins": self.competitor_asins,
            "start_week": self.start_week,
            "end_week": self.end_week,
            "evidence_refs": self.evidence_refs,
            "severity_score": self.severity_score,
            "confidence_score": self.confidence_score,
            "impact_low": self.impact_low,
            "impact_base": self.impact_base,
            "impact_high": self.impact_high,
            "impact_assumptions": self.impact_assumptions,
            "action_template_id": self.action_template_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "assigned_to": self.assigned_to,
            "due_date": self.due_date.isoformat() if self.due_date else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create Episode from dictionary."""
        return cls(
            episode_id=data.get("episode_id", cls.generate_id()),
            episode_type=data.get("episode_type", ""),
            reason_code=data.get("reason_code", ""),
            primary_asins=data.get("primary_asins", []),
            competitor_asins=data.get("competitor_asins"),
            start_week=data.get("start_week", ""),
            end_week=data.get("end_week", ""),
            evidence_refs=data.get("evidence_refs", []),
            severity_score=data.get("severity_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            impact_low=data.get("impact_low", 0.0),
            impact_base=data.get("impact_base", 0.0),
            impact_high=data.get("impact_high", 0.0),
            impact_assumptions=data.get("impact_assumptions", []),
            action_template_id=data.get("action_template_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            status=EpisodeStatus(data.get("status", "new")),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolution_notes=data.get("resolution_notes", ""),
            assigned_to=data.get("assigned_to"),
            due_date=datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None,
        )

    def __repr__(self) -> str:
        return (
            f"Episode(id={self.episode_id}, type={self.episode_type}, "
            f"score={self.composite_score:.2f}, asins={len(self.primary_asins)})"
        )
