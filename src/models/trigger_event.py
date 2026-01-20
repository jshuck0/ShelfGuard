"""
Trigger Event Data Model

A Trigger Event is a discrete, measurable market change that justifies an insight.
Examples: competitor inventory drop, price war, review spike, BuyBox loss
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class TriggerEvent:
    """
    Represents a discrete market change detected in Keepa data.

    Trigger events provide causal reasoning for AI recommendations.
    Instead of saying "raise price", we say "raise price BECAUSE competitor X is OOS".
    """

    # Identity
    event_type: str                     # "competitor_oos_imminent", "price_war_active", etc.
    severity: int                       # 1-10 (10 = most urgent)
    detected_at: datetime

    # Metric Change
    metric_name: str                    # "fba_inventory", "buy_box_price", "sales_rank", etc.
    baseline_value: float               # Value before change
    current_value: float                # Value after change
    delta_pct: float                    # Percentage change

    # Product Context
    affected_asin: str
    related_asin: Optional[str] = None  # Competitor ASIN if relevant

    # Metadata
    id: Optional[str] = None
    generated_insight_id: Optional[str] = None

    def __post_init__(self):
        """Validate severity range."""
        if not 1 <= self.severity <= 10:
            raise ValueError(f"Severity must be 1-10, got {self.severity}")

    def to_llm_context(self) -> str:
        """
        Format for LLM prompt injection.

        Returns human-readable description with key details.
        """
        direction = "increased" if self.delta_pct > 0 else "decreased"

        context = f"EVENT DETECTED: {self.event_type.upper().replace('_', ' ')}\n"
        context += f"- Severity: {self.severity}/10\n"
        context += f"- Metric: {self.metric_name} {direction} from {self.baseline_value} → {self.current_value} "
        context += f"({self.delta_pct:+.1f}%)\n"
        context += f"- Detected: {self.detected_at.strftime('%Y-%m-%d %H:%M')}\n"

        if self.related_asin:
            context += f"- Related Product: {self.related_asin}\n"

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "metric_name": self.metric_name,
            "baseline_value": float(self.baseline_value),
            "current_value": float(self.current_value),
            "delta_pct": float(self.delta_pct),
            "affected_asin": self.affected_asin,
            "related_asin": self.related_asin,
            "id": self.id,
            "generated_insight_id": self.generated_insight_id
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerEvent':
        """Create TriggerEvent from dictionary."""
        # Parse datetime if it's a string
        detected_at = data['detected_at']
        if isinstance(detected_at, str):
            detected_at = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))

        return cls(
            event_type=data['event_type'],
            severity=data['severity'],
            detected_at=detected_at,
            metric_name=data['metric_name'],
            baseline_value=data['baseline_value'],
            current_value=data['current_value'],
            delta_pct=data['delta_pct'],
            affected_asin=data['affected_asin'],
            related_asin=data.get('related_asin'),
            id=data.get('id'),
            generated_insight_id=data.get('generated_insight_id')
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TriggerEvent':
        """Create TriggerEvent from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_severity_label(self) -> str:
        """Get human-readable severity label."""
        if self.severity >= 9:
            return "CRITICAL"
        elif self.severity >= 7:
            return "HIGH"
        elif self.severity >= 5:
            return "MEDIUM"
        else:
            return "LOW"

    def get_urgency_description(self) -> str:
        """Get urgency description for UI."""
        if self.severity >= 9:
            return "Act immediately - critical risk"
        elif self.severity >= 7:
            return "Act within 48 hours"
        elif self.severity >= 5:
            return "Monitor closely - act within 7 days"
        else:
            return "Track trend - no immediate action needed"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TriggerEvent(type={self.event_type}, severity={self.severity}, "
            f"{self.metric_name}: {self.baseline_value}→{self.current_value} "
            f"({self.delta_pct:+.1f}%), asin={self.affected_asin})"
        )


# Event type categories for grouping
EVENT_CATEGORIES = {
    "inventory": [
        "competitor_oos_imminent",
        "inventory_low",
        "competitor_restocked"
    ],
    "pricing": [
        "price_war_active",
        "price_spike",
        "price_drop",
        "competitor_price_increase",
        "competitor_price_decrease"
    ],
    "buybox": [
        "buybox_share_collapse",
        "buybox_share_gained",
        "buybox_rotation"
    ],
    "rank": [
        "rank_degradation",
        "rank_improvement",
        "rank_volatility_high"
    ],
    "reviews": [
        "review_velocity_spike",
        "review_velocity_stagnant",
        "rating_decline",
        "competitor_review_surge"
    ],
    "market": [
        "new_competitor_entered",
        "competitor_exit",
        "category_growth",
        "category_decline"
    ]
}


def get_event_category(event_type: str) -> str:
    """Get category for an event type."""
    for category, events in EVENT_CATEGORIES.items():
        if event_type in events:
            return category
    return "other"
