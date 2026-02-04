"""
Alerts Module

High-signal episode filtering for immediate attention.

Alerts are generated for episodes that meet strict criteria:
- Severity >= 0.8 (critical only)
- Specific reason codes (price war, buybox collapse, etc.)

Alerts are meant to be RARE - most episodes go to the Memo, not Alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from src.models.episode import Episode
from src.workflow.reason_codes import ReasonCode


# Reason codes that can generate alerts (others go to Memo only)
ALERTABLE_REASON_CODES = {
    ReasonCode.PRICE_WAR,
    ReasonCode.BUYBOX_INSTABILITY,
    ReasonCode.COMPETITOR_OOS,  # Opportunity alert
    ReasonCode.RANK_DECLINE,
}


@dataclass
class Alert:
    """
    A high-signal alert for immediate attention.
    """
    alert_id: str
    episode_id: str
    severity: str                          # "critical", "high"
    title: str
    message: str
    reason_code: str
    primary_asins: List[str]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    @staticmethod
    def generate_id() -> str:
        """Generate a unique alert ID."""
        return f"ALT-{str(uuid.uuid4())[:8]}"

    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged."""
        return self.acknowledged_at is not None

    def get_severity_emoji(self) -> str:
        """Get emoji for severity."""
        return {
            "critical": "🔴",
            "high": "🟠",
        }.get(self.severity, "🟡")

    def acknowledge(self, by: str = "user") -> None:
        """Mark alert as acknowledged."""
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = by

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/Supabase storage."""
        return {
            "alert_id": self.alert_id,
            "episode_id": self.episode_id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "reason_code": self.reason_code,
            "primary_asins": self.primary_asins,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create Alert from dictionary."""
        return cls(
            alert_id=data.get("alert_id", cls.generate_id()),
            episode_id=data.get("episode_id", ""),
            severity=data.get("severity", "high"),
            title=data.get("title", ""),
            message=data.get("message", ""),
            reason_code=data.get("reason_code", ""),
            primary_asins=data.get("primary_asins", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            acknowledged_by=data.get("acknowledged_by"),
        )


def filter_high_severity(
    episodes: List[Episode],
    min_severity: float = 0.8
) -> List[Episode]:
    """
    Filter episodes to only high-severity ones.

    Args:
        episodes: All episodes
        min_severity: Minimum severity threshold (default 0.8 for alerts)

    Returns:
        List of high-severity episodes
    """
    return [e for e in episodes if e.severity_score >= min_severity]


def _generate_alert_title(episode: Episode) -> str:
    """Generate alert title from episode."""
    try:
        reason_code = ReasonCode(episode.reason_code)
        emoji = reason_code.emoji
        name = reason_code.display_name
    except ValueError:
        emoji = "🚨"
        name = "Market Event"

    return f"{emoji} {name} Alert"


def _generate_alert_message(episode: Episode) -> str:
    """Generate alert message from episode."""
    parts = []

    # What happened
    parts.append(f"{episode.episode_type.replace('_', ' ').title()} detected.")

    # Affected ASINs
    if episode.primary_asins:
        if len(episode.primary_asins) == 1:
            parts.append(f"Affected ASIN: {episode.primary_asins[0]}")
        else:
            parts.append(f"Affected ASINs: {', '.join(episode.primary_asins[:3])}")

    # Impact
    impact_str = episode.get_impact_range_str()
    if impact_str != "TBD":
        parts.append(f"Estimated impact: {impact_str}")

    # Urgency
    if episode.severity_score >= 0.9:
        parts.append("Action required immediately.")
    else:
        parts.append("Review and assess within 24 hours.")

    return " ".join(parts)


def _get_alert_severity(episode: Episode) -> str:
    """Determine alert severity string."""
    if episode.severity_score >= 0.9:
        return "critical"
    return "high"


def generate_alerts(
    episodes: List[Episode],
    min_severity: float = 0.8
) -> List[Alert]:
    """
    Generate alerts from high-severity episodes.

    Only episodes with:
    - Severity >= min_severity
    - Alertable reason codes

    Args:
        episodes: All episodes
        min_severity: Minimum severity for alert generation

    Returns:
        List of Alert objects
    """
    alerts = []

    # Filter to high severity
    high_severity = filter_high_severity(episodes, min_severity)

    for episode in high_severity:
        # Check if reason code is alertable
        try:
            reason_code = ReasonCode(episode.reason_code)
            if reason_code not in ALERTABLE_REASON_CODES:
                continue
        except ValueError:
            continue

        # Generate alert
        alert = Alert(
            alert_id=Alert.generate_id(),
            episode_id=episode.episode_id,
            severity=_get_alert_severity(episode),
            title=_generate_alert_title(episode),
            message=_generate_alert_message(episode),
            reason_code=episode.reason_code,
            primary_asins=episode.primary_asins,
            created_at=datetime.now(),
        )

        alerts.append(alert)

    # Sort by severity (critical first) then by episode severity score
    alerts.sort(key=lambda a: (
        0 if a.severity == "critical" else 1,
        -next((e.severity_score for e in episodes if e.episode_id == a.episode_id), 0)
    ))

    return alerts


def get_unacknowledged_alerts(alerts: List[Alert]) -> List[Alert]:
    """Filter to only unacknowledged alerts."""
    return [a for a in alerts if not a.is_acknowledged()]


def get_alert_counts(alerts: List[Alert]) -> Dict[str, int]:
    """Get counts by severity."""
    counts = {"critical": 0, "high": 0, "total": len(alerts)}
    for alert in alerts:
        if alert.severity in counts:
            counts[alert.severity] += 1
    return counts
