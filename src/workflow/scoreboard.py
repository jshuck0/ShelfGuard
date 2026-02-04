"""
Scoreboard Module

Monthly metrics aggregation for Workflow ShelfGuard.

Tracks:
- Episode counts by reason code
- High-severity alert counts
- Actions completed
- "$ Protected" range with explicit assumptions

CRITICAL: "$ Protected" is a PROXY estimate, NOT causal attribution.
Assumptions must be visible to user at all times.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict

from src.models.episode import Episode, EpisodeStatus
from src.workflow.alerts import Alert
from src.workflow.reason_codes import ReasonCode


@dataclass
class Scoreboard:
    """
    Monthly scoreboard for tracking workflow effectiveness.

    Key principle: $ Protected is a PROXY, not causal measurement.
    """
    period: str                            # "2026-01" or "2026-W05"

    # Episode counts
    episodes_detected: int = 0
    episodes_by_reason_code: Dict[str, int] = field(default_factory=dict)
    episodes_resolved: int = 0
    episodes_dismissed: int = 0

    # Alert counts
    high_severity_alerts: int = 0
    alerts_acknowledged: int = 0

    # Action tracking
    actions_created: int = 0
    actions_completed: int = 0
    actions_overdue: int = 0

    # $ Protected (PROXY - with explicit assumptions)
    protected_low: float = 0.0
    protected_base: float = 0.0
    protected_high: float = 0.0
    assumptions: List[str] = field(default_factory=list)

    # Time proxy
    hours_saved_estimate: float = 0.0
    hours_saved_assumptions: List[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)

    def get_protected_range_str(self) -> str:
        """Format $ protected as string."""
        if self.protected_low == 0 and self.protected_high == 0:
            return "$0"
        return f"${self.protected_low:,.0f} - ${self.protected_high:,.0f}"

    def get_resolution_rate(self) -> float:
        """Calculate resolution rate (0-1)."""
        total_actionable = self.episodes_resolved + self.episodes_dismissed
        if self.episodes_detected == 0:
            return 0.0
        return total_actionable / self.episodes_detected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/Supabase storage."""
        return {
            "period": self.period,
            "episodes_detected": self.episodes_detected,
            "episodes_by_reason_code": self.episodes_by_reason_code,
            "episodes_resolved": self.episodes_resolved,
            "episodes_dismissed": self.episodes_dismissed,
            "high_severity_alerts": self.high_severity_alerts,
            "alerts_acknowledged": self.alerts_acknowledged,
            "actions_created": self.actions_created,
            "actions_completed": self.actions_completed,
            "actions_overdue": self.actions_overdue,
            "protected_low": self.protected_low,
            "protected_base": self.protected_base,
            "protected_high": self.protected_high,
            "assumptions": self.assumptions,
            "hours_saved_estimate": self.hours_saved_estimate,
            "hours_saved_assumptions": self.hours_saved_assumptions,
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scoreboard":
        """Create Scoreboard from dictionary."""
        return cls(
            period=data.get("period", ""),
            episodes_detected=data.get("episodes_detected", 0),
            episodes_by_reason_code=data.get("episodes_by_reason_code", {}),
            episodes_resolved=data.get("episodes_resolved", 0),
            episodes_dismissed=data.get("episodes_dismissed", 0),
            high_severity_alerts=data.get("high_severity_alerts", 0),
            alerts_acknowledged=data.get("alerts_acknowledged", 0),
            actions_created=data.get("actions_created", 0),
            actions_completed=data.get("actions_completed", 0),
            actions_overdue=data.get("actions_overdue", 0),
            protected_low=data.get("protected_low", 0.0),
            protected_base=data.get("protected_base", 0.0),
            protected_high=data.get("protected_high", 0.0),
            assumptions=data.get("assumptions", []),
            hours_saved_estimate=data.get("hours_saved_estimate", 0.0),
            hours_saved_assumptions=data.get("hours_saved_assumptions", []),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else datetime.now(),
        )


# Default assumptions for $ Protected calculation
DEFAULT_PROTECTED_ASSUMPTIONS = [
    "Assumes issues would have caused full estimated impact if undetected",
    "Based on Keepa data only (no internal sales data)",
    "Proxy estimate - NOT causal attribution",
    "Actual impact may vary significantly",
]

# Default assumptions for hours saved
DEFAULT_HOURS_ASSUMPTIONS = [
    "Assumes 30 min manual research time saved per episode detected",
    "Does not include action execution time",
]

# Hours saved per episode (heuristic)
HOURS_PER_EPISODE_DETECTED = 0.5  # 30 minutes


def calculate_episodes_by_reason_code(episodes: List[Episode]) -> Dict[str, int]:
    """Count episodes by reason code."""
    counts: Dict[str, int] = defaultdict(int)
    for episode in episodes:
        counts[episode.reason_code] += 1
    return dict(counts)


def calculate_protected_range(
    episodes: List[Episode],
    only_resolved: bool = True
) -> tuple[float, float, float]:
    """
    Calculate $ protected range from resolved episodes.

    Args:
        episodes: All episodes
        only_resolved: If True, only count resolved episodes

    Returns:
        (low, base, high) dollar amounts
    """
    if only_resolved:
        relevant = [e for e in episodes if e.status == EpisodeStatus.RESOLVED]
    else:
        relevant = episodes

    total_low = sum(e.impact_low for e in relevant)
    total_base = sum(e.impact_base for e in relevant)
    total_high = sum(e.impact_high for e in relevant)

    return (total_low, total_base, total_high)


def calculate_hours_saved(episodes: List[Episode]) -> float:
    """
    Calculate estimated hours saved.

    Simple heuristic: 30 min per episode detected.
    """
    return len(episodes) * HOURS_PER_EPISODE_DETECTED


def calculate_scoreboard(
    episodes: List[Episode],
    alerts: List[Alert],
    period: Optional[str] = None
) -> Scoreboard:
    """
    Calculate complete scoreboard from episodes and alerts.

    Args:
        episodes: All episodes for the period
        alerts: All alerts for the period
        period: Period string (e.g., "2026-01"); defaults to current month

    Returns:
        Scoreboard object
    """
    if period is None:
        period = datetime.now().strftime("%Y-%m")

    # Episode counts
    episodes_detected = len(episodes)
    episodes_by_reason_code = calculate_episodes_by_reason_code(episodes)
    episodes_resolved = sum(1 for e in episodes if e.status == EpisodeStatus.RESOLVED)
    episodes_dismissed = sum(1 for e in episodes if e.status == EpisodeStatus.DISMISSED)

    # Alert counts
    high_severity_alerts = len(alerts)
    alerts_acknowledged = sum(1 for a in alerts if a.is_acknowledged())

    # Action counts (episodes with action templates = actions created)
    actions_created = sum(1 for e in episodes if e.action_template_id)
    actions_completed = episodes_resolved  # Resolved = action completed
    actions_overdue = sum(
        1 for e in episodes
        if e.due_date and e.due_date < datetime.now() and e.status not in [EpisodeStatus.RESOLVED, EpisodeStatus.DISMISSED]
    )

    # $ Protected calculation
    protected_low, protected_base, protected_high = calculate_protected_range(episodes, only_resolved=True)

    # Hours saved
    hours_saved = calculate_hours_saved(episodes)

    return Scoreboard(
        period=period,
        episodes_detected=episodes_detected,
        episodes_by_reason_code=episodes_by_reason_code,
        episodes_resolved=episodes_resolved,
        episodes_dismissed=episodes_dismissed,
        high_severity_alerts=high_severity_alerts,
        alerts_acknowledged=alerts_acknowledged,
        actions_created=actions_created,
        actions_completed=actions_completed,
        actions_overdue=actions_overdue,
        protected_low=protected_low,
        protected_base=protected_base,
        protected_high=protected_high,
        assumptions=DEFAULT_PROTECTED_ASSUMPTIONS.copy(),
        hours_saved_estimate=hours_saved,
        hours_saved_assumptions=DEFAULT_HOURS_ASSUMPTIONS.copy(),
        generated_at=datetime.now(),
    )


def render_scoreboard_summary(scoreboard: Scoreboard) -> str:
    """
    Render scoreboard as text summary for display.
    """
    lines = [
        f"📊 Scoreboard for {scoreboard.period}",
        f"",
        f"Episodes Detected: {scoreboard.episodes_detected}",
        f"  - Resolved: {scoreboard.episodes_resolved}",
        f"  - Dismissed: {scoreboard.episodes_dismissed}",
        f"  - Resolution Rate: {scoreboard.get_resolution_rate()*100:.0f}%",
        f"",
        f"Alerts: {scoreboard.high_severity_alerts}",
        f"  - Acknowledged: {scoreboard.alerts_acknowledged}",
        f"",
        f"Actions: {scoreboard.actions_created} created, {scoreboard.actions_completed} completed",
        f"  - Overdue: {scoreboard.actions_overdue}",
        f"",
        f"💰 $ Protected (Proxy): {scoreboard.get_protected_range_str()}",
        f"⏱️ Hours Saved (Est.): {scoreboard.hours_saved_estimate:.1f}",
        f"",
        f"⚠️ Assumptions:",
    ]

    for assumption in scoreboard.assumptions[:3]:
        lines.append(f"  • {assumption}")

    return "\n".join(lines)
