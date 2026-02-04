"""
Memo Renderer

Selects Top 10 Episodes and renders the Weekly War-Room Memo.

Selection rules (stable, deterministic):
1. Sort by composite_score (severity * confidence) descending
2. Dedupe by (reason_code, primary_asin, start_week)
3. Filter severity >= 0.5
4. Take Top 10

Tie-breakers:
1. Higher impact_base wins
2. Older start_week wins
3. Alphabetical by episode_id
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.models.episode import Episode
from src.workflow.action_templates import get_action_template, ActionTemplate
from src.workflow.reason_codes import ReasonCode


@dataclass
class MemoItem:
    """
    A single item in the Weekly Memo.

    Represents one Episode with rendered summary and action.
    """
    rank: int                              # 1-10
    episode: Episode
    headline: str                          # "Price War on Serum SKU"
    summary: str                           # 2-3 sentence explanation
    evidence_summary: str                  # "3 price drops since Jan 28"
    what_to_verify: List[str]              # Manual checks
    recommended_action: str                # From action template
    impact_range: str                      # "$2,000 - $5,000"
    urgency_emoji: str                     # "🚨", "📅", "👀"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rank": self.rank,
            "episode_id": self.episode.episode_id,
            "episode_type": self.episode.episode_type,
            "reason_code": self.episode.reason_code,
            "headline": self.headline,
            "summary": self.summary,
            "evidence_summary": self.evidence_summary,
            "what_to_verify": self.what_to_verify,
            "recommended_action": self.recommended_action,
            "impact_range": self.impact_range,
            "urgency_emoji": self.urgency_emoji,
            "severity_score": self.episode.severity_score,
            "confidence_score": self.episode.confidence_score,
            "composite_score": self.episode.composite_score,
            "primary_asins": self.episode.primary_asins,
        }


@dataclass
class WeeklyMemo:
    """
    The complete Weekly War-Room Memo.
    """
    generated_at: datetime
    period: str                            # "Week of Feb 3, 2026"
    items: List[MemoItem]                  # Top 10 (or fewer)
    total_episodes: int                    # Before filtering
    episodes_filtered: int                 # After dedup + severity filter

    # Summary stats
    threats_count: int
    opportunities_count: int
    total_impact_range: str                # Aggregated range

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "period": self.period,
            "items": [item.to_dict() for item in self.items],
            "total_episodes": self.total_episodes,
            "episodes_filtered": self.episodes_filtered,
            "threats_count": self.threats_count,
            "opportunities_count": self.opportunities_count,
            "total_impact_range": self.total_impact_range,
        }


def _sort_key(episode: Episode) -> Tuple:
    """
    Sort key for episode ranking.

    Primary: composite_score (descending)
    Tie-breakers:
    1. impact_base (descending)
    2. start_week (ascending - older first)
    3. episode_id (ascending - alphabetical)
    """
    return (
        -episode.composite_score,      # Negative for descending
        -episode.impact_base,          # Negative for descending
        episode.start_week,            # Ascending
        episode.episode_id,            # Ascending
    )


def _dedupe_key(episode: Episode) -> Tuple:
    """
    Deduplication key: (reason_code, primary_asin, start_week).

    Only keeps highest-scoring episode for each unique key.
    """
    primary_asin = episode.primary_asins[0] if episode.primary_asins else ""
    return (episode.reason_code, primary_asin, episode.start_week)


def select_top_episodes(
    episodes: List[Episode],
    min_severity: float = 0.5,
    max_episodes: int = 10
) -> List[Episode]:
    """
    Select top episodes using stable, deterministic rules.

    Args:
        episodes: All episodes from episode_builder
        min_severity: Minimum severity score (0.0-1.0)
        max_episodes: Maximum episodes to return

    Returns:
        List of top episodes, sorted by rank
    """
    if not episodes:
        return []

    # Step 1: Filter by minimum severity
    filtered = [e for e in episodes if e.severity_score >= min_severity]

    # Step 2: Sort by composite score (with tie-breakers)
    sorted_episodes = sorted(filtered, key=_sort_key)

    # Step 3: Deduplicate by (reason_code, asin, week)
    seen_keys = set()
    deduped = []
    for episode in sorted_episodes:
        key = _dedupe_key(episode)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(episode)

    # Step 4: Take top N
    return deduped[:max_episodes]


def _generate_headline(episode: Episode) -> str:
    """Generate headline from episode."""
    try:
        reason_code = ReasonCode(episode.reason_code)
        emoji = reason_code.emoji
        name = reason_code.display_name
    except ValueError:
        emoji = "❓"
        name = episode.episode_type.replace("_", " ").title()

    asin_str = episode.primary_asins[0][:10] if episode.primary_asins else "Unknown"
    return f"{emoji} {name} on {asin_str}"


def _generate_summary(episode: Episode, template: Optional[ActionTemplate]) -> str:
    """Generate 2-3 sentence summary."""
    parts = []

    # Nature
    try:
        reason_code = ReasonCode(episode.reason_code)
        nature = reason_code.nature
    except ValueError:
        nature = "issue"

    # Opening
    if nature == "opportunity":
        parts.append(f"Opportunity detected: {episode.episode_type.lower().replace('_', ' ')}.")
    else:
        parts.append(f"Threat detected: {episode.episode_type.lower().replace('_', ' ')}.")

    # Evidence count
    evidence_count = len(episode.evidence_refs) if episode.evidence_refs else 1
    parts.append(f"Based on {evidence_count} signal(s) detected in {episode.start_week}.")

    # Confidence
    if episode.confidence_score >= 0.8:
        parts.append("High confidence based on consistent evidence.")
    elif episode.confidence_score >= 0.6:
        parts.append("Moderate confidence - verify before acting.")
    else:
        parts.append("Low confidence - treat as early warning only.")

    return " ".join(parts)


def _generate_evidence_summary(episode: Episode) -> str:
    """Generate evidence summary string."""
    evidence_count = len(episode.evidence_refs) if episode.evidence_refs else 1
    return f"{evidence_count} event(s) detected during {episode.start_week}"


def _generate_what_to_verify(episode: Episode, template: Optional[ActionTemplate]) -> List[str]:
    """Generate manual verification steps."""
    verifications = [
        "Verify affected ASIN(s) in Seller Central",
        f"Check current status (as of {datetime.now().strftime('%Y-%m-%d')})",
    ]

    # Add template-specific first step if available
    if template and template.checklist:
        verifications.insert(0, template.checklist[0])

    return verifications[:3]  # Max 3 items


def render_memo_item(episode: Episode, rank: int) -> MemoItem:
    """
    Render a single MemoItem from an Episode.
    """
    # Get action template
    template = get_action_template(episode.action_template_id)

    # Generate headline
    headline = _generate_headline(episode)

    # Generate summary
    summary = _generate_summary(episode, template)

    # Evidence summary
    evidence_summary = _generate_evidence_summary(episode)

    # What to verify
    what_to_verify = _generate_what_to_verify(episode, template)

    # Recommended action
    if template:
        recommended_action = template.title
        urgency_emoji = template.get_urgency_emoji()
    else:
        recommended_action = "Investigate and assess"
        urgency_emoji = "❓"

    # Impact range
    impact_range = episode.get_impact_range_str()

    return MemoItem(
        rank=rank,
        episode=episode,
        headline=headline,
        summary=summary,
        evidence_summary=evidence_summary,
        what_to_verify=what_to_verify,
        recommended_action=recommended_action,
        impact_range=impact_range,
        urgency_emoji=urgency_emoji,
    )


def render_weekly_memo(
    episodes: List[Episode],
    min_severity: float = 0.5,
    max_episodes: int = 10
) -> WeeklyMemo:
    """
    Render the complete Weekly War-Room Memo.

    Args:
        episodes: All episodes from episode_builder
        min_severity: Minimum severity threshold
        max_episodes: Maximum items in memo

    Returns:
        WeeklyMemo object
    """
    total_episodes = len(episodes)

    # Select top episodes
    top_episodes = select_top_episodes(episodes, min_severity, max_episodes)
    episodes_filtered = len(top_episodes)

    # Render each item
    items = [
        render_memo_item(episode, rank=i+1)
        for i, episode in enumerate(top_episodes)
    ]

    # Count threats vs opportunities
    threats_count = 0
    opportunities_count = 0
    for episode in top_episodes:
        try:
            reason_code = ReasonCode(episode.reason_code)
            if reason_code.nature == "opportunity":
                opportunities_count += 1
            else:
                threats_count += 1
        except ValueError:
            threats_count += 1  # Default to threat

    # Calculate total impact range
    total_low = sum(e.impact_low for e in top_episodes)
    total_high = sum(e.impact_high for e in top_episodes)
    total_impact_range = f"${total_low:,.0f} - ${total_high:,.0f}"

    # Generate period string
    period = f"Week of {datetime.now().strftime('%b %d, %Y')}"

    return WeeklyMemo(
        generated_at=datetime.now(),
        period=period,
        items=items,
        total_episodes=total_episodes,
        episodes_filtered=episodes_filtered,
        threats_count=threats_count,
        opportunities_count=opportunities_count,
        total_impact_range=total_impact_range,
    )
