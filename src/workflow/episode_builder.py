"""
Episode Builder

Clusters related TriggerEvents into Episode objects.

Clustering rules (strict for V1):
1. Same ASIN + same week = one episode (prevents over-merging)
2. Related competitor events can be linked via competitor_asins
3. Episode type determined by highest-severity event in cluster

This module is the bridge between detector output and the workflow layer.
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from src.models.trigger_event import TriggerEvent, get_event_nature
from src.models.episode import Episode, EpisodeStatus
from src.workflow.reason_codes import (
    ReasonCode,
    map_event_type_to_reason_code,
    calculate_severity,
    calculate_confidence,
    get_reason_code_config,
)


def get_iso_week(dt: datetime) -> str:
    """
    Convert datetime to ISO week string (e.g., "2026-W05").
    """
    return dt.strftime("%G-W%V")


def cluster_events_by_asin_week(
    events: List[TriggerEvent]
) -> Dict[Tuple[str, str], List[TriggerEvent]]:
    """
    Group TriggerEvents by (affected_asin, iso_week).

    This is the core clustering logic - strict to prevent over-merging.

    Args:
        events: List of TriggerEvents from detectors

    Returns:
        Dict mapping (asin, week) to list of events
    """
    clusters: Dict[Tuple[str, str], List[TriggerEvent]] = defaultdict(list)

    for event in events:
        asin = event.affected_asin
        week = get_iso_week(event.detected_at)
        clusters[(asin, week)].append(event)

    return clusters


def determine_episode_type(events: List[TriggerEvent]) -> Tuple[str, ReasonCode]:
    """
    Determine the episode type from a cluster of events.

    Rules:
    1. Use the highest-severity event's type
    2. Map to a ReasonCode

    Args:
        events: Events in a single cluster

    Returns:
        (episode_type string, ReasonCode)
    """
    if not events:
        return ("UNKNOWN", ReasonCode.DEMAND_SHIFT)

    # Sort by severity descending
    sorted_events = sorted(events, key=lambda e: e.severity, reverse=True)
    primary_event = sorted_events[0]

    # Get episode type from event type
    episode_type = primary_event.event_type.upper()

    # Map to reason code
    reason_code = map_event_type_to_reason_code(primary_event.event_type)
    if reason_code is None:
        reason_code = ReasonCode.DEMAND_SHIFT  # Default fallback

    return (episode_type, reason_code)


def calculate_impact_range(
    events: List[TriggerEvent],
    reason_code: ReasonCode,
    weekly_revenue: float = 0.0
) -> Tuple[float, float, float, List[str]]:
    """
    Calculate impact range (low, base, high) with assumptions.

    V1: Simple heuristic based on severity and weekly revenue.
    Future: More sophisticated models per reason code.

    Args:
        events: Events in the cluster
        reason_code: Classified reason code
        weekly_revenue: Weekly revenue of affected ASIN(s)

    Returns:
        (impact_low, impact_base, impact_high, assumptions)
    """
    if not events or weekly_revenue <= 0:
        return (0.0, 0.0, 0.0, ["Insufficient data for impact estimate"])

    # Get max severity in cluster
    max_severity = max(e.severity for e in events)

    # Get max delta_pct (absolute value)
    max_delta = max(abs(e.delta_pct) for e in events if e.delta_pct)

    assumptions = []

    # Different calculation by reason code nature
    nature = reason_code.nature

    if nature == "opportunity":
        # Opportunity: potential upside
        # Low: 5% of weekly revenue
        # Base: severity% of weekly revenue
        # High: 2x base
        base_pct = max_severity / 100.0  # 8/10 severity = 8%
        impact_low = weekly_revenue * 0.05
        impact_base = weekly_revenue * base_pct
        impact_high = impact_base * 2.0
        assumptions.append(f"Assumes {base_pct*100:.0f}% revenue capture opportunity")
        assumptions.append("Based on event severity and weekly revenue")
    else:
        # Threat: potential downside
        # Low: delta% * 0.5 of weekly revenue
        # Base: delta% of weekly revenue
        # High: delta% * 1.5 of weekly revenue
        delta_factor = max_delta / 100.0 if max_delta else 0.1
        impact_low = weekly_revenue * delta_factor * 0.5
        impact_base = weekly_revenue * delta_factor
        impact_high = weekly_revenue * delta_factor * 1.5
        assumptions.append(f"Assumes {delta_factor*100:.0f}% revenue impact based on metric change")
        assumptions.append("Proxy estimate - not causal")

    return (impact_low, impact_base, impact_high, assumptions)


def collect_competitor_asins(events: List[TriggerEvent]) -> Optional[List[str]]:
    """
    Collect unique competitor ASINs from event cluster.
    """
    competitor_asins = set()
    for event in events:
        if event.related_asin:
            competitor_asins.add(event.related_asin)

    return list(competitor_asins) if competitor_asins else None


def build_episodes(
    trigger_events: List[TriggerEvent],
    portfolio_asins: Optional[List[str]] = None,
    weekly_revenue_map: Optional[Dict[str, float]] = None,
    category_weights: Optional[Dict[str, float]] = None,
) -> List[Episode]:
    """
    Main entry point: Build Episodes from TriggerEvents.

    Args:
        trigger_events: Raw output from detectors
        portfolio_asins: List of "your" ASINs (to distinguish from competitor)
        weekly_revenue_map: ASIN -> weekly revenue for impact calculation
        category_weights: ReasonCode.value -> weight for severity adjustment

    Returns:
        List of Episode objects, unsorted
    """
    if not trigger_events:
        return []

    # Default empty maps
    weekly_revenue_map = weekly_revenue_map or {}
    category_weights = category_weights or {}
    portfolio_asins = set(portfolio_asins or [])

    # Step 1: Cluster events by (asin, week)
    clusters = cluster_events_by_asin_week(trigger_events)

    episodes = []

    for (asin, week), events in clusters.items():
        # Skip if no events
        if not events:
            continue

        # Step 2: Determine episode type and reason code
        episode_type, reason_code = determine_episode_type(events)

        # Step 3: Calculate severity (with category weight)
        max_event_severity = max(e.severity for e in events)
        category_weight = category_weights.get(reason_code.value, 1.0)
        severity_score = calculate_severity(reason_code, max_event_severity, category_weight)

        # Step 4: Calculate confidence
        evidence_count = len(events)
        # Consistency: fraction of events with same nature as primary
        primary_nature = get_event_nature(events[0].event_type)
        same_nature = sum(1 for e in events if get_event_nature(e.event_type) == primary_nature)
        evidence_consistency = same_nature / len(events) if events else 1.0
        confidence_score = calculate_confidence(reason_code, evidence_count, evidence_consistency)

        # Step 5: Calculate impact range
        weekly_rev = weekly_revenue_map.get(asin, 0.0)
        impact_low, impact_base, impact_high, assumptions = calculate_impact_range(
            events, reason_code, weekly_rev
        )

        # Step 6: Get action template
        config = get_reason_code_config(reason_code)
        action_template_id = config.action_template_id

        # Step 7: Build Episode
        episode = Episode(
            episode_id=Episode.generate_id(),
            episode_type=episode_type,
            reason_code=reason_code.value,
            primary_asins=[asin],
            competitor_asins=collect_competitor_asins(events),
            start_week=week,
            end_week=week,  # Single week for V1
            evidence_refs=[e.id for e in events if e.id],
            severity_score=severity_score,
            confidence_score=confidence_score,
            impact_low=impact_low,
            impact_base=impact_base,
            impact_high=impact_high,
            impact_assumptions=assumptions,
            action_template_id=action_template_id,
            created_at=datetime.now(),
            status=EpisodeStatus.NEW,
        )

        episodes.append(episode)

    return episodes


def filter_v1_episodes(episodes: List[Episode]) -> List[Episode]:
    """
    Filter to only V1 reason codes (Keepa/public data only).
    """
    v1_codes = {
        "price_war",
        "price_compression",
        "promo_shock",
        "competitor_oos",
        "competitor_promo",
        "rank_decline",
        "rank_surge",
        "buybox_instability",
        "oos_artifact",
        "demand_shift",
    }

    return [e for e in episodes if e.reason_code in v1_codes]
