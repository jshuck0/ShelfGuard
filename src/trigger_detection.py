"""
Trigger Event Detection System

Scans historical Keepa data for discrete market changes that justify insights.
These trigger events provide causal reasoning for AI recommendations.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.trigger_event import TriggerEvent


def detect_trigger_events(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame,
    lookback_days: int = 30
) -> List[TriggerEvent]:
    """
    Detect all trigger events for a product.

    Runs 6 core detectors:
    1. Competitor inventory drops (OOS detection)
    2. Price wars (3+ drops in 7d)
    3. Review velocity spikes
    4. BuyBox share collapse
    5. Rank degradation
    6. New competitor detection

    Args:
        asin: Product ASIN to analyze
        df_historical: 90-day historical data for this ASIN
        df_competitors: Current competitor data in same category
        lookback_days: How far back to detect events

    Returns:
        List of TriggerEvent objects, sorted by severity (highest first)
    """

    events: List[TriggerEvent] = []

    # Run all detectors
    events.extend(detect_competitor_inventory_events(asin, df_competitors))
    events.extend(detect_price_war_events(asin, df_historical, lookback_days))
    events.extend(detect_review_velocity_events(asin, df_historical, lookback_days))
    events.extend(detect_buybox_events(asin, df_historical, lookback_days))
    events.extend(detect_rank_events(asin, df_historical, lookback_days))
    events.extend(detect_new_competitor_events(asin, df_competitors))

    # Sort by severity (highest first)
    events = sorted(events, key=lambda e: e.severity, reverse=True)

    return events


# ========================================
# DETECTOR 1: COMPETITOR INVENTORY
# ========================================

def detect_competitor_inventory_events(
    asin: str,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect competitor inventory issues (out of stock opportunities).

    Triggers:
    - Competitor has <5 units (imminent OOS)
    - Competitor restocked recently (threat)
    """
    events = []

    if df_competitors.empty or 'inventory_count' not in df_competitors.columns:
        return events

    # Filter out the product itself
    competitors = df_competitors[df_competitors['asin'] != asin]

    for _, comp in competitors.iterrows():
        inventory = comp.get('inventory_count', 0)
        comp_asin = comp.get('asin', '')

        # Low inventory (opportunity)
        if inventory is not None and 0 < inventory < 5:
            # Check if it was higher before (significant drop)
            inventory_7d_ago = comp.get('inventory_count_7d_ago', inventory)

            if inventory_7d_ago and inventory_7d_ago > 10:
                # Significant drop detected
                delta_pct = ((inventory - inventory_7d_ago) / inventory_7d_ago) * 100
                severity = 9 if inventory < 3 else 8

                events.append(TriggerEvent(
                    event_type="competitor_oos_imminent",
                    severity=severity,
                    detected_at=datetime.now(),
                    metric_name="fba_inventory",
                    baseline_value=float(inventory_7d_ago),
                    current_value=float(inventory),
                    delta_pct=delta_pct,
                    affected_asin=asin,
                    related_asin=comp_asin
                ))

    return events


# ========================================
# DETECTOR 2: PRICE WARS
# ========================================

def detect_price_war_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 7
) -> List[TriggerEvent]:
    """
    Detect price wars (3+ price drops in short period).

    Triggers:
    - 3+ price drops in 7 days
    - Total price decline >15%
    """
    events = []

    if df_historical.empty or 'price' not in df_historical.columns:
        return events

    # Get recent price changes
    recent = df_historical.tail(lookback_days * 24)  # Hourly data
    if len(recent) < 2:
        return events

    # Count price drops
    price_changes = recent['price'].diff()
    price_drops = (price_changes < 0).sum()

    # Total price change
    if len(recent) > 0:
        start_price = recent['price'].iloc[0]
        end_price = recent['price'].iloc[-1]

        if start_price > 0:
            total_change_pct = ((end_price - start_price) / start_price) * 100

            # Price war detected if 3+ drops OR significant decline
            if price_drops >= 3 or total_change_pct < -15:
                severity = 8 if price_drops >= 5 else 7

                events.append(TriggerEvent(
                    event_type="price_war_active",
                    severity=severity,
                    detected_at=datetime.now(),
                    metric_name="price_change_frequency",
                    baseline_value=float(price_drops),
                    current_value=float(price_drops),
                    delta_pct=total_change_pct,
                    affected_asin=asin
                ))

    return events


# ========================================
# DETECTOR 3: REVIEW VELOCITY
# ========================================

def detect_review_velocity_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 30
) -> List[TriggerEvent]:
    """
    Detect review velocity spikes (positive or concerning).

    Triggers:
    - Review count increased by 50+ in 30 days (YOUR product = good)
    - Competitor review spike (threat)
    """
    events = []

    if df_historical.empty or 'review_count' not in df_historical.columns:
        return events

    # Get current and baseline review counts
    recent = df_historical.tail(lookback_days * 24)
    if len(recent) < 2:
        return events

    current_reviews = recent['review_count'].iloc[-1]
    baseline_reviews = recent['review_count'].iloc[0]

    if pd.isna(current_reviews) or pd.isna(baseline_reviews):
        return events

    review_delta = current_reviews - baseline_reviews

    # Significant spike detected
    if review_delta > 50:
        delta_pct = (review_delta / baseline_reviews * 100) if baseline_reviews > 0 else 0

        # Your product gaining reviews is GOOD (lower severity = informational)
        severity = 6

        events.append(TriggerEvent(
            event_type="review_velocity_spike",
            severity=severity,
            detected_at=datetime.now(),
            metric_name="review_count",
            baseline_value=float(baseline_reviews),
            current_value=float(current_reviews),
            delta_pct=delta_pct,
            affected_asin=asin
        ))

    return events


# ========================================
# DETECTOR 4: BUYBOX SHARE
# ========================================

def detect_buybox_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 7
) -> List[TriggerEvent]:
    """
    Detect BuyBox share changes.

    Triggers:
    - BuyBox share dropped below 50% from >80%
    - Complete BuyBox loss (<10%)
    """
    events = []

    if df_historical.empty or 'buybox_share' not in df_historical.columns:
        return events

    recent = df_historical.tail(lookback_days * 24)
    if len(recent) < 2:
        return events

    current_bb = recent['buybox_share'].iloc[-1]
    baseline_bb = recent['buybox_share'].iloc[0]

    if pd.isna(current_bb) or pd.isna(baseline_bb):
        return events

    # Significant BuyBox loss
    if current_bb < 0.5 and baseline_bb >= 0.8:
        delta_pct = ((current_bb - baseline_bb) / baseline_bb * 100)

        # Critical if lost most of BuyBox
        severity = 10 if current_bb < 0.3 else 9

        events.append(TriggerEvent(
            event_type="buybox_share_collapse",
            severity=severity,
            detected_at=datetime.now(),
            metric_name="buybox_share",
            baseline_value=float(baseline_bb),
            current_value=float(current_bb),
            delta_pct=delta_pct,
            affected_asin=asin
        ))

    return events


# ========================================
# DETECTOR 5: RANK DEGRADATION
# ========================================

def detect_rank_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 30
) -> List[TriggerEvent]:
    """
    Detect rank (BSR) degradation.

    Triggers:
    - BSR worsened by 30%+ in 30 days
    - High rank volatility (unstable position)
    """
    events = []

    if df_historical.empty or 'bsr' not in df_historical.columns:
        return events

    recent = df_historical.tail(lookback_days * 24)
    if len(recent) < 2:
        return events

    # Remove NaN values
    bsr_values = recent['bsr'].dropna()
    if len(bsr_values) < 2:
        return events

    current_bsr = bsr_values.iloc[-1]
    baseline_bsr = bsr_values.iloc[0]

    # Calculate BSR change (positive = rank got worse)
    bsr_change_pct = ((current_bsr - baseline_bsr) / baseline_bsr * 100) if baseline_bsr > 0 else 0

    # Rank degraded significantly
    if bsr_change_pct > 30:  # Rank got 30%+ worse
        severity = 8 if bsr_change_pct > 50 else 7

        events.append(TriggerEvent(
            event_type="rank_degradation",
            severity=severity,
            detected_at=datetime.now(),
            metric_name="sales_rank",
            baseline_value=float(baseline_bsr),
            current_value=float(current_bsr),
            delta_pct=bsr_change_pct,
            affected_asin=asin
        ))

    # Check rank volatility
    if len(bsr_values) > 10:
        volatility = bsr_values.std() / bsr_values.mean() if bsr_values.mean() > 0 else 0

        if volatility > 0.5:  # High volatility
            events.append(TriggerEvent(
                event_type="rank_volatility_high",
                severity=5,
                detected_at=datetime.now(),
                metric_name="bsr_volatility",
                baseline_value=0.0,
                current_value=float(volatility),
                delta_pct=volatility * 100,
                affected_asin=asin
            ))

    return events


# ========================================
# DETECTOR 6: NEW COMPETITORS
# ========================================

def detect_new_competitor_events(
    asin: str,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect new competitors entering market.

    Triggers:
    - New ASIN with strong BSR (<10,000) appeared recently
    """
    events = []

    if df_competitors.empty:
        return events

    # Filter for recent entrants
    if 'first_seen' in df_competitors.columns:
        recent_cutoff = datetime.now() - timedelta(days=30)
        new_competitors = df_competitors[
            (df_competitors['first_seen'] >= recent_cutoff) &
            (df_competitors['asin'] != asin)
        ]

        for _, comp in new_competitors.iterrows():
            bsr = comp.get('bsr', 999999)

            # Strong new competitor (BSR < 10k)
            if bsr < 10000:
                severity = 7 if bsr < 5000 else 6

                events.append(TriggerEvent(
                    event_type="new_competitor_entered",
                    severity=severity,
                    detected_at=datetime.now(),
                    metric_name="sales_rank",
                    baseline_value=999999.0,  # Didn't exist before
                    current_value=float(bsr),
                    delta_pct=-100.0,  # New entry
                    affected_asin=asin,
                    related_asin=comp.get('asin')
                ))

    return events


# ========================================
# UTILITY FUNCTIONS
# ========================================

def filter_triggers_by_severity(
    events: List[TriggerEvent],
    min_severity: int = 5
) -> List[TriggerEvent]:
    """Filter events by minimum severity threshold."""
    return [e for e in events if e.severity >= min_severity]


def group_triggers_by_category(
    events: List[TriggerEvent]
) -> dict:
    """Group trigger events by category."""
    from src.models.trigger_event import get_event_category

    grouped = {}
    for event in events:
        category = get_event_category(event.event_type)
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(event)

    return grouped


def get_top_triggers(
    events: List[TriggerEvent],
    limit: int = 5
) -> List[TriggerEvent]:
    """Get top N triggers by severity."""
    return sorted(events, key=lambda e: e.severity, reverse=True)[:limit]
