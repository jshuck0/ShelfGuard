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

    Runs 10 core detectors for comprehensive market intelligence:
    1. Competitor inventory drops (OOS detection) - OPPORTUNITY
    2. Price wars (3+ drops in 7d) - THREAT
    3. Review velocity spikes - OPPORTUNITY
    4. BuyBox share collapse - THREAT
    5. Rank degradation - THREAT
    6. New competitor detection - THREAT
    7. Price power opportunity (underpriced) - OPPORTUNITY
    8. Rating decline - THREAT  
    9. Momentum acceleration - OPPORTUNITY
    10. Seller consolidation - OPPORTUNITY

    Args:
        asin: Product ASIN to analyze
        df_historical: 90-day historical data for this ASIN
        df_competitors: Current competitor data in same category
        lookback_days: How far back to detect events

    Returns:
        List of TriggerEvent objects, sorted by severity (highest first)
    """

    events: List[TriggerEvent] = []

    # === THREAT DETECTORS ===
    events.extend(detect_price_war_events(asin, df_historical, lookback_days))
    events.extend(detect_buybox_events(asin, df_historical, lookback_days))
    events.extend(detect_rank_events(asin, df_historical, lookback_days))
    events.extend(detect_new_competitor_events(asin, df_competitors))
    events.extend(detect_rating_decline_events(asin, df_historical, lookback_days))
    
    # === OPPORTUNITY DETECTORS ===
    events.extend(detect_competitor_inventory_events(asin, df_competitors))
    events.extend(detect_review_velocity_events(asin, df_historical, lookback_days))
    events.extend(detect_price_power_events(asin, df_historical, df_competitors))
    events.extend(detect_momentum_events(asin, df_historical, lookback_days))
    events.extend(detect_seller_consolidation_events(asin, df_competitors))

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
# DETECTOR 7: PRICE POWER OPPORTUNITY
# ========================================

def detect_price_power_events(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect pricing power opportunity (product is underpriced relative to position).

    Triggers:
    - Strong reviews + lower than median price = opportunity to raise price
    - Stable/improving rank + price below category = pricing power
    """
    events = []

    if df_historical.empty or df_competitors.empty:
        return events

    # Get current product metrics
    if len(df_historical) == 0:
        return events
    
    current = df_historical.iloc[-1] if hasattr(df_historical, 'iloc') else {}
    
    # Get current price and review count
    current_price = current.get('price', current.get('buy_box_price', 0))
    current_reviews = current.get('review_count', 0)
    
    if pd.isna(current_price) or current_price <= 0:
        return events
    
    # Get median price from competitors
    if 'buy_box_price' in df_competitors.columns:
        prices = df_competitors['buy_box_price'].dropna()
        if len(prices) > 0:
            median_price = prices.median()
            
            # Check if we have price below median + strong reviews
            if current_price < median_price * 0.9:  # 10%+ below median
                # Check review strength
                median_reviews = 0
                if 'review_count' in df_competitors.columns:
                    reviews = df_competitors['review_count'].dropna()
                    if len(reviews) > 0:
                        median_reviews = reviews.median()
                
                # Strong reviews = 50%+ above median
                if current_reviews > median_reviews * 1.5:
                    price_gap_pct = ((current_price / median_price) - 1) * 100
                    review_advantage_pct = ((current_reviews / median_reviews) - 1) * 100 if median_reviews > 0 else 100
                    
                    events.append(TriggerEvent(
                        event_type="opportunity_price_power",
                        severity=7,  # High priority opportunity
                        detected_at=datetime.now(),
                        metric_name="price_vs_median",
                        baseline_value=float(median_price),
                        current_value=float(current_price),
                        delta_pct=price_gap_pct,
                        affected_asin=asin
                    ))

    return events


# ========================================
# DETECTOR 8: RATING DECLINE
# ========================================

def detect_rating_decline_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 30
) -> List[TriggerEvent]:
    """
    Detect rating decline events.

    Triggers:
    - Rating dropped 0.3+ stars in 30 days
    - Rating fell below 4.0 stars
    """
    events = []

    if df_historical.empty or 'rating' not in df_historical.columns:
        return events

    recent = df_historical.tail(lookback_days * 24) if len(df_historical) > lookback_days * 24 else df_historical
    if len(recent) < 2:
        return events

    # Get rating values (handle Keepa's rating*10 format)
    ratings = recent['rating'].dropna()
    if len(ratings) < 2:
        return events
    
    current_rating = ratings.iloc[-1]
    baseline_rating = ratings.iloc[0]
    
    # Normalize if stored as rating*10
    if current_rating > 10:
        current_rating = current_rating / 10
    if baseline_rating > 10:
        baseline_rating = baseline_rating / 10
    
    rating_drop = baseline_rating - current_rating
    
    # Significant rating decline
    if rating_drop >= 0.3:
        severity = 8 if current_rating < 4.0 else 6
        
        events.append(TriggerEvent(
            event_type="rating_decline",
            severity=severity,
            detected_at=datetime.now(),
            metric_name="rating",
            baseline_value=float(baseline_rating),
            current_value=float(current_rating),
            delta_pct=-rating_drop * 20,  # Approximate % impact
            affected_asin=asin
        ))
    
    # Critical rating threshold
    if current_rating < 3.5 and baseline_rating >= 3.5:
        events.append(TriggerEvent(
            event_type="rating_critical",
            severity=9,
            detected_at=datetime.now(),
            metric_name="rating",
            baseline_value=float(baseline_rating),
            current_value=float(current_rating),
            delta_pct=-rating_drop * 20,
            affected_asin=asin
        ))

    return events


# ========================================
# DETECTOR 9: MOMENTUM ACCELERATION
# ========================================

def detect_momentum_events(
    asin: str,
    df_historical: pd.DataFrame,
    lookback_days: int = 30
) -> List[TriggerEvent]:
    """
    Detect positive momentum acceleration (growth opportunity).

    Triggers:
    - Rank improving 20%+ in 30 days (accelerating sales)
    - Sustained upward trajectory
    """
    events = []

    if df_historical.empty or 'bsr' not in df_historical.columns:
        return events

    recent = df_historical.tail(lookback_days * 24) if len(df_historical) > lookback_days * 24 else df_historical
    if len(recent) < 2:
        return events

    bsr_values = recent['bsr'].dropna()
    if len(bsr_values) < 2:
        return events

    current_bsr = bsr_values.iloc[-1]
    baseline_bsr = bsr_values.iloc[0]

    if baseline_bsr <= 0:
        return events

    # Calculate BSR improvement (negative = rank improved)
    bsr_change_pct = ((current_bsr - baseline_bsr) / baseline_bsr) * 100

    # Rank improved significantly (more negative = better)
    if bsr_change_pct < -20:  # Rank improved by 20%+
        severity = 7 if bsr_change_pct < -40 else 6
        
        events.append(TriggerEvent(
            event_type="momentum_acceleration",
            severity=severity,
            detected_at=datetime.now(),
            metric_name="sales_rank_improvement",
            baseline_value=float(baseline_bsr),
            current_value=float(current_bsr),
            delta_pct=bsr_change_pct,
            affected_asin=asin
        ))
    
    # Check for sustained momentum (3 consecutive weeks of improvement)
    if len(bsr_values) >= 21 * 24:  # 3 weeks of hourly data
        week1_avg = bsr_values.iloc[:7*24].mean()
        week2_avg = bsr_values.iloc[7*24:14*24].mean()
        week3_avg = bsr_values.iloc[14*24:21*24].mean()
        
        if week1_avg > week2_avg > week3_avg > 0:
            # Sustained improvement
            events.append(TriggerEvent(
                event_type="momentum_sustained",
                severity=6,
                detected_at=datetime.now(),
                metric_name="bsr_3_week_trend",
                baseline_value=float(week1_avg),
                current_value=float(week3_avg),
                delta_pct=((week3_avg - week1_avg) / week1_avg) * 100,
                affected_asin=asin
            ))

    return events


# ========================================
# DETECTOR 10: SELLER CONSOLIDATION
# ========================================

def detect_seller_consolidation_events(
    asin: str,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect seller consolidation (competitors leaving = opportunity).

    Triggers:
    - Competitor count decreased significantly
    - Market is consolidating (fewer sellers)
    """
    events = []

    if df_competitors.empty:
        return events

    # Check for competitor count change
    if 'delta30_offer_count' in df_competitors.columns or 'seller_count_change_30d' in df_competitors.columns:
        col = 'delta30_offer_count' if 'delta30_offer_count' in df_competitors.columns else 'seller_count_change_30d'
        
        for _, row in df_competitors.iterrows():
            if row.get('asin') == asin:
                delta = row.get(col, 0)
                if delta is not None and delta < -3:  # 3+ sellers left
                    events.append(TriggerEvent(
                        event_type="seller_consolidation",
                        severity=6,
                        detected_at=datetime.now(),
                        metric_name="seller_count_change",
                        baseline_value=0.0,
                        current_value=float(delta),
                        delta_pct=float(delta) * 10,  # Rough % indicator
                        affected_asin=asin
                    ))
                break
    
    # Check for low competition opportunity
    if 'offer_count' in df_competitors.columns or 'offerCountNew' in df_competitors.columns:
        col = 'offer_count' if 'offer_count' in df_competitors.columns else 'offerCountNew'
        
        for _, row in df_competitors.iterrows():
            if row.get('asin') == asin:
                count = row.get(col, 99)
                if count is not None and count <= 3:  # Very low competition
                    events.append(TriggerEvent(
                        event_type="low_competition",
                        severity=5,
                        detected_at=datetime.now(),
                        metric_name="seller_count",
                        baseline_value=10.0,  # Typical
                        current_value=float(count),
                        delta_pct=((count - 10) / 10) * 100,
                        affected_asin=asin
                    ))
                break

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
