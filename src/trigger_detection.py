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
    
    # === NEW: KEEPA INTELLIGENCE DETECTORS (2026-01-21) ===
    events.extend(detect_amazon_supply_instability(asin, df_competitors))
    events.extend(detect_backorder_crisis(asin, df_historical))
    events.extend(detect_subscription_opportunity(asin, df_historical))

    # === PHASE 2: CAUSAL INTELLIGENCE DETECTORS (2026-01-23) ===
    # Enhanced with full Keepa metric integration and Supabase category_intelligence
    events.extend(detect_platform_changes(asin, df_historical, df_competitors))
    events.extend(detect_competitor_creative_changes(asin, df_historical, df_competitors))
    events.extend(detect_share_of_voice_changes(asin, df_historical, df_competitors))
    events.extend(detect_macro_trends(asin, df_historical, df_competitors))
    events.extend(detect_competition_intensity(asin, df_historical, df_competitors))

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


# ========================================
# NEW DETECTORS: KEEPA INTELLIGENCE (2026-01-21)
# ========================================

def detect_amazon_supply_instability(
    asin: str,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect Amazon 1P supply instability using new Keepa metrics.
    
    Triggers when Amazon has gone OOS 3+ times in 30 days.
    This is a HIGH-VALUE conquest opportunity.
    
    Uses:
    - oos_count_amazon_30: Number of Amazon OOS events in 30 days
    - buybox_is_amazon: Whether Amazon owns the Buy Box
    - has_amazon_seller: Whether Amazon is a seller on the listing
    """
    events = []
    
    if df_competitors.empty:
        return events
    
    # Check for Amazon OOS data in competitor listings
    oos_col = 'oos_count_amazon_30'
    amazon_bb_col = 'buybox_is_amazon'
    has_amazon_col = 'has_amazon_seller'
    
    for _, row in df_competitors.iterrows():
        comp_asin = row.get('asin', '')
        if comp_asin == asin:
            continue  # Skip self
        
        oos_count = row.get(oos_col, 0) or 0
        amazon_owns_bb = row.get(amazon_bb_col, False)
        has_amazon = row.get(has_amazon_col, False)
        
        # Amazon supply instability detected
        if oos_count >= 3 and (amazon_owns_bb or has_amazon):
            # Higher severity for more OOS events
            if oos_count >= 7:
                severity = 10  # Critical opportunity
            elif oos_count >= 5:
                severity = 9
            else:
                severity = 8
            
            events.append(TriggerEvent(
                event_type="amazon_supply_unstable",
                severity=severity,
                detected_at=datetime.now(),
                metric_name="oos_count_amazon_30",
                baseline_value=0.0,  # Expected 0 OOS
                current_value=float(oos_count),
                delta_pct=float(oos_count) * 100,  # Each OOS = 100% signal strength
                affected_asin=asin,
                related_asin=comp_asin
            ))
    
    return events


def detect_backorder_crisis(
    asin: str,
    df_historical: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect backorder crisis using new Keepa metrics.
    
    Triggers when product is backordered - this is URGENT.
    
    Uses:
    - buybox_is_backorder: Boolean flag from Keepa
    """
    events = []
    
    if df_historical.empty:
        return events
    
    # Check most recent data point for backorder status
    if 'buybox_is_backorder' in df_historical.columns:
        latest = df_historical.iloc[-1] if len(df_historical) > 0 else None
        
        if latest is not None and latest.get('buybox_is_backorder', False):
            events.append(TriggerEvent(
                event_type="backorder_crisis",
                severity=10,  # Maximum severity - this is URGENT
                detected_at=datetime.now(),
                metric_name="buybox_is_backorder",
                baseline_value=0.0,  # Not backordered
                current_value=1.0,  # Backordered
                delta_pct=100.0,  # Full crisis
                affected_asin=asin,
                related_asin=None
            ))
    
    return events


def detect_subscription_opportunity(
    asin: str,
    df_historical: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect Subscribe & Save opportunity using new Keepa metrics.
    
    Triggers when product is S&S eligible and has strong fundamentals.
    
    Uses:
    - is_sns: Boolean flag from Keepa
    - revenue/rank metrics for validation
    """
    events = []
    
    if df_historical.empty:
        return events
    
    # Check if product is S&S eligible
    if 'is_sns' in df_historical.columns:
        latest = df_historical.iloc[-1] if len(df_historical) > 0 else None
        
        if latest is not None and latest.get('is_sns', False):
            # Check if product has good fundamentals (worth pushing S&S)
            revenue = latest.get('weekly_sales_filled', latest.get('revenue_proxy', 0)) or 0
            rank = latest.get('sales_rank_filled', latest.get('sales_rank', 999999)) or 999999
            
            # Only trigger for products worth the effort
            if revenue > 2000 or rank < 1000:
                # Estimate subscription opportunity (5-10% conversion)
                sns_opportunity = revenue * 0.08 * 4.33  # 8% * monthly
                
                events.append(TriggerEvent(
                    event_type="subscription_opportunity",
                    severity=5,  # Lower severity - opportunity, not urgent
                    detected_at=datetime.now(),
                    metric_name="is_sns",
                    baseline_value=0.0,
                    current_value=sns_opportunity,  # Estimated $ opportunity
                    delta_pct=8.0,  # 8% conversion estimate
                    affected_asin=asin,
                    related_asin=None
                ))
    
    return events


# ========================================
# PHASE 2: CAUSAL INTELLIGENCE DETECTORS
# Enhanced with full Keepa metric integration
# ========================================

def detect_platform_changes(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect Amazon platform/algorithmic changes affecting the product.
    
    ENHANCED: Uses Keepa pre-calculated metrics:
    - bb_stats_amazon_30: Amazon's BB ownership share (30-day)
    - velocity_30d: Pre-calculated BSR velocity (more accurate than manual calc)
    - buybox_is_amazon: Current BB holder
    
    Signals:
    1. Amazon buybox takeover (using bb_stats)
    2. Algorithm shift (unexplained velocity change)
    3. Platform favor/disfavor
    """
    events = []
    
    if df_historical.empty:
        return events
    
    latest = df_historical.iloc[-1] if len(df_historical) > 0 else {}
    
    # === USE KEEPA'S PRE-CALCULATED VELOCITY ===
    # velocity_30d is much more accurate than manual BSR delta calculation
    velocity_30d = latest.get('velocity_30d', None)
    
    if velocity_30d is not None and isinstance(velocity_30d, (int, float)):
        # Significant positive velocity (rank improving) = platform/algo favor
        if velocity_30d < -25:  # Rank dropped 25%+ (negative = improvement in Keepa)
            # Check for confounders
            price_change = False
            review_spike = False
            
            # Check if price dropped (which would explain rank improvement)
            if 'buy_box_price' in df_historical.columns and len(df_historical) >= 7:
                recent_price = df_historical['buy_box_price'].tail(7).mean()
                older_price = df_historical['buy_box_price'].iloc[-30:-7].mean() if len(df_historical) >= 30 else df_historical['buy_box_price'].iloc[:-7].mean()
                if older_price > 0:
                    price_change = abs((recent_price - older_price) / older_price) > 0.10
            
            if not price_change:
                events.append(TriggerEvent(
                    event_type="platform_algorithm_boost",
                    severity=5,  # Positive event
                    detected_at=datetime.now(),
                    metric_name="velocity_30d_unexplained",
                    baseline_value=0.0,
                    current_value=float(velocity_30d),
                    delta_pct=abs(velocity_30d),
                    affected_asin=asin,
                    related_asin=None
                ))
        # Significant negative velocity (rank worsening) = potential algo demotion
        elif velocity_30d > 30:  # Rank worsened 30%+ 
            events.append(TriggerEvent(
                event_type="platform_algorithm_shift",
                severity=7,
                detected_at=datetime.now(),
                metric_name="velocity_30d_decline",
                baseline_value=0.0,
                current_value=float(velocity_30d),
                delta_pct=-velocity_30d,  # Make negative for downstream
                affected_asin=asin,
                related_asin=None
            ))
    
    # === USE KEEPA'S BB STATS FOR AMAZON TAKEOVER DETECTION ===
    # bb_stats_amazon_30 is Amazon's % of BB ownership over 30 days
    bb_amazon_30 = latest.get('bb_stats_amazon_30', None)
    bb_amazon_90 = latest.get('bb_stats_amazon_90', None)
    
    if bb_amazon_30 is not None and bb_amazon_90 is not None:
        if isinstance(bb_amazon_30, (int, float)) and isinstance(bb_amazon_90, (int, float)):
            bb_delta = bb_amazon_30 - bb_amazon_90
            
            # Amazon gained significant BB share (30-day vs 90-day)
            if bb_delta > 0.25:  # 25%+ increase in Amazon's BB share
                events.append(TriggerEvent(
                    event_type="platform_amazon_takeover",
                    severity=8,
                    detected_at=datetime.now(),
                    metric_name="bb_stats_amazon_30",
                    baseline_value=float(bb_amazon_90),
                    current_value=float(bb_amazon_30),
                    delta_pct=bb_delta * 100,
                    affected_asin=asin,
                    related_asin=None
                ))
            # Amazon retreated from this ASIN
            elif bb_delta < -0.25:  # 25%+ decrease
                events.append(TriggerEvent(
                    event_type="platform_amazon_retreat",
                    severity=6,  # Opportunity for 3P sellers
                    detected_at=datetime.now(),
                    metric_name="bb_stats_amazon_30",
                    baseline_value=float(bb_amazon_90),
                    current_value=float(bb_amazon_30),
                    delta_pct=bb_delta * 100,
                    affected_asin=asin,
                    related_asin=None
                ))
    
    # === BACKORDER DETECTION USING KEEPA FLAG ===
    buybox_is_backorder = latest.get('buybox_is_backorder', False)
    if buybox_is_backorder:
        events.append(TriggerEvent(
            event_type="platform_backorder_active",
            severity=9,  # Critical supply issue
            detected_at=datetime.now(),
            metric_name="buybox_is_backorder",
            baseline_value=0.0,
            current_value=1.0,
            delta_pct=100.0,
            affected_asin=asin,
            related_asin=None
        ))
    
    # === OOS EVENT COUNT (more actionable than percentage) ===
    oos_count_30 = latest.get('oos_count_amazon_30', 0) or 0
    if isinstance(oos_count_30, (int, float)) and oos_count_30 >= 3:
        events.append(TriggerEvent(
            event_type="platform_amazon_oos_pattern",
            severity=6,
            detected_at=datetime.now(),
            metric_name="oos_count_amazon_30",
            baseline_value=0.0,
            current_value=float(oos_count_30),
            delta_pct=float(oos_count_30) * 10,  # Scale for visibility
            affected_asin=asin,
            related_asin=None
        ))
    
    return events


def detect_competitor_creative_changes(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect when competitors likely refreshed creative (images, titles, A+ content).
    
    ENHANCED: Uses Keepa pre-calculated metrics:
    - velocity_30d: BSR velocity (negative = improving)
    - bb_stats_top_seller_30: Dominant seller percentage
    - review velocity from Keepa stats
    
    Heuristic: Review velocity spike + negative velocity (rank improvement) = likely refresh
    """
    events = []
    
    if df_competitors.empty:
        return events
    
    # Look for competitors with improved metrics (suggests creative refresh)
    for _, comp in df_competitors.iterrows():
        comp_asin = comp.get('asin', '')
        if comp_asin == asin:
            continue  # Skip self
        
        # === USE KEEPA'S VELOCITY_30D (negative = rank improvement) ===
        velocity_30d = comp.get('velocity_30d', None)
        velocity_90d = comp.get('velocity_90d', None)
        
        # Check for review velocity spike
        review_30d = comp.get('review_velocity_30d', comp.get('delta30_reviews', 0)) or 0
        review_90d = comp.get('review_velocity_90d', comp.get('reviews_per_month', 0)) or 0
        
        # Velocity-based detection (if available)
        if velocity_30d is not None and isinstance(velocity_30d, (int, float)):
            rank_improved = velocity_30d < -15  # Rank improved 15%+
            rank_improvement_pct = abs(velocity_30d) if velocity_30d < 0 else 0
            
            # Compare 30d vs 90d velocity for acceleration detection
            if velocity_90d is not None and isinstance(velocity_90d, (int, float)):
                velocity_acceleration = velocity_90d - velocity_30d
                # Accelerating improvement suggests effective action
                if velocity_acceleration > 20:  # Momentum building
                    rank_improved = True
                    rank_improvement_pct = max(rank_improvement_pct, velocity_acceleration)
        else:
            # Fallback to delta30_bsr
            rank_change = comp.get('rank_change_30d', comp.get('delta30_bsr', 0)) or 0
            if isinstance(rank_change, (int, float)) and rank_change < 0:
                rank_improved = True
                rank_improvement_pct = abs(rank_change)
            else:
                rank_improved = False
                rank_improvement_pct = 0
        
        # Check for combined spike (review velocity up + rank improved)
        review_spike = review_30d > review_90d * 1.5 if review_90d > 0 else review_30d > 10
        
        if review_spike and rank_improved and rank_improvement_pct > 15:
            events.append(TriggerEvent(
                event_type="competitor_creative_refresh",
                severity=5,
                detected_at=datetime.now(),
                metric_name="competitor_conversion_improvement",
                baseline_value=float(review_90d) if review_90d else 0,
                current_value=float(review_30d),
                delta_pct=rank_improvement_pct,
                affected_asin=asin,
                related_asin=comp_asin
            ))
        
        # === NEW: DETECT COMPETITOR SELLER CONCENTRATION ===
        bb_top_seller_30 = comp.get('bb_stats_top_seller_30', None)
        if bb_top_seller_30 is not None and isinstance(bb_top_seller_30, (int, float)):
            # If one seller dominates 80%+ of buybox = potential supply control
            if bb_top_seller_30 > 0.80:
                events.append(TriggerEvent(
                    event_type="competitor_seller_concentration",
                    severity=4,
                    detected_at=datetime.now(),
                    metric_name="bb_stats_top_seller_30",
                    baseline_value=0.5,  # Normal baseline
                    current_value=float(bb_top_seller_30),
                    delta_pct=(bb_top_seller_30 - 0.5) * 100,
                    affected_asin=asin,
                    related_asin=comp_asin
                ))
    
    return events


def detect_share_of_voice_changes(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Detect Share of Voice changes (keyword ranking position shifts).
    
    ENHANCED: Uses Keepa pre-calculated velocity metrics:
    - velocity_30d: Your pre-calculated BSR velocity
    - Competitor velocity_30d for market comparison
    
    Uses relative velocity vs competitors as a proxy for search visibility changes.
    """
    events = []
    
    if df_historical.empty or df_competitors.empty:
        return events
    
    latest = df_historical.iloc[-1] if len(df_historical) > 0 else {}
    
    # === USE KEEPA'S PRE-CALCULATED VELOCITY ===
    your_velocity_30d = latest.get('velocity_30d', None)
    
    # Collect competitor velocities
    comp_velocities = []
    for _, comp in df_competitors.iterrows():
        comp_asin = comp.get('asin', '')
        if comp_asin == asin:
            continue
        
        # Prefer velocity_30d, fallback to delta30_bsr
        comp_velocity = comp.get('velocity_30d', None)
        if comp_velocity is None:
            comp_velocity = comp.get('delta30_bsr', comp.get('rank_change_30d', None))
        
        if comp_velocity is not None and isinstance(comp_velocity, (int, float)):
            comp_velocities.append(comp_velocity)
    
    # If we have velocity data, use it
    if your_velocity_30d is not None and isinstance(your_velocity_30d, (int, float)) and len(comp_velocities) >= 3:
        avg_comp_velocity = sum(comp_velocities) / len(comp_velocities)
        
        # Relative performance: negative velocity = improving, so lower is better
        relative_performance = your_velocity_30d - avg_comp_velocity
        
        # You're outperforming market (your velocity more negative than competitors)
        if relative_performance < -20:  # 20%+ better than market
            events.append(TriggerEvent(
                event_type="share_of_voice_gained",
                severity=6,
                detected_at=datetime.now(),
                metric_name="relative_velocity_30d",
                baseline_value=float(avg_comp_velocity),
                current_value=float(your_velocity_30d),
                delta_pct=relative_performance,
                affected_asin=asin,
                related_asin=None
            ))
        # You're underperforming market (your velocity more positive than competitors)
        elif relative_performance > 20:  # 20%+ worse than market
            events.append(TriggerEvent(
                event_type="share_of_voice_lost",
                severity=7,
                detected_at=datetime.now(),
                metric_name="relative_velocity_30d",
                baseline_value=float(avg_comp_velocity),
                current_value=float(your_velocity_30d),
                delta_pct=relative_performance,
                affected_asin=asin,
                related_asin=None
            ))
    else:
        # Fallback to BSR-based calculation
        rank_col = 'sales_rank_filled' if 'sales_rank_filled' in df_historical.columns else 'sales_rank' if 'sales_rank' in df_historical.columns else None
        
        if rank_col and len(df_historical) >= 14:
            recent = df_historical.tail(7)
            older = df_historical.iloc[-30:-7] if len(df_historical) >= 30 else df_historical.iloc[:-7]
            
            if len(older) > 0:
                your_recent_rank = recent[rank_col].mean() if rank_col in recent.columns else None
                your_older_rank = older[rank_col].mean() if rank_col in older.columns else None
                
                if your_recent_rank and your_older_rank and your_older_rank > 0:
                    your_rank_change_pct = ((your_recent_rank - your_older_rank) / your_older_rank) * 100
                    
                    comp_rank_changes = []
                    for _, comp in df_competitors.iterrows():
                        rank_change = comp.get('delta30_bsr', comp.get('rank_change_30d', 0)) or 0
                        if isinstance(rank_change, (int, float)):
                            comp_rank_changes.append(rank_change)
                    
                    if len(comp_rank_changes) >= 3:
                        avg_comp_rank_change = sum(comp_rank_changes) / len(comp_rank_changes)
                        relative_performance = your_rank_change_pct - avg_comp_rank_change
                        
                        if relative_performance < -20:
                            events.append(TriggerEvent(
                                event_type="share_of_voice_gained",
                                severity=6,
                                detected_at=datetime.now(),
                                metric_name="relative_rank_performance",
                                baseline_value=float(avg_comp_rank_change),
                                current_value=float(your_rank_change_pct),
                                delta_pct=relative_performance,
                                affected_asin=asin,
                                related_asin=None
                            ))
                        elif relative_performance > 20:
                            events.append(TriggerEvent(
                                event_type="share_of_voice_lost",
                                severity=7,
                                detected_at=datetime.now(),
                                metric_name="relative_rank_performance",
                                baseline_value=float(avg_comp_rank_change),
                                current_value=float(your_rank_change_pct),
                                delta_pct=relative_performance,
                                affected_asin=asin,
                                related_asin=None
                            ))
    
    return events


def detect_macro_trends(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame,
    category_id: Optional[int] = None  # NEW: For Supabase lookup
) -> List[TriggerEvent]:
    """
    Detect category-wide macro trends (market growth/decline).
    
    ENHANCED: Integrates with Supabase category_intelligence table for real benchmarks
    and uses Keepa OOS metrics for supply crisis detection.
    
    Uses aggregate competitor metrics to identify category-level shifts.
    """
    events = []
    
    if df_competitors.empty:
        return events
    
    # === TRY SUPABASE CATEGORY_INTELLIGENCE FOR REAL MARKET BENCHMARKS ===
    category_benchmarks = None
    if category_id is not None:
        try:
            from src.supabase_reader import get_supabase_client
            client = get_supabase_client()
            cat_intel = client.table("category_intelligence")\
                .select("*")\
                .eq("category_id", category_id)\
                .order("snapshot_date", desc=True)\
                .limit(2)\
                .execute()
            
            if cat_intel.data and len(cat_intel.data) >= 2:
                current = cat_intel.data[0]
                previous = cat_intel.data[1]
                
                current_revenue = current.get("total_weekly_revenue", 0) or 0
                previous_revenue = previous.get("total_weekly_revenue", 0) or 0
                
                if previous_revenue > 0:
                    category_growth_pct = ((current_revenue - previous_revenue) / previous_revenue) * 100
                    
                    if category_growth_pct > 15:
                        events.append(TriggerEvent(
                            event_type="macro_market_expansion",
                            severity=4,
                            detected_at=datetime.now(),
                            metric_name="category_intelligence_revenue",
                            baseline_value=float(previous_revenue),
                            current_value=float(current_revenue),
                            delta_pct=category_growth_pct,
                            affected_asin=asin,
                            related_asin=None
                        ))
                    elif category_growth_pct < -15:
                        events.append(TriggerEvent(
                            event_type="macro_market_contraction",
                            severity=5,
                            detected_at=datetime.now(),
                            metric_name="category_intelligence_revenue",
                            baseline_value=float(previous_revenue),
                            current_value=float(current_revenue),
                            delta_pct=category_growth_pct,
                            affected_asin=asin,
                            related_asin=None
                        ))
                    
                    category_benchmarks = current  # Save for later use
        except Exception:
            pass  # Fallback to competitor-based calculation
    
    # === FALLBACK: Calculate from df_competitors if no Supabase data ===
    if not category_benchmarks:
        revenue_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in df_competitors.columns else 'revenue_proxy' if 'revenue_proxy' in df_competitors.columns else 'monthly_revenue' if 'monthly_revenue' in df_competitors.columns else None
        
        if revenue_col:
            total_category_revenue = df_competitors[revenue_col].sum()
            delta_col = 'delta30_revenue' if 'delta30_revenue' in df_competitors.columns else None
            
            if delta_col:
                total_delta = df_competitors[delta_col].sum()
                older_revenue = total_category_revenue - total_delta
                
                if older_revenue > 0:
                    category_growth_pct = (total_delta / older_revenue) * 100
                    
                    if category_growth_pct > 15:
                        events.append(TriggerEvent(
                            event_type="macro_market_expansion",
                            severity=4,
                            detected_at=datetime.now(),
                            metric_name="category_revenue_growth",
                            baseline_value=float(older_revenue),
                            current_value=float(total_category_revenue),
                            delta_pct=category_growth_pct,
                            affected_asin=asin,
                            related_asin=None
                        ))
                    elif category_growth_pct < -15:
                        events.append(TriggerEvent(
                            event_type="macro_market_contraction",
                            severity=5,
                            detected_at=datetime.now(),
                            metric_name="category_revenue_decline",
                            baseline_value=float(older_revenue),
                            current_value=float(total_category_revenue),
                            delta_pct=category_growth_pct,
                            affected_asin=asin,
                            related_asin=None
                        ))
    
    # === USE KEEPA OOS METRICS FOR SUPPLY CRISIS DETECTION ===
    oos_count_from_keepa = 0
    oos_count_from_inventory = 0
    
    for _, comp in df_competitors.iterrows():
        # Prefer Keepa's oos_pct_30 metric (more accurate)
        oos_pct = comp.get('oos_pct_30', 0) or 0
        if isinstance(oos_pct, (int, float)) and oos_pct > 0.3:  # 30%+ OOS rate
            oos_count_from_keepa += 1
        
        # Fallback to inventory count
        inventory = comp.get('inventory_count', comp.get('stock_estimate', 99)) or 99
        if inventory < 5:
            oos_count_from_inventory += 1
    
    # Use whichever method detected more OOS (better sensitivity)
    oos_count = max(oos_count_from_keepa, oos_count_from_inventory)
    total_competitors = len(df_competitors)
    
    if total_competitors >= 5 and oos_count >= total_competitors * 0.3:
        events.append(TriggerEvent(
            event_type="macro_supply_crisis",
            severity=8,
            detected_at=datetime.now(),
            metric_name="market_oos_rate",
            baseline_value=0.0,
            current_value=float(oos_count / total_competitors) * 100,
            delta_pct=float(oos_count / total_competitors) * 100,
            affected_asin=asin,
            related_asin=None
        ))
    
    return events


def detect_competition_intensity(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    NEW DETECTOR: Analyze competition intensity using Keepa BB stats.
    
    Uses:
    - bb_stats_seller_count_30: Number of sellers rotating the buybox
    - seller_count: Total seller count
    - new_offer_count: Offer density
    """
    events = []
    
    if df_historical.empty:
        return events
    
    latest = df_historical.iloc[-1] if len(df_historical) > 0 else {}
    
    # === BUYBOX SELLER COUNT (True competition metric) ===
    bb_seller_count = latest.get('bb_stats_seller_count_30', None)
    
    if bb_seller_count is not None and isinstance(bb_seller_count, (int, float)):
        # High seller rotation = intense competition
        if bb_seller_count >= 5:
            events.append(TriggerEvent(
                event_type="competition_high_intensity",
                severity=5,
                detected_at=datetime.now(),
                metric_name="bb_stats_seller_count_30",
                baseline_value=2.0,  # Normal is 1-2 sellers
                current_value=float(bb_seller_count),
                delta_pct=(bb_seller_count - 2) * 50,  # Scale for visibility
                affected_asin=asin,
                related_asin=None
            ))
        # Low competition (opportunity)
        elif bb_seller_count <= 1:
            events.append(TriggerEvent(
                event_type="competition_low_intensity",
                severity=3,  # Opportunity
                detected_at=datetime.now(),
                metric_name="bb_stats_seller_count_30",
                baseline_value=2.0,
                current_value=float(bb_seller_count),
                delta_pct=-(2 - bb_seller_count) * 50,
                affected_asin=asin,
                related_asin=None
            ))
    
    # === OFFER COUNT SURGE (New competitors entering) ===
    current_offers = latest.get('new_offer_count', None)
    
    if current_offers is not None and isinstance(current_offers, (int, float)) and len(df_historical) >= 14:
        older_offers = df_historical.iloc[-30:-7]['new_offer_count'].mean() if len(df_historical) >= 30 else df_historical.iloc[:-7]['new_offer_count'].mean()
        
        if older_offers and older_offers > 0:
            offer_change_pct = ((current_offers - older_offers) / older_offers) * 100
            
            # Significant increase in sellers
            if offer_change_pct > 50:  # 50%+ more sellers
                events.append(TriggerEvent(
                    event_type="competition_new_entrants",
                    severity=6,
                    detected_at=datetime.now(),
                    metric_name="new_offer_count_change",
                    baseline_value=float(older_offers),
                    current_value=float(current_offers),
                    delta_pct=offer_change_pct,
                    affected_asin=asin,
                    related_asin=None
                ))
    
    return events


