"""
Revenue Attribution Engine

Decomposes portfolio revenue changes into 4 causal categories:
1. Internal Actions (Self-Inflicted): Price changes, PPC budget, coupons
2. Competitive Influence: Competitor OOS, pricing moves, new entrants
3. Platform/Algorithmic: Amazon changes (Choice badge, algorithm shifts)
4. Market/Macro: Seasonal trends, category growth

Attribution Method: Elimination (isolate each category sequentially)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import json
from supabase import Client

from src.models.revenue_attribution import (
    RevenueAttribution,
    AttributionDriver,
    InternalAction,
    CausalCategory,
    ActionType
)
from src.models.trigger_event import TriggerEvent


# ========================================
# INTERNAL ACTION DETECTION
# ========================================

def detect_price_changes(
    df_weekly: pd.DataFrame,
    lookback_days: int = 30
) -> List[InternalAction]:
    """
    Auto-detect price changes from historical data.

    Args:
        df_weekly: Historical time-series data with 'price' column
        lookback_days: Number of days to look back for changes

    Returns:
        List of detected price change actions
    """
    if df_weekly.empty or 'price' not in df_weekly.columns:
        return []

    actions = []

    # Ensure date column exists - with validation
    if 'date' not in df_weekly.columns:
        if 'week' in df_weekly.columns:
            df_weekly = df_weekly.copy()
            try:
                df_weekly['date'] = pd.to_datetime(df_weekly['week'], errors='coerce')
            except Exception:
                # If date conversion fails, return empty list
                return []
        else:
            # No date or week column available
            return []

    # Validate that 'date' column now exists and has valid data
    if 'date' not in df_weekly.columns or df_weekly['date'].isna().all():
        return []

    # Sort by date
    df_weekly = df_weekly.sort_values('date')

    # Calculate price changes
    df_weekly['price_change'] = df_weekly['price'].diff()
    df_weekly['price_change_pct'] = df_weekly['price'].pct_change() * 100

    # Get recent data
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent_df = df_weekly[df_weekly['date'] >= cutoff_date].copy()

    # Detect significant price changes (>2% or >$1)
    for idx, row in recent_df.iterrows():
        price_change = row.get('price_change', 0)
        price_change_pct = row.get('price_change_pct', 0)

        if abs(price_change_pct) > 2 or abs(price_change) > 1:
            # Significant price change detected
            asin = row.get('asin', 'PORTFOLIO')

            action = InternalAction(
                action_type=ActionType.PRICE_CHANGE,
                timestamp=row['date'],
                magnitude=price_change_pct,
                magnitude_type="percentage",
                affected_asins=[asin] if asin != 'PORTFOLIO' else [],
                description=f"Price changed from ${row['price'] - price_change:.2f} to ${row['price']:.2f} ({price_change_pct:+.1f}%)",
                metadata={
                    "old_price": float(row['price'] - price_change),
                    "new_price": float(row['price']),
                    "change_amount": float(price_change),
                    "change_pct": float(price_change_pct)
                }
            )
            actions.append(action)

    return actions


def detect_ppc_changes(
    df_weekly: pd.DataFrame,
    lookback_days: int = 30,
    default_roas: float = 4.0
) -> List[InternalAction]:
    """
    Auto-detect PPC budget changes from historical data.

    Args:
        df_weekly: Historical time-series data with PPC spend columns
        lookback_days: Number of days to look back for changes
        default_roas: Default Return on Ad Spend multiplier (revenue per $1 spent)

    Returns:
        List of detected PPC budget change actions
    """
    if df_weekly.empty:
        return []

    actions = []

    # Check for PPC spend columns (various naming conventions)
    ppc_columns = ['ppc_spend', 'ad_spend', 'advertising_spend', 'sponsored_spend']
    spend_col = None
    for col in ppc_columns:
        if col in df_weekly.columns:
            spend_col = col
            break

    if not spend_col:
        return []  # No PPC data available

    # Ensure date column exists - with validation
    if 'date' not in df_weekly.columns:
        if 'week' in df_weekly.columns:
            df_weekly = df_weekly.copy()
            try:
                df_weekly['date'] = pd.to_datetime(df_weekly['week'], errors='coerce')
            except Exception:
                return []
        else:
            return []

    # Validate that 'date' column now exists and has valid data
    if 'date' not in df_weekly.columns or df_weekly['date'].isna().all():
        return []

    # Sort by date
    df_weekly = df_weekly.sort_values('date')

    # Calculate spend changes
    df_weekly['spend_change'] = df_weekly[spend_col].diff()
    df_weekly['spend_change_pct'] = df_weekly[spend_col].pct_change() * 100

    # Get recent data
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent_df = df_weekly[df_weekly['date'] >= cutoff_date].copy()

    if recent_df.empty:
        return []

    # Detect significant PPC budget changes (>15% or >$100 absolute)
    for idx, row in recent_df.iterrows():
        spend_change = row.get('spend_change', 0)
        spend_change_pct = row.get('spend_change_pct', 0)

        # Skip if not significant
        if pd.isna(spend_change) or pd.isna(spend_change_pct):
            continue
        if abs(spend_change_pct) < 15 and abs(spend_change) < 100:
            continue

        # Calculate expected revenue impact using ROAS
        expected_impact = spend_change * default_roas

        # Create action
        action = InternalAction(
            action_type=ActionType.PPC_BUDGET,
            timestamp=row['date'],
            magnitude=spend_change_pct,
            magnitude_type="percentage",
            expected_impact=expected_impact,
            description=f"PPC budget {'increase' if spend_change > 0 else 'decrease'} of ${abs(spend_change):,.0f} ({spend_change_pct:+.1f}%)",
            metadata={
                "spend_change": float(spend_change),
                "spend_change_pct": float(spend_change_pct),
                "roas_applied": default_roas,
                "previous_spend": float(df_weekly.loc[df_weekly['date'] < row['date'], spend_col].iloc[-1]) if len(df_weekly[df_weekly['date'] < row['date']]) > 0 else 0,
                "new_spend": float(row[spend_col])
            }
        )
        actions.append(action)

    return actions


# ========================================
# ATTRIBUTION CALCULATION
# ========================================

def attribute_internal_actions(
    df_weekly: pd.DataFrame,
    internal_actions: List[InternalAction],
    baseline_revenue: float,
    current_revenue: float
) -> Tuple[float, List[AttributionDriver]]:
    """
    Attribute revenue change to internal actions (price changes, PPC, etc.).

    Method: Before/after analysis with elasticity modeling.

    Args:
        df_weekly: Historical time-series data
        internal_actions: List of detected internal actions
        baseline_revenue: Revenue before changes
        current_revenue: Current revenue

    Returns:
        (total_internal_impact, list_of_drivers)
    """
    drivers = []
    total_impact = 0.0

    for action in internal_actions:
        impact = 0.0
        confidence = 0.7  # Default confidence

        if action.action_type == ActionType.PRICE_CHANGE:
            # Use price elasticity model
            # Typical Amazon elasticity: -1.5 (1% price increase = 1.5% volume decrease)
            price_elasticity = -1.5

            price_change_pct = action.magnitude
            volume_change_pct = price_change_pct * price_elasticity

            # Revenue impact = baseline * (price_change% + volume_change%)
            # This is a simplified model; real impact depends on category
            net_change_pct = price_change_pct + volume_change_pct
            impact = baseline_revenue * (net_change_pct / 100)

            # Adjust confidence based on data availability
            if df_weekly is not None and len(df_weekly) > 4:
                confidence = 0.85
            else:
                confidence = 0.65

        elif action.action_type == ActionType.PPC_BUDGET:
            # Use ROAS model
            # Typical ROAS: 4.0 ($4 revenue per $1 spend)
            roas = action.metadata.get('roas_applied', 4.0)
            spend_delta = action.metadata.get('spend_change', 0)  # Absolute spend change in dollars

            # Revenue impact = spend change * ROAS
            impact = spend_delta * roas
            confidence = 0.8  # High confidence for PPC attribution (direct causation)

        # Create driver
        driver = AttributionDriver(
            category=CausalCategory.INTERNAL,
            description=action.description,
            impact=impact,
            confidence=confidence,
            controllable=True,
            event_type=action.action_type.value,
            timestamp=action.timestamp,
            metadata=action.metadata
        )

        drivers.append(driver)
        total_impact += impact

    return total_impact, drivers


def attribute_competitive_events(
    trigger_events: List[TriggerEvent],
    df_weekly: pd.DataFrame,
    baseline_revenue: float,
    market_snapshot: Optional[Dict] = None
) -> Tuple[float, List[AttributionDriver]]:
    """
    Attribute revenue change to competitive events (OOS, price wars, etc.).

    Method: Market share capture rate calculation.

    Args:
        trigger_events: List of detected trigger events
        df_weekly: Historical time-series data
        baseline_revenue: Revenue before changes
        market_snapshot: Market intelligence data

    Returns:
        (total_competitive_impact, list_of_drivers)
    """
    drivers = []
    total_impact = 0.0

    # Filter for competitive events
    competitive_event_types = [
        'competitor_inventory_collapse',
        'competitor_oos',
        'competitor_price_increase',
        'competitor_price_war',
        'new_competitor_entry'
    ]

    for event in trigger_events:
        if event.event_type not in competitive_event_types:
            continue

        impact = 0.0
        confidence = 0.6

        # Estimate impact based on event type
        if event.event_type == 'competitor_inventory_collapse' or event.event_type == 'competitor_oos':
            # Estimate based on severity and market position
            # High severity = more impact
            severity_multiplier = event.severity / 10.0  # Normalize to 0-1

            # Assume we capture 10-30% of lost competitor revenue
            capture_rate = 0.15 + (0.15 * severity_multiplier)

            # Estimate competitor revenue from market snapshot
            competitor_revenue = baseline_revenue * 0.5  # Assume comparable competitor

            # Estimate OOS duration (days)
            oos_duration_pct = 0.2  # Assume 20% of period (6 days out of 30)

            impact = competitor_revenue * oos_duration_pct * capture_rate
            confidence = 0.7

        elif event.event_type == 'competitor_price_increase':
            # Pricing power gain
            # Assume 5-15% revenue lift from improved price competitiveness
            lift_pct = 0.05 + (0.05 * (event.severity / 10.0))
            impact = baseline_revenue * lift_pct
            confidence = 0.65

        elif event.event_type == 'competitor_price_war':
            # Negative impact from price competition
            erosion_pct = -0.10 - (0.10 * (event.severity / 10.0))
            impact = baseline_revenue * erosion_pct
            confidence = 0.75

        elif event.event_type == 'new_competitor_entry':
            # Share loss to new entrant
            share_loss_pct = -0.05 - (0.05 * (event.severity / 10.0))
            impact = baseline_revenue * share_loss_pct
            confidence = 0.60

        # Create driver
        driver = AttributionDriver(
            category=CausalCategory.COMPETITIVE,
            description=f"{event.event_type.replace('_', ' ').title()}: {event.metric_name}",
            impact=impact,
            confidence=confidence,
            controllable=False,
            event_type=event.event_type,
            related_asin=event.related_asin,
            timestamp=None,
            metadata={
                "severity": event.severity,
                "metric_name": event.metric_name,
                "baseline_value": event.baseline_value,
                "current_value": event.current_value,
                "delta_pct": event.delta_pct
            }
        )

        drivers.append(driver)
        total_impact += impact

    return total_impact, drivers


def attribute_macro_trends(
    df_weekly: pd.DataFrame,
    baseline_revenue: float,
    market_snapshot: Optional[Dict] = None,
    category_id: Optional[int] = None
) -> Tuple[float, List[AttributionDriver]]:
    """
    Attribute revenue change to macro trends (category growth, seasonality).

    Method: Category benchmark growth share from Supabase intelligence.

    Args:
        df_weekly: Historical time-series data
        baseline_revenue: Revenue before changes
        market_snapshot: Market intelligence data with category benchmarks
        category_id: Amazon category ID for loading real benchmarks

    Returns:
        (total_macro_impact, list_of_drivers)
    """
    drivers = []
    total_impact = 0.0

    # Try to load real category benchmarks from Supabase
    category_data = None
    if category_id:
        try:
            from src.supabase_reader import load_category_benchmarks
            category_data = load_category_benchmarks(category_id, lookback_days=30)
        except Exception:
            pass

    # Fallback to market_snapshot if Supabase query failed
    if not category_data and market_snapshot and 'category_benchmarks' in market_snapshot:
        category_data = market_snapshot['category_benchmarks']

    if category_data:
        category_growth_pct = category_data.get('growth_rate_30d', 0)

        if abs(category_growth_pct) > 5:  # Significant category trend
            # Calculate your market share
            # If we have category revenue estimate, calculate actual share
            category_revenue = category_data.get('category_revenue_estimate', 0)
            if category_revenue > 0 and baseline_revenue > 0:
                your_market_share = baseline_revenue / category_revenue
                your_market_share = min(your_market_share, 0.15)  # Cap at 15% (realistic max)
            else:
                your_market_share = 0.04  # Conservative default estimate

            # Your revenue lift from category growth
            # You capture your market share of the category growth
            impact = baseline_revenue * (category_growth_pct / 100)

            # Description with direction
            growth_direction = category_data.get('growth_direction', 'unknown')
            direction_emoji = "📈" if category_growth_pct > 0 else "📉"

            driver = AttributionDriver(
                category=CausalCategory.MACRO,
                description=f"{direction_emoji} Category {growth_direction}: {'+' if category_growth_pct > 0 else ''}{category_growth_pct:.1f}%",
                impact=impact,
                confidence=0.78 if category_revenue > 0 else 0.65,  # Higher confidence with real data
                controllable=False,
                event_type="category_trend",
                metadata={
                    "category_growth_pct": category_growth_pct,
                    "market_share": your_market_share,
                    "category_revenue": category_revenue,
                    "growth_direction": growth_direction
                }
            )

            drivers.append(driver)
            total_impact += impact

    # Check for seasonality in df_weekly
    if df_weekly is not None and len(df_weekly) > 8:
        # Simple seasonality detection: compare recent 4 weeks vs previous 4 weeks
        recent = df_weekly.tail(4)
        previous = df_weekly.tail(8).head(4)

        if 'revenue' in df_weekly.columns or 'sales' in df_weekly.columns:
            revenue_col = 'revenue' if 'revenue' in df_weekly.columns else 'sales'

            recent_avg = recent[revenue_col].mean()
            previous_avg = previous[revenue_col].mean()

            if previous_avg > 0:
                seasonal_change_pct = ((recent_avg - previous_avg) / previous_avg) * 100

                # Only attribute if significant (>10%) and not explained by price/competitive
                if abs(seasonal_change_pct) > 10:
                    # Assume 50% of unexplained growth is seasonal
                    impact = baseline_revenue * (seasonal_change_pct / 100) * 0.5

                    driver = AttributionDriver(
                        category=CausalCategory.MACRO,
                        description=f"Seasonal trend ({'+' if seasonal_change_pct > 0 else ''}{seasonal_change_pct:.1f}%)",
                        impact=impact,
                        confidence=0.60,
                        controllable=False,
                        event_type="seasonality",
                        metadata={
                            "seasonal_change_pct": seasonal_change_pct,
                            "recent_avg": float(recent_avg),
                            "previous_avg": float(previous_avg)
                        }
                    )

                    drivers.append(driver)
                    total_impact += impact

    return total_impact, drivers


def attribute_platform_changes(
    trigger_events: List[TriggerEvent],
    df_weekly: pd.DataFrame,
    baseline_revenue: float
) -> Tuple[float, List[AttributionDriver]]:
    """
    Attribute revenue change to platform/algorithmic changes.

    Method: CTR differential analysis for badge changes, unexplained rank shifts.

    Args:
        trigger_events: List of detected trigger events
        df_weekly: Historical time-series data
        baseline_revenue: Revenue before changes

    Returns:
        (total_platform_impact, list_of_drivers)
    """
    drivers = []
    total_impact = 0.0

    # Filter for platform-related events
    platform_event_types = [
        'platform_choice_badge_loss',
        'platform_algorithm_shift',
        'buybox_loss',
        'search_rank_drop'
    ]

    for event in trigger_events:
        if event.event_type not in platform_event_types:
            continue

        impact = 0.0
        confidence = 0.65

        if event.event_type == 'platform_choice_badge_loss':
            # CTR differential: Choice badge typically adds +2-3% CTR
            ctr_impact_pct = -2.5
            impact = baseline_revenue * (ctr_impact_pct / 100)
            confidence = 0.70

        elif event.event_type == 'platform_algorithm_shift':
            # Unexplained rank drop
            # Severity-based impact
            rank_impact_pct = -0.05 - (0.05 * (event.severity / 10.0))
            impact = baseline_revenue * rank_impact_pct
            confidence = 0.60

        elif event.event_type == 'buybox_loss':
            # Buy Box loss is critical
            buybox_impact_pct = -0.30 - (0.20 * (event.severity / 10.0))
            impact = baseline_revenue * buybox_impact_pct
            confidence = 0.85

        # Create driver
        driver = AttributionDriver(
            category=CausalCategory.PLATFORM,
            description=f"{event.event_type.replace('_', ' ').title()}",
            impact=impact,
            confidence=confidence,
            controllable=False,  # Platform changes are generally uncontrollable
            event_type=event.event_type,
            metadata={
                "severity": event.severity,
                "metric_name": event.metric_name
            }
        )

        drivers.append(driver)
        total_impact += impact

    return total_impact, drivers


# ========================================
# MAIN ATTRIBUTION FUNCTION
# ========================================

def calculate_revenue_attribution(
    previous_revenue: float,
    current_revenue: float,
    df_weekly: pd.DataFrame,
    trigger_events: List[TriggerEvent],
    market_snapshot: Optional[Dict] = None,
    lookback_days: int = 30,
    portfolio_asins: List[str] = None,
    category_id: Optional[int] = None
) -> RevenueAttribution:
    """
    Decomposes revenue change into 4 causal categories using elimination method.

    Method:
    1. Calculate total delta: e.g., $50k growth
    2. Attribute internal actions: Your price/PPC changes → expected +$22k
    3. Attribute competitive events: Competitor OOS → estimated +$18k
    4. Attribute macro trends: Category +15% → your share = +$12k
    5. Attribute platform changes: Badge loss → estimated -$2k
    6. Residual = Unexplained variance

    Args:
        previous_revenue: Revenue at start of period
        current_revenue: Revenue at end of period
        df_weekly: Historical time-series data
        trigger_events: Detected trigger events from trigger_detection
        market_snapshot: Market intelligence data
        lookback_days: Number of days to analyze
        portfolio_asins: List of ASINs in portfolio
        category_id: Amazon category ID for loading real benchmarks

    Returns:
        RevenueAttribution object with complete breakdown
    """
    # Calculate total change
    total_delta = current_revenue - previous_revenue

    # Initialize attribution
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Step 1: Detect internal actions (price changes + PPC budget changes)
    price_actions = detect_price_changes(df_weekly, lookback_days)
    ppc_actions = detect_ppc_changes(df_weekly, lookback_days)
    internal_actions = price_actions + ppc_actions

    # Step 2: Attribute to internal actions
    internal_impact, internal_drivers = attribute_internal_actions(
        df_weekly, internal_actions, previous_revenue, current_revenue
    )

    # Step 3: Attribute to competitive events
    competitive_impact, competitive_drivers = attribute_competitive_events(
        trigger_events, df_weekly, previous_revenue, market_snapshot
    )

    # Step 4: Attribute to macro trends (with category benchmarks from Supabase)
    macro_impact, macro_drivers = attribute_macro_trends(
        df_weekly, previous_revenue, market_snapshot, category_id
    )

    # Step 5: Attribute to platform changes
    platform_impact, platform_drivers = attribute_platform_changes(
        trigger_events, df_weekly, previous_revenue
    )

    # Step 6: Calculate residual (unexplained variance)
    attributed_total = (
        internal_impact +
        competitive_impact +
        macro_impact +
        platform_impact
    )

    residual = total_delta - attributed_total

    # Calculate explained variance
    if total_delta != 0:
        explained_variance = min(1.0, abs(attributed_total) / abs(total_delta))
    else:
        explained_variance = 1.0 if attributed_total == 0 else 0.0

    # Calculate overall confidence (weighted average of driver confidences)
    all_drivers = (
        internal_drivers +
        competitive_drivers +
        macro_drivers +
        platform_drivers
    )

    if all_drivers:
        # Weight by impact magnitude
        total_weighted_confidence = sum(abs(d.impact) * d.confidence for d in all_drivers)
        total_weights = sum(abs(d.impact) for d in all_drivers)
        overall_confidence = total_weighted_confidence / total_weights if total_weights > 0 else 0.7
    else:
        overall_confidence = 0.5  # Low confidence if no drivers detected

    # Adjust confidence by explained variance
    overall_confidence = overall_confidence * explained_variance

    # Additional penalty for high residual (>20% of total delta)
    if total_delta != 0:
        residual_pct = abs(residual) / abs(total_delta)
        if residual_pct > 0.20:
            # Aggressive downweight: reduce confidence by additional 20% for each 10% of unexplained variance above threshold
            penalty_factor = max(0.5, 1.0 - ((residual_pct - 0.20) * 2.0))
            overall_confidence = overall_confidence * penalty_factor

    # Create attribution object
    attribution = RevenueAttribution(
        start_date=start_date,
        end_date=end_date,
        total_delta=total_delta,
        previous_revenue=previous_revenue,
        current_revenue=current_revenue,
        internal_contribution=internal_impact,
        competitive_contribution=competitive_impact,
        macro_contribution=macro_impact,
        platform_contribution=platform_impact,
        internal_drivers=internal_drivers,
        competitive_drivers=competitive_drivers,
        macro_drivers=macro_drivers,
        platform_drivers=platform_drivers,
        explained_variance=explained_variance,
        residual=residual,
        confidence=overall_confidence,
        portfolio_asins=portfolio_asins or [],
        attribution_method="elimination"
    )

    return attribution

# ========================================
# PERSISTENCE (Added 2026-01-23)
# ========================================

def save_revenue_attribution(
    attribution: RevenueAttribution,
    project_id: str,
    start_date: str,
    end_date: str,
    supabase: Client
) -> bool:
    """
    Persist revenue attribution to database.
    
    Stores:
    1. Top-level attribution record (revenue_attributions)
    2. Granular drivers (attribution_drivers)
    
    Args:
        attribution: Calculated attribution object
        project_id: Supabase project ID (UUID)
        start_date: Analysis start date (ISO)
        end_date: Analysis end date (ISO)
        supabase: Authenticated Supabase client
        
    Returns:
        True if successful, False otherwise
    """
    if not project_id:
        print("⚠️ Cannot save attribution: No project_id provided")
        return False
        
    try:
        # 1. Prepare main attribution record
        main_record = {
            "project_id": project_id,
            "start_date": start_date,
            "end_date": end_date,
            
            # Deltas
            "total_delta": attribution.total_delta,
            "previous_revenue": attribution.previous_revenue,
            "current_revenue": attribution.current_revenue,
            "delta_pct": attribution.delta_pct,
            
            # Categories
            "internal_contribution": attribution.internal_contribution,
            "competitive_contribution": attribution.competitive_contribution,
            "macro_contribution": attribution.macro_contribution,
            "platform_contribution": attribution.platform_contribution,
            
            # Metadata
            "explained_variance": attribution.explained_variance,
            "confidence_score": attribution.confidence_score,
            "residual": attribution.residual,
            "attribution_method": "elimination"
        }
        
        # Upsert main record (conflict on project_id + dates)
        result = supabase.table("revenue_attributions").upsert(
            main_record, 
            on_conflict="project_id,start_date,end_date"
        ).execute()
        
        if not result.data:
            print("⚠️ Failed to insert revenue_attributions record")
            return False
            
        attribution_id = result.data[0]['id']
        
        # 2. Prepare drivers
        drivers_to_insert = []
        for driver in attribution.drivers:
            driver_record = {
                "attribution_id": attribution_id,
                "category": driver.category.value,
                "description": driver.description,
                "impact": driver.impact,
                "confidence": driver.confidence,
                "controllable": driver.controllable,
                "event_type": driver.event_type,
                "related_asin": driver.related_asin,
                "event_timestamp": driver.timestamp.isoformat() if driver.timestamp else None,
                "metadata": driver.metadata or {}
            }
            drivers_to_insert.append(driver_record)
            
        # 3. Replace drivers (Delete old ones for this attribution, then insert new)
        # This acts as a "sync" for the drivers list
        if drivers_to_insert:
            # Delete existing drivers for this attribution ID (to avoid duplicates if re-running)
            supabase.table("attribution_drivers").delete().eq("attribution_id", attribution_id).execute()
            
            # Insert new drivers
            supabase.table("attribution_drivers").insert(drivers_to_insert).execute()
            
        return True
        
    except Exception as e:
        print(f"⚠️ Error saving attribution: {str(e)}")
        return False
