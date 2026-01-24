"""
Predictive Forecasting Engine for ShelfGuard Causal Intelligence Platform

Phase 2.5: Forward-looking intelligence that projects future revenue,
anticipated events, and generates Base/Optimistic/Pessimistic scenarios.
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.models.forecast_models import (
    AnticipatedEvent,
    RevenueForecast,
    Scenario,
    CombinedIntelligence,
    EventSeverity,
    DEFAULT_SEASONALITY_CURVE
)
from src.models.trigger_event import TriggerEvent


def calculate_annual_projection(
    current_monthly_revenue: float,
    historical_df: Optional[pd.DataFrame] = None,
    seasonality_curve: Optional[Dict[int, float]] = None,
    confidence_interval: float = 0.8
) -> Dict[str, Any]:
    """
    Project annual sales with seasonality and trend.
    
    Args:
        current_monthly_revenue: Current monthly revenue
        historical_df: Optional historical data for trend calculation
        seasonality_curve: Monthly multipliers (1-12)
        confidence_interval: Confidence level (default 80%)
    
    Returns:
        {
            "projected_annual": 1500000,
            "lower_bound": 1350000,
            "upper_bound": 1650000,
            "confidence": 0.8,
            "monthly_projections": [...]
        }
    """
    if seasonality_curve is None:
        seasonality_curve = DEFAULT_SEASONALITY_CURVE
    
    # Calculate base annual projection
    base_annual = current_monthly_revenue * 12
    
    # Calculate trend adjustment if historical data available
    trend_factor = 1.0
    if historical_df is not None and not historical_df.empty:
        # Try to extract monthly trend
        revenue_col = None
        for col in ['revenue_proxy', 'revenue_proxy_adjusted', 'monthly_revenue']:
            if col in historical_df.columns:
                revenue_col = col
                break
        
        if revenue_col and len(historical_df) >= 4:
            # Calculate month-over-month growth rate
            values = historical_df[revenue_col].dropna().values[-12:]  # Last 12 data points
            if len(values) >= 4:
                # Simple linear trend
                x = np.arange(len(values))
                if values.std() > 0:
                    slope = np.polyfit(x, values, 1)[0]
                    avg_value = np.mean(values)
                    if avg_value > 0:
                        monthly_growth_rate = slope / avg_value
                        # Project annual trend (compound 12 months)
                        trend_factor = (1 + monthly_growth_rate) ** 6  # 6 months average
    
    # Apply seasonality adjustment for remaining months
    current_month = datetime.now().month
    monthly_projections = []
    total_projected = 0
    
    for month_offset in range(12):
        target_month = ((current_month - 1 + month_offset) % 12) + 1
        seasonal_factor = seasonality_curve.get(target_month, 1.0)
        monthly_revenue = current_monthly_revenue * seasonal_factor * trend_factor
        monthly_projections.append({
            "month": target_month,
            "projected_revenue": monthly_revenue
        })
        total_projected += monthly_revenue
    
    # Calculate confidence bounds
    # Use coefficient of variation from historical if available, else use 10%
    volatility = 0.10
    if historical_df is not None and revenue_col and len(historical_df) >= 4:
        values = historical_df[revenue_col].dropna().values
        if len(values) > 0 and np.mean(values) > 0:
            volatility = min(np.std(values) / np.mean(values), 0.30)  # Cap at 30%
    
    # Z-score for confidence interval
    z_score = 1.28 if confidence_interval == 0.8 else 1.96  # 80% or 95%
    
    lower_bound = total_projected * (1 - volatility * z_score)
    upper_bound = total_projected * (1 + volatility * z_score)
    
    return {
        "projected_annual": total_projected,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "confidence": confidence_interval,
        "monthly_projections": monthly_projections,
        "trend_factor": trend_factor,
        "volatility": volatility
    }


def forecast_event_impacts(
    trigger_events: List[TriggerEvent],
    current_revenue: float,
    df_historical: Optional[pd.DataFrame] = None,
    attribution: Optional[Any] = None,
    forecast_horizon_days: int = 30
) -> List[AnticipatedEvent]:
    """
    Project future events and their revenue impact based on detected triggers.
    
    Examples:
    - Competitor OOS detected → Anticipate restock in 5-7 days → Impact: -$X when they return
    - Your inventory low → Anticipate stockout in N days → Impact: -$Y critical
    - Seasonality peak ending → Impact: -$Z when Q4 ends
    """
    anticipated_events = []
    today = date.today()
    
    for event in trigger_events:
        event_type = event.event_type
        
        # === COMPETITOR OOS → ANTICIPATE RESTOCK ===
        if event_type in ["competitor_oos_imminent", "amazon_supply_unstable"]:
            # Estimate days until restock (typically 5-10 days for FBA)
            days_until = 5  # Conservative estimate
            impact_pct = -0.10  # Lose ~10% of temporary gain when they return
            
            anticipated_events.append(AnticipatedEvent(
                event_type="competitor_restock",
                event_date=today + timedelta(days=days_until),
                days_until=days_until,
                impact_per_month=current_revenue * impact_pct,
                probability=0.75,
                severity=EventSeverity.MEDIUM,
                action_recommended="Capture reviews and Subscribe & Save customers while competitor is OOS",
                description=f"Competitor likely to restock soon - temporary gains will reverse",
                related_asin=event.related_asin
            ))
        
        # === YOUR INVENTORY CRISIS → STOCKOUT IMMINENT ===
        if event_type == "backorder_crisis":
            # Already in crisis - this is critical
            anticipated_events.append(AnticipatedEvent(
                event_type="stockout",
                event_date=today,
                days_until=0,
                impact_per_month=-current_revenue * 0.80,  # Lose 80%+ when OOS
                probability=0.95,
                severity=EventSeverity.CRITICAL,
                action_recommended="URGENT: Expedite inventory or negotiate with supplier. FBA backlog may take 5-7 days.",
                description=f"⚠️ BACKORDER CRISIS - Revenue at risk immediately",
                related_asin=event.affected_asin
            ))
        
        # === PRICE WAR → MARGIN EROSION CONTINUES ===
        if event_type == "price_war_active":
            # Price wars typically last 2-4 weeks
            days_until = 14
            impact_pct = -0.15  # 15% margin erosion
            
            anticipated_events.append(AnticipatedEvent(
                event_type="margin_erosion",
                event_date=today + timedelta(days=days_until),
                days_until=days_until,
                impact_per_month=current_revenue * impact_pct,
                probability=0.70,
                severity=EventSeverity.HIGH,
                action_recommended="Consider exiting price war - focus on differentiation (reviews, A+, Subscribe & Save)",
                description=f"Price war likely to continue - margin erosion expected",
                related_asin=None
            ))
        
        # === RANK DEGRADATION → CONTINUED DECLINE ===
        if event_type in ["rank_degradation", "share_of_voice_lost", "platform_algorithm_shift"]:
            days_until = 30
            impact_pct = event.delta_pct / 100 * -0.5  # 50% of the decline continues
            
            if abs(impact_pct) > 0.05:  # Only if significant
                anticipated_events.append(AnticipatedEvent(
                    event_type="rank_further_decline",
                    event_date=today + timedelta(days=days_until),
                    days_until=days_until,
                    impact_per_month=current_revenue * impact_pct,
                    probability=0.60,
                    severity=EventSeverity.MEDIUM if abs(impact_pct) < 0.15 else EventSeverity.HIGH,
                    action_recommended="Investigate root cause: Review recent PPC changes, listing edits, or competitive moves",
                    description=f"Rank decline may continue without intervention",
                    related_asin=None
                ))
        
        # === MOMENTUM ACCELERATION → GROWTH OPPORTUNITY ===
        if event_type in ["momentum_acceleration", "momentum_sustained", "share_of_voice_gained"]:
            days_until = 30
            impact_pct = abs(event.delta_pct) / 100 * 0.3  # 30% continuation
            
            anticipated_events.append(AnticipatedEvent(
                event_type="growth_continuation",
                event_date=today + timedelta(days=days_until),
                days_until=days_until,
                impact_per_month=current_revenue * impact_pct,
                probability=0.55,
                severity=EventSeverity.LOW,
                action_recommended="Double down on what's working - increase PPC on winning keywords",
                description=f"Momentum likely to continue with sustained effort",
                related_asin=None
            ))
    
    # Add seasonality event if we're near a transition
    current_month = datetime.now().month
    if current_month in [11, 12]:  # Q4 peak
        days_until_jan = (date(datetime.now().year + 1, 1, 15) - today).days
        anticipated_events.append(AnticipatedEvent(
            event_type="seasonality_end",
            event_date=date(datetime.now().year + 1, 1, 15),
            days_until=days_until_jan,
            impact_per_month=-current_revenue * 0.20,  # 20% Q4 boost reverses
            probability=0.90,
            severity=EventSeverity.MEDIUM,
            action_recommended="Don't build fixed costs on Q4 revenue - plan for post-holiday decline",
            description=f"Q4 seasonality boost will reverse in January",
            related_asin=None
        ))
    
    # Sort by days_until (most urgent first)
    anticipated_events.sort(key=lambda e: e.days_until)
    
    return anticipated_events


def build_scenarios(
    current_revenue: float,
    attribution: Optional[Any] = None,
    anticipated_events: Optional[List[AnticipatedEvent]] = None,
    market_trends: Optional[Dict] = None,
    df_historical: Optional[pd.DataFrame] = None
) -> List[Scenario]:
    """
    Generate Base/Optimistic/Pessimistic scenarios.

    Base Case: Most likely outcome (60-75% confidence)
    Optimistic: Execute all growth opportunities (30-45% confidence)
    Pessimistic: Competitive aggression / market decline (25-35% confidence)
    """
    scenarios = []

    # Calculate net event impacts
    event_impact_30d = sum(e.impact_per_month for e in (anticipated_events or []) if e.days_until <= 30)
    event_impact_60d = sum(e.impact_per_month for e in (anticipated_events or []) if e.days_until <= 60)
    event_impact_90d = sum(e.impact_per_month for e in (anticipated_events or []))

    # BUG FIX #6: Add trend-based adjustment when no events detected
    trend_adjustment = 0.0
    if len(anticipated_events or []) == 0 and df_historical is not None and len(df_historical) >= 8:
        # Calculate historical growth rate from last 8 periods
        try:
            rev_col = 'revenue_proxy' if 'revenue_proxy' in df_historical.columns else \
                      'sales' if 'sales' in df_historical.columns else \
                      'revenue' if 'revenue' in df_historical.columns else None
            if rev_col:
                recent_4wk = df_historical.iloc[-4:][rev_col].sum()
                prior_4wk = df_historical.iloc[-8:-4][rev_col].sum()
                if prior_4wk > 0:
                    growth_rate = (recent_4wk - prior_4wk) / prior_4wk
                    # Dampen growth rate for projection (assume mean reversion)
                    trend_adjustment = current_revenue * growth_rate * 0.5
        except Exception:
            trend_adjustment = 0.0

    # === BASE CASE ===
    base_30d = current_revenue + event_impact_30d * 0.7 + trend_adjustment * 0.3
    base_60d = current_revenue + event_impact_60d * 0.6 + trend_adjustment * 0.5
    base_90d = current_revenue + event_impact_90d * 0.5 + trend_adjustment * 0.7
    
    base_assumptions = []
    if event_impact_30d < 0:
        base_assumptions.append({"assumption": "Temporary factors partially reverse", "impact": event_impact_30d * 0.7})
    
    scenarios.append(Scenario(
        scenario_name="Base Case",
        probability=0.65,
        projected_revenue_30d=max(0, base_30d),
        projected_revenue_60d=max(0, base_60d),
        projected_revenue_90d=max(0, base_90d),
        assumptions=base_assumptions,
        key_risks=["Competitor restocks", "Temporary traffic declines"],
        key_opportunities=["Convert temporary gains to sticky customers"],
        narrative=f"Most likely outcome based on current trends and anticipated events."
    ))
    
    # === OPTIMISTIC CASE ===
    growth_opportunities = [e for e in (anticipated_events or []) if e.impact_per_month > 0]
    growth_potential = sum(e.impact_per_month for e in growth_opportunities) + current_revenue * 0.15
    
    opt_30d = current_revenue + growth_potential * 0.4
    opt_60d = current_revenue + growth_potential * 0.6
    opt_90d = current_revenue + growth_potential * 0.8
    
    opt_assumptions = [
        {"assumption": "Execute all growth opportunities", "impact": growth_potential * 0.3},
        {"assumption": "No major competitive threats", "impact": current_revenue * 0.05}
    ]
    
    scenarios.append(Scenario(
        scenario_name="Optimistic",
        probability=0.20,
        projected_revenue_30d=opt_30d,
        projected_revenue_60d=opt_60d,
        projected_revenue_90d=opt_90d,
        assumptions=opt_assumptions,
        key_risks=["Execution risk", "Over-investment in growth"],
        key_opportunities=["Market share capture", "Price increase potential", "Category expansion"],
        narrative=f"Best-case if growth opportunities are executed successfully."
    ))
    
    # === PESSIMISTIC CASE ===
    threats = [e for e in (anticipated_events or []) if e.impact_per_month < 0]
    threat_impact = sum(e.impact_per_month for e in threats)
    competitive_erosion = current_revenue * -0.15  # Additional competitive pressure
    
    pess_30d = current_revenue + threat_impact * 1.2 + competitive_erosion * 0.3
    pess_60d = current_revenue + threat_impact * 1.5 + competitive_erosion * 0.6
    pess_90d = current_revenue + threat_impact * 1.8 + competitive_erosion * 0.9
    
    pess_assumptions = [
        {"assumption": "All threats materialize", "impact": threat_impact},
        {"assumption": "Additional competitive pressure", "impact": competitive_erosion}
    ]
    
    scenarios.append(Scenario(
        scenario_name="Pessimistic",
        probability=0.15,
        projected_revenue_30d=max(0, pess_30d),
        projected_revenue_60d=max(0, pess_60d),
        projected_revenue_90d=max(0, pess_90d),
        assumptions=pess_assumptions,
        key_risks=["Aggressive competitor pricing", "Inventory issues", "Platform changes"],
        key_opportunities=["Defend market share", "Focus on unit economics"],
        narrative=f"Worst-case if competitive threats and market headwinds intensify."
    ))
    
    return scenarios


def calculate_sustainable_run_rate(
    current_revenue: float,
    attribution: Optional[Any] = None,
    anticipated_events: Optional[List[AnticipatedEvent]] = None
) -> Tuple[float, float, int]:
    """
    Calculate sustainable run rate after temporary factors reverse.
    
    Returns:
        (sustainable_run_rate, temporary_inflation, temporary_duration_days)
    """
    # BUG FIX #5: Clearer sustainability calculation
    # Identify temporary gains (from attribution if available)
    temporary_inflation = 0.0
    temporary_duration_days = 30

    if attribution:
        # Competitive vacuum gains are typically temporary
        competitive = getattr(attribution, 'competitive_contribution', 0) or 0
        macro = getattr(attribution, 'macro_contribution', 0) or 0

        # Only count POSITIVE contributions as temporary (gains that will reverse)
        # 70% of competitive gains are temporary (competitor will restock)
        # 50% of macro gains are temporary (seasonality, trends)
        temp_competitive = max(0, competitive) * 0.70  # Only positive gains
        temp_macro = max(0, macro) * 0.50              # Only positive gains
        temporary_inflation = temp_competitive + temp_macro

    # Calculate temporary duration from anticipated events
    if anticipated_events:
        # Find events that reverse temporary gains
        reversal_events = [e for e in anticipated_events
                          if e.event_type in ['competitor_restock', 'seasonality_end']
                          and e.days_until <= 60]
        if reversal_events:
            temporary_duration_days = max(e.days_until for e in reversal_events)
        else:
            temporary_duration_days = 30

    # Simple formula: sustainable = current - temporary_gains
    # Threats are shown separately as "risk", not mixed into sustainable rate
    sustainable_run_rate = current_revenue - temporary_inflation

    return (
        max(0, sustainable_run_rate),
        temporary_inflation,
        min(temporary_duration_days, 90)
    )


def generate_combined_intelligence(
    current_revenue: float,
    previous_revenue: float,
    attribution: Optional[Any],
    trigger_events: List[TriggerEvent],
    df_historical: Optional[pd.DataFrame] = None
) -> CombinedIntelligence:
    """
    Generate the complete CombinedIntelligence output.
    
    This is the master function that brings together:
    - Revenue attribution (what happened)
    - Anticipated events (what will happen)
    - Scenarios (what could happen)
    - Strategic synthesis (what to do)
    """
    # 1. Forecast event impacts
    anticipated_events = forecast_event_impacts(
        trigger_events=trigger_events,
        current_revenue=current_revenue,
        df_historical=df_historical,
        attribution=attribution
    )
    
    # 2. Build scenarios
    scenarios = build_scenarios(
        current_revenue=current_revenue,
        attribution=attribution,
        anticipated_events=anticipated_events,
        df_historical=df_historical
    )
    
    # 3. Calculate sustainable run rate
    sustainable, temporary, duration = calculate_sustainable_run_rate(
        current_revenue=current_revenue,
        attribution=attribution,
        anticipated_events=anticipated_events
    )
    
    # 4. Calculate revenue forecast
    annual_projection = calculate_annual_projection(
        current_monthly_revenue=current_revenue,
        historical_df=df_historical
    )
    
    base_scenario = next((s for s in scenarios if s.scenario_name == "Base Case"), None)
    
    forecast = RevenueForecast(
        current_revenue=current_revenue,
        projected_revenue=base_scenario.projected_revenue_30d if base_scenario else current_revenue,
        forecast_horizon_days=30,
        confidence_interval=0.80,
        lower_bound=scenarios[-1].projected_revenue_30d if scenarios else current_revenue * 0.85,
        upper_bound=scenarios[1].projected_revenue_30d if len(scenarios) > 1 else current_revenue * 1.15,
        event_adjustments=sum(e.impact_per_month for e in anticipated_events if e.days_until <= 30),
        projected_annual_sales=annual_projection["projected_annual"],
        annual_lower_bound=annual_projection["lower_bound"],
        annual_upper_bound=annual_projection["upper_bound"]
    )
    
    # 5. Generate critical actions
    critical_actions = []
    for event in anticipated_events:
        if event.severity in [EventSeverity.CRITICAL, EventSeverity.HIGH]:
            critical_actions.append({
                "action": event.action_recommended,
                "deadline": f"Day {event.days_until}",
                "impact": event.get_impact_str()
            })
    
    # 6. Generate strategic recommendation
    delta = current_revenue - previous_revenue if previous_revenue else 0
    delta_pct = (delta / previous_revenue * 100) if previous_revenue > 0 else 0
    
    rec_parts = []
    if delta > 0:
        rec_parts.append(f"Revenue grew ${delta:,.0f} ({delta_pct:+.1f}%).")
    else:
        rec_parts.append(f"Revenue changed ${delta:,.0f} ({delta_pct:+.1f}%).")
    
    if temporary > 0:
        rec_parts.append(f"${temporary:,.0f}/mo is from temporary factors that may reverse within {duration} days.")
    
    rec_parts.append(f"Sustainable run rate: ${sustainable:,.0f}/mo.")
    
    if critical_actions:
        rec_parts.append(f"⚠️ {len(critical_actions)} urgent action(s) required.")
    
    strategic_recommendation = " ".join(rec_parts)
    
    return CombinedIntelligence(
        attribution=attribution,
        forecast=forecast,
        anticipated_events=anticipated_events,
        scenarios=scenarios,
        sustainable_run_rate=sustainable,
        temporary_inflation=temporary,
        temporary_duration_days=duration,
        critical_actions=critical_actions,
        strategic_recommendation=strategic_recommendation
    )
