"""
Predictive Forecasting Data Models for ShelfGuard Causal Intelligence Platform

Phase 2.5: Forward-looking intelligence structures that project future revenue,
anticipated events, and scenario analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from enum import Enum


class EventSeverity(Enum):
    """Severity levels for anticipated events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AnticipatedEvent:
    """
    Future event with revenue impact projection.
    
    Examples:
    - Competitor X restock (threat) - Day 3 - Impact: -$18k/mo
    - Your stockout (critical) - Day 7 - Impact: -$50k/mo
    - Q4 seasonality ends - Day 14 - Impact: -$12k/mo
    """
    event_type: str  # "competitor_restock", "stockout", "seasonality_end", "price_increase"
    event_date: date  # When event will occur
    days_until: int  # Days from now
    impact_per_month: float  # Revenue impact (+ or -)
    probability: float  # 0.0-1.0 confidence
    severity: EventSeverity  # critical, high, medium, low
    action_recommended: str  # Recommended action
    description: str  # Human-readable description
    related_asin: Optional[str] = None  # Related competitor ASIN if applicable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/display."""
        return {
            "event_type": self.event_type,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "days_until": self.days_until,
            "impact_per_month": self.impact_per_month,
            "probability": self.probability,
            "severity": self.severity.value if isinstance(self.severity, EventSeverity) else self.severity,
            "action_recommended": self.action_recommended,
            "description": self.description,
            "related_asin": self.related_asin
        }
    
    def get_impact_str(self) -> str:
        """Format impact as string with sign."""
        if self.impact_per_month >= 0:
            return f"+${self.impact_per_month:,.0f}/mo"
        return f"-${abs(self.impact_per_month):,.0f}/mo"
    
    def get_urgency_badge(self) -> str:
        """Get urgency badge for UI."""
        if self.severity == EventSeverity.CRITICAL:
            return "🚨 CRITICAL"
        elif self.severity == EventSeverity.HIGH:
            return "🔴 HIGH"
        elif self.severity == EventSeverity.MEDIUM:
            return "🟡 MEDIUM"
        return "🟢 LOW"


@dataclass
class RevenueForecast:
    """
    Revenue projection with confidence intervals.
    
    Provides 30/60/90 day forecasts and annualized projections.
    """
    current_revenue: float  # Starting point (monthly revenue)
    projected_revenue: float  # Forecasted revenue (at horizon)
    forecast_horizon_days: int  # 30, 60, or 90 days
    
    # Confidence intervals
    confidence_interval: float = 0.8  # 0.8 = 80% CI
    lower_bound: float = 0.0  # Pessimistic estimate
    upper_bound: float = 0.0  # Optimistic estimate
    
    # Components
    base_trend: float = 0.0  # Trend-based projection
    seasonality_adjustment: float = 0.0  # Seasonal factor (multiplier)
    event_adjustments: float = 0.0  # Net impact of anticipated events
    
    # Annual projection
    projected_annual_sales: float = 0.0  # Annualized estimate
    annual_lower_bound: float = 0.0
    annual_upper_bound: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/display."""
        return {
            "current_revenue": self.current_revenue,
            "projected_revenue": self.projected_revenue,
            "forecast_horizon_days": self.forecast_horizon_days,
            "confidence_interval": self.confidence_interval,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "base_trend": self.base_trend,
            "seasonality_adjustment": self.seasonality_adjustment,
            "event_adjustments": self.event_adjustments,
            "projected_annual_sales": self.projected_annual_sales,
            "annual_lower_bound": self.annual_lower_bound,
            "annual_upper_bound": self.annual_upper_bound
        }
    
    def get_change_pct(self) -> float:
        """Calculate percentage change from current to projected."""
        if self.current_revenue > 0:
            return ((self.projected_revenue - self.current_revenue) / self.current_revenue) * 100
        return 0.0
    
    def get_annual_range_str(self) -> str:
        """Format annual projection as range string."""
        return f"${self.annual_lower_bound/1e6:.1f}M - ${self.annual_upper_bound/1e6:.1f}M"


@dataclass
class Scenario:
    """
    What-if scenario analysis.
    
    Provides Base/Optimistic/Pessimistic projected outcomes.
    """
    scenario_name: str  # "Base Case", "Optimistic", "Pessimistic"
    probability: float  # Likelihood of this scenario (0.0-1.0)
    
    # Revenue projections
    projected_revenue_30d: float = 0.0
    projected_revenue_60d: float = 0.0
    projected_revenue_90d: float = 0.0
    
    # Assumptions behind this scenario
    assumptions: List[Dict] = field(default_factory=list)
    # e.g., [{"assumption": "Competitor X undercuts", "impact": -20000}]
    
    # Risk factors
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    
    # Narrative
    narrative: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/display."""
        return {
            "scenario_name": self.scenario_name,
            "probability": self.probability,
            "projected_revenue_30d": self.projected_revenue_30d,
            "projected_revenue_60d": self.projected_revenue_60d,
            "projected_revenue_90d": self.projected_revenue_90d,
            "assumptions": self.assumptions,
            "key_risks": self.key_risks,
            "key_opportunities": self.key_opportunities,
            "narrative": self.narrative
        }


@dataclass
class CombinedIntelligence:
    """
    Unified causal + predictive intelligence output.
    
    This is the master data structure that combines:
    - Backward-looking attribution (what happened)
    - Forward-looking forecasting (what will happen)
    - Strategic synthesis (what to do about it)
    """
    # Backward-looking (causal) - from RevenueAttribution
    attribution: Optional[Any] = None  # RevenueAttribution object
    
    # Forward-looking (predictive)
    forecast: Optional[RevenueForecast] = None
    anticipated_events: List[AnticipatedEvent] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    
    # Synthesis
    sustainable_run_rate: float = 0.0  # Revenue after temporary factors reverse
    temporary_inflation: float = 0.0  # Revenue from temporary factors
    temporary_duration_days: int = 0  # How long temporary factors last
    
    # Recommendations
    critical_actions: List[Dict] = field(default_factory=list)
    # e.g., [{"action": "Reorder inventory", "deadline": "Day 5", "impact": "$50k/mo"}]
    
    strategic_recommendation: str = ""  # Combined narrative
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/display."""
        return {
            "attribution": self.attribution.to_dict() if self.attribution and hasattr(self.attribution, 'to_dict') else None,
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "anticipated_events": [e.to_dict() for e in self.anticipated_events],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "sustainable_run_rate": self.sustainable_run_rate,
            "temporary_inflation": self.temporary_inflation,
            "temporary_duration_days": self.temporary_duration_days,
            "critical_actions": self.critical_actions,
            "strategic_recommendation": self.strategic_recommendation
        }
    
    def get_next_critical_event(self) -> Optional[AnticipatedEvent]:
        """Get the next critical event by date."""
        critical_events = [e for e in self.anticipated_events if e.severity == EventSeverity.CRITICAL]
        if critical_events:
            return min(critical_events, key=lambda e: e.days_until)
        return None
    
    def get_net_30d_impact(self) -> float:
        """Calculate net revenue impact from anticipated events in next 30 days."""
        return sum(e.impact_per_month for e in self.anticipated_events if e.days_until <= 30)


# Seasonality curve for CPG products (monthly multipliers)
# Index 1-12 = January-December
DEFAULT_SEASONALITY_CURVE = {
    1: 0.85,   # January - post-holiday slump
    2: 0.85,   # February
    3: 0.90,   # March
    4: 0.95,   # April
    5: 0.95,   # May
    6: 1.00,   # June
    7: 0.95,   # July
    8: 0.95,   # August
    9: 1.00,   # September
    10: 1.05,  # October - Halloween
    11: 1.15,  # November - Thanksgiving/Black Friday
    12: 1.25,  # December - Holiday peak
}
