"""
Unified Intelligence Data Model

Combines strategic classification, predictive forecasting, and actionable insights
into a single unified output from the intelligence pipeline.

Enhanced with Phase 2-2.5 Causal Intelligence:
- Revenue Attribution (why revenue changed)
- Predictive Forecasting (anticipated events, scenarios, sustainable run rate)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from src.models.product_status import ProductStatus
from src.models.trigger_event import TriggerEvent

# Type hints for Phase 2/2.5 models (avoid circular imports)
if TYPE_CHECKING:
    from src.revenue_attribution import RevenueAttribution
    from src.models.forecast_models import AnticipatedEvent, Scenario, CombinedIntelligence


@dataclass
class UnifiedIntelligence:
    """
    UNIFIED AI ENGINE OUTPUT

    Combines:
    1. Strategic Classification (product status, strategic state)
    2. Predictive Intelligence (30-day risk/growth forecasts)
    3. Actionable Insight (specific recommendation with $ amounts)
    4. Trigger Events (causal reasoning)
    5. Network Intelligence Context (category benchmarks, competitive position)

    This is the single output from the complete intelligence pipeline.
    """

    # === IDENTITY ===
    asin: str
    timestamp: datetime = field(default_factory=datetime.now)

    # === TRIGGER EVENTS (Causal Reasoning) ===
    trigger_events: List[TriggerEvent] = field(default_factory=list)
    primary_trigger: Optional[str] = None  # Most important trigger event type

    # === STRATEGIC CLASSIFICATION ===
    product_status: ProductStatus = ProductStatus.STABLE_CASH_COW
    strategic_state: Optional[str] = None  # FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL (backward compat)
    confidence: float = 0.7
    reasoning: str = ""

    # Visual properties
    state_emoji: str = "📊"
    state_color: str = "#6c757d"
    primary_outcome: str = ""

    # === PREDICTIVE INTELLIGENCE ===
    thirty_day_risk: float = 0.0            # Risk forecast
    thirty_day_growth: float = 0.0          # Growth forecast
    net_expected_value: float = 0.0         # Risk-adjusted upside

    # Risk components
    price_erosion_risk: float = 0.0
    share_erosion_risk: float = 0.0
    stockout_risk: float = 0.0
    daily_burn_rate: float = 0.0

    # Predictive state (forward-looking)
    predictive_state: str = "HOLD"          # DEFEND, EXPLOIT, REPLENISH, HOLD
    predictive_emoji: str = "✅"
    cost_of_inaction: str = ""

    # === ACTIONABLE INSIGHT ===
    recommendation: str = ""                 # Specific action with $ amount
    projected_upside_monthly: float = 0.0    # Revenue opportunity
    downside_risk_monthly: float = 0.0       # Potential loss
    action_type: str = "PROFIT_CAPTURE"      # PROFIT_CAPTURE or RISK_MITIGATION
    time_horizon_days: int = 14              # When to act
    confidence_factors: List[str] = field(default_factory=list)

    # === NETWORK INTELLIGENCE CONTEXT ===
    category_benchmarks: Dict[str, Any] = field(default_factory=dict)
    competitive_position: Dict[str, Any] = field(default_factory=dict)
    brand_intelligence: Dict[str, Any] = field(default_factory=dict)

    # === VALIDATION & QUALITY ===
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    data_quality: str = "MEDIUM"             # HIGH, MEDIUM, LOW
    source: str = "llm"                      # "llm" or "fallback"

    # === USER INTERACTION ===
    user_dismissed: bool = False
    user_feedback: Optional[str] = None
    insight_id: Optional[str] = None

    # === PHASE 2: CAUSAL ATTRIBUTION ===
    revenue_attribution: Optional[Any] = None  # RevenueAttribution object

    # === PHASE 2.5: PREDICTIVE FORECASTING ===
    anticipated_events: List[Any] = field(default_factory=list)  # List[AnticipatedEvent]
    scenarios: List[Any] = field(default_factory=list)  # List[Scenario]
    sustainable_run_rate: float = 0.0  # Monthly revenue after temporary factors removed
    combined_intelligence: Optional[Any] = None  # CombinedIntelligence object

    # === NETWORK INTELLIGENCE SUMMARY STRINGS ===
    category_benchmarks_summary: str = ""
    competitive_position_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and database storage."""
        return {
            # Identity
            "asin": self.asin,
            "timestamp": self.timestamp.isoformat(),

            # Trigger events (convert to dicts)
            "trigger_events": [event.to_dict() for event in self.trigger_events],
            "primary_trigger": self.primary_trigger,

            # Strategic classification
            "product_status": self.product_status.value,
            "status_priority": self.product_status.priority,
            "strategic_state": self.strategic_state,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "state_emoji": self.state_emoji,
            "state_color": self.state_color,
            "primary_outcome": self.primary_outcome,

            # Predictive intelligence
            "thirty_day_risk": self.thirty_day_risk,
            "thirty_day_growth": self.thirty_day_growth,
            "net_expected_value": self.net_expected_value,
            "price_erosion_risk": self.price_erosion_risk,
            "share_erosion_risk": self.share_erosion_risk,
            "stockout_risk": self.stockout_risk,
            "daily_burn_rate": self.daily_burn_rate,
            "predictive_state": self.predictive_state,
            "predictive_emoji": self.predictive_emoji,
            "cost_of_inaction": self.cost_of_inaction,

            # Actionable insight
            "recommendation": self.recommendation,
            "projected_upside_monthly": self.projected_upside_monthly,
            "downside_risk_monthly": self.downside_risk_monthly,
            "action_type": self.action_type,
            "time_horizon_days": self.time_horizon_days,
            "confidence_factors": self.confidence_factors,

            # Network intelligence
            "category_benchmarks": self.category_benchmarks,
            "competitive_position": self.competitive_position,
            "brand_intelligence": self.brand_intelligence,

            # Validation
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "data_quality": self.data_quality,
            "source": self.source,

            # User interaction
            "user_dismissed": self.user_dismissed,
            "user_feedback": self.user_feedback,
            "insight_id": self.insight_id,

            # Phase 2: Causal Attribution
            "revenue_attribution": self.revenue_attribution.to_dict() if self.revenue_attribution and hasattr(self.revenue_attribution, 'to_dict') else None,

            # Phase 2.5: Predictive Forecasting
            "anticipated_events": [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.anticipated_events],
            "scenarios": [s.to_dict() if hasattr(s, 'to_dict') else s for s in self.scenarios],
            "sustainable_run_rate": self.sustainable_run_rate,
            "combined_intelligence": self.combined_intelligence.to_dict() if self.combined_intelligence and hasattr(self.combined_intelligence, 'to_dict') else None,
            "category_benchmarks_summary": self.category_benchmarks_summary,
            "competitive_position_summary": self.competitive_position_summary
        }

    def to_database_record(self) -> Dict[str, Any]:
        """Convert to format for database insertion."""
        import json

        return {
            # Core identity
            "asin": self.asin,
            "created_at": self.timestamp,

            # Unified status
            "product_status": self.product_status.value,
            "status_priority": self.product_status.priority,
            "strategic_state": self.strategic_state,

            # LLM output
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "llm_model": "gpt-4o-mini",

            # Trigger events (store as JSONB)
            "trigger_events": json.dumps([e.to_dict() for e in self.trigger_events]),
            "primary_trigger": self.primary_trigger,

            # Financial impact
            "projected_upside_monthly": self.projected_upside_monthly,
            "downside_risk_monthly": self.downside_risk_monthly,
            "net_expected_value": self.net_expected_value,

            # Predictive intelligence
            "thirty_day_risk": self.thirty_day_risk,
            "thirty_day_growth": self.thirty_day_growth,
            "price_erosion_risk": self.price_erosion_risk,
            "share_erosion_risk": self.share_erosion_risk,
            "stockout_risk": self.stockout_risk,

            # Action metadata
            "action_type": self.action_type,
            "time_horizon_days": self.time_horizon_days,
            "confidence_factors": self.confidence_factors,

            # Validation
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,

            # User interaction
            "user_dismissed": self.user_dismissed,
            "user_feedback": self.user_feedback
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedIntelligence':
        """Create UnifiedIntelligence from dictionary."""
        # Parse timestamp
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp = timestamp or datetime.now()

        # Parse product status
        product_status = data.get('product_status', 'stable_cash_cow')
        if isinstance(product_status, str):
            product_status = ProductStatus(product_status)

        # Parse trigger events
        trigger_events = []
        for event_data in data.get('trigger_events', []):
            if isinstance(event_data, dict):
                trigger_events.append(TriggerEvent.from_dict(event_data))
            elif isinstance(event_data, TriggerEvent):
                trigger_events.append(event_data)

        return cls(
            asin=data['asin'],
            timestamp=timestamp,
            trigger_events=trigger_events,
            primary_trigger=data.get('primary_trigger'),
            product_status=product_status,
            strategic_state=data.get('strategic_state'),
            confidence=data.get('confidence', 0.7),
            reasoning=data.get('reasoning', ''),
            state_emoji=data.get('state_emoji', '📊'),
            state_color=data.get('state_color', '#6c757d'),
            primary_outcome=data.get('primary_outcome', ''),
            thirty_day_risk=data.get('thirty_day_risk', 0.0),
            thirty_day_growth=data.get('thirty_day_growth', 0.0),
            net_expected_value=data.get('net_expected_value', 0.0),
            price_erosion_risk=data.get('price_erosion_risk', 0.0),
            share_erosion_risk=data.get('share_erosion_risk', 0.0),
            stockout_risk=data.get('stockout_risk', 0.0),
            daily_burn_rate=data.get('daily_burn_rate', 0.0),
            predictive_state=data.get('predictive_state', 'HOLD'),
            predictive_emoji=data.get('predictive_emoji', '✅'),
            cost_of_inaction=data.get('cost_of_inaction', ''),
            recommendation=data.get('recommendation', ''),
            projected_upside_monthly=data.get('projected_upside_monthly', 0.0),
            downside_risk_monthly=data.get('downside_risk_monthly', 0.0),
            action_type=data.get('action_type', 'PROFIT_CAPTURE'),
            time_horizon_days=data.get('time_horizon_days', 14),
            confidence_factors=data.get('confidence_factors', []),
            category_benchmarks=data.get('category_benchmarks', {}),
            competitive_position=data.get('competitive_position', {}),
            brand_intelligence=data.get('brand_intelligence', {}),
            validation_passed=data.get('validation_passed', True),
            validation_errors=data.get('validation_errors', []),
            data_quality=data.get('data_quality', 'MEDIUM'),
            source=data.get('source', 'llm'),
            user_dismissed=data.get('user_dismissed', False),
            user_feedback=data.get('user_feedback'),
            insight_id=data.get('insight_id'),
            # Phase 2: Causal Attribution
            revenue_attribution=data.get('revenue_attribution'),
            # Phase 2.5: Predictive Forecasting
            anticipated_events=data.get('anticipated_events', []),
            scenarios=data.get('scenarios', []),
            sustainable_run_rate=data.get('sustainable_run_rate', 0.0),
            combined_intelligence=data.get('combined_intelligence'),
            category_benchmarks_summary=data.get('category_benchmarks_summary', ''),
            competitive_position_summary=data.get('competitive_position_summary', '')
        )

    def get_priority_label(self) -> str:
        """Get human-readable priority label."""
        priority = self.product_status.priority
        if priority == 100:
            return "CRITICAL"
        elif priority == 75:
            return "OPPORTUNITY"
        elif priority == 50:
            return "WATCH"
        else:
            return "STABLE"

    def should_display_in_action_queue(self) -> bool:
        """Determine if this insight should appear in default Action Queue view."""
        return (
            self.validation_passed and
            not self.user_dismissed and
            self.product_status.priority >= 50  # Hide STABLE by default
        )

    def get_urgency_description(self) -> str:
        """Get urgency description based on time horizon."""
        if self.time_horizon_days <= 1:
            return "Act immediately"
        elif self.time_horizon_days <= 3:
            return "Act within 48-72 hours"
        elif self.time_horizon_days <= 7:
            return "Act within 1 week"
        elif self.time_horizon_days <= 14:
            return "Act within 2 weeks"
        else:
            return "Monitor and plan"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UnifiedIntelligence(asin={self.asin}, "
            f"status={self.product_status.value}, "
            f"priority={self.product_status.priority}, "
            f"recommendation='{self.recommendation[:50]}...', "
            f"upside=${self.projected_upside_monthly:.0f})"
        )
