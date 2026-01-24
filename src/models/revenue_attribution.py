"""
Revenue Attribution Data Models

Data structures for causal intelligence and revenue attribution.
Decomposes revenue changes into 4 causal categories:
1. Self-Inflicted (Internal Actions): Price changes, PPC, coupons
2. Competitive Influence: Competitor OOS, pricing moves, new entrants
3. Platform/Algorithmic: Amazon changes (Choice badge, algorithm shifts)
4. Market/Macro: Seasonal trends, category growth, viral moments
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class CausalCategory(Enum):
    """The 4 causal categories for revenue attribution."""
    INTERNAL = "internal"           # Self-inflicted (user-controlled)
    COMPETITIVE = "competitive"     # Competitor actions
    PLATFORM = "platform"           # Amazon algorithm/policy changes
    MACRO = "macro"                 # Market-wide trends


class ActionType(Enum):
    """Types of internal actions users can take."""
    PRICE_CHANGE = "price_change"
    PPC_BUDGET = "ppc_budget"
    COUPON_ACTIVATION = "coupon_activation"
    PROMOTION = "promotion"
    INVENTORY_REPLENISHMENT = "inventory_replenishment"
    LISTING_OPTIMIZATION = "listing_optimization"


@dataclass
class InternalAction:
    """
    User-controlled changes that affect revenue.

    Auto-detected from historical data or manually logged.
    """
    action_type: ActionType
    timestamp: datetime
    magnitude: float                 # e.g., +15% or +$5.00
    magnitude_type: str              # "percentage" or "absolute"
    affected_asins: List[str] = field(default_factory=list)
    expected_impact: float = 0.0     # Revenue impact forecast
    actual_impact: float = 0.0       # Measured impact after attribution
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type.value,
            "timestamp": self.timestamp.isoformat(),
            "magnitude": self.magnitude,
            "magnitude_type": self.magnitude_type,
            "affected_asins": self.affected_asins,
            "expected_impact": self.expected_impact,
            "actual_impact": self.actual_impact,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class AttributionDriver:
    """
    Individual event/action contributing to revenue change.

    Can be internal action, competitive event, platform change, or macro trend.
    """
    category: CausalCategory
    description: str                 # Human-readable description
    impact: float                    # Revenue impact (+ or -)
    confidence: float                # 0.0-1.0 confidence score
    controllable: bool               # Can user control this factor?
    event_type: str = ""             # Specific event type (e.g., "competitor_oos")
    related_asin: Optional[str] = None  # Related competitor ASIN if applicable
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_confidence_badge(self) -> str:
        """Get confidence badge emoji and color."""
        if self.confidence >= 0.8:
            return "🟢"  # High confidence
        elif self.confidence >= 0.6:
            return "🟡"  # Medium confidence
        else:
            return "🔴"  # Low confidence

    def get_confidence_label(self) -> str:
        """Get confidence label text."""
        if self.confidence >= 0.8:
            return f"High ({self.confidence:.0%})"
        elif self.confidence >= 0.6:
            return f"Medium ({self.confidence:.0%})"
        else:
            return f"Low ({self.confidence:.0%})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "description": self.description,
            "impact": self.impact,
            "confidence": self.confidence,
            "controllable": self.controllable,
            "event_type": self.event_type,
            "related_asin": self.related_asin,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }


@dataclass
class RevenueAttribution:
    """
    Revenue attribution breakdown for a portfolio or ASIN.

    Decomposes revenue change into 4 causal categories with confidence scoring.
    """
    # Time period
    start_date: datetime
    end_date: datetime

    # Total revenue change
    total_delta: float               # Total revenue change
    previous_revenue: float          # Starting revenue
    current_revenue: float           # Ending revenue
    delta_pct: float = 0.0           # Percentage change

    # The 4 Categories (Portfolio-Level)
    internal_contribution: float = 0.0      # Self-inflicted
    competitive_contribution: float = 0.0   # Competitor moves
    macro_contribution: float = 0.0         # Market trends
    platform_contribution: float = 0.0      # Amazon algorithm/policy

    # Detailed breakdown per category
    internal_drivers: List[AttributionDriver] = field(default_factory=list)
    competitive_drivers: List[AttributionDriver] = field(default_factory=list)
    macro_drivers: List[AttributionDriver] = field(default_factory=list)
    platform_drivers: List[AttributionDriver] = field(default_factory=list)

    # Confidence metrics
    explained_variance: float = 0.0  # % of delta attributed (0.0-1.0)
    residual: float = 0.0            # Unexplained delta
    confidence: float = 0.0          # Overall attribution confidence score

    # Metadata
    portfolio_asins: List[str] = field(default_factory=list)
    attribution_method: str = "elimination"  # "elimination", "regression", "ml"

    def __post_init__(self):
        """Calculate derived fields."""
        if self.previous_revenue != 0:
            self.delta_pct = (self.total_delta / self.previous_revenue) * 100

    def get_all_drivers(self) -> List[AttributionDriver]:
        """Get all drivers sorted by absolute impact (largest first)."""
        all_drivers = (
            self.internal_drivers +
            self.competitive_drivers +
            self.macro_drivers +
            self.platform_drivers
        )
        return sorted(all_drivers, key=lambda d: abs(d.impact), reverse=True)

    def get_top_drivers(self, n: int = 5) -> List[AttributionDriver]:
        """Get top N drivers by impact."""
        return self.get_all_drivers()[:n]

    def get_earned_growth(self) -> float:
        """Get 'earned' growth (from internal actions only)."""
        return self.internal_contribution

    def get_opportunistic_growth(self) -> float:
        """Get 'opportunistic' growth (from external factors)."""
        return (
            self.competitive_contribution +
            self.macro_contribution
        )

    def get_earned_percentage(self) -> float:
        """Get percentage of growth from internal actions (capped at 100%)."""
        if self.total_delta == 0:
            return 0.0
        # Cap at 100% to prevent impossible display values
        raw_pct = (self.internal_contribution / self.total_delta) * 100
        return min(100.0, max(-100.0, raw_pct))

    def get_opportunistic_percentage(self) -> float:
        """Get percentage of growth from opportunistic factors (capped at 100%)."""
        if self.total_delta == 0:
            return 0.0
        # Cap at 100% to prevent impossible display values
        raw_pct = (self.get_opportunistic_growth() / self.total_delta) * 100
        return min(100.0, max(-100.0, raw_pct))

    def get_variance_badge(self) -> str:
        """Get explained variance badge."""
        if self.explained_variance >= 0.8:
            return "🟢"  # High confidence
        elif self.explained_variance >= 0.6:
            return "🟡"  # Medium confidence
        else:
            return "🔴"  # Low confidence

    def get_variance_label(self) -> str:
        """Get explained variance label."""
        if self.explained_variance >= 0.8:
            return f"High Confidence ({self.explained_variance:.0%})"
        elif self.explained_variance >= 0.6:
            return f"Medium Confidence ({self.explained_variance:.0%})"
        else:
            return f"Low Confidence ({self.explained_variance:.0%})"

    def get_residual_percentage(self) -> float:
        """Get residual as percentage of total delta."""
        if self.total_delta == 0:
            return 0.0
        return abs(self.residual) / abs(self.total_delta) * 100

    def has_high_residual(self, threshold: float = 20.0) -> bool:
        """Check if residual exceeds threshold percentage."""
        return self.get_residual_percentage() > threshold

    def get_residual_warning(self) -> Optional[str]:
        """
        Get warning message if residual is high.

        Returns None if residual is acceptable (<20% of total delta).
        Returns warning message if residual is high (>20%).
        """
        residual_pct = self.get_residual_percentage()

        if residual_pct > 20:
            return f"⚠️ {residual_pct:.0f}% unexplained variance - review data quality. Attribution confidence may be lower than reported."
        return None

    def get_data_quality_summary(self) -> str:
        """Get overall data quality assessment."""
        residual_pct = self.get_residual_percentage()
        explained = self.explained_variance

        if explained >= 0.8 and residual_pct < 20:
            return "✅ High data quality - attribution is reliable"
        elif explained >= 0.6 and residual_pct < 30:
            return "🟡 Medium data quality - attribution is directionally correct"
        else:
            return f"🔴 Low data quality - {residual_pct:.0f}% unexplained variance. Use with caution."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            # Time period
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),

            # Total change
            "total_delta": self.total_delta,
            "previous_revenue": self.previous_revenue,
            "current_revenue": self.current_revenue,
            "delta_pct": self.delta_pct,

            # Category totals
            "internal_contribution": self.internal_contribution,
            "competitive_contribution": self.competitive_contribution,
            "macro_contribution": self.macro_contribution,
            "platform_contribution": self.platform_contribution,

            # Detailed drivers
            "internal_drivers": [d.to_dict() for d in self.internal_drivers],
            "competitive_drivers": [d.to_dict() for d in self.competitive_drivers],
            "macro_drivers": [d.to_dict() for d in self.macro_drivers],
            "platform_drivers": [d.to_dict() for d in self.platform_drivers],

            # Confidence
            "explained_variance": self.explained_variance,
            "residual": self.residual,
            "confidence": self.confidence,

            # Metadata
            "portfolio_asins": self.portfolio_asins,
            "attribution_method": self.attribution_method,

            # Derived metrics
            "earned_growth": self.get_earned_growth(),
            "opportunistic_growth": self.get_opportunistic_growth(),
            "earned_percentage": self.get_earned_percentage(),
            "opportunistic_percentage": self.get_opportunistic_percentage()
        }

    def to_summary_text(self) -> str:
        """Generate executive summary text."""
        if self.total_delta >= 0:
            direction = "grew"
            sign = "+"
        else:
            direction = "declined"
            sign = ""

        summary = f"""
**Revenue {direction} {sign}${abs(self.total_delta):,.0f} ({sign}{self.delta_pct:.1f}%)**

**Attribution Breakdown:**
- **Internal Actions:** ${self.internal_contribution:,.0f} ({self.get_earned_percentage():.0f}% of total)
- **Competitive Factors:** ${self.competitive_contribution:,.0f} ({(self.competitive_contribution/self.total_delta*100) if self.total_delta else 0:.0f}% of total)
- **Market Trends:** ${self.macro_contribution:,.0f} ({(self.macro_contribution/self.total_delta*100) if self.total_delta else 0:.0f}% of total)
- **Platform Changes:** ${self.platform_contribution:,.0f} ({(self.platform_contribution/self.total_delta*100) if self.total_delta else 0:.0f}% of total)

**Confidence:** {self.get_variance_label()}
**Unexplained Variance:** ${abs(self.residual):,.0f}
        """.strip()

        return summary
