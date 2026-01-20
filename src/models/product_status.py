"""
Product Status Taxonomy - The Unified "One Truth"

This enum replaces all conflicting status fields with a single, granular classification system.
"""

from enum import Enum
from typing import Dict


class ProductStatus(Enum):
    """
    UNIFIED STATUS CLASSIFICATION

    This is THE SINGLE SOURCE OF TRUTH for product state.
    All UI elements, alerts, and recommendations derive from this field.

    4 Priority Tiers:
    - CRITICAL (100): Immediate action required (losing money NOW)
    - OPPORTUNITY (75): Optimization available (capture upside)
    - WATCH (50): Volatile/changing (monitor closely)
    - STABLE (0): Healthy/cash cow (filter from default view)
    """

    # CRITICAL - Priority 100
    CRITICAL_MARGIN_COLLAPSE = "critical_margin_collapse"
    CRITICAL_INVENTORY_RISK = "critical_inventory_risk"
    CRITICAL_BUYBOX_LOSS = "critical_buybox_loss"
    CRITICAL_VELOCITY_CRASH = "critical_velocity_crash"

    # OPPORTUNITY - Priority 75
    OPPORTUNITY_PRICE_POWER = "opportunity_price_power"
    OPPORTUNITY_AD_WASTE = "opportunity_ad_waste"
    OPPORTUNITY_REVIEW_GAP = "opportunity_review_gap"
    OPPORTUNITY_COMPETITOR_WEAKNESS = "opportunity_competitor_weakness"

    # WATCH - Priority 50
    WATCH_NEW_COMPETITOR = "watch_new_competitor"
    WATCH_PRICE_WAR = "watch_price_war"
    WATCH_SEASONAL_ANOMALY = "watch_seasonal_anomaly"
    WATCH_RANK_VOLATILITY = "watch_rank_volatility"

    # STABLE - Priority 0 (hidden by default)
    STABLE_FORTRESS = "stable_fortress"
    STABLE_CASH_COW = "stable_cash_cow"
    STABLE_NICHE = "stable_niche"

    @property
    def priority(self) -> int:
        """Get priority level for filtering and sorting."""
        return STATUS_PRIORITY.get(self, 0)

    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return STATUS_DISPLAY_NAMES.get(self, self.value)

    @property
    def description(self) -> str:
        """Get detailed description."""
        return STATUS_DESCRIPTIONS.get(self, "")

    @property
    def color(self) -> str:
        """Get UI color code."""
        return STATUS_COLORS.get(self, "#999999")

    @property
    def icon(self) -> str:
        """Get emoji icon."""
        return STATUS_ICONS.get(self, "📊")

    def is_critical(self) -> bool:
        """Check if this status requires immediate action."""
        return self.priority == 100

    def is_opportunity(self) -> bool:
        """Check if this status represents growth potential."""
        return self.priority == 75

    def is_stable(self) -> bool:
        """Check if this status is stable (hidden by default)."""
        return self.priority == 0


# Priority mapping
STATUS_PRIORITY: Dict[ProductStatus, int] = {
    # Critical
    ProductStatus.CRITICAL_MARGIN_COLLAPSE: 100,
    ProductStatus.CRITICAL_INVENTORY_RISK: 100,
    ProductStatus.CRITICAL_BUYBOX_LOSS: 100,
    ProductStatus.CRITICAL_VELOCITY_CRASH: 100,

    # Opportunity
    ProductStatus.OPPORTUNITY_PRICE_POWER: 75,
    ProductStatus.OPPORTUNITY_AD_WASTE: 75,
    ProductStatus.OPPORTUNITY_REVIEW_GAP: 75,
    ProductStatus.OPPORTUNITY_COMPETITOR_WEAKNESS: 75,

    # Watch
    ProductStatus.WATCH_NEW_COMPETITOR: 50,
    ProductStatus.WATCH_PRICE_WAR: 50,
    ProductStatus.WATCH_SEASONAL_ANOMALY: 50,
    ProductStatus.WATCH_RANK_VOLATILITY: 50,

    # Stable
    ProductStatus.STABLE_FORTRESS: 0,
    ProductStatus.STABLE_CASH_COW: 0,
    ProductStatus.STABLE_NICHE: 0,
}

# Display names
STATUS_DISPLAY_NAMES: Dict[ProductStatus, str] = {
    # Critical
    ProductStatus.CRITICAL_MARGIN_COLLAPSE: "Critical: Margin Collapse",
    ProductStatus.CRITICAL_INVENTORY_RISK: "Critical: Stockout Risk",
    ProductStatus.CRITICAL_BUYBOX_LOSS: "Critical: BuyBox Loss",
    ProductStatus.CRITICAL_VELOCITY_CRASH: "Critical: Revenue Crash",

    # Opportunity
    ProductStatus.OPPORTUNITY_PRICE_POWER: "Opportunity: Price Power",
    ProductStatus.OPPORTUNITY_AD_WASTE: "Opportunity: Ad Waste",
    ProductStatus.OPPORTUNITY_REVIEW_GAP: "Opportunity: Review Gap",
    ProductStatus.OPPORTUNITY_COMPETITOR_WEAKNESS: "Opportunity: Competitor Weakness",

    # Watch
    ProductStatus.WATCH_NEW_COMPETITOR: "Watch: New Competitor",
    ProductStatus.WATCH_PRICE_WAR: "Watch: Price War",
    ProductStatus.WATCH_SEASONAL_ANOMALY: "Watch: Seasonal Anomaly",
    ProductStatus.WATCH_RANK_VOLATILITY: "Watch: Rank Volatility",

    # Stable
    ProductStatus.STABLE_FORTRESS: "Stable: Market Fortress",
    ProductStatus.STABLE_CASH_COW: "Stable: Cash Cow",
    ProductStatus.STABLE_NICHE: "Stable: Niche Player",
}

# Descriptions
STATUS_DESCRIPTIONS: Dict[ProductStatus, str] = {
    # Critical
    ProductStatus.CRITICAL_MARGIN_COLLAPSE: "Margin <5% and declining - immediate intervention required",
    ProductStatus.CRITICAL_INVENTORY_RISK: "Out of stock in <7 days - expedite shipment now",
    ProductStatus.CRITICAL_BUYBOX_LOSS: "BuyBox share <30% - investigate pricing/stock immediately",
    ProductStatus.CRITICAL_VELOCITY_CRASH: "Revenue -50%+ in 30 days - root cause analysis needed",

    # Opportunity
    ProductStatus.OPPORTUNITY_PRICE_POWER: "Can raise price due to competitive advantage or weakness",
    ProductStatus.OPPORTUNITY_AD_WASTE: "Can cut ad spend 15%+ without rank impact",
    ProductStatus.OPPORTUNITY_REVIEW_GAP: "Launch Vine campaign - reviews below category average",
    ProductStatus.OPPORTUNITY_COMPETITOR_WEAKNESS: "Competitor out of stock or pricing high",

    # Watch
    ProductStatus.WATCH_NEW_COMPETITOR: "New ASIN with strong BSR entered market",
    ProductStatus.WATCH_PRICE_WAR: "3+ price drops in 7 days detected - monitor margin",
    ProductStatus.WATCH_SEASONAL_ANOMALY: "Unusual pattern vs historical baseline",
    ProductStatus.WATCH_RANK_VOLATILITY: "BSR variance >50% - unstable position",

    # Stable
    ProductStatus.STABLE_FORTRESS: "Market leader with defended position - maintain",
    ProductStatus.STABLE_CASH_COW: "Consistent revenue, low volatility - harvest",
    ProductStatus.STABLE_NICHE: "Small but profitable segment - preserve",
}

# UI colors
STATUS_COLORS: Dict[ProductStatus, str] = {
    # Critical - Red
    ProductStatus.CRITICAL_MARGIN_COLLAPSE: "#dc3545",
    ProductStatus.CRITICAL_INVENTORY_RISK: "#dc3545",
    ProductStatus.CRITICAL_BUYBOX_LOSS: "#dc3545",
    ProductStatus.CRITICAL_VELOCITY_CRASH: "#dc3545",

    # Opportunity - Green
    ProductStatus.OPPORTUNITY_PRICE_POWER: "#28a745",
    ProductStatus.OPPORTUNITY_AD_WASTE: "#28a745",
    ProductStatus.OPPORTUNITY_REVIEW_GAP: "#28a745",
    ProductStatus.OPPORTUNITY_COMPETITOR_WEAKNESS: "#28a745",

    # Watch - Orange
    ProductStatus.WATCH_NEW_COMPETITOR: "#fd7e14",
    ProductStatus.WATCH_PRICE_WAR: "#fd7e14",
    ProductStatus.WATCH_SEASONAL_ANOMALY: "#fd7e14",
    ProductStatus.WATCH_RANK_VOLATILITY: "#fd7e14",

    # Stable - Gray
    ProductStatus.STABLE_FORTRESS: "#6c757d",
    ProductStatus.STABLE_CASH_COW: "#6c757d",
    ProductStatus.STABLE_NICHE: "#6c757d",
}

# Icons
STATUS_ICONS: Dict[ProductStatus, str] = {
    # Critical
    ProductStatus.CRITICAL_MARGIN_COLLAPSE: "🚨",
    ProductStatus.CRITICAL_INVENTORY_RISK: "📦",
    ProductStatus.CRITICAL_BUYBOX_LOSS: "🎯",
    ProductStatus.CRITICAL_VELOCITY_CRASH: "📉",

    # Opportunity
    ProductStatus.OPPORTUNITY_PRICE_POWER: "💰",
    ProductStatus.OPPORTUNITY_AD_WASTE: "✂️",
    ProductStatus.OPPORTUNITY_REVIEW_GAP: "⭐",
    ProductStatus.OPPORTUNITY_COMPETITOR_WEAKNESS: "🎣",

    # Watch
    ProductStatus.WATCH_NEW_COMPETITOR: "👀",
    ProductStatus.WATCH_PRICE_WAR: "⚔️",
    ProductStatus.WATCH_SEASONAL_ANOMALY: "🌡️",
    ProductStatus.WATCH_RANK_VOLATILITY: "📊",

    # Stable
    ProductStatus.STABLE_FORTRESS: "🏰",
    ProductStatus.STABLE_CASH_COW: "🐄",
    ProductStatus.STABLE_NICHE: "🎯",
}


def get_status_from_string(status_str: str) -> ProductStatus:
    """Convert string to ProductStatus enum."""
    try:
        return ProductStatus(status_str)
    except ValueError:
        # Default to stable cash cow if invalid
        return ProductStatus.STABLE_CASH_COW
