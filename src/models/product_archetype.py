"""
Product Archetype Data Models

Defines the category archetypes and SKU roles that provide
context-aware interpretation of market signals.

The same signal means different things depending on:
1. Category Archetype: What kind of product economics does this have?
2. SKU Role: What strategic role does this SKU play in the portfolio?

Example:
- Price drop on Gillette razor handles (acquisition hero) = GOOD
- Price drop on Gillette cartridges (profit engine) = BAD
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime


class CategoryArchetype(Enum):
    """
    The 7 category economic archetypes.

    Each archetype has different economics that change how signals
    should be interpreted.
    """
    REPLENISHABLE_COMMODITY = "replenishable_commodity"  # Toilet paper, batteries
    RAZOR_AND_BLADES = "razor_and_blades"  # Printers/ink, razors/cartridges
    DURABLE_CONSIDERATION = "durable_consideration"  # Furniture, appliances
    SEASONAL_SPIKE = "seasonal_spike"  # Halloween, Christmas items
    REGULATED_COMPLIANCE = "regulated_compliance"  # Supplements, baby products
    FASHION_STYLE = "fashion_style"  # Clothing, accessories
    COMMODITY_BUNDLE = "commodity_bundle"  # Multipacks, variety packs
    UNKNOWN = "unknown"

    @property
    def description(self) -> str:
        """Get human-readable description of the archetype."""
        descriptions = {
            self.REPLENISHABLE_COMMODITY: "High purchase frequency, S&S critical, stockout catastrophic",
            self.RAZOR_AND_BLADES: "Base is loss leader, consumable is profit center",
            self.DURABLE_CONSIDERATION: "Research-heavy buyers, reviews critical, returns expensive",
            self.SEASONAL_SPIKE: "80%+ revenue in <90 day window, timing is everything",
            self.REGULATED_COMPLIANCE: "Trust/safety critical, unauthorized sellers = existential risk",
            self.FASHION_STYLE: "Trend-driven, high returns normal, short lifecycle",
            self.COMMODITY_BUNDLE: "Bundle economics differ from single unit",
            self.UNKNOWN: "Archetype not yet classified",
        }
        return descriptions.get(self, "Unknown archetype")

    @property
    def badge(self) -> str:
        """Get emoji badge for the archetype."""
        badges = {
            self.REPLENISHABLE_COMMODITY: "🔄",
            self.RAZOR_AND_BLADES: "🪒",
            self.DURABLE_CONSIDERATION: "🏠",
            self.SEASONAL_SPIKE: "🎃",
            self.REGULATED_COMPLIANCE: "⚕️",
            self.FASHION_STYLE: "👗",
            self.COMMODITY_BUNDLE: "📦",
            self.UNKNOWN: "❓",
        }
        return badges.get(self, "❓")


class SKURole(Enum):
    """
    The 7 SKU strategic roles.

    Each role has different objectives that change how signals
    should be interpreted and what actions to recommend.
    """
    ACQUISITION_HERO = "acquisition_hero"  # Drive trial, capture new customers
    PROFIT_ENGINE = "profit_engine"  # Generate margin, fund growth
    DEFENDER = "defender"  # Block competitors, hold shelf space
    PROMO_LEVER = "promo_lever"  # Drive traffic, create urgency
    BUNDLE_LADDER = "bundle_ladder"  # Increase AOV, demonstrate value
    TRAFFIC_SPONGE = "traffic_sponge"  # Capture keywords, feed funnel
    CLEANUP_LONGTAIL = "cleanup_longtail"  # Serve niche demand
    UNKNOWN = "unknown"

    @property
    def description(self) -> str:
        """Get human-readable description of the role."""
        descriptions = {
            self.ACQUISITION_HERO: "Drive trial - margin secondary to volume",
            self.PROFIT_ENGINE: "Protect margin - never discount",
            self.DEFENDER: "Block competitors - match their price exactly",
            self.PROMO_LEVER: "Designed for discounting - measure halo effect",
            self.BUNDLE_LADDER: "Price anchored to components - increase AOV",
            self.TRAFFIC_SPONGE: "Capture keywords - conversion rate secondary",
            self.CLEANUP_LONGTAIL: "Minimal investment - watch for revival signals",
            self.UNKNOWN: "Role not yet classified",
        }
        return descriptions.get(self, "Unknown role")

    @property
    def badge(self) -> str:
        """Get emoji badge for the role."""
        badges = {
            self.ACQUISITION_HERO: "🦸",
            self.PROFIT_ENGINE: "💰",
            self.DEFENDER: "🛡️",
            self.PROMO_LEVER: "🏷️",
            self.BUNDLE_LADDER: "📊",
            self.TRAFFIC_SPONGE: "🧲",
            self.CLEANUP_LONGTAIL: "🧹",
            self.UNKNOWN: "❓",
        }
        return badges.get(self, "❓")

    @property
    def price_sensitivity(self) -> str:
        """How sensitive is this role to price changes?"""
        sensitivities = {
            self.ACQUISITION_HERO: "low",  # Price drops OK
            self.PROFIT_ENGINE: "high",  # Protect price
            self.DEFENDER: "medium",  # Match competitor
            self.PROMO_LEVER: "low",  # Designed for deals
            self.BUNDLE_LADDER: "medium",  # Anchored to components
            self.TRAFFIC_SPONGE: "low",  # Volume over margin
            self.CLEANUP_LONGTAIL: "low",  # Minimal investment
            self.UNKNOWN: "medium",
        }
        return sensitivities.get(self, "medium")


@dataclass
class ArchetypeFeatures:
    """
    Deterministic features extracted from product data.
    Used as input to archetype classification.

    These features are extracted from Keepa data and product metadata
    without any LLM involvement.
    """
    # Category signals
    category_tree: str = ""
    is_consumable: bool = False
    is_subscription_eligible: bool = False
    is_hazmat: bool = False
    is_adult: bool = False
    requires_approval: bool = False

    # Pricing signals
    price: float = 0.0
    price_vs_category_median: float = 0.0  # ratio (1.0 = at median)
    has_subscribe_save: bool = False
    typical_discount_pct: float = 0.0

    # Velocity signals
    purchase_frequency_days: Optional[float] = None  # avg days between repeat purchase
    review_velocity_30d: int = 0
    return_rate_estimate: Optional[float] = None

    # Variation signals
    is_variation: bool = False
    variation_count: int = 0
    is_parent: bool = False
    has_size_variations: bool = False
    has_color_variations: bool = False

    # Bundle signals
    is_multipack: bool = False
    pack_count: int = 1

    # Seasonal signals
    sales_coefficient_of_variation: float = 0.0  # high = seasonal
    peak_months: List[int] = field(default_factory=list)

    # Competitive signals
    seller_count: int = 0
    has_amazon_offer: bool = False
    brand_registry_enrolled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category_tree": self.category_tree,
            "is_consumable": self.is_consumable,
            "is_subscription_eligible": self.is_subscription_eligible,
            "is_hazmat": self.is_hazmat,
            "is_adult": self.is_adult,
            "requires_approval": self.requires_approval,
            "price": self.price,
            "price_vs_category_median": self.price_vs_category_median,
            "has_subscribe_save": self.has_subscribe_save,
            "typical_discount_pct": self.typical_discount_pct,
            "purchase_frequency_days": self.purchase_frequency_days,
            "review_velocity_30d": self.review_velocity_30d,
            "return_rate_estimate": self.return_rate_estimate,
            "is_variation": self.is_variation,
            "variation_count": self.variation_count,
            "is_parent": self.is_parent,
            "has_size_variations": self.has_size_variations,
            "has_color_variations": self.has_color_variations,
            "is_multipack": self.is_multipack,
            "pack_count": self.pack_count,
            "sales_coefficient_of_variation": self.sales_coefficient_of_variation,
            "peak_months": self.peak_months,
            "seller_count": self.seller_count,
            "has_amazon_offer": self.has_amazon_offer,
            "brand_registry_enrolled": self.brand_registry_enrolled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchetypeFeatures":
        """Create from dictionary."""
        return cls(
            category_tree=data.get("category_tree", ""),
            is_consumable=data.get("is_consumable", False),
            is_subscription_eligible=data.get("is_subscription_eligible", False),
            is_hazmat=data.get("is_hazmat", False),
            is_adult=data.get("is_adult", False),
            requires_approval=data.get("requires_approval", False),
            price=data.get("price", 0.0),
            price_vs_category_median=data.get("price_vs_category_median", 0.0),
            has_subscribe_save=data.get("has_subscribe_save", False),
            typical_discount_pct=data.get("typical_discount_pct", 0.0),
            purchase_frequency_days=data.get("purchase_frequency_days"),
            review_velocity_30d=data.get("review_velocity_30d", 0),
            return_rate_estimate=data.get("return_rate_estimate"),
            is_variation=data.get("is_variation", False),
            variation_count=data.get("variation_count", 0),
            is_parent=data.get("is_parent", False),
            has_size_variations=data.get("has_size_variations", False),
            has_color_variations=data.get("has_color_variations", False),
            is_multipack=data.get("is_multipack", False),
            pack_count=data.get("pack_count", 1),
            sales_coefficient_of_variation=data.get("sales_coefficient_of_variation", 0.0),
            peak_months=data.get("peak_months", []),
            seller_count=data.get("seller_count", 0),
            has_amazon_offer=data.get("has_amazon_offer", False),
            brand_registry_enrolled=data.get("brand_registry_enrolled", False),
        )


@dataclass
class ProductArchetype:
    """
    The complete archetype classification for a product.

    Combines category archetype and SKU role to provide
    context for signal interpretation.
    """
    asin: str

    # Classifications
    category_archetype: CategoryArchetype = CategoryArchetype.UNKNOWN
    sku_role: SKURole = SKURole.UNKNOWN

    # Confidence scores (0.0 - 1.0)
    archetype_confidence: float = 0.0
    role_confidence: float = 0.0

    # Rationale (from LLM or rules)
    archetype_rationale: str = ""
    role_rationale: str = ""

    # Override tracking
    is_user_override: bool = False
    override_timestamp: Optional[datetime] = None
    override_by: Optional[str] = None

    # Features used for classification
    features: Optional[ArchetypeFeatures] = None

    # Timestamps
    classified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and storage."""
        return {
            "asin": self.asin,
            "category_archetype": self.category_archetype.value,
            "sku_role": self.sku_role.value,
            "archetype_confidence": self.archetype_confidence,
            "role_confidence": self.role_confidence,
            "archetype_rationale": self.archetype_rationale,
            "role_rationale": self.role_rationale,
            "is_user_override": self.is_user_override,
            "override_timestamp": self.override_timestamp.isoformat() if self.override_timestamp else None,
            "override_by": self.override_by,
            "features": self.features.to_dict() if self.features else None,
            "classified_at": self.classified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductArchetype":
        """Create from dictionary (e.g., from database)."""
        features = None
        if data.get("features"):
            features = ArchetypeFeatures.from_dict(data["features"])

        override_ts = None
        if data.get("override_timestamp"):
            override_ts = datetime.fromisoformat(data["override_timestamp"])

        classified_at = datetime.now()
        if data.get("classified_at"):
            classified_at = datetime.fromisoformat(data["classified_at"])

        return cls(
            asin=data.get("asin", ""),
            category_archetype=CategoryArchetype(data.get("category_archetype", "unknown")),
            sku_role=SKURole(data.get("sku_role", "unknown")),
            archetype_confidence=data.get("archetype_confidence", 0.0),
            role_confidence=data.get("role_confidence", 0.0),
            archetype_rationale=data.get("archetype_rationale", ""),
            role_rationale=data.get("role_rationale", ""),
            is_user_override=data.get("is_user_override", False),
            override_timestamp=override_ts,
            override_by=data.get("override_by"),
            features=features,
            classified_at=classified_at,
        )

    def get_archetype_badge(self) -> str:
        """Get emoji badge for archetype."""
        return self.category_archetype.badge

    def get_role_badge(self) -> str:
        """Get emoji badge for SKU role."""
        return self.sku_role.badge

    def get_combined_badge(self) -> str:
        """Get combined badge showing both archetype and role."""
        return f"{self.get_archetype_badge()}{self.get_role_badge()}"

    def get_display_name(self) -> str:
        """Get human-readable display name."""
        archetype_name = self.category_archetype.value.replace("_", " ").title()
        role_name = self.sku_role.value.replace("_", " ").title()
        return f"{archetype_name} / {role_name}"

    def to_prompt_context(self) -> str:
        """
        Format for LLM prompt injection.

        This provides the archetype context that helps the LLM
        interpret signals correctly.
        """
        lines = [
            f"=== PRODUCT ARCHETYPE: {self.asin} ===",
            f"Category: {self.category_archetype.badge} {self.category_archetype.value.upper()}",
            f"  → {self.category_archetype.description}",
            f"Role: {self.sku_role.badge} {self.sku_role.value.upper()}",
            f"  → {self.sku_role.description}",
            f"Price Sensitivity: {self.sku_role.price_sensitivity.upper()}",
        ]

        if self.archetype_rationale:
            lines.append(f"Classification Reason: {self.archetype_rationale}")

        if self.is_user_override:
            lines.append("(User Override)")

        return "\n".join(lines)

    def is_price_drop_concerning(self) -> bool:
        """
        Should a price drop be flagged as concerning for this product?

        Returns False for acquisition heroes, promo levers, etc.
        Returns True for profit engines.
        """
        if self.sku_role == SKURole.PROFIT_ENGINE:
            return True
        if self.sku_role in [SKURole.ACQUISITION_HERO, SKURole.PROMO_LEVER, SKURole.TRAFFIC_SPONGE]:
            return False
        if self.category_archetype == CategoryArchetype.SEASONAL_SPIKE:
            return False  # Expected for clearance
        return True  # Default to concerning

    def is_competitor_oos_critical(self) -> bool:
        """
        Is competitor OOS a critical opportunity for this product?

        Returns True for replenishables and razor-and-blades.
        """
        return self.category_archetype in [
            CategoryArchetype.REPLENISHABLE_COMMODITY,
            CategoryArchetype.RAZOR_AND_BLADES,
        ]

    def is_rating_drop_critical(self) -> bool:
        """
        Is a rating drop critical for this product?

        Returns True for durables and regulated products.
        """
        return self.category_archetype in [
            CategoryArchetype.DURABLE_CONSIDERATION,
            CategoryArchetype.REGULATED_COMPLIANCE,
        ]


# Type alias for convenience
Archetype = ProductArchetype
