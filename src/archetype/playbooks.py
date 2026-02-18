"""
Archetype-Specific Playbooks.

These playbooks define how signals should be interpreted
based on the product's archetype and SKU role.

The same signal (e.g., price drop) means different things:
- Gillette razor handles (acquisition hero): GOOD
- Gillette cartridges (profit engine): BAD
- Halloween costume (seasonal): EXPECTED

This module provides the interpretation layer.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.models.product_archetype import CategoryArchetype, SKURole


class SignalType(Enum):
    """Types of market signals that can be detected."""
    PRICE_DROP = "price_drop"
    PRICE_INCREASE = "price_increase"
    COMPETITOR_OOS = "competitor_oos"
    YOUR_OOS = "your_oos"
    RATING_DROP = "rating_drop"
    RATING_INCREASE = "rating_increase"
    NEW_COMPETITOR = "new_competitor"
    COMPETITOR_EXIT = "competitor_exit"
    RANK_IMPROVING = "rank_improving"
    RANK_DECLINING = "rank_declining"
    REVIEW_SURGE = "review_surge"
    AMAZON_ENTERED = "amazon_entered"


class SignalSentiment(Enum):
    """How to interpret a signal in context."""
    POSITIVE = "positive"      # Good for you
    NEGATIVE = "negative"      # Bad for you
    NEUTRAL = "neutral"        # Depends on context
    EXPECTED = "expected"      # Normal for this archetype
    CRITICAL = "critical"      # Requires immediate action
    SUSPICIOUS = "suspicious"  # Investigate further


@dataclass
class SignalInterpretation:
    """Interpretation of a signal for a specific archetype/role."""
    sentiment: SignalSentiment
    urgency: str  # "immediate", "this_week", "monitor"
    explanation: str
    recommended_action: str
    weight_multiplier: float = 1.0  # How much to weight this signal

    @property
    def urgency_badge(self) -> str:
        """Get emoji for urgency level."""
        badges = {
            "immediate": "🚨",
            "this_week": "⚠️",
            "monitor": "👀",
        }
        return badges.get(self.urgency, "📊")

    @property
    def sentiment_badge(self) -> str:
        """Get emoji for sentiment."""
        badges = {
            SignalSentiment.POSITIVE: "✅",
            SignalSentiment.NEGATIVE: "❌",
            SignalSentiment.NEUTRAL: "➖",
            SignalSentiment.EXPECTED: "📅",
            SignalSentiment.CRITICAL: "🚨",
            SignalSentiment.SUSPICIOUS: "🔍",
        }
        return badges.get(self.sentiment, "❓")


# =============================================================================
# ARCHETYPE PLAYBOOKS
# =============================================================================

ARCHETYPE_PLAYBOOKS: Dict[CategoryArchetype, Dict[SignalType, SignalInterpretation]] = {

    # =========================================================================
    # REPLENISHABLE COMMODITY
    # =========================================================================
    CategoryArchetype.REPLENISHABLE_COMMODITY: {
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="this_week",
            explanation="Price competition normal in commodities - focus on S&S lock-in",
            recommended_action="If competitor drops, consider matching to protect S&S subscribers",
            weight_multiplier=0.8,
        ),
        SignalType.COMPETITOR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CRITICAL: Window to lock in S&S subscribers - they'll switch permanently",
            recommended_action="Increase PPC aggressively, push Subscribe & Save hard",
            weight_multiplier=2.0,
        ),
        SignalType.YOUR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CATASTROPHIC: Customers will find alternatives and may not return",
            recommended_action="Emergency restock, consider FBM backup, notify S&S customers",
            weight_multiplier=2.5,
        ),
        SignalType.RATING_DROP: SignalInterpretation(
            sentiment=SignalSentiment.NEGATIVE,
            urgency="this_week",
            explanation="Moderate concern - commodities are less review-sensitive than durables",
            recommended_action="Monitor but don't overreact - price/availability matter more",
            weight_multiplier=0.7,
        ),
        SignalType.NEW_COMPETITOR: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Competition normal in commodities - focus on S&S retention",
            recommended_action="Monitor pricing, protect S&S customers",
            weight_multiplier=0.8,
        ),
    },

    # =========================================================================
    # RAZOR AND BLADES
    # =========================================================================
    CategoryArchetype.RAZOR_AND_BLADES: {
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="this_week",
            explanation="DEPENDS: Price drop on handle = good (acquisition). On consumable = bad (margin)",
            recommended_action="Check if this is handle or consumable before reacting",
            weight_multiplier=1.0,
        ),
        SignalType.COMPETITOR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CRITICAL for handles: Competitor's installed base is up for grabs",
            recommended_action="Aggressive conquest on handles to capture locked-in customers",
            weight_multiplier=2.0,
        ),
        SignalType.YOUR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="Handle OOS = lose customer forever. Consumable OOS = they'll wait (locked in)",
            recommended_action="Prioritize handle inventory over consumables",
            weight_multiplier=1.5,
        ),
        SignalType.AMAZON_ENTERED: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="Amazon on handles = ecosystem threat. On consumables = margin compression",
            recommended_action="Defend installed base, consider exclusive bundles",
            weight_multiplier=1.8,
        ),
    },

    # =========================================================================
    # DURABLE CONSIDERATION
    # =========================================================================
    CategoryArchetype.DURABLE_CONSIDERATION: {
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.SUSPICIOUS,
            urgency="monitor",
            explanation="Price drops on durables can signal quality doubt to research-heavy buyers",
            recommended_action="If dropping price, frame as 'limited time' not 'new lower price'",
            weight_multiplier=1.2,
        ),
        SignalType.RATING_DROP: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CRITICAL: Research-heavy buyers scrutinize ratings - conversion cliff ahead",
            recommended_action="Identify root cause, respond to negative reviews, consider quality fixes",
            weight_multiplier=2.5,
        ),
        SignalType.REVIEW_SURGE: SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="monitor",
            explanation="Excellent: More reviews = more trust for consideration-heavy buyers",
            recommended_action="Leverage in ads, highlight review count in creative",
            weight_multiplier=1.5,
        ),
        SignalType.NEW_COMPETITOR: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Less urgent: Long consideration cycles limit new competitor impact",
            recommended_action="Monitor but don't panic - established reviews are a moat",
            weight_multiplier=0.6,
        ),
        SignalType.PRICE_INCREASE: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Durables have more pricing power - quality perception supports premium",
            recommended_action="Test price increases carefully, monitor conversion rate",
            weight_multiplier=0.8,
        ),
    },

    # =========================================================================
    # SEASONAL SPIKE
    # =========================================================================
    CategoryArchetype.SEASONAL_SPIKE: {
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.EXPECTED,
            urgency="monitor",
            explanation="EXPECTED: Post-season clearance is normal. Focus on inventory recovery.",
            recommended_action="Maximize recovery, don't fight clearance pricing",
            weight_multiplier=0.3,
        ),
        SignalType.YOUR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CATASTROPHIC during season: Can't make up lost sales, window is closing",
            recommended_action="Emergency air freight if in-season, accept loss if post-season",
            weight_multiplier=3.0,
        ),
        SignalType.RANK_IMPROVING: SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="immediate",
            explanation="TIME-SENSITIVE: Capture momentum before season window closes",
            recommended_action="Increase spend immediately - can't save budget for next year",
            weight_multiplier=1.5,
        ),
        SignalType.RANK_DECLINING: SignalInterpretation(
            sentiment=SignalSentiment.EXPECTED,
            urgency="monitor",
            explanation="Post-season decline is expected - don't fight the calendar",
            recommended_action="Begin clearance pricing, shift focus to next season prep",
            weight_multiplier=0.3,
        ),
        SignalType.COMPETITOR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="immediate",
            explanation="Capture their seasonal demand NOW - window is limited",
            recommended_action="Max out PPC, they can't restock fast enough for this season",
            weight_multiplier=1.8,
        ),
    },

    # =========================================================================
    # REGULATED COMPLIANCE
    # =========================================================================
    CategoryArchetype.REGULATED_COMPLIANCE: {
        SignalType.NEW_COMPETITOR: SignalInterpretation(
            sentiment=SignalSentiment.SUSPICIOUS,
            urgency="immediate",
            explanation="WARNING: New seller may be unauthorized/counterfeit - existential risk",
            recommended_action="Verify authorization, report if unauthorized, alert Brand Registry",
            weight_multiplier=2.0,
        ),
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.SUSPICIOUS,
            urgency="this_week",
            explanation="SUSPICIOUS: Aggressive pricing may indicate diverted/counterfeit goods",
            recommended_action="Check seller authorization, monitor for quality complaints",
            weight_multiplier=1.5,
        ),
        SignalType.RATING_DROP: SignalInterpretation(
            sentiment=SignalSentiment.CRITICAL,
            urgency="immediate",
            explanation="CRITICAL: Quality complaints can trigger regulatory scrutiny",
            recommended_action="Review complaint specifics, assess regulatory exposure, escalate internally",
            weight_multiplier=3.0,
        ),
        SignalType.REVIEW_SURGE: SignalInterpretation(
            sentiment=SignalSentiment.SUSPICIOUS,
            urgency="this_week",
            explanation="Monitor content - regulatory-sensitive reviews need attention",
            recommended_action="Read reviews for compliance issues, respond appropriately",
            weight_multiplier=1.5,
        ),
    },

    # =========================================================================
    # FASHION STYLE
    # =========================================================================
    CategoryArchetype.FASHION_STYLE: {
        SignalType.RANK_IMPROVING: SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="this_week",
            explanation="Check if trend-driven (viral?) or organic - ride the wave if trending",
            recommended_action="Increase inventory and spend if trend-driven, capitalize quickly",
            weight_multiplier=1.3,
        ),
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Fashion arbitrage is normal - trend may be passing",
            recommended_action="If trend is fading, accept lower margins to clear inventory",
            weight_multiplier=0.7,
        ),
        SignalType.YOUR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.NEGATIVE,
            urgency="this_week",
            explanation="Lost sales, but fashion buyers have many alternatives",
            recommended_action="Restock key sizes/colors, accept some styles may have passed",
            weight_multiplier=1.0,
        ),
        SignalType.RANK_DECLINING: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Trend lifecycle is normal - may be time to clear and move on",
            recommended_action="Begin markdown if trend passing, don't chase fading styles",
            weight_multiplier=0.8,
        ),
    },

    # =========================================================================
    # COMMODITY BUNDLE
    # =========================================================================
    CategoryArchetype.COMMODITY_BUNDLE: {
        SignalType.PRICE_DROP: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Check bundle-to-single ratio - bundle value perception must be maintained",
            recommended_action="If single unit dropped, may need to adjust bundle price",
            weight_multiplier=0.9,
        ),
        SignalType.RANK_DECLINING: SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="this_week",
            explanation="Customers may prefer singles (less commitment) - check single unit trends",
            recommended_action="Consider smaller bundle sizes or trial packs",
            weight_multiplier=1.0,
        ),
        SignalType.COMPETITOR_OOS: SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="this_week",
            explanation="Opportunity to capture bulk/value buyers",
            recommended_action="Highlight value proposition in ads",
            weight_multiplier=1.2,
        ),
    },
}


# =============================================================================
# SKU ROLE MODIFIERS
# =============================================================================

SKU_ROLE_MODIFIERS: Dict[SKURole, Dict[str, Any]] = {

    SKURole.ACQUISITION_HERO: {
        "price_sensitivity": "low",
        "margin_priority": "low",
        "ppc_investment": "high",
        "price_drop_reaction": "accept",
        "description": "Drive trial - margin secondary to volume",
    },

    SKURole.PROFIT_ENGINE: {
        "price_sensitivity": "high",
        "margin_priority": "high",
        "ppc_investment": "low",
        "price_drop_reaction": "resist",
        "description": "Protect margin - never discount",
    },

    SKURole.DEFENDER: {
        "price_sensitivity": "medium",
        "margin_priority": "low",
        "ppc_investment": "low",
        "price_drop_reaction": "match_exactly",
        "description": "Block competitors - match their price exactly",
    },

    SKURole.PROMO_LEVER: {
        "price_sensitivity": "low",
        "margin_priority": "medium",
        "ppc_investment": "medium",
        "price_drop_reaction": "embrace",
        "description": "Designed for discounting - measure halo effect",
    },

    SKURole.BUNDLE_LADDER: {
        "price_sensitivity": "medium",
        "margin_priority": "medium",
        "ppc_investment": "medium",
        "price_drop_reaction": "anchored",
        "description": "Price anchored to components",
    },

    SKURole.TRAFFIC_SPONGE: {
        "price_sensitivity": "low",
        "margin_priority": "low",
        "ppc_investment": "very_high",
        "price_drop_reaction": "accept",
        "description": "Capture keywords - conversion rate secondary",
    },

    SKURole.CLEANUP_LONGTAIL: {
        "price_sensitivity": "low",
        "margin_priority": "low",
        "ppc_investment": "minimal",
        "price_drop_reaction": "accept",
        "description": "Minimal investment - watch for revival signals",
    },
}


# =============================================================================
# MAIN INTERPRETATION FUNCTION
# =============================================================================

def get_signal_interpretation(
    archetype: CategoryArchetype,
    signal_type: SignalType,
    sku_role: Optional[SKURole] = None,
) -> SignalInterpretation:
    """
    Get the interpretation of a signal for a specific archetype.

    Args:
        archetype: The product's category archetype
        signal_type: The type of signal detected
        sku_role: Optional SKU role for additional context

    Returns:
        SignalInterpretation with sentiment, urgency, and recommendations
    """
    # Get base interpretation from archetype playbook
    archetype_playbook = ARCHETYPE_PLAYBOOKS.get(archetype, {})
    interpretation = archetype_playbook.get(signal_type)

    if interpretation is None:
        # Default interpretation
        interpretation = SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation="Standard signal - interpret based on general market dynamics",
            recommended_action="Monitor and assess based on overall portfolio strategy",
            weight_multiplier=1.0,
        )

    # Apply SKU role modifiers for price-related signals
    if sku_role and signal_type == SignalType.PRICE_DROP:
        interpretation = _apply_role_modifier_to_price_drop(interpretation, sku_role)

    return interpretation


def _apply_role_modifier_to_price_drop(
    base_interpretation: SignalInterpretation,
    sku_role: SKURole,
) -> SignalInterpretation:
    """
    Modify price drop interpretation based on SKU role.
    """
    role_config = SKU_ROLE_MODIFIERS.get(sku_role, {})
    reaction = role_config.get("price_drop_reaction", "monitor")
    description = role_config.get("description", "")

    if reaction == "accept":
        return SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="monitor",
            explanation=f"Price drop acceptable for {sku_role.value}: {description}",
            recommended_action="Accept price pressure, focus on volume/acquisition",
            weight_multiplier=0.5,
        )

    elif reaction == "resist":
        return SignalInterpretation(
            sentiment=SignalSentiment.NEGATIVE,
            urgency="immediate",
            explanation=f"Price drop threatens {sku_role.value} margin: {description}",
            recommended_action="Protect price, consider reducing visibility vs dropping price",
            weight_multiplier=2.0,
        )

    elif reaction == "match_exactly":
        return SignalInterpretation(
            sentiment=SignalSentiment.NEUTRAL,
            urgency="this_week",
            explanation=f"Match competitor price exactly for {sku_role.value}: {description}",
            recommended_action="Match price to maintain position, don't undercut",
            weight_multiplier=1.0,
        )

    elif reaction == "embrace":
        return SignalInterpretation(
            sentiment=SignalSentiment.POSITIVE,
            urgency="monitor",
            explanation=f"Discounting is expected for {sku_role.value}: {description}",
            recommended_action="Run the promotion, measure halo effect on other SKUs",
            weight_multiplier=0.3,
        )

    return base_interpretation


def get_archetype_guidance(archetype: CategoryArchetype, sku_role: Optional[SKURole] = None) -> str:
    """
    Get general strategic guidance for an archetype.

    Returns a string suitable for LLM prompt injection.
    """
    guidance_lines = [f"=== STRATEGIC GUIDANCE FOR {archetype.value.upper()} ==="]

    if archetype == CategoryArchetype.REPLENISHABLE_COMMODITY:
        guidance_lines.extend([
            "Focus on: Subscribe & Save penetration, stockout prevention",
            "Price drops are normal - don't overreact",
            "Competitor OOS is CRITICAL - capture S&S customers permanently",
        ])

    elif archetype == CategoryArchetype.RAZOR_AND_BLADES:
        guidance_lines.extend([
            "Focus on: Installed base growth (handles), consumable margin protection",
            "Handle price drops = GOOD (acquisition cost)",
            "Consumable price drops = BAD (margin destruction)",
            "Handle OOS = CATASTROPHIC (lose customer forever)",
        ])

    elif archetype == CategoryArchetype.DURABLE_CONSIDERATION:
        guidance_lines.extend([
            "Focus on: Reviews, ratings, trust signals",
            "Rating drops are CRITICAL - affects research-heavy buyers",
            "Price drops can signal quality doubt",
            "New competitors have slow impact (long consideration cycle)",
        ])

    elif archetype == CategoryArchetype.SEASONAL_SPIKE:
        guidance_lines.extend([
            "Focus on: Timing, inventory, seasonal window capture",
            "Price drops are EXPECTED post-season (clearance)",
            "Rank declines are EXPECTED post-season",
            "In-season OOS is CATASTROPHIC - can't recover",
        ])

    elif archetype == CategoryArchetype.REGULATED_COMPLIANCE:
        guidance_lines.extend([
            "Focus on: Trust, safety, authorized sellers",
            "New competitors are SUSPICIOUS - verify authorization",
            "Aggressive pricing is SUSPICIOUS - may be diverted goods",
            "Quality complaints can trigger regulatory scrutiny",
        ])

    elif archetype == CategoryArchetype.FASHION_STYLE:
        guidance_lines.extend([
            "Focus on: Trends, visual appeal, inventory turns",
            "Rank changes may be trend-driven - capitalize quickly",
            "Price drops are normal as trends pass",
            "High returns are expected - size/fit friction",
        ])

    elif archetype == CategoryArchetype.COMMODITY_BUNDLE:
        guidance_lines.extend([
            "Focus on: Value perception, bundle-to-single ratio",
            "Monitor single unit pricing - affects bundle value perception",
            "Customers may prefer singles (less commitment)",
        ])

    if sku_role:
        role_config = SKU_ROLE_MODIFIERS.get(sku_role, {})
        guidance_lines.append(f"\nSKU Role: {sku_role.value.upper()}")
        guidance_lines.append(f"Strategy: {role_config.get('description', 'Standard')}")
        guidance_lines.append(f"Price Sensitivity: {role_config.get('price_sensitivity', 'medium')}")
        guidance_lines.append(f"PPC Investment: {role_config.get('ppc_investment', 'medium')}")

    return "\n".join(guidance_lines)


def adjust_trigger_severity(
    base_severity: int,
    signal_type: SignalType,
    archetype: CategoryArchetype,
    sku_role: Optional[SKURole] = None,
) -> int:
    """
    Adjust trigger event severity based on archetype context.

    Args:
        base_severity: Original severity (1-10)
        signal_type: Type of signal detected
        archetype: Product's category archetype
        sku_role: Product's SKU role

    Returns:
        Adjusted severity (1-10)
    """
    interpretation = get_signal_interpretation(archetype, signal_type, sku_role)

    # Apply weight multiplier
    adjusted = base_severity * interpretation.weight_multiplier

    # Clamp to 1-10
    return max(1, min(10, int(adjusted)))
