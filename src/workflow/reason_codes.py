"""
Reason Codes for Workflow ShelfGuard

Deterministic classification of Episodes into actionable reason codes.
Each reason code maps to:
1. A severity baseline
2. A confidence calculation method
3. An action template

V1 Reason Codes (Keepa/public data only):
- Price-related: PRICE_WAR, PRICE_COMPRESSION, PROMO_SHOCK
- Competition: COMPETITOR_OOS, COMPETITOR_PROMO
- Rank: RANK_DECLINE, RANK_SURGE
- BuyBox: BUYBOX_INSTABILITY
- Platform: OOS_ARTIFACT, DEMAND_SHIFT
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class ReasonCode(Enum):
    """
    V1 Reason Codes - mapped to exactly 6 detectors.

    Each code represents a specific, actionable market condition.
    """
    # Price-Related
    PRICE_WAR = "price_war"
    PRICE_COMPRESSION = "price_compression"
    PROMO_SHOCK = "promo_shock"

    # Competition-Related
    COMPETITOR_OOS = "competitor_oos"
    COMPETITOR_PROMO = "competitor_promo"

    # Rank-Related
    RANK_DECLINE = "rank_decline"
    RANK_SURGE = "rank_surge"

    # BuyBox-Related
    BUYBOX_INSTABILITY = "buybox_instability"

    # Platform-Related
    OOS_ARTIFACT = "oos_artifact"
    DEMAND_SHIFT = "demand_shift"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return {
            ReasonCode.PRICE_WAR: "Price War",
            ReasonCode.PRICE_COMPRESSION: "Price Compression",
            ReasonCode.PROMO_SHOCK: "Promo Shock",
            ReasonCode.COMPETITOR_OOS: "Competitor Out of Stock",
            ReasonCode.COMPETITOR_PROMO: "Competitor Promotion",
            ReasonCode.RANK_DECLINE: "Rank Decline",
            ReasonCode.RANK_SURGE: "Rank Surge",
            ReasonCode.BUYBOX_INSTABILITY: "BuyBox Instability",
            ReasonCode.OOS_ARTIFACT: "OOS Artifact",
            ReasonCode.DEMAND_SHIFT: "Demand Shift",
        }.get(self, self.value.replace("_", " ").title())

    @property
    def nature(self) -> str:
        """Is this a threat or opportunity?"""
        opportunities = {
            ReasonCode.COMPETITOR_OOS,
            ReasonCode.RANK_SURGE,
            ReasonCode.OOS_ARTIFACT,
        }
        return "opportunity" if self in opportunities else "threat"

    @property
    def emoji(self) -> str:
        """Emoji for quick visual identification."""
        return {
            ReasonCode.PRICE_WAR: "⚔️",
            ReasonCode.PRICE_COMPRESSION: "📉",
            ReasonCode.PROMO_SHOCK: "🏷️",
            ReasonCode.COMPETITOR_OOS: "📦",
            ReasonCode.COMPETITOR_PROMO: "🎯",
            ReasonCode.RANK_DECLINE: "📊",
            ReasonCode.RANK_SURGE: "🚀",
            ReasonCode.BUYBOX_INSTABILITY: "🔄",
            ReasonCode.OOS_ARTIFACT: "👻",
            ReasonCode.DEMAND_SHIFT: "📈",
        }.get(self, "❓")


@dataclass
class ReasonCodeConfig:
    """
    Configuration for a reason code including:
    - Base severity (adjusted by category module)
    - Confidence calculation inputs
    - Default action template mapping
    """
    code: ReasonCode
    base_severity: float               # 0.0-1.0
    default_confidence: float          # 0.0-1.0 (when evidence is ambiguous)
    action_template_id: str
    evidence_requirements: List[str]   # TriggerEvent types that support this code


# V1 Reason Code Configurations
REASON_CODE_CONFIGS: Dict[ReasonCode, ReasonCodeConfig] = {
    ReasonCode.PRICE_WAR: ReasonCodeConfig(
        code=ReasonCode.PRICE_WAR,
        base_severity=0.85,
        default_confidence=0.7,
        action_template_id="investigate_price_war",
        evidence_requirements=["price_war_active"],
    ),
    ReasonCode.PRICE_COMPRESSION: ReasonCodeConfig(
        code=ReasonCode.PRICE_COMPRESSION,
        base_severity=0.70,
        default_confidence=0.6,
        action_template_id="monitor_price_compression",
        evidence_requirements=["price_war_active", "competitor_price_decrease"],
    ),
    ReasonCode.PROMO_SHOCK: ReasonCodeConfig(
        code=ReasonCode.PROMO_SHOCK,
        base_severity=0.65,
        default_confidence=0.7,
        action_template_id="assess_promo_response",
        evidence_requirements=["competitor_price_decrease", "price_drop"],
    ),
    ReasonCode.COMPETITOR_OOS: ReasonCodeConfig(
        code=ReasonCode.COMPETITOR_OOS,
        base_severity=0.80,
        default_confidence=0.85,
        action_template_id="conquest_oos",
        evidence_requirements=["competitor_oos_imminent", "amazon_supply_unstable"],
    ),
    ReasonCode.COMPETITOR_PROMO: ReasonCodeConfig(
        code=ReasonCode.COMPETITOR_PROMO,
        base_severity=0.60,
        default_confidence=0.6,
        action_template_id="monitor_competitor_promo",
        evidence_requirements=["competitor_price_decrease"],
    ),
    ReasonCode.RANK_DECLINE: ReasonCodeConfig(
        code=ReasonCode.RANK_DECLINE,
        base_severity=0.75,
        default_confidence=0.8,
        action_template_id="diagnose_rank_drop",
        evidence_requirements=["rank_degradation", "rank_volatility_high"],
    ),
    ReasonCode.RANK_SURGE: ReasonCodeConfig(
        code=ReasonCode.RANK_SURGE,
        base_severity=0.50,  # Lower severity (opportunity, not threat)
        default_confidence=0.8,
        action_template_id="capitalize_momentum",
        evidence_requirements=["momentum_acceleration", "momentum_sustained"],
    ),
    ReasonCode.BUYBOX_INSTABILITY: ReasonCodeConfig(
        code=ReasonCode.BUYBOX_INSTABILITY,
        base_severity=0.90,
        default_confidence=0.9,
        action_template_id="stabilize_buybox",
        evidence_requirements=["buybox_share_collapse"],
    ),
    ReasonCode.OOS_ARTIFACT: ReasonCodeConfig(
        code=ReasonCode.OOS_ARTIFACT,
        base_severity=0.55,
        default_confidence=0.5,  # Hard to distinguish from real demand
        action_template_id="verify_oos_artifact",
        evidence_requirements=["platform_amazon_oos_pattern", "platform_backorder_active"],
    ),
    ReasonCode.DEMAND_SHIFT: ReasonCodeConfig(
        code=ReasonCode.DEMAND_SHIFT,
        base_severity=0.65,
        default_confidence=0.6,
        action_template_id="analyze_demand_shift",
        evidence_requirements=["platform_algorithm_shift", "share_of_voice_lost"],
    ),
}


def get_reason_code_config(code: ReasonCode) -> ReasonCodeConfig:
    """Get configuration for a reason code."""
    return REASON_CODE_CONFIGS.get(code, REASON_CODE_CONFIGS[ReasonCode.DEMAND_SHIFT])


def map_event_type_to_reason_code(event_type: str) -> Optional[ReasonCode]:
    """
    Map a TriggerEvent type to the most appropriate ReasonCode.

    This is the bridge between detector output and episode classification.
    """
    mapping = {
        # Price-related
        "price_war_active": ReasonCode.PRICE_WAR,
        "price_drop": ReasonCode.PROMO_SHOCK,
        "price_spike": ReasonCode.PRICE_COMPRESSION,

        # Competition-related
        "competitor_oos_imminent": ReasonCode.COMPETITOR_OOS,
        "amazon_supply_unstable": ReasonCode.COMPETITOR_OOS,
        "competitor_price_decrease": ReasonCode.COMPETITOR_PROMO,
        "new_competitor_entered": ReasonCode.COMPETITOR_PROMO,

        # Rank-related
        "rank_degradation": ReasonCode.RANK_DECLINE,
        "rank_volatility_high": ReasonCode.RANK_DECLINE,
        "momentum_acceleration": ReasonCode.RANK_SURGE,
        "momentum_sustained": ReasonCode.RANK_SURGE,

        # BuyBox-related
        "buybox_share_collapse": ReasonCode.BUYBOX_INSTABILITY,

        # Platform-related
        "platform_algorithm_shift": ReasonCode.DEMAND_SHIFT,
        "platform_amazon_oos_pattern": ReasonCode.OOS_ARTIFACT,
        "platform_backorder_active": ReasonCode.OOS_ARTIFACT,
        "platform_amazon_takeover": ReasonCode.DEMAND_SHIFT,
        "platform_amazon_retreat": ReasonCode.COMPETITOR_OOS,
        "share_of_voice_lost": ReasonCode.DEMAND_SHIFT,
        "share_of_voice_gained": ReasonCode.RANK_SURGE,
    }

    return mapping.get(event_type)


def calculate_severity(
    reason_code: ReasonCode,
    event_severity: int,
    category_weight: float = 1.0
) -> float:
    """
    Calculate normalized severity score (0.0-1.0).

    Args:
        reason_code: The classified reason code
        event_severity: Raw severity from TriggerEvent (1-10)
        category_weight: Modifier from category module (e.g., 1.2 for skincare price wars)

    Returns:
        Normalized severity score
    """
    config = get_reason_code_config(reason_code)

    # Normalize event severity to 0-1
    normalized_event = event_severity / 10.0

    # Weighted average of base severity and event severity
    raw_severity = (config.base_severity * 0.4) + (normalized_event * 0.6)

    # Apply category modifier
    adjusted = raw_severity * category_weight

    # Clamp to 0-1
    return min(1.0, max(0.0, adjusted))


def calculate_confidence(
    reason_code: ReasonCode,
    evidence_count: int,
    evidence_consistency: float = 1.0
) -> float:
    """
    Calculate confidence score (0.0-1.0).

    Confidence increases with:
    - More evidence (TriggerEvents supporting this code)
    - Higher consistency (events all point same direction)

    Args:
        reason_code: The classified reason code
        evidence_count: Number of supporting TriggerEvents
        evidence_consistency: 0-1 measure of how aligned the evidence is

    Returns:
        Confidence score
    """
    config = get_reason_code_config(reason_code)

    # Base confidence from config
    base = config.default_confidence

    # Evidence count bonus (diminishing returns)
    evidence_bonus = min(0.2, evidence_count * 0.05)

    # Consistency multiplier
    consistency_factor = 0.8 + (evidence_consistency * 0.2)

    confidence = (base + evidence_bonus) * consistency_factor

    # Clamp to 0-1
    return min(1.0, max(0.0, confidence))
