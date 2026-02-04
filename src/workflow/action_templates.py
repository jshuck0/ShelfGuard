"""
Action Templates

Standardized action library for Workflow ShelfGuard.

Each action template defines:
- Title (what to do)
- Owner role (who does it)
- Checklist (step-by-step)
- Urgency (when to do it)
- Default due date offset

Templates are deterministic - same Episode type always gets same action.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Urgency(Enum):
    """When to act."""
    IMMEDIATE = "immediate"   # < 24 hours
    THIS_WEEK = "this_week"   # 1-7 days
    MONITOR = "monitor"       # Track and reassess


class OwnerRole(Enum):
    """Who should own this action."""
    BRAND_MANAGER = "Brand Manager"
    OPS = "Operations"
    FINANCE = "Finance"
    MARKETING = "Marketing"
    ECOMMERCE = "E-commerce Manager"


@dataclass
class ActionTemplate:
    """
    A standardized action template.

    Templates are deterministic - same Episode type always gets same action.
    """
    template_id: str
    title: str                             # "Investigate Price War"
    description: str                       # Longer explanation
    owner_role: OwnerRole
    checklist: List[str]                   # Step-by-step actions
    urgency: Urgency
    default_due_offset_days: int           # Days from now
    category_modifier: float = 1.0         # Weight adjustment per category

    def get_urgency_emoji(self) -> str:
        """Get emoji for urgency."""
        return {
            Urgency.IMMEDIATE: "🚨",
            Urgency.THIS_WEEK: "📅",
            Urgency.MONITOR: "👀",
        }.get(self.urgency, "❓")

    def get_owner_short(self) -> str:
        """Short form of owner role."""
        return {
            OwnerRole.BRAND_MANAGER: "BM",
            OwnerRole.OPS: "Ops",
            OwnerRole.FINANCE: "Fin",
            OwnerRole.MARKETING: "Mkt",
            OwnerRole.ECOMMERCE: "Ecom",
        }.get(self.owner_role, "?")


# =============================================================================
# V1 ACTION TEMPLATES
# =============================================================================

ACTION_TEMPLATES: Dict[str, ActionTemplate] = {
    # Price War
    "investigate_price_war": ActionTemplate(
        template_id="investigate_price_war",
        title="Investigate Price War",
        description="Multiple price drops detected. Assess if this is a coordinated price war or isolated event.",
        owner_role=OwnerRole.BRAND_MANAGER,
        urgency=Urgency.IMMEDIATE,
        default_due_offset_days=3,
        checklist=[
            "Identify which competitors are dropping prices",
            "Check competitor inventory levels (are they liquidating?)",
            "Review your current margin at new price points",
            "Decide: match, hold, or strategic retreat",
            "If matching: update pricing in Seller Central",
            "Document decision and rationale",
        ],
    ),

    # Price Compression
    "monitor_price_compression": ActionTemplate(
        template_id="monitor_price_compression",
        title="Monitor Price Compression",
        description="Price floor is eroding. Track trend and prepare contingency.",
        owner_role=OwnerRole.FINANCE,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=7,
        checklist=[
            "Calculate margin impact at current trajectory",
            "Identify price floor (minimum viable margin)",
            "Set price alert triggers for next drop",
            "Prepare memo for leadership if action needed",
        ],
    ),

    # Promo Shock
    "assess_promo_response": ActionTemplate(
        template_id="assess_promo_response",
        title="Assess Promo Response",
        description="Competitor launched promotion. Evaluate if response is needed.",
        owner_role=OwnerRole.MARKETING,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=5,
        checklist=[
            "Confirm competitor promo is live (check listing)",
            "Estimate promo duration (typical patterns)",
            "Calculate potential share loss if no response",
            "Evaluate counter-promo options (coupon, deal, PPC)",
            "If responding: launch within 48 hours",
        ],
    ),

    # Competitor OOS
    "conquest_oos": ActionTemplate(
        template_id="conquest_oos",
        title="Conquest OOS Opportunity",
        description="Competitor is out of stock. Capture their customers before they restock.",
        owner_role=OwnerRole.ECOMMERCE,
        urgency=Urgency.IMMEDIATE,
        default_due_offset_days=2,
        checklist=[
            "Verify competitor OOS status",
            "Increase PPC bids on competitor keywords (+30-50%)",
            "Check your inventory - can you absorb demand spike?",
            "Consider temporary price increase (capture margin)",
            "Monitor competitor restock signals",
            "Scale back when competitor restocks",
        ],
    ),

    # Competitor Promo
    "monitor_competitor_promo": ActionTemplate(
        template_id="monitor_competitor_promo",
        title="Monitor Competitor Promo",
        description="Competitor launched promotion. Track impact on your sales.",
        owner_role=OwnerRole.BRAND_MANAGER,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=7,
        checklist=[
            "Note promo type and discount level",
            "Track your daily sales for comparison",
            "Set reminder to check when promo likely ends",
            "Document for competitive intelligence file",
        ],
    ),

    # Rank Decline
    "diagnose_rank_drop": ActionTemplate(
        template_id="diagnose_rank_drop",
        title="Diagnose Rank Drop",
        description="BSR is declining. Identify root cause before it accelerates.",
        owner_role=OwnerRole.ECOMMERCE,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=5,
        checklist=[
            "Check recent price changes (yours and competitors)",
            "Review conversion rate in Business Reports",
            "Check for suppressed listing or content issues",
            "Review recent reviews (negative feedback spike?)",
            "Check inventory availability and Buy Box %",
            "Identify primary driver and address",
        ],
    ),

    # Rank Surge
    "capitalize_momentum": ActionTemplate(
        template_id="capitalize_momentum",
        title="Capitalize on Momentum",
        description="BSR is improving. Ensure you can sustain and amplify the growth.",
        owner_role=OwnerRole.BRAND_MANAGER,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=7,
        checklist=[
            "Verify inventory can support continued growth",
            "Consider increasing PPC spend to amplify",
            "Check if price increase is viable (test +5%)",
            "Ensure review collection is optimized",
            "Document what drove the improvement",
        ],
    ),

    # BuyBox Instability
    "stabilize_buybox": ActionTemplate(
        template_id="stabilize_buybox",
        title="Stabilize Buy Box",
        description="Buy Box share is unstable. Identify cause and restore control.",
        owner_role=OwnerRole.OPS,
        urgency=Urgency.IMMEDIATE,
        default_due_offset_days=2,
        checklist=[
            "Check who is winning Buy Box (Seller Central)",
            "Identify if new seller or existing seller undercutting",
            "Review your pricing vs competitors",
            "Check fulfillment method (FBA vs FBM impact)",
            "If unauthorized seller: escalate to Brand Registry",
            "If price issue: adjust or match",
        ],
    ),

    # OOS Artifact
    "verify_oos_artifact": ActionTemplate(
        template_id="verify_oos_artifact",
        title="Verify OOS Artifact",
        description="Platform OOS detected. Confirm if this is a data artifact or real supply issue.",
        owner_role=OwnerRole.OPS,
        urgency=Urgency.THIS_WEEK,
        default_due_offset_days=3,
        checklist=[
            "Check actual inventory in Seller Central",
            "Check if listing is suppressed or inactive",
            "Verify no backend catalog issues",
            "If artifact: document for future reference",
            "If real OOS: expedite replenishment",
        ],
    ),

    # Demand Shift
    "analyze_demand_shift": ActionTemplate(
        template_id="analyze_demand_shift",
        title="Analyze Demand Shift",
        description="Organic demand pattern changed. Investigate if trend or anomaly.",
        owner_role=OwnerRole.BRAND_MANAGER,
        urgency=Urgency.MONITOR,
        default_due_offset_days=14,
        checklist=[
            "Compare to prior year same period (seasonality?)",
            "Check category-level trends (macro shift?)",
            "Review search volume trends (Google Trends)",
            "Assess if product lifecycle issue",
            "Document findings for quarterly review",
        ],
    ),
}


def get_action_template(template_id: str) -> Optional[ActionTemplate]:
    """Get action template by ID."""
    return ACTION_TEMPLATES.get(template_id)


def get_all_templates() -> List[ActionTemplate]:
    """Get all action templates."""
    return list(ACTION_TEMPLATES.values())


def get_template_for_reason_code(reason_code: str) -> Optional[ActionTemplate]:
    """
    Get the default action template for a reason code.

    Mapping is defined in reason_codes.py; this is a convenience function.
    """
    from src.workflow.reason_codes import ReasonCode, get_reason_code_config

    try:
        code = ReasonCode(reason_code)
        config = get_reason_code_config(code)
        return get_action_template(config.action_template_id)
    except (ValueError, KeyError):
        return None
