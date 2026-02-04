"""
Skincare Category Module Configuration

V1 MVP Module: Beauty → Facial Skincare (Serums + Moisturizers)

This module defines:
- Thresholds tuned for skincare dynamics (high margins, premium positioning)
- Reason code weights (price wars hurt more in high-margin categories)
- Action template mappings specific to skincare

Key skincare characteristics:
- High margins (50-70% gross)
- Premium pricing common
- Brand loyalty matters
- Review quality critical
- Seasonal patterns (holiday gifting, summer skincare)
"""

SKINCARE_CONFIG = {
    # Module identity
    "module_id": "skincare_serum_moisturizer",
    "display_name": "Beauty → Facial Skincare (Serums + Moisturizers)",
    "description": "High-margin skincare products including serums, moisturizers, and treatments",

    # Amazon category IDs (for filtering)
    "amazon_category_ids": [
        11060451,   # Skin Care
        11060711,   # Face Serums
        11060691,   # Face Moisturizers
        11060671,   # Face Creams
    ],

    # Detector thresholds (tuned for skincare)
    "thresholds": {
        # Price war detection
        "price_war_min_drops": 3,              # Number of drops to trigger
        "price_war_total_decline_pct": 15.0,   # Total decline to trigger (%)
        "price_war_lookback_days": 7,          # Window for detection

        # Rank momentum
        "rank_change_significant_pct": 30.0,   # BSR change to flag (%)
        "rank_volatility_threshold": 0.5,      # Coefficient of variation

        # BuyBox
        "buybox_collapse_threshold": 0.50,     # Below this = collapse
        "buybox_stable_threshold": 0.80,       # Above this = stable

        # Competition
        "new_competitor_bsr_threshold": 10000, # New entrant with BSR below this = threat
        "seller_consolidation_threshold": -3,  # Sellers leaving

        # OOS
        "oos_count_trigger": 3,                # OOS events to flag
        "oos_pct_critical": 0.20,              # 20%+ OOS = critical
    },

    # Reason code weights (category-specific)
    # > 1.0 = more important in skincare
    # < 1.0 = less alarming in skincare
    "reason_code_weights": {
        "price_war": 1.2,           # High margin category, price wars hurt more
        "price_compression": 1.1,   # Erosion matters
        "promo_shock": 0.9,         # Common in skincare, less alarming
        "competitor_oos": 1.1,      # Good conquest opportunity
        "competitor_promo": 0.9,    # Common
        "rank_decline": 1.0,        # Standard
        "rank_surge": 1.0,          # Standard
        "buybox_instability": 1.15, # Critical for brand perception
        "oos_artifact": 0.8,        # Less urgent
        "demand_shift": 1.0,        # Standard
    },

    # Action template mappings
    # Can override default template per reason code
    "action_templates": {
        "price_war": "investigate_price_war",       # Use default
        "price_compression": "monitor_price_compression",
        "promo_shock": "assess_promo_response",
        "competitor_oos": "conquest_oos",
        "competitor_promo": "monitor_competitor_promo",
        "rank_decline": "diagnose_rank_drop",
        "rank_surge": "capitalize_momentum",
        "buybox_instability": "stabilize_buybox",
        "oos_artifact": "verify_oos_artifact",
        "demand_shift": "analyze_demand_shift",
    },

    # Skincare-specific guidance (for LLM explanation enhancement)
    "category_context": {
        "margin_profile": "High margin (50-70% gross typical)",
        "price_sensitivity": "Moderate - premium positioning common",
        "review_importance": "Critical - drives purchase decisions",
        "seasonal_patterns": [
            "Q4 holiday gifting peak",
            "Summer sun care awareness",
            "New Year skincare resolutions",
        ],
        "key_competitors": [
            "CeraVe", "La Roche-Posay", "Neutrogena",
            "The Ordinary", "Paula's Choice", "Olay",
        ],
    },

    # Impact estimation modifiers
    "impact_modifiers": {
        "weekly_revenue_multiplier": 1.0,  # No adjustment
        "opportunity_capture_rate": 0.08,  # 8% of competitor OOS revenue
        "threat_impact_rate": 0.15,        # 15% revenue at risk for threats
    },
}


def get_skincare_threshold(threshold_name: str) -> float:
    """Get a specific threshold value."""
    return SKINCARE_CONFIG["thresholds"].get(threshold_name, 0.0)


def get_skincare_weight(reason_code: str) -> float:
    """Get reason code weight for skincare."""
    return SKINCARE_CONFIG["reason_code_weights"].get(reason_code, 1.0)


def get_skincare_action_template(reason_code: str) -> str:
    """Get action template ID for a reason code in skincare."""
    return SKINCARE_CONFIG["action_templates"].get(reason_code, "investigate_price_war")
