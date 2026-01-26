# src/analyst/config.py

"""
KEEPA_CONFIG - The Complete Physics of the Amazon 3P Marketplace

This file is the "Dictionary" that teaches the AI to speak Amazon fluently.
It defines:
1. What metrics mean (Targets, Levers, Flags)
2. Which direction is "good" (Directionality)
3. How things cause other things (Causal Chains)
4. What thresholds trigger alerts (Thresholds)
5. What formulas convert raw data (Formulas)

ARCHITECTURE:
- This is the PRIOR (Global defaults / Common sense)
- calibrator.py calculates the POSTERIOR (Product-specific physics)
- The Agent combines both to make decisions
"""

from typing import Dict, Any

# ==============================================================================
# MASTER CONFIG: THE PHYSICS OF THE AMAZON 3P MARKETPLACE
# ==============================================================================

KEEPA_CONFIG: Dict[str, Any] = {
    
    # ==========================================================================
    # BUCKET 1: TARGETS (The Health Vitals)
    # These are the OUTCOMES we care about. The Profiler checks these for anomalies.
    # ==========================================================================
    "TARGETS": {
        "sales_rank": {
            "col": "sales_rank",
            "direction": "lower_is_better",  # CRITICAL: Rank 1 > Rank 1000
            "desc": "Best Seller Rank (BSR)",
            "unit": "ordinal",
            "alert_velocity": 0.20,  # 20% weekly change triggers alert
        },
        "estimated_units": {
            "col": "estimated_units",
            "direction": "higher_is_better",
            "desc": "Est. Units Sold (from BSR formula)",
            "unit": "count",
            "alert_velocity": 0.25,
        },
        "monthly_sold": {
            "col": "monthly_sold",
            "direction": "higher_is_better",
            "desc": "Amazon's Official Monthly Sold Badge",
            "unit": "count",
            "alert_velocity": 0.20,
        },
        "revenue": {
            "col": "weekly_revenue",
            "direction": "higher_is_better",
            "desc": "Weekly Revenue ($)",
            "unit": "currency",
            "alert_velocity": 0.25,
        },
        "rating": {
            "col": "rating",
            "direction": "higher_is_better",
            "desc": "Star Rating (1-5)",
            "unit": "score",
            "alert_velocity": 0.05,  # Rating moves slowly - 0.05 star change is notable
            "critical_threshold": 4.0,  # Below 4.0 = Conversion death
        },
        "review_count": {
            "col": "review_count",
            "direction": "higher_is_better",
            "desc": "Total Review Count",
            "unit": "count",
            "alert_velocity": 0.10,
        },
    },
    
    # ==========================================================================
    # BUCKET 2: LEVERS (The Causal Drivers)
    # These are the INPUTS that cause Targets to move.
    # The Causal Sensor scans these columns to find "why" something happened.
    # ==========================================================================
    "LEVERS": {
        # --- PRICE FORCES ---
        "buy_box_price": {
            "col": "buy_box_price",
            "type": "price",
            "desc": "The price customers actually pay",
            "causality": "inverse",  # Lower price -> Higher sales (usually)
        },
        "new_price": {
            "col": "new_price",
            "type": "price", 
            "desc": "Lowest 3P New price (market floor)",
            "causality": "inverse",
        },
        "amazon_price": {
            "col": "amazon_price",
            "type": "price",
            "desc": "Amazon 1P price (market ceiling)",
            "causality": "anchor",  # Sets consumer expectations
        },
        "filled_price": {
            "col": "filled_price",
            "type": "price",
            "desc": "Best available price (cascaded priority)",
            "causality": "inverse",
        },
        
        # --- SUPPLY FORCES ---
        "new_offer_count": {
            "col": "new_offer_count",
            "type": "competition",
            "desc": "Number of 3P sellers",
            "causality": "inverse",  # More sellers -> Price war -> Lower margin
            "spike_threshold": 3,  # +3 sellers in a week = potential hijacker
        },
        "oos_pct_30": {
            "col": "oos_pct_30",
            "type": "availability",
            "desc": "Out of Stock % last 30 days",
            "causality": "inverse",  # Higher OOS -> Lost sales
            "critical_threshold": 0.10,  # 10% OOS is bad
        },
        "oos_pct_90": {
            "col": "oos_pct_90",
            "type": "availability",
            "desc": "Out of Stock % last 90 days",
            "causality": "inverse",
        },
        
        # --- MARKETPLACE FORCES ---
        "amazon_bb_share": {
            "col": "bb_stats_amazon_90",
            "type": "platform",
            "desc": "Amazon 1P Buy Box Share (90 day)",
            "causality": "inverse",  # Higher Amazon share -> Less 3P opportunity
            "warning_threshold": 0.40,  # 40% = Competitive pressure
            "critical_threshold": 0.80,  # 80% = Amazon dominance
        },
        "fba_bb_share": {
            "col": "bb_stats_fba_90",
            "type": "platform",
            "desc": "FBA Seller Buy Box Share (90 day)",
            "causality": "positive",  # FBA sellers winning is good for 3P ecosystem
        },
        
        # --- CONTENT FORCES ---
        "has_aplus": {
            "col": "has_aplus",
            "type": "content",
            "desc": "Has A+ Content",
            "causality": "positive",  # A+ boosts conversion
        },
        "has_video": {
            "col": "has_video",
            "type": "content",
            "desc": "Has Product Video",
            "causality": "positive",
        },
        "image_count": {
            "col": "image_count",
            "type": "content",
            "desc": "Number of Product Images",
            "causality": "positive",
            "min_optimal": 5,  # Best practice is 5-7 images
        },
    },
    
    # ==========================================================================
    # BUCKET 3: FLAGS (Binary State Indicators)
    # These are TRUE/FALSE states that require immediate attention.
    # ==========================================================================
    "FLAGS": {
        "is_amazon_choice": {
            "col": "is_amazon_choice",
            "positive": True,
            "desc": "Amazon's Choice Badge",
        },
        "is_bestseller": {
            "col": "is_bestseller",
            "positive": True,
            "desc": "Best Seller Badge",
        },
        "is_fba": {
            "col": "is_fba",
            "positive": True,  # For 3P sellers, FBA is usually advantageous
            "desc": "Fulfilled by Amazon",
        },
        "is_variation": {
            "col": "is_variation",
            "positive": None,  # Neutral - just informational
            "desc": "Part of a variation family",
        },
        "is_adult": {
            "col": "is_adult",
            "positive": False,  # Limits advertising
            "desc": "Adult product flag",
        },
        "is_sns": {
            "col": "is_sns",
            "positive": True,  # Subscribe & Save = Recurring revenue
            "desc": "Subscribe & Save eligible",
        },
    },
    
    # ==========================================================================
    # BUCKET 4: THRESHOLDS (When to Alert)
    # These define the numerical boundaries for significance testing.
    # ==========================================================================
    "THRESHOLDS": {
        # --- Significance Detection ---
        "price_change_pct": 0.03,      # 3% price change is significant
        "rank_change_pct": 0.10,       # 10% rank change is significant
        "review_delta": 5,              # 5+ new reviews in a period
        "bb_share_change": 0.10,       # 10% Buy Box shift
        "seller_spike": 3,              # +3 sellers = hijacker alert
        
        # --- Severity Levels ---
        "severity": {
            "minor": 0.05,    # 5% change
            "moderate": 0.15, # 15% change
            "major": 0.30,    # 30% change
            "critical": 0.50, # 50% change
        },
        
        # --- Business Thresholds ---
        "rating_danger_zone": 4.0,     # Below 4.0 = Conversion cliff
        "oos_warning": 0.05,           # 5% OOS
        "oos_critical": 0.20,          # 20% OOS = Emergency
        "amazon_threat_warning": 0.40, # 40% Amazon BB share
        "amazon_threat_critical": 0.80,# 80% = Category dominated
        
        # --- Velocity Thresholds (Weekly) ---
        "velocity_crash": -0.30,       # 30% drop in units
        "velocity_surge": 0.50,        # 50% increase in units
    },
    
    # ==========================================================================
    # BUCKET 5: CAUSAL CHAINS (If X then Y)
    # These define the expected relationships between Levers and Targets.
    # Used by the Causal Sensor to validate hypotheses.
    # ==========================================================================
    "CAUSAL_CHAINS": {
        # --- POSITIVE CHAINS (Good outcomes) ---
        "price_elasticity_success": {
            "desc": "Price Cut drove Volume (Expected Behavior)",
            "cause": ("buy_box_price", "decrease", 0.03),  # 3% price drop
            "effect": ("sales_rank", "decrease", 0.10),    # 10% rank improvement
            "confidence": "HIGH",
            "mechanism": "Lower price -> Higher conversion -> More sales -> Lower BSR",
        },
        "review_momentum": {
            "desc": "Review Growth drove Trust",
            "cause": ("review_count", "increase", 5),
            "effect": ("sales_rank", "decrease", 0.05),
            "confidence": "MEDIUM",
            "mechanism": "More reviews -> Higher trust -> Better conversion",
        },
        "competitor_attrition": {
            "desc": "Competitors left, we captured share",
            "cause": ("new_offer_count", "decrease", 2),
            "effect": ("sales_rank", "decrease", 0.15),
            "confidence": "HIGH",
            "mechanism": "Fewer competitors -> Less price pressure -> Buy Box stability",
        },
        "stock_recovery": {
            "desc": "Inventory restored, sales resumed",
            "cause": ("oos_pct_30", "decrease", 0.10),
            "effect": ("estimated_units", "increase", 0.20),
            "confidence": "HIGH",
            "mechanism": "Back in stock -> Can fulfill orders -> Sales resume",
        },
        
        # --- NEGATIVE CHAINS (Bad outcomes) ---
        "price_elasticity_failure": {
            "desc": "Price Cut FAILED to improve Rank (Inelastic Product)",
            "cause": ("buy_box_price", "decrease", 0.10),
            "effect": ("sales_rank", "stable", 0.05),  # Rank didn't move
            "confidence": "HIGH",
            "flag": "INELASTIC_PRODUCT",
            "mechanism": "Price insensitive demand - Brand loyalty or necessity product",
        },
        "rating_death_spiral": {
            "desc": "Bad Reviews killed Conversion",
            "cause": ("rating", "decrease", 0.2),  # Drop 0.2 stars
            "effect": ("sales_rank", "increase", 0.20),  # Rank got worse
            "confidence": "HIGH",
            "flag": "QUALITY_CRISIS",
            "mechanism": "Lower rating -> Lower trust -> Lower conversion -> Higher BSR",
        },
        "amazon_takeover": {
            "desc": "Amazon 1P entered and crushed 3P sales",
            "cause": ("amazon_bb_share", "increase", 0.30),
            "effect": ("sales_rank", "increase", 0.25),  # 3P rank got worse
            "confidence": "HIGH",
            "flag": "AMAZON_THREAT",
            "mechanism": "Amazon wins Buy Box -> 3P loses visibility -> Sales crash",
        },
        "hijacker_invasion": {
            "desc": "New sellers flooded the listing",
            "cause": ("new_offer_count", "increase", 5),
            "effect": ("buy_box_price", "decrease", 0.15),
            "confidence": "HIGH",
            "flag": "PRICE_WAR",
            "mechanism": "More sellers -> Price competition -> Race to bottom",
        },
        "stockout_cascade": {
            "desc": "Inventory outage killed momentum",
            "cause": ("oos_pct_30", "increase", 0.15),
            "effect": ("sales_rank", "increase", 0.30),  # Rank crashed
            "confidence": "HIGH",
            "flag": "SUPPLY_CRISIS",
            "mechanism": "OOS -> Lost sales -> Algorithm penalizes -> Rank crashes",
        },
        "competitor_undercut": {
            "desc": "Competitor dropped price, stole sales",
            "cause": ("new_price", "decrease", 0.10),  # Market price dropped
            "effect": ("sales_rank", "increase", 0.15),  # Our rank got worse
            "confidence": "MEDIUM",
            "condition": "Only valid if buy_box_price stayed flat (we didn't match)",
            "mechanism": "Competitor undercuts -> Loses Buy Box -> Loses sales",
        },
    },
    
    # ==========================================================================
    # BUCKET 6: FORMULAS (Mathematical Conversions)
    # These are the exact formulas used to derive metrics.
    # ==========================================================================
    "FORMULAS": {
        "bsr_to_units": {
            "name": "BSR to Monthly Units (Power Law)",
            "formula": "units = 145000 * (sales_rank ^ -0.9)",
            "constant": 145000,
            "exponent": -0.9,
            "note": "Derived from top-category calibration. Adjust constant per category.",
            "python": "estimated_units = 145000 * (sales_rank ** -0.9)",
        },
        "weekly_revenue": {
            "name": "Weekly Revenue Calculation",
            "formula": "revenue = filled_price * estimated_units / 4.33",
            "weeks_per_month": 4.33,
            "python": "weekly_revenue = filled_price * (estimated_units / 4.33)",
        },
        "price_gap": {
            "name": "Price Gap vs Market",
            "formula": "gap = (my_price - market_price) / market_price",
            "interpretation": {
                "positive": "Premium pricing",
                "negative": "Discount pricing",
                "zero": "Market rate",
            },
            "python": "price_gap_pct = (buy_box_price - new_price) / new_price",
        },
        "velocity_trend": {
            "name": "Week-over-Week Velocity",
            "formula": "velocity = (current_units - prev_units) / prev_units",
            "python": "velocity_pct = df['estimated_units'].pct_change()",
        },
        "referral_fee": {
            "name": "Amazon Referral Fee (Category Dependent)",
            "default_rate": 0.15,  # 15% for most categories
            "beauty_rate": 0.08,   # 8% for Beauty first $10
            "grocery_rate": 0.08,  # 8% for Grocery first $15
            "python": "referral_fee = price * 0.15",
        },
    },
    
    # ==========================================================================
    # BUCKET 7: PHASE TRANSITIONS (State Changes)
    # These define when a product moves from one strategic state to another.
    # ==========================================================================
    "PHASE_TRANSITIONS": {
        "launch_to_growth": {
            "trigger": "review_count > 50 AND rating > 4.0",
            "from_state": "LAUNCH",
            "to_state": "GROWTH",
            "desc": "Product has enough reviews to scale",
        },
        "growth_to_maturity": {
            "trigger": "rank_velocity < 0.05 AND rank < 1000",
            "from_state": "GROWTH",
            "to_state": "MATURITY",
            "desc": "Rank stabilized at high level",
        },
        "maturity_to_decline": {
            "trigger": "rank_velocity > 0.10 for 4 consecutive weeks",
            "from_state": "MATURITY",
            "to_state": "DECLINE",
            "desc": "Consistent rank degradation",
        },
        "decline_to_crisis": {
            "trigger": "rating < 4.0 OR oos_pct > 0.20",
            "from_state": "DECLINE",
            "to_state": "CRISIS",
            "desc": "Quality or supply emergency",
        },
    },
    
    # ==========================================================================
    # BUCKET 8: DEFAULT ELASTICITIES (Bayesian Priors)
    # These are the starting assumptions. calibrator.py will update them.
    # ==========================================================================
    "DEFAULT_ELASTICITIES": {
        "price_elasticity": {
            "prior": -1.5,  # 1% price drop -> 1.5% unit increase (elastic)
            "range": (-3.0, 0.0),  # All products should have negative elasticity
            "confidence": "LOW",  # Will be calibrated
        },
        "review_elasticity": {
            "prior": 0.05,  # 1% more reviews -> 0.05% more sales
            "range": (0.0, 0.2),
            "confidence": "MEDIUM",
        },
        "competition_elasticity": {
            "prior": -0.10,  # 1 more seller -> 0.1% sales drop
            "range": (-0.5, 0.0),
            "confidence": "LOW",
        },
        "amazon_impact": {
            "prior": -0.30,  # Amazon taking Buy Box -> 30% sales drop
            "range": (-0.8, 0.0),
            "confidence": "MEDIUM",
        },
    },
    
    # ==========================================================================
    # BUCKET 9: COLUMN MAPPINGS (Raw -> Processed)
    # Maps the actual column names in df_weekly to their roles.
    # ==========================================================================
    "COLUMN_MAP": {
        # Time-series columns (from build_keepa_weekly_table)
        "week_start": "week_start",
        "sales_rank": "sales_rank",
        "amazon_price": "amazon_price",
        "new_price": "new_price",
        "buy_box_price": "buy_box_price",
        "filled_price": "filled_price",
        "new_offer_count": "new_offer_count",
        "rating": "rating",
        "review_count": "review_count",
        "estimated_units": "estimated_units",
        "weekly_revenue": "weekly_revenue",
        
        # Product-level columns (from seed discovery)
        "title": "title",
        "brand": "brand",
        "category": "category",
        "parent_asin": "parent_asin",
        "image_count": "image_count",
        "has_aplus": "has_aplus",
        "has_video": "has_video",
        "is_fba": "is_fba",
        "is_sns": "is_sns",
        "monthly_sold": "monthly_sold",
        "oos_pct_30": "oos_pct_30",
        "oos_pct_90": "oos_pct_90",
        "bb_stats_amazon_90": "bb_stats_amazon_90",
        "bb_stats_fba_90": "bb_stats_fba_90",
    },
    
    # ==========================================================================
    # BUCKET 10: PROFILER DIRECTIVES
    # Tells the Profiler exactly what to check and in what order.
    # ==========================================================================
    "PROFILER_CHECKS": {
        "order": [
            "data_health",      # Check for missing data first
            "stationarity",     # Is the data stable enough to analyze?
            "seasonality",      # Are there predictable patterns?
            "distribution",     # Are there outliers?
            "trend",            # What direction is the product heading?
        ],
        "data_health": {
            "min_weeks": 4,     # Need at least 4 weeks of data
            "max_missing_pct": 0.20,  # Max 20% missing values
        },
        "stationarity": {
            "test": "ADF",      # Augmented Dickey-Fuller
            "p_threshold": 0.05,
        },
        "seasonality": {
            "period": 4,        # Check for monthly (4-week) cycles
            "min_amplitude": 0.10,  # 10% seasonal swing to be notable
        },
    },
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_target_direction(metric_name: str) -> str:
    """Returns 'lower_is_better' or 'higher_is_better' for a target metric."""
    targets = KEEPA_CONFIG.get("TARGETS", {})
    if metric_name in targets:
        return targets[metric_name].get("direction", "higher_is_better")
    return "higher_is_better"  # Default assumption


def get_threshold(threshold_name: str) -> float:
    """Returns a threshold value from config."""
    thresholds = KEEPA_CONFIG.get("THRESHOLDS", {})
    return thresholds.get(threshold_name, 0.10)  # Default 10%


def get_causal_chain(chain_name: str) -> dict:
    """Returns a causal chain definition."""
    chains = KEEPA_CONFIG.get("CAUSAL_CHAINS", {})
    return chains.get(chain_name, {})


def is_improvement(metric_name: str, old_value: float, new_value: float) -> bool:
    """
    Determines if a change represents an improvement, respecting directionality.
    
    Example:
        is_improvement("sales_rank", 1000, 500) -> True (rank improved)
        is_improvement("rating", 4.0, 4.5) -> True (rating improved)
    """
    direction = get_target_direction(metric_name)
    if direction == "lower_is_better":
        return new_value < old_value
    else:
        return new_value > old_value


def calculate_change_severity(change_pct: float) -> str:
    """Categorizes a percentage change into severity levels."""
    abs_change = abs(change_pct)
    severity = KEEPA_CONFIG["THRESHOLDS"]["severity"]
    
    if abs_change >= severity["critical"]:
        return "CRITICAL"
    elif abs_change >= severity["major"]:
        return "MAJOR"
    elif abs_change >= severity["moderate"]:
        return "MODERATE"
    elif abs_change >= severity["minor"]:
        return "MINOR"
    else:
        return "NEGLIGIBLE"
