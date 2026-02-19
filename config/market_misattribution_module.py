"""
Market Misattribution Shield — Regime Detector Config
======================================================
All thresholds for the 5 regime detectors live here.
Tune these values against your golden brand arena before shipping.

Design rules:
- Every threshold has a comment explaining what it does and when to tune it
- Thresholds are conservative by default (prefer fewer false positives)
- Confidence rubric parameters are separate from regime thresholds
- Nothing in here should require an LLM to interpret
"""

from dataclasses import dataclass, field
from typing import Dict


# ─── REGIME DETECTOR THRESHOLDS ───────────────────────────────────────────────

REGIME_THRESHOLDS: Dict = {

    # ── A: TIER COMPRESSION / TIER SHIFT ──────────────────────────────────────
    # Detects when the market-wide price floor has dropped, compressing margins
    # across the arena regardless of your own pricing actions.
    "tier_compression": {
        # Base window for computing price baseline (days)
        # Use 28 days to smooth out single-week promo noise
        "base_window_days": 28,

        # Minimum % decline in tier median price_per_unit to flag compression
        # vs the base window (e.g. 0.05 = 5% cheaper than 28-day median)
        "compression_threshold_pct": 0.05,

        # How many consecutive weeks the tier must be compressed to confirm regime
        # 1 week = noise; 2+ weeks = structural shift
        "persistence_weeks": 2,

        # Minimum fraction of the arena that must be discounted to call it tier-wide
        # (vs just one competitor discounting)
        "arena_coverage_min": 0.30,  # 30% of tracked ASINs

        # Confidence: upgrade to HIGH if coverage exceeds this
        "arena_coverage_high": 0.50,
    },

    # ── B: PROMO WAR PROXY ────────────────────────────────────────────────────
    # Detects synchronized discounting across multiple competitors —
    # the signal that someone fired first and others are matching.
    "promo_war": {
        # Minimum % price drop (vs each ASIN's own 28-day base) to count as "discounted"
        "discount_threshold_pct": 0.07,   # 7% below own base = promotional

        # Minimum number of distinct brands (not ASINs) discounting simultaneously
        # to call it a promo war (not just one brand doing a sale)
        "min_brands_discounting": 3,

        # Rank gain concentrated check:
        # at least this fraction of rank gainers must be in the discounted set
        "rank_gain_from_discounters_min": 0.50,

        # Lookback window for detecting synchronization (weeks)
        "sync_window_weeks": 2,

        # Confidence: HIGH if brand count >= this
        "high_confidence_brand_count": 5,
    },

    # ── C: COMPETITOR COMPOUNDING ─────────────────────────────────────────────
    # Detects a specific competitor (or set) gaining persistently while your
    # BSR deteriorates — the signal that a rival is actually eating your share.
    "competitor_compounding": {
        # Minimum BSR improvement (%) for a competitor ASIN to be "compounding"
        # Negative = better rank (lower BSR number). -0.15 = 15% rank improvement.
        "competitor_rank_gain_pct": -0.15,

        # Competitor must sustain this gain for at least N weeks
        "persistence_weeks": 2,

        # Your own BSR must be declining (or flat) while competitor gains
        # 0.0 = any worsening; 0.05 = only flag if your rank is 5%+ worse
        "your_rank_decline_threshold": 0.0,

        # Maximum price_per_unit premium a compounding competitor can have
        # vs tier median (if they're higher priced AND gaining, it's organic demand)
        # Use None to disable this check
        "max_price_premium_pct": 0.15,  # Skip if competitor is >15% above tier

        # Confidence: HIGH if competitor's rank gain persists 3+ weeks
        "high_confidence_weeks": 3,
    },

    # ── D: DEMAND TAILWIND / HEADWIND ─────────────────────────────────────────
    # Detects category-wide rank movement not explained by individual pricing —
    # organic demand swings (seasonality, macro, Amazon algorithm shift).
    "demand_tailwind": {
        # Fraction of the arena that must show coordinated rank movement
        # to attribute it to demand vs individual actions
        "arena_fraction_min": 0.50,   # Half the arena moving = demand signal

        # Minimum median BSR change across the arena (%) to flag
        # 0.10 = 10% median rank change (either direction)
        "median_rank_change_pct": 0.10,

        # Stability check: skip if any single brand accounts for >40% of rank movers
        # (concentration = competitive event, not demand)
        "max_single_brand_share": 0.40,

        # Lookback window (weeks)
        "lookback_weeks": 4,

        # Confidence: HIGH if 60%+ of arena moves in same direction
        "high_confidence_fraction": 0.60,
    },

    # ── E: NEW ENTRANT / ASSORTMENT SHOCK ─────────────────────────────────────
    # Detects a new ASIN appearing with fast rank acceleration —
    # a new brand/product entering and potentially disrupting the arena.
    "new_entrant": {
        # An ASIN is "new" if first seen within this many days
        "new_asin_days": 56,   # 8 weeks

        # New entrant must reach this BSR threshold to be considered a threat
        # (excludes deep long-tail new entries that don't matter)
        "threat_bsr_threshold": 20000,

        # BSR improvement rate: entrant must improve rank by at least this %
        # per week to signal momentum (not just a slow ramp)
        "rank_velocity_pct_per_week": -0.10,  # 10% better each week

        # Review accumulation proxy: if review_count grows faster than this
        # per week, flag as aggressive launch (ads + vine likely)
        # Set to None to skip if review_count data is sparse
        "review_velocity_weekly": 20,  # 20+ new reviews/week = aggressive

        # Confidence: HIGH if entrant hits both rank + review criteria
    },
}


# ─── ASIN ROLE ASSIGNMENT THRESHOLDS ─────────────────────────────────────────
# Used by features/asin_metrics.py to classify ASINs into roles.

ASIN_ROLE_THRESHOLDS: Dict = {
    # BSR cutoffs for Core / Challenger / Long-tail
    # Core: top sellers that drive category conversation
    "core_bsr_max": 5000,

    # Challenger: competitive but not dominant
    "challenger_bsr_max": 25000,

    # Long-tail: everything else

    # Revenue share: ASIN accounts for this fraction of arena revenue to be "Core"
    # Used as a fallback when BSR is unreliable
    "core_revenue_share_min": 0.03,  # 3% of arena revenue

    # Review count minimum to be considered "established" (not a new entrant)
    "established_review_min": 100,
}


# ─── AD WASTE RISK FLAG THRESHOLDS ───────────────────────────────────────────
# Used to flag per-ASIN "Ad Waste Risk" without implying spend exists.

AD_WASTE_RISK_THRESHOLDS: Dict = {
    # HIGH risk: ASIN price is above tier median by this much (uncompetitive)
    "price_above_tier_pct": 0.10,   # 10% more expensive than tier median

    # HIGH risk: BSR deteriorating by this % or more in last 30d
    "bsr_deterioration_pct": 0.20,

    # HIGH risk: competitor discounting must be persistent (from promo_war config)
    # (uses promo_war.discount_threshold_pct as the reference)

    # LOW risk: ASIN is competitive (at or below tier median) AND rank improving
    "competitive_price_below_tier_pct": 0.05,  # 5% below = competitive
    "rank_improving_pct": -0.05,   # 5% better rank = improving momentum
}


# ─── CONFIDENCE RUBRIC PARAMETERS ────────────────────────────────────────────
# Used by scoring/confidence.py to produce HIGH/MED/LOW per driver.

CONFIDENCE_RUBRIC: Dict = {
    # Minimum fraction of arena ASINs with complete data in last 14 days
    # Below this = downgrade all confidence by one tier
    "data_completeness_min": 0.70,

    # Minimum number of ASINs in the tracked peer set
    # Below this = LOW confidence regardless of signal strength
    "min_peer_set_size": 10,

    # monthly_sold coverage: if fewer than this fraction of ASINs have
    # a non-zero monthly_sold, downgrade any "demand" claims and use rank language
    "monthly_sold_coverage_min": 0.40,

    # Cross-signal agreement: price move AND rank move must align
    # for a confidence upgrade (e.g. promo war: price down + rank up = aligned)
    "cross_signal_agreement_required": True,

    # Pack size sanity: comparable ASINs must be within this factor of each other
    # (e.g. 2.0 = only compare ASINs with pack size within 2x)
    "pack_size_comparability_factor": 2.0,

    # Minimum weeks of data for an ASIN to contribute to confidence scoring
    "min_data_weeks": 4,
}


# ─── BRIEF BANDING ────────────────────────────────────────────────────────────
# Defines how numeric values are expressed as bands (never exact numbers).

BRIEF_BANDS: Dict = {
    # BSR change bands
    "rank_change": [
        (-999999, -0.20, "significant gain"),
        (-0.20, -0.05, "modest gain"),
        (-0.05, 0.05, "roughly flat"),
        (0.05, 0.20, "modest decline"),
        (0.20, 999999, "significant decline"),
    ],
    # Price vs category median bands
    "price_vs_tier": [
        (-999, -0.10, "well below category median"),
        (-0.10, -0.03, "below category median"),
        (-0.03, 0.03, "at category median"),
        (0.03, 0.10, "above category median"),
        (0.10, 999, "well above category median"),
    ],
    # Revenue change bands
    "revenue_change": [
        (-999, -0.20, "down significantly"),
        (-0.20, -0.05, "down modestly"),
        (-0.05, 0.05, "roughly flat"),
        (0.05, 0.20, "up modestly"),
        (0.20, 999, "up significantly"),
    ],
}


def get_regime_threshold(regime: str, key: str, default=None):
    """Get a specific threshold value by regime and key name."""
    return REGIME_THRESHOLDS.get(regime, {}).get(key, default)


def get_confidence_param(key: str, default=None):
    """Get a confidence rubric parameter."""
    return CONFIDENCE_RUBRIC.get(key, default)


def band_value(value: float, band_type: str) -> str:
    """
    Convert a numeric value to its band label.

    Args:
        value: The numeric value (e.g. 0.12 for 12% rank change)
        band_type: One of "rank_change", "price_vs_tier", "revenue_change"

    Returns:
        Band label string (e.g. "modest decline")
    """
    bands = BRIEF_BANDS.get(band_type, [])
    for low, high, label in bands:
        if low <= value < high:
            return label
    return "unknown"


# ─── MODULE TAXONOMY ─────────────────────────────────────────────────────────

# Token set for inferring skincare module from category_path breadcrumb
SKINCARE_MODULE_TOKENS = {
    "skincare", "skin care", "facial", "face moisturizer", "face serum",
    "beauty", "body wash", "lotion", "cleanser", "toner", "sunscreen",
    "moisturizer", "serum",
}

# Keyword → product_type mapping (ordered — first match wins)
# More specific patterns must come before generic ones
PRODUCT_TYPE_KEYWORDS = [
    ("micellar", "micellar water"),
    ("retinol", "retinol"),
    ("exfoliant", "exfoliant"),
    ("exfoliating", "exfoliant"),
    (" aha ", "exfoliant"),
    (" bha ", "exfoliant"),
    ("glycolic", "exfoliant"),
    ("salicylic", "exfoliant"),
    ("sunscreen", "sunscreen"),
    (" spf ", "sunscreen"),
    ("sun protect", "sunscreen"),
    ("body wash", "body wash"),
    ("shower gel", "body wash"),
    (" benzoyl", "acne_treatment"),
    ("spot treat", "acne_treatment"),
    (" acne ", "acne_treatment"),
    ("face wash", "cleanser"),
    ("gel cleanser", "cleanser"),
    ("foaming wash", "cleanser"),
    ("cleanser", "cleanser"),
    ("toner", "toner"),
    ("essence", "toner"),
    ("eye cream", "eye cream"),
    ("eye gel", "eye cream"),
    ("sheet mask", "mask"),
    ("clay mask", "mask"),
    (" mask", "mask"),
    ("niacinamide", "serum"),
    ("vitamin c", "serum"),
    ("serum", "serum"),
    ("ampoule", "serum"),
    ("balm", "balm"),
    ("scrub", "scrub"),
    ("body lotion", "lotion"),
    ("body milk", "lotion"),
    ("lotion", "lotion"),
    ("moisturizer", "moisturizer"),
    ("moisturising", "moisturizer"),
    ("moisturizing", "moisturizer"),
    ("face cream", "moisturizer"),
    ("day cream", "moisturizer"),
    ("night cream", "moisturizer"),
    ("hydrat", "moisturizer"),
    ("body oil", "oil"),
    ("face oil", "oil"),
    (" oil", "oil"),
]


# item_type_keyword → product_type fallback mapping
# Keepa provides this Amazon-catalogued keyword. Reuses existing product_type names.
ITEM_TYPE_KEYWORD_MAP = {
    "facial_cleansing_product": "cleanser",
    "facial_cleanser": "cleanser",
    "face_cleanser": "cleanser",
    "facial_moisturizer": "moisturizer",
    "face_moisturizer": "moisturizer",
    "facial_cream": "moisturizer",
    "body_moisturizer": "lotion",
    "body_lotion": "lotion",
    "facial_serum": "serum",
    "face_serum": "serum",
    "facial_toner": "toner",
    "face_toner": "toner",
    "sunscreen": "sunscreen",
    "sun_protection": "sunscreen",
    "facial_mask": "mask",
    "face_mask": "mask",
    "sheet_mask": "mask",
    "eye_cream": "eye cream",
    "eye_treatment": "eye cream",
    "body_wash": "body wash",
    "shower_gel": "body wash",
    "facial_scrub": "scrub",
    "body_scrub": "scrub",
    "facial_oil": "oil",
    "body_oil": "oil",
    "acne_treatment": "acne_treatment",
    "spot_treatment": "acne_treatment",
    "retinol_treatment": "retinol",
    "exfoliant": "exfoliant",
    "lip_balm": "balm",
    "face_balm": "balm",
    "micellar_water": "micellar water",
}


def classify_title(title: str, item_type_keyword: str = "") -> str:
    """
    Map ASIN title to product_type using fixed keyword dictionary.
    Falls back to item_type_keyword (Amazon-catalogued) when title parsing
    returns "other".

    Args:
        title: ASIN title string (from Keepa data)
        item_type_keyword: Optional Keepa item_type_keyword for fallback

    Returns:
        product_type string e.g. "serum", "moisturizer", "cleanser", "other"
    """
    if not title:
        title = ""
    title_lower = " " + title.lower() + " "  # pad for whole-word boundary matching
    for keyword, ptype in PRODUCT_TYPE_KEYWORDS:
        if keyword in title_lower:
            return ptype

    # Fallback: use item_type_keyword from Amazon catalogue
    if item_type_keyword:
        _itk = item_type_keyword.strip().lower().replace(" ", "_")
        mapped = ITEM_TYPE_KEYWORD_MAP.get(_itk)
        if mapped:
            return mapped

    return "other"


# Keyword → concern/active mapping (all matches collected, most specific first)
# Used by tag_concerns() — separate from product_type classification.
CONCERN_KEYWORDS = [
    ("retinol",       "retinol"),
    ("retinoic",      "retinol"),
    ("vitamin c",     "vitamin_c"),
    ("ascorbic",      "vitamin_c"),
    ("l-ascorbic",    "vitamin_c"),
    ("niacinamide",   "niacinamide"),
    (" b3 ",          "niacinamide"),
    ("azelaic",       "azelaic"),
    ("glycolic",      "aha_glycolic"),
    (" aha ",         "aha_glycolic"),
    ("salicylic",     "bha_salicylic"),
    (" bha ",         "bha_salicylic"),
    ("benzoyl",       "benzoyl_peroxide"),
    ("hyaluronic",    "hyaluronic"),
    (" ha ",          "hyaluronic"),
    ("ceramide",      "ceramides_barrier"),
    ("barrier cream", "ceramides_barrier"),
    (" spf ",         "spf"),
    (" acne ",        "acne"),
    ("blemish",       "acne"),
]


def tag_concerns(title: str, ingredients: str = "") -> list:
    """
    Return up to 2 concern/active labels for an ASIN.
    Scans title AND optional structured activeIngredients string.
    Ingredients string is appended so title-based ordering is preserved.
    Returns list of 0-2 unique concern strings, e.g. ["niacinamide", "aha_glycolic"].
    """
    if not title and not ingredients:
        return []
    combined = " " + (title or "").lower() + " " + (ingredients or "").lower() + " "
    found = []
    seen: set = set()
    for keyword, concern in CONCERN_KEYWORDS:
        if keyword in combined and concern not in seen:
            found.append(concern)
            seen.add(concern)
            if len(found) == 2:
                break
    return found


def infer_module(category_path: str) -> str:
    """
    Infer module ID from the category breadcrumb string.

    Args:
        category_path: Full category breadcrumb, e.g.
                       "Health & Household > Skin Care > Face Moisturizers"

    Returns:
        "skincare" or "generic"
    """
    if not category_path:
        return "generic"
    path_lower = category_path.lower()
    if any(token in path_lower for token in SKINCARE_MODULE_TOKENS):
        return "skincare"
    return "generic"
