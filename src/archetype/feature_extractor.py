"""
Deterministic Feature Extractor for Archetype Classification.

This module extracts features from product data that are used
to classify products into archetypes. The extraction is 100%
deterministic - no LLM calls here.

Features are extracted from:
- Product title (keyword detection)
- Category tree
- Keepa metrics (price, S&S, variations, etc.)
- Historical time series (seasonality detection)
"""

import re
import pandas as pd
from typing import Dict, List, Optional

from src.models.product_archetype import ArchetypeFeatures


# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================

# Keywords that suggest consumable/replenishable products
CONSUMABLE_KEYWORDS = [
    "refill", "replacement", "cartridge", "pod", "capsule",
    "disposable", "pack of", "count", "supply", "tablets",
    "sheets", "wipes", "tissues", "bags", "filters",
    "batteries", "ink", "toner", "gel", "cream", "lotion",
    "shampoo", "conditioner", "soap", "detergent", "cleaner",
]

# Keywords that suggest regulated products
REGULATED_KEYWORDS = [
    "supplement", "vitamin", "medication", "drug", "pharmaceutical",
    "organic", "natural", "fda", "baby", "infant", "formula",
    "food", "dietary", "health", "medical", "prescription",
    "topical", "ingestible", "probiotic", "herbal",
]

# Keywords that suggest seasonal products
SEASONAL_KEYWORDS: Dict[str, List[int]] = {
    "halloween": [10],
    "christmas": [11, 12],
    "xmas": [11, 12],
    "holiday": [11, 12],
    "easter": [3, 4],
    "summer": [5, 6, 7, 8],
    "pool": [5, 6, 7, 8],
    "beach": [5, 6, 7, 8],
    "winter": [11, 12, 1, 2],
    "back to school": [7, 8],
    "school": [7, 8],
    "valentine": [2],
    "valentines": [2],
    "mothers day": [5],
    "mother's day": [5],
    "fathers day": [6],
    "father's day": [6],
    "thanksgiving": [11],
    "new year": [12, 1],
    "spring": [3, 4, 5],
    "fall": [9, 10],
    "autumn": [9, 10],
    "pumpkin": [9, 10],
    "spooky": [10],
    "costume": [10],
    "ornament": [11, 12],
    "wreath": [11, 12],
}

# Keywords that suggest fashion/style
FASHION_KEYWORDS = [
    "dress", "shirt", "pants", "jeans", "shorts", "skirt",
    "blouse", "sweater", "jacket", "coat", "hoodie",
    "shoes", "boots", "sneakers", "sandals", "heels",
    "jewelry", "necklace", "bracelet", "earring", "ring",
    "accessory", "handbag", "purse", "wallet", "belt",
    "fashion", "style", "outfit", "clothing", "apparel", "wear",
    "hat", "cap", "scarf", "gloves", "sunglasses",
]

# Keywords that suggest durable goods
DURABLE_KEYWORDS = [
    "furniture", "sofa", "couch", "chair", "table", "desk",
    "bed", "mattress", "dresser", "cabinet", "shelf",
    "appliance", "refrigerator", "washer", "dryer", "dishwasher",
    "microwave", "oven", "stove", "grill",
    "tv", "television", "monitor", "computer", "laptop",
    "camera", "printer", "speaker", "headphone",
    "vacuum", "fan", "heater", "ac", "air conditioner",
]

# Multipack indicators (regex patterns)
MULTIPACK_PATTERNS = [
    r"(\d+)\s*[-]?\s*pack",
    r"(\d+)\s*[-]?\s*count",
    r"(\d+)\s*[-]?\s*ct\b",
    r"pack of (\d+)",
    r"(\d+)\s*[-]?\s*pieces?",
    r"(\d+)\s*[-]?\s*pcs?\b",
    r"(\d+)\s*[-]?\s*pk\b",
    r"(\d+)\s*[-]?\s*set",
    r"bundle of (\d+)",
    r"variety (\d+)",
    r"(\d+)\s*[-]?\s*rolls?",
    r"(\d+)\s*[-]?\s*oz\b",  # Weight often indicates multipack
]


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_archetype_features(
    row: pd.Series,
    df_historical: Optional[pd.DataFrame] = None,
    category_median_price: float = 0.0,
) -> ArchetypeFeatures:
    """
    Extract deterministic features for archetype classification.

    Args:
        row: Product data row (from portfolio snapshot or market snapshot)
        df_historical: Historical time series (optional, for seasonality)
        category_median_price: Median price in category (for relative pricing)

    Returns:
        ArchetypeFeatures with all available signals
    """
    features = ArchetypeFeatures()

    # Get title for keyword analysis
    title = str(row.get("title", "")).lower()

    # --- Category signals ---
    features.category_tree = str(row.get("category_tree", row.get("category", "")))

    # Check consumable keywords
    features.is_consumable = any(kw in title for kw in CONSUMABLE_KEYWORDS)

    # Check subscription eligibility
    features.is_subscription_eligible = bool(
        row.get("is_sns", row.get("isSNS", row.get("sns_eligible", False)))
    )
    features.has_subscribe_save = features.is_subscription_eligible

    # Check hazmat/adult flags
    features.is_hazmat = bool(row.get("is_hazmat", row.get("isHazmat", False)))
    features.is_adult = bool(row.get("is_adult", row.get("isAdult", False)))

    # Check for regulated keywords
    features.requires_approval = any(kw in title.lower() for kw in REGULATED_KEYWORDS)

    # Also check category tree for regulated categories
    category_lower = features.category_tree.lower()
    if any(cat in category_lower for cat in ["health", "baby", "supplement", "vitamin", "personal care"]):
        features.requires_approval = True

    # --- Pricing signals ---
    features.price = float(row.get("filled_price", row.get("price", row.get("buy_box_price", 0))) or 0)

    if category_median_price > 0:
        features.price_vs_category_median = features.price / category_median_price
    else:
        features.price_vs_category_median = 1.0  # Assume at median if unknown

    # --- Variation signals ---
    features.is_variation = bool(row.get("is_variation", row.get("isVariation", False)))
    features.variation_count = int(row.get("variation_count", row.get("variationCount", 0)) or 0)

    # Check if this is a parent ASIN
    parent_asin = row.get("parent_asin", row.get("parentAsin", ""))
    asin = row.get("asin", "")
    features.is_parent = bool(parent_asin and parent_asin == asin)

    # Check for size/color variations in title
    features.has_size_variations = any(
        x in title for x in ["small", "medium", "large", "xl", "xxl", "size", "s/m", "l/xl"]
    )
    features.has_color_variations = any(
        x in title for x in [
            "black", "white", "red", "blue", "green", "yellow", "pink",
            "purple", "orange", "brown", "gray", "grey", "color", "colour"
        ]
    )

    # --- Bundle signals ---
    pack_count = _extract_pack_count(title)
    features.is_multipack = pack_count > 1
    features.pack_count = pack_count

    # --- Seasonal signals ---
    peak_months = _detect_seasonal_keywords(title)
    features.peak_months = list(set(peak_months))

    # Calculate sales seasonality from historical data
    if df_historical is not None and len(df_historical) >= 12:
        features.sales_coefficient_of_variation = _calculate_seasonality_cv(df_historical)

    # --- Competitive signals ---
    features.seller_count = int(
        row.get("new_offer_count", row.get("offerCountNew", row.get("seller_count", 0))) or 0
    )

    # Check for Amazon presence
    amazon_bb_share = row.get("bb_stats_amazon_90", row.get("amazon_bb_share", 0)) or 0
    features.has_amazon_offer = float(amazon_bb_share) > 0.1

    # Check for brand registry (proxy: has A+ content)
    features.brand_registry_enrolled = bool(row.get("has_aplus", row.get("hasAplus", False)))

    # --- Review velocity ---
    features.review_velocity_30d = int(row.get("review_velocity_30d", row.get("new_reviews_30d", 0)) or 0)

    return features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_pack_count(title: str) -> int:
    """
    Extract pack count from title using regex patterns.

    Examples:
        "12-Pack AA Batteries" -> 12
        "Pack of 6 Notebooks" -> 6
        "Single Toothbrush" -> 1
    """
    title_lower = title.lower()

    for pattern in MULTIPACK_PATTERNS:
        match = re.search(pattern, title_lower, re.IGNORECASE)
        if match:
            try:
                count = int(match.group(1))
                if 1 < count <= 1000:  # Sanity check
                    return count
            except (ValueError, IndexError):
                continue

    return 1  # Default to single unit


def _detect_seasonal_keywords(title: str) -> List[int]:
    """
    Detect seasonal keywords in title and return associated months.

    Returns list of month numbers (1-12) associated with detected keywords.
    """
    title_lower = title.lower()
    peak_months = []

    for keyword, months in SEASONAL_KEYWORDS.items():
        if keyword in title_lower:
            peak_months.extend(months)

    return peak_months


def _calculate_seasonality_cv(df_historical: pd.DataFrame) -> float:
    """
    Calculate coefficient of variation for sales to detect seasonality.

    High CV (>0.5) suggests seasonal product.
    Low CV (<0.2) suggests stable/replenishable.
    """
    # Find revenue column
    revenue_col = None
    for col in ["weekly_revenue", "revenue", "estimated_units", "monthly_sold"]:
        if col in df_historical.columns:
            revenue_col = col
            break

    if revenue_col is None:
        return 0.0

    sales = df_historical[revenue_col].dropna()
    if len(sales) < 4 or sales.mean() == 0:
        return 0.0

    return float(sales.std() / sales.mean())


# =============================================================================
# ARCHETYPE LIKELIHOOD FUNCTIONS
# =============================================================================

def is_likely_fashion(features: ArchetypeFeatures, title: str = "") -> bool:
    """Check if product is likely fashion/style category."""
    title_lower = title.lower()

    # Check fashion keywords
    has_fashion_keyword = any(kw in title_lower for kw in FASHION_KEYWORDS)

    # Check for size/color variations (common in fashion)
    has_fashion_signals = (
        features.has_size_variations or
        features.has_color_variations
    )

    # Check category
    category_lower = features.category_tree.lower()
    is_fashion_category = any(
        cat in category_lower for cat in ["clothing", "fashion", "shoes", "jewelry", "accessories"]
    )

    return has_fashion_keyword or (has_fashion_signals and is_fashion_category)


def is_likely_seasonal(features: ArchetypeFeatures) -> bool:
    """Check if product is likely seasonal."""
    # High sales variance suggests seasonality
    high_variance = features.sales_coefficient_of_variation > 0.5

    # Explicit seasonal keywords detected
    has_seasonal_keywords = len(features.peak_months) > 0

    return high_variance or has_seasonal_keywords


def is_likely_bundle(features: ArchetypeFeatures) -> bool:
    """Check if product is a bundle/multipack."""
    return features.is_multipack or features.pack_count > 1


def is_likely_consumable(features: ArchetypeFeatures) -> bool:
    """Check if product is consumable/replenishable."""
    return (
        features.is_consumable or
        features.is_subscription_eligible or
        features.has_subscribe_save
    )


def is_likely_durable(features: ArchetypeFeatures, title: str = "") -> bool:
    """Check if product is likely a durable good."""
    title_lower = title.lower()

    # Check durable keywords
    has_durable_keyword = any(kw in title_lower for kw in DURABLE_KEYWORDS)

    # High price suggests durable
    high_price = features.price > 100

    # Check category
    category_lower = features.category_tree.lower()
    is_durable_category = any(
        cat in category_lower for cat in [
            "furniture", "appliances", "electronics", "office", "home", "kitchen"
        ]
    )

    return has_durable_keyword or (high_price and is_durable_category)


def is_likely_regulated(features: ArchetypeFeatures) -> bool:
    """Check if product is in a regulated category."""
    return features.requires_approval


def get_archetype_signals(features: ArchetypeFeatures, title: str = "") -> Dict[str, float]:
    """
    Get signal strength (0.0-1.0) for each archetype.

    Returns dict of archetype -> confidence score.
    Used to help classification when signals are mixed.
    """
    signals = {
        "replenishable_commodity": 0.0,
        "razor_and_blades": 0.0,
        "durable_consideration": 0.0,
        "seasonal_spike": 0.0,
        "regulated_compliance": 0.0,
        "fashion_style": 0.0,
        "commodity_bundle": 0.0,
    }

    # Consumable signals
    if is_likely_consumable(features):
        signals["replenishable_commodity"] = 0.7
        if features.has_subscribe_save:
            signals["replenishable_commodity"] = 0.9

    # Bundle signals
    if is_likely_bundle(features):
        signals["commodity_bundle"] = 0.8

    # Seasonal signals
    if is_likely_seasonal(features):
        signals["seasonal_spike"] = 0.7
        if len(features.peak_months) > 0:
            signals["seasonal_spike"] = 0.85

    # Fashion signals
    if is_likely_fashion(features, title):
        signals["fashion_style"] = 0.75

    # Durable signals
    if is_likely_durable(features, title):
        signals["durable_consideration"] = 0.7
        if features.price > 200:
            signals["durable_consideration"] = 0.85

    # Regulated signals
    if is_likely_regulated(features):
        signals["regulated_compliance"] = 0.8

    # Razor and blades is harder to detect - look for consumable + related products pattern
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["cartridge", "refill", "replacement", "ink", "toner", "pod"]):
        # This is likely a consumable in a razor-and-blades model
        signals["razor_and_blades"] = 0.6

    return signals
