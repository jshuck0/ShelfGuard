"""
ShelfGuard Extended Keepa Data Schema
======================================
Defines the extended fields available from Keepa Product Finder API
that power the Strategic Triangulation Engine.

This module documents which fields to request and how to map them
for use in the AI Engine's signal extraction.

Token Cost: 10 + 1 per 100 ASINs for Product Finder queries
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# KEEPA DOMAIN IDS
# =============================================================================

class AmazonDomain(Enum):
    """Amazon marketplace domain IDs for Keepa API."""
    US = 1       # amazon.com
    UK = 2       # amazon.co.uk
    DE = 3       # amazon.de
    FR = 4       # amazon.fr
    JP = 5       # amazon.co.jp
    CA = 6       # amazon.ca
    IT = 8       # amazon.it
    ES = 9       # amazon.es
    IN = 10      # amazon.in
    MX = 11      # amazon.com.mx
    BR = 12      # amazon.com.br


# =============================================================================
# EXTENDED FIELD DEFINITIONS
# =============================================================================

@dataclass
class KeepaField:
    """Defines a Keepa Product Finder field."""
    name: str
    data_type: str
    description: str
    signal_category: str  # Maps to ai_engine signal categories
    example_value: Any
    unit: Optional[str] = None


# Fields organized by signal category for the Triangulation Engine

MARKET_PRESSURE_FIELDS = {
    # === COMPETITION METRICS ===
    "current_COUNT_NEW": KeepaField(
        name="current_COUNT_NEW",
        data_type="Integer",
        description="Current count of new offers",
        signal_category="market_pressure",
        example_value=8,
        unit="sellers"
    ),
    "delta30_COUNT_NEW": KeepaField(
        name="delta30_COUNT_NEW",
        data_type="Integer",
        description="30-day change in new offer count",
        signal_category="market_pressure",
        example_value=3,
        unit="sellers"
    ),
    "delta90_COUNT_NEW": KeepaField(
        name="delta90_COUNT_NEW",
        data_type="Integer",
        description="90-day change in new offer count",
        signal_category="market_pressure",
        example_value=5,
        unit="sellers"
    ),
    "avg30_COUNT_NEW": KeepaField(
        name="avg30_COUNT_NEW",
        data_type="Integer",
        description="30-day average new offer count",
        signal_category="market_pressure",
        example_value=7,
        unit="sellers"
    ),
    
    # === BUY BOX COMPETITION ===
    "buyBoxStatsSellerCount30": KeepaField(
        name="buyBoxStatsSellerCount30",
        data_type="Integer",
        description="Number of sellers with Buy Box ownership in last 30 days",
        signal_category="market_pressure",
        example_value=3,
        unit="sellers"
    ),
    "buyBoxStatsSellerCount90": KeepaField(
        name="buyBoxStatsSellerCount90",
        data_type="Integer",
        description="Number of sellers with Buy Box ownership in last 90 days",
        signal_category="market_pressure",
        example_value=5,
        unit="sellers"
    ),
    "buyBoxStatsTopSeller30": KeepaField(
        name="buyBoxStatsTopSeller30",
        data_type="Integer",
        description="Buy Box share % of top seller (Amazon incl.) in last 30 days",
        signal_category="market_pressure",
        example_value=85,
        unit="%"
    ),
}

VELOCITY_MOMENTUM_FIELDS = {
    # === SALES RANK TRENDS ===
    "current_SALES": KeepaField(
        name="current_SALES",
        data_type="Integer",
        description="Current sales rank",
        signal_category="velocity_momentum",
        example_value=1500,
        unit="rank"
    ),
    "delta30_SALES": KeepaField(
        name="delta30_SALES",
        data_type="Integer",
        description="30-day absolute change in sales rank",
        signal_category="velocity_momentum",
        example_value=-200,
        unit="rank"
    ),
    "delta90_SALES": KeepaField(
        name="delta90_SALES",
        data_type="Integer",
        description="90-day absolute change in sales rank",
        signal_category="velocity_momentum",
        example_value=-500,
        unit="rank"
    ),
    "deltaPercent30_SALES": KeepaField(
        name="deltaPercent30_SALES",
        data_type="Integer",
        description="30-day percentage change in sales rank",
        signal_category="velocity_momentum",
        example_value=-10,
        unit="%"
    ),
    "deltaPercent90_SALES": KeepaField(
        name="deltaPercent90_SALES",
        data_type="Integer",
        description="90-day percentage change in sales rank",
        signal_category="velocity_momentum",
        example_value=-15,
        unit="%"
    ),
    "avg30_SALES": KeepaField(
        name="avg30_SALES",
        data_type="Integer",
        description="30-day average sales rank",
        signal_category="velocity_momentum",
        example_value=1800,
        unit="rank"
    ),
    "avg90_SALES": KeepaField(
        name="avg90_SALES",
        data_type="Integer",
        description="90-day average sales rank",
        signal_category="velocity_momentum",
        example_value=2000,
        unit="rank"
    ),
    
    # === SALES VOLUME ===
    "monthlySold": KeepaField(
        name="monthlySold",
        data_type="Integer",
        description="How often product was bought in past month (Amazon's metric)",
        signal_category="velocity_momentum",
        example_value=5000,
        unit="units"
    ),
    "deltaPercent90_monthlySold": KeepaField(
        name="deltaPercent90_monthlySold",
        data_type="Integer",
        description="90-day percentage change in monthly sales",
        signal_category="velocity_momentum",
        example_value=10,
        unit="%"
    ),
    
    # === STOCK STATUS ===
    "outOfStockPercentage90": KeepaField(
        name="outOfStockPercentage90",
        data_type="Integer",
        description="90-day Amazon out-of-stock percentage (0-100)",
        signal_category="velocity_momentum",
        example_value=5,
        unit="%"
    ),
    "outOfStockCountAmazon30": KeepaField(
        name="outOfStockCountAmazon30",
        data_type="Integer",
        description="Number of times Amazon was out of stock in last 30 days",
        signal_category="velocity_momentum",
        example_value=2,
        unit="occurrences"
    ),
}

UNIT_ECONOMICS_FIELDS = {
    # === AMAZON PRICING ===
    "current_AMAZON": KeepaField(
        name="current_AMAZON",
        data_type="Integer",
        description="Current Amazon price (in cents)",
        signal_category="unit_economics",
        example_value=4599,
        unit="cents"
    ),
    "avg30_AMAZON": KeepaField(
        name="avg30_AMAZON",
        data_type="Integer",
        description="30-day average Amazon price (in cents)",
        signal_category="unit_economics",
        example_value=4499,
        unit="cents"
    ),
    "avg90_AMAZON": KeepaField(
        name="avg90_AMAZON",
        data_type="Integer",
        description="90-day average Amazon price (in cents)",
        signal_category="unit_economics",
        example_value=4399,
        unit="cents"
    ),
    "delta30_AMAZON": KeepaField(
        name="delta30_AMAZON",
        data_type="Integer",
        description="30-day absolute change in Amazon price (in cents)",
        signal_category="unit_economics",
        example_value=100,
        unit="cents"
    ),
    "deltaPercent30_AMAZON": KeepaField(
        name="deltaPercent30_AMAZON",
        data_type="Integer",
        description="30-day percentage change in Amazon price",
        signal_category="unit_economics",
        example_value=2,
        unit="%"
    ),
    
    # === BUY BOX PRICING ===
    "current_BUY_BOX_SHIPPING": KeepaField(
        name="current_BUY_BOX_SHIPPING",
        data_type="Integer",
        description="Current Buy Box price including shipping (in cents)",
        signal_category="unit_economics",
        example_value=4799,
        unit="cents"
    ),
    "avg90_BUY_BOX_SHIPPING": KeepaField(
        name="avg90_BUY_BOX_SHIPPING",
        data_type="Integer",
        description="90-day average Buy Box price (in cents)",
        signal_category="unit_economics",
        example_value=4699,
        unit="cents"
    ),
    
    # === NEW FBA PRICING ===
    "current_NEW_FBA": KeepaField(
        name="current_NEW_FBA",
        data_type="Integer",
        description="Current lowest FBA price (in cents)",
        signal_category="unit_economics",
        example_value=4699,
        unit="cents"
    ),
    "avg90_NEW_FBA": KeepaField(
        name="avg90_NEW_FBA",
        data_type="Integer",
        description="90-day average lowest FBA price (in cents)",
        signal_category="unit_economics",
        example_value=4599,
        unit="cents"
    ),
    
    # === LOWEST PRICE FLAGS ===
    "isLowest_AMAZON": KeepaField(
        name="isLowest_AMAZON",
        data_type="Boolean",
        description="Whether current Amazon price is all-time lowest",
        signal_category="unit_economics",
        example_value=False,
        unit=None
    ),
    "isLowest90_AMAZON": KeepaField(
        name="isLowest90_AMAZON",
        data_type="Boolean",
        description="Whether current Amazon price is 90-day lowest",
        signal_category="unit_economics",
        example_value=True,
        unit=None
    ),
    
    # === COUPONS & PROMOTIONS ===
    "couponOneTimePercent": KeepaField(
        name="couponOneTimePercent",
        data_type="Integer",
        description="One-time coupon percentage (if active)",
        signal_category="unit_economics",
        example_value=10,
        unit="%"
    ),
    "couponSNSPercent": KeepaField(
        name="couponSNSPercent",
        data_type="Integer",
        description="Subscribe & Save coupon percentage",
        signal_category="unit_economics",
        example_value=15,
        unit="%"
    ),
}

BRAND_STRENGTH_FIELDS = {
    # === REVIEWS ===
    "current_COUNT_REVIEWS": KeepaField(
        name="current_COUNT_REVIEWS",
        data_type="Integer",
        description="Current review count",
        signal_category="brand_strength",
        example_value=1250,
        unit="reviews"
    ),
    "delta30_COUNT_REVIEWS": KeepaField(
        name="delta30_COUNT_REVIEWS",
        data_type="Integer",
        description="30-day change in review count",
        signal_category="brand_strength",
        example_value=45,
        unit="reviews"
    ),
    "deltaPercent30_COUNT_REVIEWS": KeepaField(
        name="deltaPercent30_COUNT_REVIEWS",
        data_type="Integer",
        description="30-day percentage change in review count",
        signal_category="brand_strength",
        example_value=4,
        unit="%"
    ),
    
    # === RATING ===
    "current_RATING": KeepaField(
        name="current_RATING",
        data_type="Integer",
        description="Current rating (stored as rating * 10, e.g., 45 = 4.5 stars)",
        signal_category="brand_strength",
        example_value=45,
        unit="rating*10"
    ),
    "delta30_RATING": KeepaField(
        name="delta30_RATING",
        data_type="Integer",
        description="30-day change in rating (*10)",
        signal_category="brand_strength",
        example_value=1,
        unit="rating*10"
    ),
    
    # === VARIATION REVIEWS ===
    "variationReviewCount": KeepaField(
        name="variationReviewCount",
        data_type="Integer",
        description="Number of reviews specific to this variation",
        signal_category="brand_strength",
        example_value=250,
        unit="reviews"
    ),
    
    # === PRODUCT AGE ===
    "trackingSince": KeepaField(
        name="trackingSince",
        data_type="Integer",
        description="When Keepa started tracking (in Keepa Time minutes)",
        signal_category="brand_strength",
        example_value=3411319,
        unit="Keepa minutes"
    ),
    
    # === CONTENT QUALITY ===
    "imageCount": KeepaField(
        name="imageCount",
        data_type="Integer",
        description="Number of product images",
        signal_category="brand_strength",
        example_value=7,
        unit="images"
    ),
    "videoCount": KeepaField(
        name="videoCount",
        data_type="Integer",
        description="Number of product videos",
        signal_category="brand_strength",
        example_value=1,
        unit="videos"
    ),
    "hasAPlus": KeepaField(
        name="hasAPlus",
        data_type="Boolean",
        description="Whether product has A+ content",
        signal_category="brand_strength",
        example_value=True,
        unit=None
    ),
}

BUYBOX_CONTROL_FIELDS = {
    # === BUY BOX OWNERSHIP ===
    "buyBoxStatsAmazon30": KeepaField(
        name="buyBoxStatsAmazon30",
        data_type="Integer",
        description="Amazon's Buy Box ownership percentage in last 30 days",
        signal_category="buybox_control",
        example_value=75,
        unit="%"
    ),
    "buyBoxStatsAmazon90": KeepaField(
        name="buyBoxStatsAmazon90",
        data_type="Integer",
        description="Amazon's Buy Box ownership percentage in last 90 days",
        signal_category="buybox_control",
        example_value=72,
        unit="%"
    ),
    "buyBoxIsAmazon": KeepaField(
        name="buyBoxIsAmazon",
        data_type="Boolean",
        description="Whether Amazon currently holds the Buy Box",
        signal_category="buybox_control",
        example_value=True,
        unit=None
    ),
    "buyBoxIsFBA": KeepaField(
        name="buyBoxIsFBA",
        data_type="Boolean",
        description="Whether current Buy Box is FBA",
        signal_category="buybox_control",
        example_value=True,
        unit=None
    ),
    
    # === BUY BOX STATUS ===
    "buyBoxIsPreorder": KeepaField(
        name="buyBoxIsPreorder",
        data_type="Boolean",
        description="Whether Buy Box is a preorder",
        signal_category="buybox_control",
        example_value=False,
        unit=None
    ),
    "buyBoxIsBackorder": KeepaField(
        name="buyBoxIsBackorder",
        data_type="Boolean",
        description="Whether Buy Box is backordered",
        signal_category="buybox_control",
        example_value=False,
        unit=None
    ),
    
    # === AMAZON AVAILABILITY ===
    "availabilityAmazon": KeepaField(
        name="availabilityAmazon",
        data_type="Integer",
        description="Amazon offer availability (-1=none, 0=in stock, 1=preorder, 2=unknown, 3=backorder, 4=delayed)",
        signal_category="buybox_control",
        example_value=0,
        unit="status code"
    ),
    "backInStock_AMAZON": KeepaField(
        name="backInStock_AMAZON",
        data_type="Boolean",
        description="Whether Amazon offer came back in stock in last 60 days",
        signal_category="buybox_control",
        example_value=False,
        unit=None
    ),
}


# =============================================================================
# ALL FIELDS COMBINED
# =============================================================================

ALL_EXTENDED_FIELDS = {
    **MARKET_PRESSURE_FIELDS,
    **VELOCITY_MOMENTUM_FIELDS,
    **UNIT_ECONOMICS_FIELDS,
    **BRAND_STRENGTH_FIELDS,
    **BUYBOX_CONTROL_FIELDS,
}


# =============================================================================
# QUERY BUILDER HELPERS
# =============================================================================

def get_recommended_fields_for_triangulation() -> List[str]:
    """
    Get list of field names recommended for strategic triangulation.
    
    These are the minimum fields needed for the AI Engine to provide
    accurate classifications.
    """
    return [
        # Market Pressure (Competition)
        "current_COUNT_NEW",
        "delta30_COUNT_NEW",
        "buyBoxStatsSellerCount30",
        "buyBoxStatsTopSeller30",
        
        # Velocity Momentum (Demand)
        "current_SALES",
        "delta30_SALES",
        "delta90_SALES",
        "deltaPercent30_SALES",
        "deltaPercent90_SALES",
        "monthlySold",
        "outOfStockPercentage90",
        
        # Unit Economics (Margin)
        "current_AMAZON",
        "avg90_AMAZON",
        "delta30_AMAZON",
        "deltaPercent30_AMAZON",
        "current_BUY_BOX_SHIPPING",
        "current_NEW_FBA",
        "couponOneTimePercent",
        
        # Brand Strength (Reviews)
        "current_COUNT_REVIEWS",
        "delta30_COUNT_REVIEWS",
        "current_RATING",
        "trackingSince",
        "hasAPlus",
        
        # Buy Box Control
        "buyBoxStatsAmazon30",
        "buyBoxIsAmazon",
        "buyBoxIsFBA",
        "availabilityAmazon",
    ]


def build_product_finder_query(
    asins: Optional[List[str]] = None,
    brand: Optional[str] = None,
    root_category: Optional[int] = None,
    min_sales_rank: Optional[int] = None,
    max_sales_rank: Optional[int] = None,
    min_reviews: Optional[int] = None,
    page: int = 0,
    per_page: int = 100
) -> Dict[str, Any]:
    """
    Build a Keepa Product Finder query optimized for the Triangulation Engine.
    
    Args:
        asins: Optional list of specific ASINs to query
        brand: Optional brand name filter
        root_category: Optional category ID
        min_sales_rank: Minimum sales rank filter
        max_sales_rank: Maximum sales rank filter
        min_reviews: Minimum review count
        page: Page number (0-indexed)
        per_page: Results per page (50-10000)
        
    Returns:
        Query dictionary for Keepa Product Finder API
    """
    query = {
        "page": page,
        "perPage": min(max(50, per_page), 10000),  # Enforce Keepa limits
    }
    
    if root_category:
        query["rootCategory"] = [root_category]
    
    if brand:
        query["brand"] = [brand]
    
    if min_sales_rank is not None:
        query["current_SALES_gte"] = min_sales_rank
    
    if max_sales_rank is not None:
        query["current_SALES_lte"] = max_sales_rank
    
    if min_reviews is not None:
        query["current_COUNT_REVIEWS_gte"] = min_reviews
    
    # Sort by sales rank ascending (best sellers first)
    query["sort"] = [["current_SALES", "asc"]]
    
    return query


def normalize_keepa_price(price_cents: int) -> float:
    """Convert Keepa price (cents) to dollars."""
    if price_cents is None or price_cents < 0:
        return None
    return price_cents / 100.0


def keepa_minutes_to_days(keepa_minutes: int) -> int:
    """Convert Keepa time minutes to days since 2011-01-01."""
    if keepa_minutes is None or keepa_minutes <= 0:
        return None
    return keepa_minutes // (60 * 24)


def keepa_rating_to_stars(rating_times_10: int) -> float:
    """Convert Keepa rating (stored as rating*10) to star rating."""
    if rating_times_10 is None or rating_times_10 <= 0:
        return None
    return rating_times_10 / 10.0


# =============================================================================
# DATA TRANSFORMATION FOR AI ENGINE
# =============================================================================

def transform_product_finder_row(raw_row: Dict) -> Dict:
    """
    Transform a raw Keepa Product Finder row into the format
    expected by the StrategicTriangulator.
    
    Handles:
    - Price normalization (cents → dollars)
    - Rating normalization (x10 → stars)
    - Keepa time conversion
    - Missing value handling
    """
    transformed = {**raw_row}  # Copy original
    
    # Normalize prices (cents to dollars)
    price_fields = [
        "current_AMAZON", "avg30_AMAZON", "avg90_AMAZON",
        "current_BUY_BOX_SHIPPING", "avg90_BUY_BOX_SHIPPING",
        "current_NEW_FBA", "avg90_NEW_FBA",
        "delta30_AMAZON", "delta90_AMAZON"
    ]
    for field in price_fields:
        if field in raw_row and raw_row[field] is not None:
            transformed[field] = normalize_keepa_price(raw_row[field])
    
    # Normalize rating
    if "current_RATING" in raw_row and raw_row["current_RATING"]:
        transformed["rating"] = keepa_rating_to_stars(raw_row["current_RATING"])
    
    # Calculate days tracked
    if "trackingSince" in raw_row and raw_row["trackingSince"]:
        transformed["days_tracked"] = keepa_minutes_to_days(raw_row["trackingSince"])
    
    # Map to existing field names for compatibility with current engine
    field_mapping = {
        "current_COUNT_NEW": "new_offer_count",
        "current_COUNT_REVIEWS": "review_count",
        "current_SALES": "sales_rank",
    }
    for keepa_field, engine_field in field_mapping.items():
        if keepa_field in raw_row:
            transformed[engine_field] = raw_row[keepa_field]
    
    return transformed


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Show available fields
    print("="*60)
    print("KEEPA EXTENDED FIELDS FOR TRIANGULATION ENGINE")
    print("="*60)
    
    categories = {
        "Market Pressure": MARKET_PRESSURE_FIELDS,
        "Velocity Momentum": VELOCITY_MOMENTUM_FIELDS,
        "Unit Economics": UNIT_ECONOMICS_FIELDS,
        "Brand Strength": BRAND_STRENGTH_FIELDS,
        "Buy Box Control": BUYBOX_CONTROL_FIELDS,
    }
    
    for cat_name, fields in categories.items():
        print(f"\n{cat_name}:")
        print("-" * 40)
        for name, field in fields.items():
            print(f"  {name}: {field.description}")
    
    print("\n" + "="*60)
    print("RECOMMENDED FIELDS FOR TRIANGULATION:")
    print("="*60)
    for field in get_recommended_fields_for_triangulation():
        print(f"  • {field}")
