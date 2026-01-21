"""
ShelfGuard Universal Data Healer
==================================
Comprehensive synthetic data filling and interpolation for ALL numerical metrics.

Ensures every time-series metric (Price, Rank, Reviews, Offers, Ratings, etc.) 
is continuous and gap-free before hitting the AI Logic Engine.

The 3-Step Healing Process:
1. Linear Interpolate (for smooth trends)
2. Forward Fill (for step functions like reviews)
3. Backward Fill (for early data gaps)
4. Default Fallback (worst-case assumptions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# SAFE INTERPOLATION HELPER (Pandas 2.0+ compatible)
# =============================================================================

def _safe_interpolate(series: pd.Series, method: str = 'linear', limit: int = None) -> pd.Series:
    """
    Safely interpolate a series, handling edge cases that raise ValueError in pandas 2.0+.
    
    Args:
        series: Pandas Series to interpolate
        method: Interpolation method (default: 'linear')
        limit: Maximum number of consecutive NaN values to fill
    
    Returns:
        Interpolated series (or original if interpolation not possible)
    """
    if len(series) <= 1 or series.isna().all():
        return series
    try:
        return series.interpolate(method=method, limit=limit)
    except ValueError:
        # Fallback to forward fill if interpolate fails
        return series.ffill(limit=limit)


# =============================================================================
# METRIC GROUP DEFINITIONS
# =============================================================================

@dataclass
class MetricGroup:
    """Defines how a group of metrics should be filled."""
    name: str
    columns: List[str]
    fill_strategy: str  # "interpolate", "ffill", "bfill"
    default_value: float
    max_gap_limit: Optional[int] = None  # Maximum gap to fill (in rows)
    description: str = ""


# Group A: Financials (Price, Fees, Estimated Revenue)
FINANCIAL_METRICS = MetricGroup(
    name="Financials",
    columns=[
        "filled_price",
        "buy_box_price",
        "amazon_price",
        "new_price",
        "new_fba_price",
        "fba_fees",
        "weekly_sales_filled",
        "revenue_proxy",        # ADDED: Monthly revenue estimate (critical for Command Center)
        "monthly_units",        # ADDED: Monthly units estimate
        "avg_weekly_revenue",   # ADDED: Weekly revenue from historical data
        "estimated_units",
        "eff_p",
        "synthetic_cogs",
        "landed_logistics",
        "net_margin",
        "price",                # ADDED: Base price from discovery
    ],
    fill_strategy="interpolate",
    default_value=0.0,
    max_gap_limit=4,  # 4 weeks max gap
    description="Financial metrics should interpolate smoothly"
)

# Group B: Performance (Sales Rank / BSR)
PERFORMANCE_METRICS = MetricGroup(
    name="Performance",
    columns=[
        "sales_rank",
        "sales_rank_filled",
        "bsr",                  # ADDED: Discovery uses 'bsr' for Best Seller Rank
        "current_SALES",
        "avg30_SALES",
        "avg90_SALES",
        "avg180_SALES",
    ],
    fill_strategy="interpolate",
    default_value=1_000_000,  # Assume worst case if missing
    max_gap_limit=3,  # 3 weeks max gap
    description="Sales rank should interpolate, defaulting to worst rank"
)

# Group C: Social & Competitive (Rating, Review Count, Offer Count)
SOCIAL_COMPETITIVE_METRICS = MetricGroup(
    name="Social & Competitive",
    columns=[
        "rating",
        "current_RATING",
        "review_count",
        "current_COUNT_REVIEWS",
        "new_offer_count",
        "used_offer_count",
        "current_COUNT_NEW",
        "current_COUNT_USED",
        "delta30_COUNT_REVIEWS",
        "delta90_COUNT_REVIEWS",
        "delta30_COUNT_NEW",
        "delta90_COUNT_NEW",
    ],
    fill_strategy="ffill",  # Forward fill (reviews/offers don't decrease smoothly)
    default_value=0.0,
    max_gap_limit=8,  # 8 weeks max gap for social metrics
    description="Social metrics use forward fill (step functions)"
)

# Special handling for specific metrics
# These defaults are used when no data is available after fill strategies
# Chosen to be "neutral" values that don't trigger aggressive AI recommendations
SPECIAL_DEFAULTS = {
    # Ratings: 0 means "no rating data" - AI should handle gracefully
    "rating": 0.0,
    "current_RATING": 0.0,
    # Reviews: 0 means "no review data" - AI should not assume product is new
    "review_count": 0,
    "current_COUNT_REVIEWS": 0,
    # Sellers: Assume at least 1 seller (the product is being sold)
    "new_offer_count": 1,
    "current_COUNT_NEW": 1,
    "used_offer_count": 0,
    "current_COUNT_USED": 0,
    "seller_count": 1,  # NEW: True seller count from sellerIds
    # Buy Box: Assume 50% if unknown (neutral, doesn't trigger "zero BB" alerts)
    "amazon_bb_share": 0.5,
    "buy_box_switches": 0,
    "buybox_is_amazon": None,  # NEW: None means "unknown" - don't assume
    "buybox_is_fba": None,     # NEW: None means "unknown"
    "buybox_is_backorder": False,  # NEW: Assume not backordered
    "has_amazon_seller": None,     # NEW: None means "unknown"
    "bb_seller_count_30": 0,       # NEW: Buy Box seller count
    "bb_top_seller_30": 0.0,       # NEW: Top seller's BB share
    "bb_stats_amazon_30": None,    # NEW: Keepa's Amazon BB stat
    "bb_stats_amazon_90": None,    # NEW: Keepa's Amazon BB stat
    # Velocity: Assume stable (no decay/growth) if unknown
    "velocity_decay": 1.0,
    "velocity_trend_30d": 0.0,
    "velocity_trend_90d": 0.0,
    "velocity_30d": None,  # NEW: Keepa's pre-calculated velocity (None = no data)
    "velocity_90d": None,  # NEW: Keepa's pre-calculated velocity (None = no data)
    # BSR/Rank: Assume worst case if missing (high rank = low sales)
    "bsr": 1_000_000,
    "sales_rank": 1_000_000,
    "sales_rank_filled": 1_000_000,
    # Revenue: 0 if unknown (will be calculated by ensure_revenue_proxy)
    "revenue_proxy": 0.0,
    "weekly_sales_filled": 0.0,
    "monthly_units": 0.0,
    "estimated_units": 0.0,
    "monthly_sold": 0,  # NEW: Amazon's actual monthly sold estimate
    "units_source": "unknown",  # NEW: Track data source
    # Pack size
    "number_of_items": 1,  # NEW: Pack size (1 = single item)
    "price_per_unit": 0.0,  # NEW: Price per unit
    # Competitive: Neutral defaults
    "competitor_count": 0,
    "competitor_oos_pct": 0.0,
    "price_gap_vs_competitor": 0.0,
    # AI Risk: No risk if unknown
    "thirty_day_risk": 0.0,
    "thirty_day_growth": 0.0,
    "price_erosion_risk": 0.0,
    "share_erosion_risk": 0.0,
    "stockout_risk": 0.0,
    "daily_burn_rate": 0.0,
    "cost_of_inaction": 0.0,
    # OOS: Assume in stock if unknown
    "outOfStockPercentage30": 0.0,
    "outOfStockPercentage90": 0.0,
    "oos_pct_30": 0.0,      # NEW: Alternate naming
    "oos_pct_90": 0.0,      # NEW: Alternate naming
    "oos_count_amazon_30": 0,  # NEW: OOS event count (more actionable)
    "oos_count_amazon_90": 0,  # NEW: OOS event count
    "days_until_stockout": 999,  # 999 = no stockout predicted
    # Subscribe & Save
    "is_sns": False,  # NEW: S&S eligibility
}

# Group D: Buy Box & Ownership Metrics
BUYBOX_METRICS = MetricGroup(
    name="Buy Box & Ownership",
    columns=[
        "amazon_bb_share",
        "buy_box_switches",
        "buyBoxStatsAmazon30",
        "buyBoxStatsAmazon90",
        "buyBoxStatsTopSeller30",
        "buyBoxStatsSellerCount30",
        # NEW (2026-01-21): Rich BB metrics from Keepa
        "bb_stats_amazon_30",
        "bb_stats_amazon_90",
        "bb_stats_top_seller_30",
        "bb_seller_count_30",
        "bb_top_seller_30",
    ],
    fill_strategy="ffill",
    default_value=0.5,  # Assume 50% if unknown
    max_gap_limit=4,
    description="Buy Box metrics use forward fill"
)

# Group D2: Buy Box Boolean Flags (don't fill - keep as None if unknown)
BUYBOX_FLAGS = MetricGroup(
    name="Buy Box Flags",
    columns=[
        "buybox_is_amazon",
        "buybox_is_fba",
        "buybox_is_backorder",
        "buybox_is_unqualified",
        "has_amazon_seller",
    ],
    fill_strategy="ffill",  # Forward fill for booleans
    default_value=None,  # Keep as None if unknown - don't assume
    max_gap_limit=1,  # Only fill 1 week for booleans
    description="Buy Box boolean flags - None means unknown"
)

# Group E: Velocity & Decay Metrics
VELOCITY_METRICS = MetricGroup(
    name="Velocity & Decay",
    columns=[
        "velocity_decay",
        "velocity_trend_30d",   # 30-day velocity trend (critical for AI)
        "velocity_trend_90d",   # 90-day velocity trend (critical for AI)
        "velocity_30d",         # NEW: Keepa's pre-calculated velocity
        "velocity_90d",         # NEW: Keepa's pre-calculated velocity
        "forecast_change",
        "deltaPercent30_SALES",
        "deltaPercent90_SALES",
        "deltaPercent30_COUNT_NEW",
        "deltaPercent90_COUNT_NEW",
    ],
    fill_strategy="interpolate",
    default_value=0.0,  # Neutral velocity (no change) - changed from 1.0 for trend metrics
    max_gap_limit=2,
    description="Velocity metrics interpolate smoothly"
)

# Group F: Competitive Intelligence Metrics (calculated by dashboard/AI)
COMPETITIVE_METRICS = MetricGroup(
    name="Competitive Intelligence",
    columns=[
        "competitor_count",
        "competitor_oos_pct",
        "price_gap_vs_competitor",
        "price_gap_vs_median",
        "median_competitor_price",
        "median_competitor_reviews",
        "review_advantage_pct",
        "best_competitor_rank",
        "worst_competitor_rank",
    ],
    fill_strategy="ffill",
    default_value=0.0,  # Neutral competitive position
    max_gap_limit=4,
    description="Competitive metrics use forward fill"
)

# Group G: AI/Risk Metrics (calculated by AI engine)
AI_RISK_METRICS = MetricGroup(
    name="AI Risk & Growth",
    columns=[
        "thirty_day_risk",
        "thirty_day_growth",
        "price_erosion_risk",
        "share_erosion_risk",
        "stockout_risk",
        "price_lift_opportunity",
        "conquest_opportunity",
        "daily_burn_rate",
        "cost_of_inaction",
    ],
    fill_strategy="ffill",
    default_value=0.0,  # No risk/growth if unknown
    max_gap_limit=4,
    description="AI-calculated risk metrics"
)

# Group H: Out of Stock Metrics (from Keepa)
OOS_METRICS = MetricGroup(
    name="Out of Stock",
    columns=[
        "outOfStockPercentage30",
        "outOfStockPercentage90",
        "oos_pct_30",           # NEW: Alternate naming
        "oos_pct_90",           # NEW: Alternate naming
        "oos_count_amazon_30",  # NEW: OOS event count (more actionable)
        "oos_count_amazon_90",  # NEW: OOS event count
        "days_until_stockout",
    ],
    fill_strategy="ffill",
    default_value=0.0,  # Assume in stock if unknown
    max_gap_limit=4,
    description="OOS metrics use forward fill"
)

# Group I: Product Attributes (static per product)
PRODUCT_ATTRIBUTES = MetricGroup(
    name="Product Attributes",
    columns=[
        "number_of_items",      # Pack size
        "price_per_unit",       # Per-unit price
        "monthly_sold",         # Amazon's sold estimate
        "seller_count",         # True seller count from sellerIds
        "is_sns",               # Subscribe & Save eligibility
    ],
    fill_strategy="ffill",
    default_value=1,  # Assume 1 for pack size
    max_gap_limit=8,
    description="Product attributes - mostly static"
)

# All metric groups
ALL_METRIC_GROUPS = [
    FINANCIAL_METRICS,
    PERFORMANCE_METRICS,
    SOCIAL_COMPETITIVE_METRICS,
    BUYBOX_METRICS,
    BUYBOX_FLAGS,       # NEW: Buy Box boolean flags
    VELOCITY_METRICS,
    COMPETITIVE_METRICS,
    AI_RISK_METRICS,
    OOS_METRICS,
    PRODUCT_ATTRIBUTES, # NEW: Pack size, monthly_sold, etc.
]


# =============================================================================
# CORE HEALING FUNCTION
# =============================================================================

def clean_and_interpolate_metrics(
    df: pd.DataFrame,
    group_by_column: str = "asin",
    verbose: bool = False,
    force: bool = False
) -> pd.DataFrame:
    """
    Universal Data Healer: Fill and interpolate ALL numerical metrics.
    
    The 3-Step Process:
    1. Detect numerical columns and their metric groups
    2. Apply group-specific fill strategies (interpolate/ffill/bfill)
    3. Apply default fallbacks for remaining gaps
    
    OPTIMIZATION: Skips healing if DataFrame was already healed (uses attrs flag).
    This prevents redundant healing when data passes through multiple pipeline stages.
    
    Args:
        df: DataFrame with time-series data
        group_by_column: Column to group by (usually "asin")
        verbose: Print detailed fill statistics
        force: Force healing even if already healed (default False)
        
    Returns:
        DataFrame with all numerical gaps filled and 'healed' attr set to True
    """
    # OPTIMIZATION: Skip if already healed (prevents 4x redundant healing)
    if not force and hasattr(df, 'attrs') and df.attrs.get('healed', False):
        if verbose:
            print("[DATA HEALER] Skipping - DataFrame already healed")
        return df
    
    df = df.copy()
    fill_stats = {}
    
    if verbose:
        print("\n" + "="*60)
        print("UNIVERSAL DATA HEALER - Starting")
        print("="*60)
    
    # CRITICAL: Ensure revenue_proxy exists FIRST (before metric group processing)
    # This creates the column from available sources if missing
    df = ensure_revenue_proxy(df, verbose=verbose)
    
    # Process each metric group
    for group in ALL_METRIC_GROUPS:
        if verbose:
            print(f"\n[{group.name}] Strategy: {group.fill_strategy.upper()}")
        
        for col in group.columns:
            if col not in df.columns:
                continue
            
            # Skip if column is not numerical
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Count gaps before filling
            gaps_before = df[col].isna().sum()
            if gaps_before == 0:
                continue
            
            # Apply fill strategy
            df = _apply_fill_strategy(
                df, 
                col, 
                group.fill_strategy, 
                group.max_gap_limit,
                group_by_column
            )
            
            # Apply default fallback for remaining gaps
            # FIX: Skip fillna if default is None (None means "keep as unknown")
            default = SPECIAL_DEFAULTS.get(col, group.default_value)
            if default is not None:
                df[col] = df[col].fillna(default)
            
            # Count gaps after filling
            gaps_after = df[col].isna().sum()
            fill_count = gaps_before - gaps_after
            
            fill_stats[col] = {
                "gaps_before": gaps_before,
                "gaps_filled": fill_count,
                "gaps_remaining": gaps_after,
                "strategy": group.fill_strategy,
                "default_used": default,
            }
            
            if verbose and fill_count > 0:
                print(f"  + {col}: Filled {fill_count} gaps (default={default})")
    
    if verbose:
        print("\n" + "="*60)
        print(f"HEALING COMPLETE - {len(fill_stats)} metrics processed")
        print("="*60)
        
        # Summary statistics
        total_filled = sum(s["gaps_filled"] for s in fill_stats.values())
        total_remaining = sum(s["gaps_remaining"] for s in fill_stats.values())
        print(f"\nTotal gaps filled: {total_filled:,}")
        print(f"Remaining gaps: {total_remaining:,}")
        
        if total_remaining > 0:
            print("\n[!] Warning: Some gaps could not be filled")
            for col, stats in fill_stats.items():
                if stats["gaps_remaining"] > 0:
                    print(f"  - {col}: {stats['gaps_remaining']} gaps remain")
    
    # OPTIMIZATION: Mark DataFrame as healed to prevent redundant healing
    df.attrs['healed'] = True
    
    return df


def _apply_fill_strategy(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    max_gap_limit: Optional[int],
    group_by_column: str
) -> pd.DataFrame:
    """
    Apply the specified fill strategy to a column.
    
    Strategies:
    - interpolate: Linear interpolation for smooth trends
    - ffill: Forward fill for step functions
    - bfill: Backward fill for early gaps
    """
    if strategy == "interpolate":
        # Linear interpolation within groups
        # FIX: Handle pandas 2.0+ where interpolate on all-NaN raises ValueError
        def safe_interpolate(x, limit):
            if len(x) <= 1 or x.isna().all():
                return x
            try:
                return x.interpolate(method='linear', limit=limit)
            except ValueError:
                # Fallback to ffill if interpolate fails (e.g., all NaN)
                return x.ffill(limit=limit)
        
        df[column] = df.groupby(group_by_column)[column].transform(
            lambda x: safe_interpolate(x, max_gap_limit)
        )
        # Forward fill remaining gaps
        df[column] = df.groupby(group_by_column)[column].ffill(limit=max_gap_limit)
        # Backward fill early gaps
        df[column] = df.groupby(group_by_column)[column].bfill(limit=1)
        
    elif strategy == "ffill":
        # Forward fill (for step functions like review counts)
        df[column] = df.groupby(group_by_column)[column].ffill(limit=max_gap_limit)
        # Backward fill early gaps
        df[column] = df.groupby(group_by_column)[column].bfill(limit=1)
        
    elif strategy == "bfill":
        # Backward fill (rarely used)
        df[column] = df.groupby(group_by_column)[column].bfill(limit=max_gap_limit)
        # Forward fill remaining
        df[column] = df.groupby(group_by_column)[column].ffill(limit=1)
    
    return df


# =============================================================================
# REVENUE PROXY CREATION (CRITICAL FOR COMMAND CENTER)
# =============================================================================

def ensure_revenue_proxy(
    df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Ensure revenue_proxy column exists and has valid values.
    
    Creates revenue_proxy from available columns if missing:
    1. avg_weekly_revenue * 4.33 (best - from historical data)
    2. price * monthly_units (good - from discovery)
    3. filled_price * estimated_units (fallback)
    4. 0.0 (last resort)
    
    Also creates weekly_sales_filled as alias for backward compatibility.
    """
    df = df.copy()
    
    # Check if revenue_proxy already exists with valid data
    has_valid_revenue = (
        'revenue_proxy' in df.columns and 
        df['revenue_proxy'].notna().any() and 
        (df['revenue_proxy'] > 0).any()
    )
    
    if not has_valid_revenue:
        if verbose:
            print(f"  Creating revenue_proxy from available columns...")
        
        # Priority 1: avg_weekly_revenue * 4.33 (from 90-day historical)
        if 'avg_weekly_revenue' in df.columns:
            df['revenue_proxy'] = pd.to_numeric(df['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
            if verbose:
                print(f"    Created from avg_weekly_revenue (historical)")
        
        # Priority 2: price * monthly_units (from discovery/BSR formula)
        elif 'price' in df.columns and 'monthly_units' in df.columns:
            df['revenue_proxy'] = (
                pd.to_numeric(df['price'], errors='coerce').fillna(0) *
                pd.to_numeric(df['monthly_units'], errors='coerce').fillna(0)
            )
            if verbose:
                print(f"    Created from price * monthly_units")
        
        # Priority 3: filled_price * estimated_units (dashboard columns)
        elif 'filled_price' in df.columns and 'estimated_units' in df.columns:
            df['revenue_proxy'] = (
                pd.to_numeric(df['filled_price'], errors='coerce').fillna(0) *
                pd.to_numeric(df['estimated_units'], errors='coerce').fillna(0)
            )
            if verbose:
                print(f"    Created from filled_price * estimated_units")
        
        # Priority 4: buy_box_price * estimated_units
        elif 'buy_box_price' in df.columns and 'estimated_units' in df.columns:
            df['revenue_proxy'] = (
                pd.to_numeric(df['buy_box_price'], errors='coerce').fillna(0) *
                pd.to_numeric(df['estimated_units'], errors='coerce').fillna(0)
            )
            if verbose:
                print(f"    Created from buy_box_price * estimated_units")
        
        # Last resort: default to 0
        else:
            df['revenue_proxy'] = 0.0
            if verbose:
                print(f"    No revenue sources found, defaulting to 0")
    else:
        # Column exists - ensure it's numeric
        df['revenue_proxy'] = pd.to_numeric(df['revenue_proxy'], errors='coerce').fillna(0)
    
    # Also create weekly_sales_filled as alias (backward compatibility)
    if 'weekly_sales_filled' not in df.columns:
        df['weekly_sales_filled'] = df['revenue_proxy'].copy()
    
    # Report data quality
    if verbose:
        products_with_revenue = (df['revenue_proxy'] > 0).sum()
        total_products = len(df)
        pct = products_with_revenue / total_products * 100 if total_products > 0 else 0
        print(f"  Revenue data quality: {products_with_revenue}/{total_products} ({pct:.0f}%) products have revenue")
    
    return df


def apply_variation_deduplication(
    df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply variation deduplication for BSR-derived metrics.
    
    When products share a parent_asin, they are variations of ONE listing.
    They all inherit the parent's BSR, so BSR-derived metrics (revenue, units)
    would be counted N times if we just sum. Fix: divide by sibling count.
    
    Example: ALOHA has 22 variations all sharing parent B0FP2N64VD with BSR=25
    Without fix: 22 products × $72k each = $1.5M (22x overcounted)
    With fix: $72k / 22 per product = $72k total (correct)
    
    Creates adjusted columns:
    - revenue_proxy_adjusted: revenue divided by sibling count
    - monthly_units_adjusted: units divided by sibling count
    - _sibling_count: number of products sharing same parent
    """
    if df.empty:
        df["revenue_proxy_adjusted"] = 0.0
        df["monthly_units_adjusted"] = 0.0
        df["_sibling_count"] = 1
        return df
    
    if "parent_asin" in df.columns:
        # Count siblings per parent ASIN (products with same parent)
        valid_parents = df["parent_asin"].notna() & (df["parent_asin"] != "")
        parent_counts = df[valid_parents].groupby("parent_asin").size()
        
        # Create sibling count column (default 1 for products without parent)
        df["_sibling_count"] = df["parent_asin"].map(parent_counts).fillna(1).astype(int)
        df.loc[~valid_parents, "_sibling_count"] = 1
        
        # Adjust revenue by dividing by sibling count
        if "revenue_proxy" in df.columns:
            df["revenue_proxy_adjusted"] = df["revenue_proxy"] / df["_sibling_count"]
        else:
            df["revenue_proxy_adjusted"] = 0.0
        
        # Also adjust units (also BSR-derived)
        if "monthly_units" in df.columns:
            df["monthly_units_adjusted"] = df["monthly_units"] / df["_sibling_count"]
        elif "estimated_units" in df.columns:
            df["monthly_units_adjusted"] = df["estimated_units"] / df["_sibling_count"]
        else:
            df["monthly_units_adjusted"] = 0.0
        
        # Adjust weekly sales if present
        if "weekly_sales_filled" in df.columns:
            df["weekly_sales_adjusted"] = df["weekly_sales_filled"] / df["_sibling_count"]
        
        if verbose:
            products_with_siblings = (df["_sibling_count"] > 1).sum()
            if products_with_siblings > 0:
                print(f"  Variation deduplication: {products_with_siblings} products share parent ASINs")
                avg_siblings = df[df["_sibling_count"] > 1]["_sibling_count"].mean()
                print(f"    Average sibling count: {avg_siblings:.1f}")
    else:
        # No parent_asin column - use raw values
        df["revenue_proxy_adjusted"] = df.get("revenue_proxy", 0)
        df["monthly_units_adjusted"] = df.get("monthly_units", df.get("estimated_units", 0))
        df["_sibling_count"] = 1
        if verbose:
            print("  Variation deduplication: No parent_asin column, using raw values")
    
    return df


def heal_market_snapshot(
    df: pd.DataFrame,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Comprehensive healer specifically for market snapshot DataFrames.
    
    Ensures ALL columns required by the Command Center dashboard exist with valid values.
    This is the single entry point for healing discovery/market data before dashboard processing.
    
    Critical columns ensured:
    - Financial: revenue_proxy, weekly_sales_filled, price, filled_price, buy_box_price
    - Performance: bsr, sales_rank, sales_rank_filled
    - Velocity: velocity_trend_30d, velocity_trend_90d, data_quality, data_weeks
    - Social: review_count, rating, new_offer_count
    - Competitive: competitor_oos_pct, price_gap_vs_competitor
    - AI: thirty_day_risk, thirty_day_growth, predictive_state
    - Deduplication: revenue_proxy_adjusted, monthly_units_adjusted, _sibling_count
    """
    df = df.copy()
    
    if verbose:
        print("\n" + "="*60)
        print("MARKET SNAPSHOT HEALER - Ensuring Dashboard Compatibility")
        print("="*60)
    
    # 1. REVENUE (most critical - prevents KeyError)
    df = ensure_revenue_proxy(df, verbose=verbose)
    
    # 2. PRICE COLUMNS (dashboard expects multiple price columns)
    price_sources = ['price', 'buy_box_price', 'filled_price', 'amazon_price', 'new_price']
    price_value = None
    for col in price_sources:
        if col in df.columns and df[col].notna().any():
            price_value = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Ensure all price columns exist
    for col in ['price', 'buy_box_price', 'filled_price']:
        if col not in df.columns:
            df[col] = price_value if price_value is not None else 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 3. BSR/RANK COLUMNS
    bsr_sources = ['bsr', 'sales_rank', 'sales_rank_filled']
    bsr_value = None
    for col in bsr_sources:
        if col in df.columns and df[col].notna().any():
            bsr_value = pd.to_numeric(df[col], errors='coerce')
            break
    
    for col in ['bsr', 'sales_rank_filled']:
        if col not in df.columns:
            df[col] = bsr_value if bsr_value is not None else 1_000_000
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1_000_000)
    
    # 4. VELOCITY COLUMNS (critical for AI engine)
    velocity_cols = {
        'velocity_trend_30d': 0.0,
        'velocity_trend_90d': 0.0,
        'data_quality': 'VERY_LOW',
        'data_weeks': 0,
    }
    for col, default in velocity_cols.items():
        if col not in df.columns:
            df[col] = default
        elif col in ['velocity_trend_30d', 'velocity_trend_90d']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        elif col == 'data_weeks':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(int)
    
    # 5. SOCIAL METRICS
    social_cols = {
        'review_count': 0,
        'rating': 0.0,
        'new_offer_count': 1,
    }
    for col, default in social_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 6. COMPETITIVE METRICS (calculated by dashboard, but ensure they exist)
    competitive_cols = {
        'competitor_oos_pct': 0.0,
        'price_gap_vs_competitor': 0.0,
        'competitor_count': 0,
    }
    for col, default in competitive_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 7. OOS METRICS (from Keepa)
    oos_cols = {
        'outOfStockPercentage90': 0.0,
        'outOfStockPercentage30': 0.0,
    }
    for col, default in oos_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 8. BUY BOX (critical for AI - prevents "zero buy box" false positives)
    if 'amazon_bb_share' not in df.columns:
        df['amazon_bb_share'] = 0.5  # Neutral default
    else:
        df['amazon_bb_share'] = pd.to_numeric(df['amazon_bb_share'], errors='coerce').fillna(0.5)
    
    # 8b. NEW BUY BOX METRICS (2026-01-21)
    buybox_numeric_cols = {
        'bb_stats_amazon_30': 0.5,
        'bb_stats_amazon_90': 0.5,
        'bb_stats_top_seller_30': 0.0,
        'bb_seller_count_30': 0,
        'bb_top_seller_30': 0.0,
    }
    for col, default in buybox_numeric_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 8c. BUY BOX BOOLEAN FLAGS - keep as None if unknown
    buybox_bool_cols = ['buybox_is_amazon', 'buybox_is_fba', 'buybox_is_backorder', 
                        'buybox_is_unqualified', 'has_amazon_seller']
    for col in buybox_bool_cols:
        if col not in df.columns:
            df[col] = None  # None means "unknown"
    
    # 8d. SELLER COUNT (from sellerIds array)
    if 'seller_count' not in df.columns:
        # Fallback to new_offer_count if available
        if 'new_offer_count' in df.columns:
            df['seller_count'] = df['new_offer_count']
        else:
            df['seller_count'] = 1
    else:
        df['seller_count'] = pd.to_numeric(df['seller_count'], errors='coerce').fillna(1).astype(int)
    
    # 8e. OOS COUNTS (more actionable than %)
    oos_count_cols = {
        'oos_count_amazon_30': 0,
        'oos_count_amazon_90': 0,
        'oos_pct_30': 0.0,
        'oos_pct_90': 0.0,
    }
    for col, default in oos_count_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 8f. VELOCITY (pre-calculated by Keepa)
    velocity_cols = {'velocity_30d': None, 'velocity_90d': None}
    for col, default in velocity_cols.items():
        if col not in df.columns:
            df[col] = default
    
    # 8g. PRODUCT ATTRIBUTES
    if 'number_of_items' not in df.columns:
        df['number_of_items'] = 1
    else:
        df['number_of_items'] = pd.to_numeric(df['number_of_items'], errors='coerce').fillna(1).clip(lower=1).astype(int)
    
    if 'monthly_sold' not in df.columns:
        df['monthly_sold'] = 0
    else:
        df['monthly_sold'] = pd.to_numeric(df['monthly_sold'], errors='coerce').fillna(0).astype(int)
    
    if 'is_sns' not in df.columns:
        df['is_sns'] = False
    
    if 'units_source' not in df.columns:
        df['units_source'] = 'bsr_formula'
    
    # 8h. PRICE PER UNIT (for fair comparison across pack sizes)
    if 'price_per_unit' not in df.columns:
        if 'price' in df.columns and 'number_of_items' in df.columns:
            df['price_per_unit'] = df['price'] / df['number_of_items'].clip(lower=1)
        else:
            df['price_per_unit'] = df.get('price', 0)
    
    # 9. UNITS COLUMNS
    units_cols = {
        'monthly_units': 0.0,
        'estimated_units': 0.0,
    }
    for col, default in units_cols.items():
        if col not in df.columns:
            # Try to calculate from revenue/price
            if 'revenue_proxy' in df.columns and 'price' in df.columns:
                price_safe = df['price'].replace(0, np.nan)
                df[col] = (df['revenue_proxy'] / price_safe).fillna(default)
            else:
                df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    # 10. VARIATION DEDUPLICATION
    # When products share a parent_asin, they are variations of ONE listing.
    # They all inherit the parent's BSR, so BSR-derived metrics (revenue, units)
    # would be counted N times if we just sum. Fix: divide by sibling count.
    df = apply_variation_deduplication(df, verbose=verbose)
    
    if verbose:
        cols_created = []
        critical_cols = ['revenue_proxy', 'weekly_sales_filled', 'price', 'bsr', 
                        'velocity_trend_30d', 'review_count', 'amazon_bb_share']
        for col in critical_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                cols_created.append(f"  [OK] {col}: {non_null}/{len(df)} valid")
        print("\nColumn Status:")
        for status in cols_created:
            print(status)
        print("="*60)
    
    return df


# =============================================================================
# SPECIALIZED HEALERS
# =============================================================================

def heal_price_metrics(
    df: pd.DataFrame,
    price_cols: List[str] = None,
    max_weeks: int = 4,
    group_by: str = "asin"
) -> pd.DataFrame:
    """
    Specialized healer for price metrics.
    
    Price hierarchy: buy_box_price → amazon_price → new_price → new_fba_price
    
    Args:
        df: DataFrame with price columns
        price_cols: List of price columns (in priority order)
        max_weeks: Maximum weeks to forward fill
        group_by: Column to group by
        
    Returns:
        DataFrame with filled prices
    """
    df = df.copy()
    
    if price_cols is None:
        price_cols = ["buy_box_price", "amazon_price", "new_price", "new_fba_price"]
    
    # Build effective price from hierarchy
    eff_p = pd.Series(np.nan, index=df.index)
    for col in price_cols:
        if col in df.columns:
            eff_p = eff_p.fillna(df[col])
    
    df["eff_p"] = eff_p
    
    # Forward fill with limit
    df["filled_price"] = df.groupby(group_by)["eff_p"].ffill(limit=max_weeks)
    
    # Interpolate remaining gaps
    df["filled_price"] = df.groupby(group_by)["filled_price"].transform(
        lambda x: _safe_interpolate(x, method='linear', limit=2)
    )
    
    # Final fallback to 0
    df["filled_price"] = df["filled_price"].fillna(0.0)
    
    return df


def heal_rank_metrics(
    df: pd.DataFrame,
    rank_col: str = "sales_rank",
    max_weeks: int = 3,
    group_by: str = "asin",
    worst_rank: float = 1_000_000
) -> pd.DataFrame:
    """
    Specialized healer for sales rank (BSR) metrics.
    
    Uses linear interpolation for short gaps, worst-case fallback for long gaps.
    
    Args:
        df: DataFrame with rank column
        rank_col: Name of rank column
        max_weeks: Maximum weeks to interpolate
        group_by: Column to group by
        worst_rank: Worst-case rank to assume
        
    Returns:
        DataFrame with filled ranks
    """
    df = df.copy()
    
    if rank_col not in df.columns:
        return df
    
    # Interpolate gaps
    df[f"{rank_col}_filled"] = df.groupby(group_by)[rank_col].transform(
        lambda x: _safe_interpolate(x, method='linear', limit=max_weeks)
    )
    
    # Forward fill short gaps
    df[f"{rank_col}_filled"] = df.groupby(group_by)[f"{rank_col}_filled"].ffill(limit=1)
    
    # Fallback to worst rank
    df[f"{rank_col}_filled"] = df[f"{rank_col}_filled"].fillna(worst_rank)
    
    return df


def heal_review_metrics(
    df: pd.DataFrame,
    review_col: str = "review_count",
    rating_col: str = "rating",
    group_by: str = "asin"
) -> pd.DataFrame:
    """
    Specialized healer for review and rating metrics.
    
    Reviews use forward fill (they only increase).
    Ratings use forward fill with interpolation (they can fluctuate slightly).
    
    Args:
        df: DataFrame with review/rating columns
        review_col: Review count column
        rating_col: Rating column
        group_by: Column to group by
        
    Returns:
        DataFrame with filled review metrics
    """
    df = df.copy()
    
    # Reviews: Forward fill only (they only go up)
    if review_col in df.columns:
        df[review_col] = df.groupby(group_by)[review_col].ffill()
        df[review_col] = df[review_col].fillna(0)
    
    # Ratings: Forward fill + light interpolation
    if rating_col in df.columns:
        df[rating_col] = df.groupby(group_by)[rating_col].ffill()
        df[rating_col] = df.groupby(group_by)[rating_col].transform(
            lambda x: _safe_interpolate(x, method='linear', limit=2)
        )
        df[rating_col] = df[rating_col].fillna(0.0)
    
    return df


def heal_competitive_metrics(
    df: pd.DataFrame,
    offer_count_cols: List[str] = None,
    buybox_cols: List[str] = None,
    group_by: str = "asin"
) -> pd.DataFrame:
    """
    Specialized healer for competitive metrics (offers, Buy Box).
    
    Args:
        df: DataFrame with competitive columns
        offer_count_cols: Offer count columns
        buybox_cols: Buy Box metric columns
        group_by: Column to group by
        
    Returns:
        DataFrame with filled competitive metrics
    """
    df = df.copy()
    
    if offer_count_cols is None:
        offer_count_cols = ["new_offer_count", "used_offer_count", "current_COUNT_NEW"]
    
    if buybox_cols is None:
        buybox_cols = ["amazon_bb_share", "buy_box_switches"]
    
    # Offer counts: Forward fill (step function)
    for col in offer_count_cols:
        if col in df.columns:
            df[col] = df.groupby(group_by)[col].ffill()
            # Default: Assume at least 1 new seller, 0 used
            default = 1 if "new" in col.lower() else 0
            df[col] = df[col].fillna(default)
    
    # Buy Box: Forward fill + interpolate
    for col in buybox_cols:
        if col in df.columns:
            df[col] = df.groupby(group_by)[col].ffill()
            df[col] = df.groupby(group_by)[col].transform(
                lambda x: _safe_interpolate(x, method='linear', limit=2)
            )
            # Default: Assume 50% if unknown (or 0 for switches)
            default = 0.5 if "share" in col.lower() else 0
            df[col] = df[col].fillna(default)
    
    return df


# =============================================================================
# DATA QUALITY REPORTING
# =============================================================================

def generate_data_quality_report(df: pd.DataFrame, group_by: str = "asin") -> Dict:
    """
    Generate a comprehensive data quality report.
    
    Reports on:
    - Gap counts per metric
    - Gap patterns per product
    - Completeness scores
    """
    report = {
        "total_rows": len(df),
        "total_products": df[group_by].nunique() if group_by in df.columns else 1,
        "metrics": {},
        "product_completeness": {},
    }
    
    # Analyze each numerical column
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        gaps = df[col].isna().sum()
        if gaps > 0:
            report["metrics"][col] = {
                "gaps": gaps,
                "gap_pct": gaps / len(df) * 100,
                "completeness": (len(df) - gaps) / len(df) * 100,
            }
    
    # Analyze per-product completeness
    if group_by in df.columns:
        for product, group in df.groupby(group_by):
            total_values = len(group) * len(numerical_cols)
            missing_values = group[numerical_cols].isna().sum().sum()
            completeness = (total_values - missing_values) / total_values * 100
            
            if completeness < 95:  # Only report products with <95% completeness
                report["product_completeness"][product] = {
                    "completeness_pct": completeness,
                    "missing_values": missing_values,
                }
    
    return report


def validate_healing(df: pd.DataFrame, critical_cols: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that critical columns have been successfully healed.
    
    Args:
        df: DataFrame to validate
        critical_cols: List of critical columns that must have no gaps
        
    Returns:
        (is_valid, list_of_issues)
    """
    if critical_cols is None:
        critical_cols = [
            "filled_price",
            "sales_rank",
            "review_count",
            "new_offer_count",
        ]
    
    issues = []
    
    for col in critical_cols:
        if col not in df.columns:
            issues.append(f"Missing critical column: {col}")
            continue
        
        gaps = df[col].isna().sum()
        if gaps > 0:
            issues.append(f"{col} has {gaps} remaining gaps")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Create test data with gaps
    test_data = {
        "asin": ["B001"] * 10 + ["B002"] * 10,
        "week_start": pd.date_range("2024-01-01", periods=10, freq="W").tolist() * 2,
        "filled_price": [10.0, np.nan, np.nan, 12.0, 13.0, np.nan, 14.0, 15.0, 16.0, 17.0] * 2,
        "sales_rank": [1000, 1100, np.nan, np.nan, 1300, 1400, np.nan, 1600, 1700, 1800] * 2,
        "review_count": [10, 10, np.nan, 12, 12, 13, 13, np.nan, 15, 16] * 2,
        "new_offer_count": [5, np.nan, np.nan, 6, 7, 7, np.nan, 8, 8, 9] * 2,
        "rating": [4.5, np.nan, 4.6, 4.6, np.nan, 4.7, 4.7, 4.8, np.nan, 4.9] * 2,
    }
    
    df_test = pd.DataFrame(test_data)
    
    print("\n" + "="*60)
    print("UNIVERSAL DATA HEALER TEST")
    print("="*60)
    
    print("\nBefore healing:")
    print(df_test[["asin", "filled_price", "sales_rank", "review_count"]].head(10))
    print(f"\nTotal NaN values: {df_test.isna().sum().sum()}")
    
    # Apply universal healing
    df_healed = clean_and_interpolate_metrics(df_test, verbose=True)
    
    print("\n\nAfter healing:")
    print(df_healed[["asin", "filled_price", "sales_rank", "review_count"]].head(10))
    print(f"\nTotal NaN values: {df_healed.isna().sum().sum()}")
    
    # Validate
    is_valid, issues = validate_healing(df_healed)
    print(f"\n\nValidation: {'[PASSED]' if is_valid else '[FAILED]'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
