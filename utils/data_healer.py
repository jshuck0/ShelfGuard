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
        "estimated_units",
        "eff_p",
        "synthetic_cogs",
        "landed_logistics",
        "net_margin",
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
SPECIAL_DEFAULTS = {
    "rating": 0.0,
    "current_RATING": 0.0,
    "review_count": 0,
    "current_COUNT_REVIEWS": 0,
    "new_offer_count": 1,  # Assume at least 1 seller
    "current_COUNT_NEW": 1,
    "used_offer_count": 0,
    "current_COUNT_USED": 0,
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
    ],
    fill_strategy="ffill",
    default_value=0.5,  # Assume 50% if unknown
    max_gap_limit=4,
    description="Buy Box metrics use forward fill"
)

# Group E: Velocity & Decay Metrics
VELOCITY_METRICS = MetricGroup(
    name="Velocity & Decay",
    columns=[
        "velocity_decay",
        "forecast_change",
        "deltaPercent30_SALES",
        "deltaPercent90_SALES",
        "deltaPercent30_COUNT_NEW",
        "deltaPercent90_COUNT_NEW",
    ],
    fill_strategy="interpolate",
    default_value=1.0,  # Neutral decay factor
    max_gap_limit=2,
    description="Velocity metrics interpolate smoothly"
)

# All metric groups
ALL_METRIC_GROUPS = [
    FINANCIAL_METRICS,
    PERFORMANCE_METRICS,
    SOCIAL_COMPETITIVE_METRICS,
    BUYBOX_METRICS,
    VELOCITY_METRICS,
]


# =============================================================================
# CORE HEALING FUNCTION
# =============================================================================

def clean_and_interpolate_metrics(
    df: pd.DataFrame,
    group_by_column: str = "asin",
    verbose: bool = False
) -> pd.DataFrame:
    """
    Universal Data Healer: Fill and interpolate ALL numerical metrics.
    
    The 3-Step Process:
    1. Detect numerical columns and their metric groups
    2. Apply group-specific fill strategies (interpolate/ffill/bfill)
    3. Apply default fallbacks for remaining gaps
    
    Args:
        df: DataFrame with time-series data
        group_by_column: Column to group by (usually "asin")
        verbose: Print detailed fill statistics
        
    Returns:
        DataFrame with all numerical gaps filled
    """
    df = df.copy()
    fill_stats = {}
    
    if verbose:
        print("\n" + "="*60)
        print("UNIVERSAL DATA HEALER - Starting")
        print("="*60)
    
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
            default = SPECIAL_DEFAULTS.get(col, group.default_value)
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
        df[column] = df.groupby(group_by_column)[column].transform(
            lambda x: x.interpolate(method='linear', limit=max_gap_limit) if len(x) > 1 else x
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
        lambda x: x.interpolate(method='linear', limit=2) if len(x) > 1 else x
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
        lambda x: x.interpolate(method='linear', limit=max_weeks) if len(x) > 1 else x
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
            lambda x: x.interpolate(method='linear', limit=2) if len(x) > 1 else x
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
                lambda x: x.interpolate(method='linear', limit=2) if len(x) > 1 else x
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
