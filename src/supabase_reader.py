"""
ShelfGuard Supabase Data Reader
================================
The Oracle's Data Access Layer

This module provides fast, cached access to harvested data from Supabase.
Replaces direct Keepa API calls in the dashboard for instant page loads.

Key Features:
- Cached reads with configurable TTL
- Fallback to session state for new projects
- Trend comparison queries
- Project-scoped data access
- SINGLETON PATTERN: Single connection reused across all calls

REFACTORED: Uses singleton pattern to avoid creating 20+ connections per request.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Optional, Dict, Tuple
from datetime import date, datetime, timedelta
from supabase import Client


# ============================================
# SUPABASE SINGLETON - Reuses single connection
# ============================================

_supabase_client: Optional[Client] = None
_known_snapshot_columns: Optional[set] = None  # Cached column names for product_snapshots


def get_supabase_client() -> Client:
    """
    Get or create Supabase client using singleton pattern.
    
    PERFORMANCE: This avoids creating 20+ connections per request.
    The client is cached in a module-level variable and reused.
    
    Returns: Authenticated Supabase client (singleton)
    """
    global _supabase_client
    
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(st.secrets["url"], st.secrets["key"])
    
    return _supabase_client


def create_supabase_client() -> Client:
    """
    DEPRECATED: Use get_supabase_client() instead.
    
    Kept for backward compatibility - now just calls get_supabase_client().
    """
    return get_supabase_client()


def _get_known_snapshot_columns() -> set:
    """
    Discover which columns exist in product_snapshots.
    Cached so we only query once per process lifetime.
    Prevents upsert failures when schema hasn't been migrated yet.
    """
    global _known_snapshot_columns
    if _known_snapshot_columns is not None:
        return _known_snapshot_columns
    try:
        sb = get_supabase_client()
        probe = sb.table("product_snapshots").select("*").limit(1).execute()
        if probe.data:
            _known_snapshot_columns = set(probe.data[0].keys())
        else:
            # Table is empty — fall back to a conservative set.
            # All columns that existed before Phase A migration.
            _known_snapshot_columns = {
                "id", "asin", "snapshot_date", "title", "brand", "parent_asin",
                "main_image", "sales_rank", "buy_box_price", "amazon_price",
                "new_fba_price", "filled_price", "new_offer_count", "review_count",
                "rating", "amazon_bb_share", "buy_box_switches", "estimated_units",
                "estimated_weekly_revenue", "competitor_oos_pct",
                "category_id", "category_name", "category_tree", "category_root",
                "fetched_at", "source",
                # Extended signals (added previously)
                "monthly_sold", "number_of_items", "price_per_unit",
                "buybox_is_amazon", "buybox_is_fba", "buybox_is_backorder",
                "has_amazon_seller", "seller_count",
                "oos_count_amazon_30", "oos_count_amazon_90", "oos_pct_30", "oos_pct_90",
                "velocity_30d", "velocity_90d",
                "bb_seller_count_30", "bb_top_seller_30",
                "units_source", "is_sns",
            }
    except Exception:
        _known_snapshot_columns = set()  # Will not filter anything
    return _known_snapshot_columns


def _strip_unknown_columns(record: dict) -> dict:
    """Remove keys that don't exist as columns in product_snapshots."""
    known = _get_known_snapshot_columns()
    if not known:
        return record  # Can't filter — pass through
    return {k: v for k, v in record.items() if k in known}


@st.cache_data(ttl=300)  # 5 minute cache
def load_latest_snapshots(asins: Tuple[str, ...]) -> pd.DataFrame:
    """
    Load the most recent snapshot for each ASIN.
    
    Args:
        asins: Tuple of ASIN strings (tuple for cache hashability)
        
    Returns:
        DataFrame with latest snapshot data for each ASIN
    """
    if not asins:
        return pd.DataFrame()
    
    try:
        supabase = create_supabase_client()
        
        # Query latest snapshots for these ASINs
        # Using raw SQL via RPC would be more efficient, but this works
        asin_list = list(asins)
        
        # Fetch all recent snapshots and filter to latest per ASIN
        result = supabase.table("product_snapshots").select("*").in_(
            "asin", asin_list
        ).order("snapshot_date", desc=True).execute()
        
        if not result.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Keep only the latest snapshot per ASIN
        df = df.sort_values("snapshot_date", ascending=False)
        df = df.drop_duplicates(subset=["asin"], keep="first")
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Failed to load snapshots from Supabase: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_snapshot_history(
    asins: Tuple[str, ...],
    days: int = 30
) -> pd.DataFrame:
    """
    Load snapshot history for trend analysis.
    
    Args:
        asins: Tuple of ASIN strings
        days: Number of days of history to load
        
    Returns:
        DataFrame with snapshot history (multiple rows per ASIN)
    """
    if not asins:
        return pd.DataFrame()
    
    try:
        supabase = create_supabase_client()
        
        cutoff_date = (date.today() - timedelta(days=days)).isoformat()
        asin_list = list(asins)
        
        result = supabase.table("product_snapshots").select("*").in_(
            "asin", asin_list
        ).gte("snapshot_date", cutoff_date).order("snapshot_date", desc=True).execute()
        
        if not result.data:
            return pd.DataFrame()
        
        return pd.DataFrame(result.data)
        
    except Exception as e:
        st.warning(f"⚠️ Failed to load snapshot history: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=60)  # 1 minute cache for trends
def load_snapshot_trends(asins: Tuple[str, ...]) -> pd.DataFrame:
    """
    Load trend data comparing today vs yesterday.
    
    Args:
        asins: Tuple of ASIN strings
        
    Returns:
        DataFrame with trend metrics (price_delta, rank_delta, etc.)
    """
    if not asins:
        return pd.DataFrame()
    
    try:
        # Load last 2 days of snapshots
        history = load_snapshot_history(asins, days=2)
        
        if history.empty:
            return pd.DataFrame()
        
        # Pivot to compare today vs yesterday
        history["snapshot_date"] = pd.to_datetime(history["snapshot_date"]).dt.date
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        today_df = history[history["snapshot_date"] == today].set_index("asin")
        yesterday_df = history[history["snapshot_date"] == yesterday].set_index("asin")
        
        if today_df.empty:
            # Use most recent available data
            today_df = history.sort_values("snapshot_date", ascending=False).drop_duplicates(
                subset=["asin"]
            ).set_index("asin")
        
        # Calculate deltas
        trends = []
        for asin in today_df.index:
            row = {"asin": asin}
            
            # Current values
            row["current_price"] = today_df.loc[asin, "buy_box_price"] if "buy_box_price" in today_df.columns else None
            row["current_rank"] = today_df.loc[asin, "sales_rank"] if "sales_rank" in today_df.columns else None
            row["current_bb_share"] = today_df.loc[asin, "amazon_bb_share"] if "amazon_bb_share" in today_df.columns else None
            row["current_revenue"] = today_df.loc[asin, "estimated_weekly_revenue"] if "estimated_weekly_revenue" in today_df.columns else None
            
            # Calculate deltas if we have yesterday's data
            if asin in yesterday_df.index:
                prev_price = yesterday_df.loc[asin, "buy_box_price"] if "buy_box_price" in yesterday_df.columns else None
                prev_rank = yesterday_df.loc[asin, "sales_rank"] if "sales_rank" in yesterday_df.columns else None
                prev_bb = yesterday_df.loc[asin, "amazon_bb_share"] if "amazon_bb_share" in yesterday_df.columns else None
                
                # Price change %
                if prev_price and prev_price > 0 and row["current_price"]:
                    row["price_change_pct"] = (row["current_price"] - prev_price) / prev_price * 100
                
                # Rank improvement % (lower is better, so positive = improvement)
                if prev_rank and prev_rank > 0 and row["current_rank"]:
                    row["rank_improvement_pct"] = (prev_rank - row["current_rank"]) / prev_rank * 100
                
                # BB share change (absolute points)
                if prev_bb is not None and row["current_bb_share"] is not None:
                    row["bb_share_change_pct"] = (row["current_bb_share"] - prev_bb) * 100
            
            trends.append(row)
        
        return pd.DataFrame(trends)
        
    except Exception as e:
        st.warning(f"⚠️ Failed to calculate trends: {str(e)}")
        return pd.DataFrame()


def load_project_data(
    project_asins: List[str],
    use_cache: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load project data from Supabase with fallback to session state.
    
    This is the main entry point for the dashboard.
    
    Args:
        project_asins: List of ASINs for the project
        use_cache: Whether to use Streamlit cache
        
    Returns:
        Tuple of (DataFrame with product data, stats dict)
    """
    stats = {
        "source": "unknown",
        "asin_count": len(project_asins),
        "cache_hit": False
    }
    
    if not project_asins:
        return pd.DataFrame(), stats
    
    # Convert to tuple for cache hashability
    asin_tuple = tuple(sorted([a.strip().upper() for a in project_asins]))
    
    # Try Supabase first
    df = load_latest_snapshots(asin_tuple)
    
    if not df.empty:
        stats["source"] = "supabase"
        stats["cache_hit"] = True
        stats["snapshot_date"] = df["snapshot_date"].max() if "snapshot_date" in df.columns else None
        
        # Map columns to expected dashboard format
        df = _normalize_snapshot_to_dashboard(df)
        
        return df, stats
    
    # Fallback to session state (for newly created projects)
    session_data = st.session_state.get("active_project_data", pd.DataFrame())
    
    if not session_data.empty:
        stats["source"] = "session_state"
        stats["cache_hit"] = False
        return session_data, stats
    
    # No data available
    stats["source"] = "none"
    return pd.DataFrame(), stats


def _normalize_snapshot_to_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize snapshot DataFrame columns to match dashboard expectations.
    
    Maps product_snapshots schema to the format expected by shelfguard_app.py
    
    CRITICAL: Also converts numeric columns to proper dtypes to avoid
    'cannot use method nlargest with dtype object' errors.
    """
    import numpy as np
    
    # Column mapping: snapshot_column -> dashboard_column
    # Most columns pass through with the same name; only a few need renaming.
    column_map = {
        "buy_box_price": "buy_box_price",
        "amazon_price": "amazon_price",
        "new_fba_price": "new_fba_price",
        "sales_rank": "sales_rank_filled",
        "amazon_bb_share": "amazon_bb_share",
        "buy_box_switches": "buy_box_switches",
        "new_offer_count": "new_offer_count",
        "review_count": "review_count",
        "rating": "rating",
        "estimated_units": "estimated_units",
        "estimated_weekly_revenue": "weekly_sales_filled",
        "filled_price": "filled_price",
        "title": "title",
        "brand": "brand",
        "parent_asin": "parent_asin",
        "main_image": "main_image",
        "asin": "asin",
        # Extended product signals (passthrough — same names)
        "monthly_sold": "monthly_sold",
        "number_of_items": "number_of_items",
        "price_per_unit": "price_per_unit",
        "buybox_is_amazon": "buybox_is_amazon",
        "buybox_is_fba": "buybox_is_fba",
        "buybox_is_backorder": "buybox_is_backorder",
        "has_amazon_seller": "has_amazon_seller",
        "seller_count": "seller_count",
        "oos_count_amazon_30": "oos_count_amazon_30",
        "oos_count_amazon_90": "oos_count_amazon_90",
        "oos_pct_30": "oos_pct_30",
        "oos_pct_90": "oos_pct_90",
        "velocity_30d": "velocity_30d",
        "velocity_90d": "velocity_90d",
        "bb_seller_count_30": "bb_seller_count_30",
        "bb_top_seller_30": "bb_top_seller_30",
        "units_source": "units_source",
        "is_sns": "is_sns",
        # Phase A signals (passthrough — same names)
        "return_rate": "return_rate",
        "sales_rank_drops_30": "sales_rank_drops_30",
        "sales_rank_drops_90": "sales_rank_drops_90",
        "monthly_sold_delta": "monthly_sold_delta",
        "top_comp_bb_share_30": "top_comp_bb_share_30",
        "active_ingredients_raw": "active_ingredients_raw",
        "item_type_keyword": "item_type_keyword",
        "has_buybox_stats": "has_buybox_stats",
        "has_monthly_sold_history": "has_monthly_sold_history",
        "has_active_ingredients": "has_active_ingredients",
        "has_sales_rank_drops": "has_sales_rank_drops",
    }
    
    # Rename columns that exist
    rename_map = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # CRITICAL: Convert numeric columns to proper dtypes
    # This prevents 'cannot use method nlargest with dtype object' errors
    numeric_columns_fill_zero = [
        "weekly_sales_filled", "revenue_proxy", "filled_price", "buy_box_price",
        "amazon_price", "new_fba_price", "sales_rank_filled", "amazon_bb_share",
        "estimated_units", "rating", "review_count", "new_offer_count",
        # Extended signals
        "monthly_sold", "number_of_items", "price_per_unit", "seller_count",
        "oos_count_amazon_30", "oos_count_amazon_90", "oos_pct_30", "oos_pct_90",
        "velocity_30d", "velocity_90d", "bb_seller_count_30", "bb_top_seller_30",
    ]
    
    for col in numeric_columns_fill_zero:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Phase A numeric columns: coerce to numeric but PRESERVE NaN.
    # None/NaN means "no data" which is semantically different from 0.
    # (e.g. top_comp_bb_share_30=None means "no BB signal" vs 0="zero share")
    phase_a_numeric_preserve_nan = [
        "return_rate", "sales_rank_drops_30", "sales_rank_drops_90",
        "monthly_sold_delta", "top_comp_bb_share_30",
    ]

    for col in phase_a_numeric_preserve_nan:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # NaN preserved

    # CRITICAL: Convert boolean columns to proper dtypes
    # This prevents type errors when using .any(), .sum(), etc.
    boolean_columns = [
        "buybox_is_amazon", "buybox_is_fba", "buybox_is_backorder",
        "has_amazon_seller", "is_sns", "amazon_unstable",
        # Phase A data presence flags
        "has_buybox_stats", "has_monthly_sold_history",
        "has_active_ingredients", "has_sales_rank_drops",
    ]

    for col in boolean_columns:
        if col in df.columns:
            # Convert to boolean, handling NaN, floats (0.0/1.0), strings, etc.
            df[col] = df[col].apply(lambda x:
                False if pd.isna(x) or x is None
                else bool(int(x)) if isinstance(x, (int, float))
                else bool(x) if isinstance(x, bool)
                else str(x).lower() in ('true', '1', 'yes')
            )

    # Ensure required columns exist with proper values
    if "weekly_sales_filled" not in df.columns or df["weekly_sales_filled"].sum() == 0:
        if "filled_price" in df.columns and "estimated_units" in df.columns:
            df["weekly_sales_filled"] = pd.to_numeric(df["estimated_units"], errors='coerce').fillna(0) * \
                                        pd.to_numeric(df["filled_price"], errors='coerce').fillna(0)
        else:
            df["weekly_sales_filled"] = 0.0

    # Ensure weekly_sales_filled is numeric
    df["weekly_sales_filled"] = pd.to_numeric(df["weekly_sales_filled"], errors='coerce').fillna(0)

    # === STANDARDIZED REVENUE COLUMNS (2026-01-30) ===
    # Base unit: WEEKLY (weekly_revenue is the source of truth)
    # Monthly is always calculated as weekly * 4.33
    df["weekly_revenue"] = df["weekly_sales_filled"].copy()  # Source of truth
    df["monthly_revenue"] = df["weekly_revenue"] * 4.33  # Derived

    # Legacy alias for backward compatibility (to be deprecated)
    df["revenue_proxy"] = df["weekly_revenue"].copy()

    # === DATA HEALER: Fill any remaining gaps ===
    # Ensures all metrics have values before reaching AI engine
    try:
        from utils.data_healer import clean_and_interpolate_metrics
        df = clean_and_interpolate_metrics(df, group_by_column="asin", verbose=False)
    except ImportError:
        pass  # Data healer not available

    return df


def check_data_freshness(asins: List[str]) -> Dict:
    """
    Check how fresh the cached data is for a set of ASINs.
    
    Returns:
        Dict with freshness information
    """
    if not asins:
        return {"has_data": False, "freshness_hours": None}
    
    try:
        supabase = create_supabase_client()
        
        # Get most recent snapshot date for any of these ASINs
        result = supabase.table("product_snapshots").select(
            "snapshot_date, fetched_at"
        ).in_("asin", asins).order("fetched_at", desc=True).limit(1).execute()
        
        if not result.data:
            return {"has_data": False, "freshness_hours": None}
        
        from datetime import datetime
        fetched_at = result.data[0].get("fetched_at")
        
        if fetched_at:
            fetched_dt = pd.to_datetime(fetched_at)
            hours_ago = (datetime.now(fetched_dt.tzinfo) - fetched_dt).total_seconds() / 3600
            
            return {
                "has_data": True,
                "freshness_hours": round(hours_ago, 1),
                "snapshot_date": result.data[0].get("snapshot_date"),
                "is_stale": hours_ago > 24
            }
        
        return {"has_data": True, "freshness_hours": None}
        
    except Exception as e:
        return {"has_data": False, "error": str(e)}


def get_market_snapshot_from_cache(
    project_asins: List[str],
    seed_brand: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a market snapshot from cached data.

    Separates "your" products (matching seed_brand) from competitors.

    Args:
        project_asins: All ASINs in the project
        seed_brand: Brand to identify as "yours" (optional)

    Returns:
        Tuple of (market_snapshot DataFrame, market_stats dict)
        Note: Revenue columns include both weekly (base) and monthly (derived).
    """
    df, stats = load_project_data(project_asins)

    if df.empty:
        return df, {"total_revenue": 0, "weekly_revenue": 0, "monthly_revenue": 0, "your_revenue": 0, "competitor_revenue": 0}

    # === STANDARDIZED REVENUE COLUMNS (2026-01-30) ===
    # Base unit: WEEKLY (weekly_revenue is the source of truth)
    # Monthly is always calculated as weekly * 4.33
    WEEKS_PER_MONTH = 4.33

    # Ensure weekly_revenue exists (should come from load_project_data)
    if "weekly_revenue" not in df.columns:
        if "weekly_sales_filled" in df.columns:
            df["weekly_revenue"] = pd.to_numeric(df["weekly_sales_filled"], errors='coerce').fillna(0)
        elif "estimated_weekly_revenue" in df.columns:
            df["weekly_revenue"] = pd.to_numeric(df["estimated_weekly_revenue"], errors='coerce').fillna(0)
        else:
            df["weekly_revenue"] = 0.0

    # Ensure monthly_revenue exists
    if "monthly_revenue" not in df.columns:
        df["monthly_revenue"] = df["weekly_revenue"] * WEEKS_PER_MONTH

    # Legacy alias (to be deprecated)
    df["revenue_proxy"] = df["weekly_revenue"].copy()

    # ==========================================================================
    # VARIATION DEDUPLICATION: Prevent revenue overcounting for child variations
    # ==========================================================================
    # When products share the same parent_asin, they are variations of ONE listing.
    # They all inherit the parent's BSR, so revenue would be counted N times if
    # we just sum all products. Fix: divide revenue by sibling count.
    #
    # Example: ALOHA has 22 variations all sharing parent B0FP2N64VD with BSR=25
    # Without fix: 22 products × $72k each = $1.5M (22x overcounted)
    # With fix: $72k / 22 per product = $72k total (correct)
    # ==========================================================================

    if "parent_asin" in df.columns:
        # Count siblings per parent ASIN (products with same parent)
        parent_counts = df[df["parent_asin"].notna() & (df["parent_asin"] != "")].groupby("parent_asin").size()

        # Create a sibling count column (default 1 for products without parent)
        df["_sibling_count"] = df["parent_asin"].map(parent_counts).fillna(1).astype(int)
        df.loc[df["parent_asin"].isna() | (df["parent_asin"] == ""), "_sibling_count"] = 1

        # Adjust BOTH weekly and monthly revenue by sibling count
        df["weekly_revenue_adjusted"] = df["weekly_revenue"] / df["_sibling_count"]
        df["monthly_revenue_adjusted"] = df["monthly_revenue"] / df["_sibling_count"]

        # Legacy alias
        df["revenue_proxy_adjusted"] = df["weekly_revenue_adjusted"].copy()

        # Also adjust units if present (also BSR-derived)
        if "monthly_units" in df.columns:
            df["monthly_units_adjusted"] = df["monthly_units"] / df["_sibling_count"]
        elif "weekly_units" in df.columns:
            df["weekly_units_adjusted"] = df["weekly_units"] / df["_sibling_count"]
            df["monthly_units_adjusted"] = df["weekly_units_adjusted"] * WEEKS_PER_MONTH
        elif "estimated_units" in df.columns:
            df["monthly_units_adjusted"] = df["estimated_units"] / df["_sibling_count"]
        else:
            df["monthly_units_adjusted"] = 0.0
    else:
        df["weekly_revenue_adjusted"] = df["weekly_revenue"]
        df["monthly_revenue_adjusted"] = df["monthly_revenue"]
        df["revenue_proxy_adjusted"] = df["weekly_revenue"]
        df["monthly_units_adjusted"] = df.get("monthly_units", df.get("estimated_units", 0))
        df["_sibling_count"] = 1

    # Calculate market metrics using ADJUSTED revenue
    # Primary: Weekly (base unit), Secondary: Monthly (derived)
    total_weekly_revenue = df["weekly_revenue_adjusted"].sum()
    total_monthly_revenue = df["monthly_revenue_adjusted"].sum()
    total_units = df["monthly_units_adjusted"].sum() if "monthly_units_adjusted" in df.columns else 0

    market_stats = {
        "total_revenue": total_weekly_revenue,  # Weekly, deduplicated (DEFAULT)
        "weekly_revenue": total_weekly_revenue,  # Explicit weekly
        "monthly_revenue": total_monthly_revenue,  # Explicit monthly
        "total_units": total_units,  # Monthly units, deduplicated
        "total_products": len(df),
        "source": stats.get("source", "unknown")
    }

    # Split by brand if seed_brand provided
    if seed_brand and "brand" in df.columns:
        df["is_yours"] = df["brand"].str.lower().str.contains(seed_brand.lower(), na=False)
        your_df = df[df["is_yours"]]
        competitor_df = df[~df["is_yours"]]

        # Use adjusted revenue for market share calculations (weekly = base unit)
        market_stats["your_revenue"] = your_df["weekly_revenue_adjusted"].sum() if not your_df.empty else 0
        market_stats["your_weekly_revenue"] = market_stats["your_revenue"]
        market_stats["your_monthly_revenue"] = your_df["monthly_revenue_adjusted"].sum() if not your_df.empty else 0
        market_stats["competitor_revenue"] = competitor_df["weekly_revenue_adjusted"].sum() if not competitor_df.empty else 0
        market_stats["competitor_weekly_revenue"] = market_stats["competitor_revenue"]
        market_stats["competitor_monthly_revenue"] = competitor_df["monthly_revenue_adjusted"].sum() if not competitor_df.empty else 0
        market_stats["your_product_count"] = len(your_df)
        market_stats["competitor_product_count"] = len(competitor_df)

        if total_weekly_revenue > 0:
            market_stats["market_share"] = (market_stats["your_revenue"] / total_weekly_revenue) * 100

    return df, market_stats


def cache_market_snapshot(
    market_snapshot: pd.DataFrame,
    df_weekly: Optional[pd.DataFrame] = None,
    category_context: Optional[Dict] = None
) -> int:
    """
    Cache market snapshot data to product_snapshots table.

    Called during project creation to enable instant loads on return visits.

    ENHANCEMENT 2.3: Now accepts category_context to consolidate all fields
    in a single write, avoiding duplicate writes from NetworkIntelligenceAccumulator.

    Args:
        market_snapshot: DataFrame from discovery phase
        df_weekly: Optional detailed weekly data (preferred if available)
        category_context: Optional dict with category_id, category_name, category_tree, category_root

    Returns:
        Number of records cached
    """
    import numpy as np
    from datetime import date, datetime

    # Use df_weekly if available (has more detail), otherwise use market_snapshot
    source_df = df_weekly if df_weekly is not None and not df_weekly.empty else market_snapshot

    if source_df.empty:
        return 0

    # CRITICAL FIX: df_weekly has multiple rows per ASIN (one per week).
    # We need to aggregate to get the MOST RECENT NON-NULL values per ASIN.
    # The most recent week might have NaN for some metrics, so we forward-fill then take last.
    if df_weekly is not None and not df_weekly.empty and "week_start" in df_weekly.columns:
        # Sort by week ascending
        df_sorted = df_weekly.sort_values(["asin", "week_start"], ascending=True)
        
        # Forward-fill NaN values within each ASIN group, then take the last row
        # This ensures we get the most recent non-null value for each metric
        df_filled = df_sorted.groupby("asin", group_keys=False).apply(
            lambda g: g.ffill()
        )
        
        # Take the last (most recent) row per ASIN after forward-fill
        source_df = df_filled.groupby("asin", as_index=False).last()

    try:
        supabase = create_supabase_client()
        snapshot_date = date.today().isoformat()

        # Extract category context if provided
        category_id = category_context.get("category_id") if category_context else None
        category_name = category_context.get("category_name") if category_context else None
        category_tree = category_context.get("category_tree") if category_context else None
        category_root = category_context.get("category_root") if category_context else None

        # Build snapshot records
        # IMPORTANT: Discovery phase uses different column names than Keepa client:
        # Discovery: bsr, price, revenue_proxy, monthly_units
        # Keepa:     sales_rank_filled, filled_price, weekly_sales_filled, estimated_units
        records = []
        for _, row in source_df.iterrows():
            asin = row.get("asin")
            if not asin:
                continue

            # Convert row to dict for easier access
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

            # Extract price with proper fallback (handles NaN correctly)
            price_val = _get_first_valid(
                row_dict,
                ["buy_box_price", "filled_price", "price", "avg_price"],
                _safe_float
            )
            
            # Extract sales rank with proper fallback
            rank_val = _get_first_valid(
                row_dict,
                ["sales_rank_filled", "sales_rank", "bsr"],
                _safe_int
            )
            
            # Extract revenue with proper fallback
            # Priority: weekly_sales_filled (already weekly) > estimated_weekly_revenue (already weekly)
            # Then fallback to revenue_proxy (MONTHLY - needs conversion)
            revenue_val = _get_first_valid(
                row_dict,
                ["weekly_sales_filled", "estimated_weekly_revenue"],
                _safe_float
            )

            # If no weekly value found, check for monthly revenue_proxy and convert
            if revenue_val is None:
                monthly_revenue = _safe_float(row_dict.get("revenue_proxy"))
                if monthly_revenue and monthly_revenue > 0:
                    revenue_val = monthly_revenue / 4.33  # Convert monthly to weekly

            # Extract units with proper fallback
            units_val = _get_first_valid(
                row_dict,
                ["estimated_units", "monthly_units"],
                _safe_int
            )

            # If we have price and units but no revenue, calculate it
            if revenue_val is None and price_val and units_val:
                revenue_val = price_val * units_val / 4.33  # Weekly from monthly units

            # Extract additional metrics with fallback chains (use row_dict)
            bb_share_val = _get_first_valid(
                row_dict,
                ["amazon_bb_share", "buybox_share", "bb_share"],
                _safe_float
            )
            review_count_val = _get_first_valid(
                row_dict,
                ["review_count", "reviews", "current_COUNT_REVIEWS"],
                _safe_int
            )
            rating_val = _get_first_valid(
                row_dict,
                ["rating", "current_RATING"],
                _safe_float
            )
            offer_count_val = _get_first_valid(
                row_dict,
                ["new_offer_count", "offer_count", "current_COUNT_NEW"],
                _safe_int
            )

            # Apply data healer defaults for missing metrics
            # This prevents AI from seeing NULL as "zero buy box ownership"
            # and making incorrect TERMINAL recommendations
            DEFAULT_BB_SHARE = 0.5  # Neutral assumption when unknown
            DEFAULT_REVIEW_COUNT = None  # Keep as None - AI should see "no data"
            DEFAULT_RATING = None  # Keep as None
            DEFAULT_OFFER_COUNT = 1  # At least 1 seller
            
            record = {
                "asin": str(asin).strip().upper(),
                "snapshot_date": snapshot_date,
                "buy_box_price": price_val,
                "amazon_price": _safe_float(row_dict.get("amazon_price")),
                "new_fba_price": _safe_float(row_dict.get("new_fba_price")),
                "sales_rank": rank_val,
                # CRITICAL: Apply defaults to prevent AI misclassification
                "amazon_bb_share": bb_share_val if bb_share_val is not None else DEFAULT_BB_SHARE,
                "buy_box_switches": _safe_int(row_dict.get("buy_box_switches")) or 0,
                "new_offer_count": offer_count_val if offer_count_val is not None else DEFAULT_OFFER_COUNT,
                "review_count": review_count_val,  # Keep None - handled by AI
                "rating": rating_val,  # Keep None - handled by AI
                "estimated_units": units_val,
                "estimated_weekly_revenue": revenue_val,
                "filled_price": price_val,  # Use same as buy_box_price
                "title": str(row_dict.get("title", ""))[:500] if row_dict.get("title") else None,
                "brand": str(row_dict.get("brand", "")) if row_dict.get("brand") else None,
                "parent_asin": str(row_dict.get("parent_asin", "")) if row_dict.get("parent_asin") else None,
                "main_image": str(row_dict.get("main_image", "")) if row_dict.get("main_image") else None,
                "source": "discovery",
                "fetched_at": datetime.now().isoformat(),
                # === NEW CRITICAL METRICS (2026-01-21) ===
                "monthly_sold": _safe_int(row_dict.get("monthly_sold")),
                "number_of_items": _safe_int(row_dict.get("number_of_items")) or 1,
                "price_per_unit": _safe_float(row_dict.get("price_per_unit")),
                "buybox_is_amazon": _safe_bool(row_dict.get("buybox_is_amazon")),
                "buybox_is_fba": _safe_bool(row_dict.get("buybox_is_fba")),
                "buybox_is_backorder": _safe_bool(row_dict.get("buybox_is_backorder")) or False,
                "has_amazon_seller": _safe_bool(row_dict.get("has_amazon_seller")),
                "seller_count": _safe_int(row_dict.get("seller_count")) or 1,
                "oos_count_amazon_30": _safe_int(row_dict.get("oos_count_amazon_30")),
                "oos_count_amazon_90": _safe_int(row_dict.get("oos_count_amazon_90")),
                "oos_pct_30": _safe_float(row_dict.get("oos_pct_30")),
                "oos_pct_90": _safe_float(row_dict.get("oos_pct_90")),
                "velocity_30d": _safe_float(row_dict.get("velocity_30d")),
                "velocity_90d": _safe_float(row_dict.get("velocity_90d")),
                "bb_seller_count_30": _safe_int(row_dict.get("bb_seller_count_30")),
                "bb_top_seller_30": _safe_float(row_dict.get("bb_top_seller_30")),
                "units_source": str(row_dict.get("units_source", "bsr_formula")),
                "is_sns": _safe_bool(row_dict.get("is_sns")) or False,
                # === PHASE A SIGNALS (2026-02-18) ===
                "return_rate": _safe_int(row_dict.get("return_rate")),
                "sales_rank_drops_30": _safe_int(row_dict.get("sales_rank_drops_30")),
                "sales_rank_drops_90": _safe_int(row_dict.get("sales_rank_drops_90")),
                "monthly_sold_delta": _safe_int(row_dict.get("monthly_sold_delta")),
                "top_comp_bb_share_30": _safe_float(row_dict.get("top_comp_bb_share_30")),
                "active_ingredients_raw": str(row_dict.get("active_ingredients_raw", ""))[:500] if row_dict.get("active_ingredients_raw") else None,
                "item_type_keyword": str(row_dict.get("item_type_keyword", ""))[:200] if row_dict.get("item_type_keyword") else None,
                "has_buybox_stats": _safe_bool(row_dict.get("has_buybox_stats")),
                "has_monthly_sold_history": _safe_bool(row_dict.get("has_monthly_sold_history")),
                "has_active_ingredients": _safe_bool(row_dict.get("has_active_ingredients")),
                "has_sales_rank_drops": _safe_bool(row_dict.get("has_sales_rank_drops")),
            }

            # ENHANCEMENT 2.3: Add category metadata if provided (consolidates with accumulator)
            if category_id is not None:
                record["category_id"] = category_id
            if category_name:
                record["category_name"] = category_name
            if category_tree:
                record["category_tree"] = category_tree
            if category_root:
                record["category_root"] = category_root

            records.append(record)
        
        if not records:
            return 0
        
        # Deduplicate by ASIN (keep first occurrence)
        seen_asins = set()
        unique_records = []
        for r in records:
            if r["asin"] not in seen_asins:
                seen_asins.add(r["asin"])
                unique_records.append(r)
        
        # Upsert in chunks
        chunk_size = 100
        total_cached = 0

        for i in range(0, len(unique_records), chunk_size):
            chunk = unique_records[i:i + chunk_size]

            # Final sanitization: Replace any remaining NaN/inf values with None
            import math
            sanitized_chunk = []
            for record in chunk:
                sanitized_record = {}
                for key, val in record.items():
                    # Check for NaN/inf in numeric types
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        sanitized_record[key] = None
                    else:
                        sanitized_record[key] = val
                # Strip columns that don't exist in schema (pre-migration safety)
                sanitized_chunk.append(_strip_unknown_columns(sanitized_record))

            try:
                supabase.table("product_snapshots").upsert(
                    sanitized_chunk,
                    on_conflict="asin,snapshot_date"
                ).execute()
                total_cached += len(sanitized_chunk)
            except Exception as e:
                # Log but don't fail - caching is optional
                st.warning(f"Cache batch failed: {e}")
                # Show which ASINs failed for debugging
                failed_asins = [r.get("asin", "unknown") for r in sanitized_chunk]
                st.caption(f"Failed ASINs: {failed_asins[:5]}...")
        
        return total_cached
        
    except Exception as e:
        # Caching failed - not critical, just log it
        st.caption(f"⚠️ Cache write failed: {e}")
        return 0


def cache_weekly_timeseries(
    df_weekly: pd.DataFrame,
    category_context: Optional[Dict] = None
) -> int:
    """
    Cache the FULL weekly time series to product_snapshots.
    
    Unlike cache_market_snapshot (which only stores latest), this stores
    ALL weekly rows, enabling the profiler to work on return visits.
    
    Data is stored at the ASIN level (no project_id), so User A's query
    benefits User B who searches for the same ASINs later.
    
    Args:
        df_weekly: Weekly time series data with week_start column
        category_context: Optional dict with category_id, category_name, etc.
        
    Returns:
        Number of records cached
    """
    import numpy as np
    import math
    from datetime import datetime
    
    if df_weekly is None or df_weekly.empty:
        return 0
    
    if "week_start" not in df_weekly.columns:
        # Fall back to regular snapshot caching
        return cache_market_snapshot(df_weekly, category_context=category_context)
    
    try:
        supabase = create_supabase_client()
        
        # Extract category context if provided
        category_id = category_context.get("category_id") if category_context else None
        category_name = category_context.get("category_name") if category_context else None
        category_tree = category_context.get("category_tree") if category_context else None
        category_root = category_context.get("category_root") if category_context else None
        
        # Build records for each week of each ASIN
        records = []
        for _, row in df_weekly.iterrows():
            asin = row.get("asin")
            week_start = row.get("week_start")
            
            if not asin or pd.isna(week_start):
                continue
            
            # Convert week_start to date string for snapshot_date
            if isinstance(week_start, pd.Timestamp):
                snapshot_date = week_start.date().isoformat()
            elif isinstance(week_start, datetime):
                snapshot_date = week_start.date().isoformat()
            else:
                try:
                    snapshot_date = pd.to_datetime(week_start).date().isoformat()
                except:
                    continue
            
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            
            # Extract price with fallback
            price_val = _get_first_valid(
                row_dict,
                ["buy_box_price", "filled_price", "price", "eff_p"],
                _safe_float
            )
            
            # Extract sales rank with fallback
            rank_val = _get_first_valid(
                row_dict,
                ["sales_rank_filled", "sales_rank", "bsr"],
                _safe_int
            )
            
            # Extract revenue
            revenue_val = _get_first_valid(
                row_dict,
                ["weekly_sales_filled", "weekly_revenue", "estimated_weekly_revenue"],
                _safe_float
            )
            
            # Extract units
            units_val = _get_first_valid(
                row_dict,
                ["estimated_units", "weekly_units"],
                _safe_int
            )
            
            record = {
                "asin": str(asin).strip().upper(),
                "snapshot_date": snapshot_date,
                "buy_box_price": price_val,
                "amazon_price": _safe_float(row_dict.get("amazon_price")),
                "new_fba_price": _safe_float(row_dict.get("new_fba_price")),
                "sales_rank": rank_val,
                "amazon_bb_share": _safe_float(row_dict.get("amazon_bb_share")),
                "buy_box_switches": _safe_int(row_dict.get("buy_box_switches")) or 0,
                "new_offer_count": _safe_int(row_dict.get("new_offer_count")),
                "review_count": _safe_int(row_dict.get("review_count")),
                "rating": _safe_float(row_dict.get("rating")),
                "estimated_units": units_val,
                "estimated_weekly_revenue": revenue_val,
                "filled_price": price_val,
                "title": str(row_dict.get("title", ""))[:500] if row_dict.get("title") else None,
                "brand": str(row_dict.get("brand", "")) if row_dict.get("brand") else None,
                "parent_asin": str(row_dict.get("parent_asin", "")) if row_dict.get("parent_asin") else None,
                "main_image": str(row_dict.get("main_image", "")) if row_dict.get("main_image") else None,
                "source": "keepa_weekly",
                "fetched_at": datetime.now().isoformat(),
                # Extended product signals
                "monthly_sold": _safe_int(row_dict.get("monthly_sold")),
                "number_of_items": _safe_int(row_dict.get("number_of_items")) or 1,
                "price_per_unit": _safe_float(row_dict.get("price_per_unit")),
                "buybox_is_amazon": _safe_bool(row_dict.get("buybox_is_amazon")),
                "buybox_is_fba": _safe_bool(row_dict.get("buybox_is_fba")),
                "buybox_is_backorder": _safe_bool(row_dict.get("buybox_is_backorder")) or False,
                "has_amazon_seller": _safe_bool(row_dict.get("has_amazon_seller")),
                "seller_count": _safe_int(row_dict.get("seller_count")) or 1,
                "oos_count_amazon_30": _safe_int(row_dict.get("oos_count_amazon_30")),
                "oos_count_amazon_90": _safe_int(row_dict.get("oos_count_amazon_90")),
                "oos_pct_30": _safe_float(row_dict.get("oos_pct_30")),
                "oos_pct_90": _safe_float(row_dict.get("oos_pct_90")),
                "velocity_30d": _safe_float(row_dict.get("velocity_30d")),
                "velocity_90d": _safe_float(row_dict.get("velocity_90d")),
                "bb_seller_count_30": _safe_int(row_dict.get("bb_seller_count_30")),
                "bb_top_seller_30": _safe_float(row_dict.get("bb_top_seller_30")),
                "units_source": str(row_dict.get("units_source", "bsr_formula")),
                "is_sns": _safe_bool(row_dict.get("is_sns")) or False,
                # Phase A signals
                "return_rate": _safe_int(row_dict.get("return_rate")),
                "sales_rank_drops_30": _safe_int(row_dict.get("sales_rank_drops_30")),
                "sales_rank_drops_90": _safe_int(row_dict.get("sales_rank_drops_90")),
                "monthly_sold_delta": _safe_int(row_dict.get("monthly_sold_delta")),
                "top_comp_bb_share_30": _safe_float(row_dict.get("top_comp_bb_share_30")),
                "active_ingredients_raw": str(row_dict.get("active_ingredients_raw", ""))[:500] if row_dict.get("active_ingredients_raw") else None,
                "item_type_keyword": str(row_dict.get("item_type_keyword", ""))[:200] if row_dict.get("item_type_keyword") else None,
                "has_buybox_stats": _safe_bool(row_dict.get("has_buybox_stats")),
                "has_monthly_sold_history": _safe_bool(row_dict.get("has_monthly_sold_history")),
                "has_active_ingredients": _safe_bool(row_dict.get("has_active_ingredients")),
                "has_sales_rank_drops": _safe_bool(row_dict.get("has_sales_rank_drops")),
            }
            
            # Add category context if provided
            if category_id is not None:
                record["category_id"] = category_id
            if category_name:
                record["category_name"] = category_name
            if category_tree:
                record["category_tree"] = category_tree
            if category_root:
                record["category_root"] = category_root
            
            records.append(record)
        
        if not records:
            return 0
        
        # Upsert in chunks
        chunk_size = 100
        total_cached = 0
        
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            
            # Sanitize NaN/inf values
            sanitized_chunk = []
            for record in chunk:
                sanitized_record = {}
                for key, val in record.items():
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        sanitized_record[key] = None
                    else:
                        sanitized_record[key] = val
                # Strip columns that don't exist in schema (pre-migration safety)
                sanitized_chunk.append(_strip_unknown_columns(sanitized_record))
            
            try:
                supabase.table("product_snapshots").upsert(
                    sanitized_chunk,
                    on_conflict="asin,snapshot_date"
                ).execute()
                total_cached += len(sanitized_chunk)
            except Exception as e:
                # Log but don't fail
                st.warning(f"Weekly cache batch failed: {e}")
        
        return total_cached
        
    except Exception as e:
        st.caption(f"⚠️ Weekly cache write failed: {e}")
        return 0


@st.cache_data(ttl=300)  # 5 minute cache
def load_weekly_timeseries(
    asins: Tuple[str, ...],
    days: int = 90
) -> pd.DataFrame:
    """
    Load weekly time series for a set of ASINs from product_snapshots.
    
    This enables the profiler/brain to work on return visits without
    hitting Keepa again. Data is shared across all users (ASIN-level).
    
    Args:
        asins: Tuple of ASIN strings (tuple for cache hashability)
        days: Number of days of history to load (default 90)
        
    Returns:
        DataFrame with weekly data (multiple rows per ASIN)
    """
    if not asins:
        return pd.DataFrame()
    
    try:
        supabase = create_supabase_client()
        
        cutoff_date = (date.today() - timedelta(days=days)).isoformat()
        asin_list = list(asins)
        
        # Fetch all snapshots for these ASINs in the date range
        result = supabase.table("product_snapshots").select("*").in_(
            "asin", asin_list
        ).gte("snapshot_date", cutoff_date).order("snapshot_date", desc=False).execute()
        
        if not result.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(result.data)
        
        # Normalize column names for profiler compatibility
        column_map = {
            "snapshot_date": "week_start",
            "buy_box_price": "filled_price",  # Use buy_box_price as filled_price
            "sales_rank": "sales_rank",
            "estimated_weekly_revenue": "weekly_revenue",
            "estimated_units": "estimated_units",
        }
        
        # Rename columns that exist
        rename_map = {}
        for src, dst in column_map.items():
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Ensure week_start is datetime
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"])
        
        # If filled_price doesn't exist, try to derive it
        if "filled_price" not in df.columns:
            for col in ["buy_box_price", "amazon_price", "new_fba_price"]:
                if col in df.columns:
                    df["filled_price"] = df[col]
                    break
        
        # Ensure derived columns that the brief/regime detectors expect
        if "sales_rank" in df.columns and "sales_rank_filled" not in df.columns:
            df["sales_rank_filled"] = pd.to_numeric(df["sales_rank"], errors="coerce")

        if "filled_price" in df.columns and "price_per_unit" not in df.columns:
            _items = pd.to_numeric(df.get("number_of_items", 1), errors="coerce").clip(lower=1).fillna(1)
            df["price_per_unit"] = pd.to_numeric(df["filled_price"], errors="coerce") / _items

        if "weekly_revenue" not in df.columns and "filled_price" in df.columns:
            # Derive weekly_revenue from BSR-based monthly units estimate
            _bsr = pd.to_numeric(df.get("sales_rank_filled", df.get("sales_rank")), errors="coerce").clip(lower=1)
            _monthly = pd.to_numeric(df.get("monthly_sold", 0), errors="coerce").fillna(0)
            _bsr_units = 145000.0 * (_bsr ** -0.9)
            _mu = _monthly.where(_monthly > 0, _bsr_units)
            df["weekly_revenue"] = (_mu * (7 / 30)) * pd.to_numeric(df["filled_price"], errors="coerce").fillna(0)

        # Sort by ASIN and date for time series analysis
        if "week_start" in df.columns and "asin" in df.columns:
            df = df.sort_values(["asin", "week_start"])
        
        return df
        
    except Exception as e:
        # Silent fail - return empty DataFrame
        error_str = str(e)
        if "PGRST205" not in error_str and "could not find" not in error_str.lower():
            st.caption(f"⚠️ Could not load weekly data: {error_str[:50]}")
        return pd.DataFrame()


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, handling NaN and None."""
    import numpy as np
    import pandas as pd
    if val is None:
        return None
    if pd.isna(val):  # Handles np.nan, pd.NA, None, etc.
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    """Safely convert to int, handling NaN and None."""
    import numpy as np
    import pandas as pd
    if val is None:
        return None
    if pd.isna(val):  # Handles np.nan, pd.NA, None, etc.
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_bool(val) -> Optional[bool]:
    """Safely convert to bool, handling NaN, None, and float booleans (0.0/1.0)."""
    import numpy as np
    import pandas as pd
    if val is None:
        return None
    if pd.isna(val):  # Handles np.nan, pd.NA, None, etc.
        return None
    try:
        # Handle float booleans (0.0 → False, 1.0 → True)
        if isinstance(val, (int, float)):
            return bool(int(val))
        # Handle string booleans
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        # Handle actual booleans
        return bool(val)
    except (ValueError, TypeError):
        return None


def _get_first_valid(row: dict, columns: list, converter) -> Optional:
    """
    Get the first valid (non-null, non-NaN) value from a list of column names.
    
    This fixes the issue where Python's `or` operator treats NaN as truthy,
    preventing fallback to subsequent columns.
    
    Returns 0 if all columns have 0 (0 is a valid value, not "missing").
    """
    for col in columns:
        val = converter(row.get(col))
        if val is not None:
            return val
    return None


# ========================================
# HISTORICAL METRICS (FIX 1.1)
# ========================================

@st.cache_data(ttl=300)  # 5 minute cache
def load_historical_metrics_from_db(project_id: str) -> pd.DataFrame:
    """
    Load 90-day historical metrics from the database for velocity extraction.

    This fixes the disconnect where historical_metrics was written but never read
    for the main dashboard velocity calculations.

    Args:
        project_id: The project UUID

    Returns:
        DataFrame with weekly historical data for velocity calculations
    """
    if not project_id:
        return pd.DataFrame()

    try:
        supabase = create_supabase_client()

        # Fetch historical metrics for this project
        result = supabase.table("historical_metrics").select("*").eq(
            "project_id", project_id
        ).order("datetime", desc=True).execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)

        # Normalize column names to match expected format for velocity extraction
        # NOTE: backfill.py writes: datetime, sales_rank, buy_box_price, amazon_price, new_fba_price
        # We map these to the dashboard expected names
        column_map = {
            "datetime": "week_start",
            # Map actual columns written by backfill.py
            "buy_box_price": "filled_price",
            "sales_rank": "sales_rank_filled",
            # Also support alternate names if they exist (from other sources)
            "price": "filled_price",
            "bsr": "sales_rank_filled",
            "revenue": "weekly_sales_filled",
            "units": "estimated_units",
            # Support the new columns added to schema
            "filled_price": "filled_price",
            "sales_rank_filled": "sales_rank_filled"
        }

        # Rename columns that exist (avoid overwriting if target already exists)
        rename_map = {}
        for src, dst in column_map.items():
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        df = df.rename(columns=rename_map)

        # Ensure required columns exist
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"])

        # If filled_price still doesn't exist, try to derive it
        if "filled_price" not in df.columns:
            for col in ["buy_box_price", "amazon_price", "new_fba_price"]:
                if col in df.columns:
                    df["filled_price"] = df[col]
                    break

        return df

    except Exception as e:
        # Silently handle missing table or empty results - this is expected before backfill
        error_str = str(e)
        if "PGRST205" in error_str or "could not find" in error_str.lower():
            # Table doesn't exist yet - silent fallback
            pass
        else:
            # Only show warning for unexpected errors
            st.caption(f"⚠️ Historical metrics: {error_str[:80]}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_historical_metrics_by_asins(asins: Tuple[str, ...], days: int = 90) -> pd.DataFrame:
    """
    Load historical metrics for a set of ASINs (alternative to project-based lookup).

    Args:
        asins: Tuple of ASIN strings (tuple for cache hashability)
        days: Number of days of history to load

    Returns:
        DataFrame with historical metrics
    """
    if not asins:
        return pd.DataFrame()

    try:
        supabase = create_supabase_client()

        cutoff_date = (date.today() - timedelta(days=days)).isoformat()
        asin_list = list(asins)

        result = supabase.table("historical_metrics").select("*").in_(
            "asin", asin_list
        ).gte("datetime", cutoff_date).order("datetime", desc=True).execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)

        # Normalize column names to match expected format for velocity extraction
        # NOTE: backfill.py writes: datetime, sales_rank, buy_box_price, amazon_price, new_fba_price
        column_map = {
            "datetime": "week_start",
            # Map actual columns written by backfill.py
            "buy_box_price": "filled_price",
            "sales_rank": "sales_rank_filled",
            # Also support alternate names if they exist
            "price": "filled_price",
            "bsr": "sales_rank_filled",
            "revenue": "weekly_sales_filled",
            "units": "estimated_units",
            "filled_price": "filled_price",
            "sales_rank_filled": "sales_rank_filled"
        }

        # Rename columns that exist (avoid overwriting if target already exists)
        rename_map = {}
        for src, dst in column_map.items():
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        df = df.rename(columns=rename_map)

        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"])

        # If filled_price still doesn't exist, try to derive it
        if "filled_price" not in df.columns:
            for col in ["buy_box_price", "amazon_price", "new_fba_price"]:
                if col in df.columns:
                    df["filled_price"] = df[col]
                    break

        return df

    except Exception as e:
        # Silently handle missing table - expected before backfill runs
        error_str = str(e)
        if "PGRST205" in error_str or "could not find" in error_str.lower():
            pass  # Silent fallback
        return pd.DataFrame()


# ========================================
# NETWORK INTELLIGENCE (FIX 1.2)
# ========================================

def get_market_snapshot_with_network_intelligence(
    project_asins: List[str],
    seed_brand: Optional[str] = None,
    category_id: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced market snapshot that includes network intelligence benchmarks.

    This fixes the disconnect where category_intelligence was written but never
    read back for the dashboard display.

    Args:
        project_asins: All ASINs in the project
        seed_brand: Brand to identify as "yours" (optional)
        category_id: Category ID for fetching benchmarks (optional)

    Returns:
        Tuple of (market_snapshot DataFrame, market_stats dict with network context)
    """
    # Get base market snapshot
    df, stats = get_market_snapshot_from_cache(project_asins, seed_brand)

    if df.empty:
        return df, stats

    # Try to enrich with network intelligence
    try:
        # Get category_id from data if not provided
        if not category_id and 'category_id' in df.columns:
            category_id = df['category_id'].iloc[0] if pd.notna(df['category_id'].iloc[0]) else None

        if category_id:
            supabase = create_supabase_client()

            # Fetch category benchmarks
            result = supabase.table('category_intelligence').select('*').eq(
                'category_id', int(category_id)
            ).order('snapshot_date', desc=True).limit(1).execute()

            if result.data and len(result.data) > 0:
                benchmarks = result.data[0]
                
                # VALIDATION: Check if benchmarks look reasonable against actual portfolio data
                # If median price is >10x different from portfolio avg, the benchmark is stale/incorrect
                benchmark_median_price = benchmarks.get('median_price', 0)
                benchmark_valid = True
                
                if benchmark_median_price and benchmark_median_price > 0:
                    price_col = 'buy_box_price' if 'buy_box_price' in df.columns else 'filled_price' if 'filled_price' in df.columns else 'price' if 'price' in df.columns else None
                    if price_col and not df[price_col].dropna().empty:
                        portfolio_avg_price = df[price_col].dropna().mean()
                        if portfolio_avg_price > 0:
                            price_ratio = max(portfolio_avg_price / benchmark_median_price, benchmark_median_price / portfolio_avg_price)
                            if price_ratio > 10:
                                # Benchmark is wildly off - mark as stale
                                benchmark_valid = False
                                print(f"⚠️ Stale benchmark detected for category {category_id}: median ${benchmark_median_price:.2f} vs portfolio avg ${portfolio_avg_price:.2f}")

                # Add benchmarks to stats
                stats['network_intelligence'] = {
                    'category_id': category_id,
                    'median_price': benchmarks.get('median_price'),
                    'median_bsr': benchmarks.get('median_bsr'),
                    'median_review_count': benchmarks.get('median_review_count'),
                    'median_rating': benchmarks.get('median_rating'),
                    # BUY BOX & COMPETITION BENCHMARKS (new)
                    'avg_bb_share': benchmarks.get('avg_bb_share'),
                    'avg_offer_count': benchmarks.get('avg_offer_count'),
                    'total_asins_tracked': benchmarks.get('total_asins_tracked', 0),
                    'data_quality': 'STALE' if not benchmark_valid else benchmarks.get('data_quality', 'LOW'),
                    'snapshot_date': benchmarks.get('snapshot_date'),
                    'benchmark_valid': benchmark_valid
                }

                # Calculate competitive position for each product
                if benchmarks.get('median_price'):
                    price_col = 'buy_box_price' if 'buy_box_price' in df.columns else 'filled_price' if 'filled_price' in df.columns else None
                    if price_col:
                        median_price = float(benchmarks['median_price'])
                        df['price_vs_category_median'] = ((df[price_col] / median_price - 1) * 100).fillna(0)

                if benchmarks.get('median_review_count'):
                    if 'review_count' in df.columns:
                        median_reviews = float(benchmarks['median_review_count'])
                        df['reviews_vs_category_median'] = ((df['review_count'] / median_reviews - 1) * 100).fillna(0)

                stats['has_network_context'] = True
            else:
                stats['has_network_context'] = False
                stats['network_intelligence'] = {'message': 'No category benchmarks available yet'}
        else:
            stats['has_network_context'] = False
            stats['network_intelligence'] = {'message': 'No category ID available'}

    except Exception as e:
        stats['has_network_context'] = False
        stats['network_intelligence'] = {'error': str(e)}

    return df, stats


# =============================================================================
# P3: CATEGORY BENCHMARKS & HISTORICAL TRACKING
# =============================================================================

def load_category_benchmarks(category_id: int, lookback_days: int = 30) -> Optional[Dict]:
    """
    Load category-level intelligence and benchmarks from Supabase.

    Args:
        category_id: Amazon category ID
        lookback_days: Number of days to look back for trend calculation

    Returns:
        Dictionary with category benchmarks or None if not available
    """
    if not SUPABASE_CACHE_ENABLED:
        return None

    try:
        supabase = get_supabase_client()
        if not supabase:
            return None

        # Query category_intelligence table for latest benchmarks
        # Assuming table structure: category_id, timestamp, metrics (JSONB)
        result = supabase.table('category_intelligence')\
            .select('*')\
            .eq('category_id', category_id)\
            .order('timestamp', desc=True)\
            .limit(lookback_days)\
            .execute()

        if not result.data:
            return None

        # Calculate trends from historical data
        data_points = result.data
        latest = data_points[0]

        # Extract current metrics
        benchmarks = {
            'category_id': category_id,
            'timestamp': latest.get('timestamp'),
            'median_price': latest.get('median_price', 0),
            'median_review_count': latest.get('median_review_count', 0),
            'median_rating': latest.get('median_rating', 0),
            'total_products': latest.get('total_products', 0),
            'category_revenue_estimate': latest.get('category_revenue_estimate', 0)
        }

        # Calculate growth rate (comparing first vs last in lookback period)
        if len(data_points) >= 2:
            oldest = data_points[-1]
            latest_revenue = latest.get('category_revenue_estimate', 0)
            oldest_revenue = oldest.get('category_revenue_estimate', 0)

            if oldest_revenue > 0:
                days_between = (
                    pd.to_datetime(latest.get('timestamp')) -
                    pd.to_datetime(oldest.get('timestamp'))
                ).days or 1

                # Normalize to 30-day growth rate
                growth = ((latest_revenue / oldest_revenue) - 1) * 100
                growth_rate_30d = growth * (30 / days_between)

                benchmarks['growth_rate_30d'] = growth_rate_30d
                benchmarks['growth_direction'] = 'expanding' if growth_rate_30d > 5 else 'contracting' if growth_rate_30d < -5 else 'stable'
            else:
                benchmarks['growth_rate_30d'] = 0
                benchmarks['growth_direction'] = 'unknown'
        else:
            benchmarks['growth_rate_30d'] = 0
            benchmarks['growth_direction'] = 'insufficient_data'

        return benchmarks

    except Exception as e:
        print(f"⚠️ Error loading category benchmarks: {str(e)}")
        return None


def load_keyword_rank_history(
    asin: str,
    lookback_days: int = 30
) -> Optional[pd.DataFrame]:
    """
    Load keyword ranking history for Share of Voice tracking.

    Args:
        asin: Product ASIN
        lookback_days: Number of days of history to load

    Returns:
        DataFrame with columns: keyword, rank, search_volume, timestamp
        Returns None if data unavailable
    """
    if not SUPABASE_CACHE_ENABLED:
        return None

    try:
        supabase = get_supabase_client()
        if not supabase:
            return None

        # Query keyword_ranks table
        # Assuming table structure: asin, keyword, rank, search_volume, timestamp
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

        result = supabase.table('keyword_ranks')\
            .select('*')\
            .eq('asin', asin)\
            .gte('timestamp', cutoff_date)\
            .order('timestamp', desc=True)\
            .execute()

        if not result.data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(result.data)

        # Ensure required columns exist
        required_cols = ['keyword', 'rank', 'search_volume', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            return None

        return df

    except Exception as e:
        print(f"⚠️ Error loading keyword rank history: {str(e)}")
        return None


def calculate_share_of_voice_metrics(
    keyword_rank_df: pd.DataFrame
) -> Dict:
    """
    Calculate Share of Voice metrics from keyword rank history.

    Args:
        keyword_rank_df: DataFrame with keyword ranking history

    Returns:
        Dictionary with SOV metrics and trend
    """
    if keyword_rank_df is None or keyword_rank_df.empty:
        return {}

    try:
        # Calculate weighted average rank (weighted by search volume)
        keyword_rank_df['weighted_rank'] = keyword_rank_df['rank'] * keyword_rank_df['search_volume']

        # Get recent (7d) and older (30d) periods
        keyword_rank_df['timestamp'] = pd.to_datetime(keyword_rank_df['timestamp'])
        cutoff_7d = datetime.now() - timedelta(days=7)
        cutoff_30d = datetime.now() - timedelta(days=30)

        recent_df = keyword_rank_df[keyword_rank_df['timestamp'] >= cutoff_7d]
        older_df = keyword_rank_df[
            (keyword_rank_df['timestamp'] < cutoff_7d) &
            (keyword_rank_df['timestamp'] >= cutoff_30d)
        ]

        # Calculate weighted average ranks
        if not recent_df.empty and recent_df['search_volume'].sum() > 0:
            recent_avg_rank = recent_df['weighted_rank'].sum() / recent_df['search_volume'].sum()
        else:
            recent_avg_rank = 50  # Default fallback

        if not older_df.empty and older_df['search_volume'].sum() > 0:
            older_avg_rank = older_df['weighted_rank'].sum() / older_df['search_volume'].sum()
        else:
            older_avg_rank = recent_avg_rank  # No change

        # Calculate change percentage
        # Note: Lower rank is better, so improvement = negative % change
        if older_avg_rank > 0:
            rank_change_pct = ((recent_avg_rank - older_avg_rank) / older_avg_rank) * 100
        else:
            rank_change_pct = 0

        # Determine trend
        if rank_change_pct < -20:
            trend = "significant_improvement"
        elif rank_change_pct < -5:
            trend = "improving"
        elif rank_change_pct > 20:
            trend = "significant_decline"
        elif rank_change_pct > 5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            'recent_avg_rank': recent_avg_rank,
            'older_avg_rank': older_avg_rank,
            'rank_change_pct': rank_change_pct,
            'trend': trend,
            'keywords_tracked': len(keyword_rank_df['keyword'].unique()),
            'total_search_volume': keyword_rank_df['search_volume'].sum()
        }

    except Exception as e:
        print(f"⚠️ Error calculating SOV metrics: {str(e)}")
        return {}


# ============================================================================
# HISTORICAL ATTRIBUTION TRACKING
# ============================================================================

def save_attribution_to_history(
    project_id: str,
    asin: str,
    attribution: 'RevenueAttribution',
    attribution_date: Optional[date] = None
) -> bool:
    """
    Save revenue attribution to Supabase for historical tracking.

    Args:
        project_id: Project UUID
        asin: Product ASIN
        attribution: RevenueAttribution object
        attribution_date: Date of attribution (defaults to today)

    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()

        if attribution_date is None:
            attribution_date = date.today()

        # Prepare data for insertion
        record = {
            'project_id': project_id,
            'asin': asin,
            'attribution_date': attribution_date.isoformat(),

            # Total change
            'total_revenue_delta': float(attribution.total_delta),
            'previous_revenue': float(attribution.previous_revenue),
            'current_revenue': float(attribution.current_revenue),

            # The 4 Categories
            'internal_contribution': float(attribution.internal_contribution),
            'competitive_contribution': float(attribution.competitive_contribution),
            'macro_contribution': float(attribution.macro_contribution),
            'platform_contribution': float(attribution.platform_contribution),

            # Metadata
            'explained_variance': float(attribution.explained_variance),
            'residual': float(attribution.residual),
            'confidence': float(attribution.confidence),

            # Detailed drivers (JSON)
            'internal_drivers': [d.to_dict() for d in attribution.internal_drivers],
            'competitive_drivers': [d.to_dict() for d in attribution.competitive_drivers],
            'macro_drivers': [d.to_dict() for d in attribution.macro_drivers],
            'platform_drivers': [d.to_dict() for d in attribution.platform_drivers]
        }

        # Upsert (insert or update if exists)
        result = supabase.table('revenue_attributions')\
            .upsert(record)\
            .execute()

        return len(result.data) > 0

    except Exception as e:
        print(f"⚠️ Error saving attribution to history: {str(e)}")
        return False


def load_attribution_history(
    project_id: str,
    asin: str,
    lookback_days: int = 90
) -> Optional[pd.DataFrame]:
    """
    Load historical attribution records for trend analysis.

    Args:
        project_id: Project UUID
        asin: Product ASIN
        lookback_days: How far back to retrieve (default 90 days)

    Returns:
        DataFrame with historical attribution data or None if error
    """
    try:
        supabase = get_supabase_client()

        cutoff_date = date.today() - timedelta(days=lookback_days)

        result = supabase.table('revenue_attributions')\
            .select('*')\
            .eq('project_id', project_id)\
            .eq('asin', asin)\
            .gte('attribution_date', cutoff_date.isoformat())\
            .order('attribution_date', desc=False)\
            .execute()

        if not result.data:
            return None

        df = pd.DataFrame(result.data)

        # Convert date strings to datetime
        df['attribution_date'] = pd.to_datetime(df['attribution_date'])

        return df

    except Exception as e:
        print(f"⚠️ Error loading attribution history: {str(e)}")
        return None


def calculate_attribution_trends(
    attribution_history_df: pd.DataFrame
) -> Dict:
    """
    Analyze attribution trends over time to identify patterns.

    Examples:
    - "Dependency on competitor OOS growing: 20% → 36%"
    - "Internal contribution declining: 60% → 45%"
    - "Platform impact increasing: -$2k → -$8k"

    Args:
        attribution_history_df: Historical attribution data

    Returns:
        Dictionary with trend analysis
    """
    try:
        if attribution_history_df is None or len(attribution_history_df) < 2:
            return {}

        # Sort by date
        df = attribution_history_df.sort_values('attribution_date')

        # Get oldest and most recent records
        oldest = df.iloc[0]
        latest = df.iloc[-1]

        # Calculate percentage contributions
        def calc_percentages(row):
            total = row['total_revenue_delta']
            if total == 0:
                return {'internal': 0, 'competitive': 0, 'macro': 0, 'platform': 0}

            return {
                'internal': (row['internal_contribution'] / total) * 100 if total != 0 else 0,
                'competitive': (row['competitive_contribution'] / total) * 100 if total != 0 else 0,
                'macro': (row['macro_contribution'] / total) * 100 if total != 0 else 0,
                'platform': (row['platform_contribution'] / total) * 100 if total != 0 else 0
            }

        oldest_pct = calc_percentages(oldest)
        latest_pct = calc_percentages(latest)

        # Calculate trends
        trends = {}

        # Internal contribution trend
        internal_change = latest_pct['internal'] - oldest_pct['internal']
        if abs(internal_change) > 10:
            trends['internal_trend'] = {
                'change_pct': internal_change,
                'direction': 'increasing' if internal_change > 0 else 'decreasing',
                'old_value': oldest_pct['internal'],
                'new_value': latest_pct['internal'],
                'interpretation': f"Internal actions {'growing' if internal_change > 0 else 'declining'} as revenue driver: {oldest_pct['internal']:.0f}% → {latest_pct['internal']:.0f}%"
            }

        # Competitive dependency trend
        competitive_change = latest_pct['competitive'] - oldest_pct['competitive']
        if abs(competitive_change) > 10:
            trends['competitive_trend'] = {
                'change_pct': competitive_change,
                'direction': 'increasing' if competitive_change > 0 else 'decreasing',
                'old_value': oldest_pct['competitive'],
                'new_value': latest_pct['competitive'],
                'interpretation': f"Competitive dependency {'growing' if competitive_change > 0 else 'declining'}: {oldest_pct['competitive']:.0f}% → {latest_pct['competitive']:.0f}%",
                'risk_level': 'HIGH' if latest_pct['competitive'] > 40 else 'MEDIUM' if latest_pct['competitive'] > 20 else 'LOW'
            }

        # Macro trend
        macro_change = latest_pct['macro'] - oldest_pct['macro']
        if abs(macro_change) > 10:
            trends['macro_trend'] = {
                'change_pct': macro_change,
                'direction': 'increasing' if macro_change > 0 else 'decreasing',
                'old_value': oldest_pct['macro'],
                'new_value': latest_pct['macro'],
                'interpretation': f"Market tailwind dependency {'growing' if macro_change > 0 else 'declining'}: {oldest_pct['macro']:.0f}% → {latest_pct['macro']:.0f}%"
            }

        # Platform impact trend
        platform_change = latest_pct['platform'] - oldest_pct['platform']
        if abs(platform_change) > 5:
            trends['platform_trend'] = {
                'change_pct': platform_change,
                'direction': 'increasing' if platform_change > 0 else 'decreasing',
                'old_value': oldest_pct['platform'],
                'new_value': latest_pct['platform'],
                'interpretation': f"Platform impact {'growing' if platform_change > 0 else 'declining'}: {oldest_pct['platform']:.0f}% → {latest_pct['platform']:.0f}%"
            }

        # Overall attribution quality trend
        explained_var_change = latest['explained_variance'] - oldest['explained_variance']
        trends['quality_trend'] = {
            'change': explained_var_change,
            'direction': 'improving' if explained_var_change > 0 else 'declining',
            'old_value': oldest['explained_variance'],
            'new_value': latest['explained_variance'],
            'interpretation': f"Attribution quality {'improving' if explained_var_change > 0 else 'declining'}: {oldest['explained_variance']:.0%} → {latest['explained_variance']:.0%}"
        }

        # Revenue volatility
        revenue_std = df['total_revenue_delta'].std()
        revenue_mean = df['total_revenue_delta'].mean()
        volatility = (revenue_std / abs(revenue_mean)) * 100 if revenue_mean != 0 else 0

        trends['volatility'] = {
            'coefficient_of_variation': volatility,
            'interpretation': f"Revenue volatility: {'HIGH' if volatility > 50 else 'MEDIUM' if volatility > 25 else 'LOW'} ({volatility:.0f}%)"
        }

        return trends

    except Exception as e:
        print(f"⚠️ Error calculating attribution trends: {str(e)}")
        return {}


# ============================================================================
# FORECAST ACCURACY TRACKING
# ============================================================================

def save_forecast_to_history(
    project_id: str,
    asin: str,
    forecast: 'RevenueForecast',
    forecast_date: Optional[date] = None
) -> bool:
    """
    Save revenue forecast to Supabase for accuracy tracking.

    Args:
        project_id: Project UUID
        asin: Product ASIN
        forecast: RevenueForecast object
        forecast_date: Date forecast was made (defaults to today)

    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()

        if forecast_date is None:
            forecast_date = date.today()

        # Prepare data for insertion
        record = {
            'project_id': project_id,
            'asin': asin,
            'forecast_date': forecast_date.isoformat(),
            'forecast_horizon_days': forecast.forecast_horizon_days,

            # Predictions
            'predicted_revenue': float(forecast.projected_revenue),
            'predicted_lower_bound': float(forecast.lower_bound),
            'predicted_upper_bound': float(forecast.upper_bound),
            'confidence_interval': float(forecast.confidence_interval),

            # Actuals will be filled later
            'actual_revenue': None,
            'actual_date': None,

            # Accuracy metrics will be calculated later
            'absolute_error': None,
            'percentage_error': None,
            'within_confidence_interval': None,

            # Metadata
            'forecast_metadata': {
                'base_trend': forecast.base_trend,
                'seasonality_adjustment': forecast.seasonality_adjustment,
                'event_adjustments': forecast.event_adjustments
            }
        }

        # Upsert (insert or update if exists)
        result = supabase.table('forecast_accuracy')\
            .upsert(record)\
            .execute()

        return len(result.data) > 0

    except Exception as e:
        print(f"⚠️ Error saving forecast to history: {str(e)}")
        return False


def update_forecast_actuals(
    project_id: str,
    asin: str,
    actual_revenue: float,
    actual_date: date
) -> int:
    """
    Update forecast records with actual outcomes and calculate accuracy.

    Finds all forecasts that should have materialized by actual_date
    and fills in the actual_revenue, then calculates accuracy metrics.

    Args:
        project_id: Project UUID
        asin: Product ASIN
        actual_revenue: Actual revenue achieved
        actual_date: Date of actual measurement

    Returns:
        Number of forecast records updated
    """
    try:
        supabase = get_supabase_client()

        # Find forecasts that should have materialized
        # (forecast_date + forecast_horizon_days <= actual_date)
        result = supabase.table('forecast_accuracy')\
            .select('*')\
            .eq('project_id', project_id)\
            .eq('asin', asin)\
            .is_('actual_revenue', 'null')\
            .execute()

        if not result.data:
            return 0

        updated_count = 0

        for forecast_record in result.data:
            forecast_date_str = forecast_record['forecast_date']
            forecast_horizon = forecast_record['forecast_horizon_days']

            # Parse forecast date
            forecast_date_obj = datetime.fromisoformat(forecast_date_str).date() if isinstance(forecast_date_str, str) else forecast_date_str

            # Calculate target date
            target_date = forecast_date_obj + timedelta(days=forecast_horizon)

            # Check if actual_date has passed the target
            if actual_date >= target_date:
                # Calculate accuracy metrics
                predicted = forecast_record['predicted_revenue']
                absolute_error = abs(actual_revenue - predicted)
                percentage_error = (absolute_error / predicted * 100) if predicted != 0 else 0

                # Check if within confidence interval
                lower_bound = forecast_record['predicted_lower_bound']
                upper_bound = forecast_record['predicted_upper_bound']
                within_ci = lower_bound <= actual_revenue <= upper_bound

                # Update record
                update_result = supabase.table('forecast_accuracy')\
                    .update({
                        'actual_revenue': float(actual_revenue),
                        'actual_date': actual_date.isoformat(),
                        'absolute_error': float(absolute_error),
                        'percentage_error': float(percentage_error),
                        'within_confidence_interval': within_ci
                    })\
                    .eq('id', forecast_record['id'])\
                    .execute()

                if update_result.data:
                    updated_count += 1

        return updated_count

    except Exception as e:
        print(f"⚠️ Error updating forecast actuals: {str(e)}")
        return 0


def calculate_forecast_accuracy_metrics(
    project_id: str,
    asin: str,
    lookback_days: int = 90
) -> Dict:
    """
    Calculate forecast accuracy metrics (MAPE, hit rate, bias).

    Args:
        project_id: Project UUID
        asin: Product ASIN
        lookback_days: How far back to analyze (default 90 days)

    Returns:
        Dictionary with accuracy metrics
    """
    try:
        supabase = get_supabase_client()

        cutoff_date = date.today() - timedelta(days=lookback_days)

        # Get all forecasts with actuals filled in
        result = supabase.table('forecast_accuracy')\
            .select('*')\
            .eq('project_id', project_id)\
            .eq('asin', asin)\
            .gte('forecast_date', cutoff_date.isoformat())\
            .not_.is_('actual_revenue', 'null')\
            .execute()

        if not result.data or len(result.data) == 0:
            return {
                'has_data': False,
                'message': 'No forecast actuals available yet'
            }

        df = pd.DataFrame(result.data)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = df['percentage_error'].mean()

        # Calculate hit rate (% of forecasts within confidence interval)
        hit_rate = (df['within_confidence_interval'].sum() / len(df)) * 100

        # Calculate bias (average signed error)
        df['signed_error'] = df['actual_revenue'] - df['predicted_revenue']
        bias = df['signed_error'].mean()
        bias_pct = (bias / df['predicted_revenue'].mean() * 100) if df['predicted_revenue'].mean() != 0 else 0

        # Accuracy by horizon
        accuracy_by_horizon = {}
        for horizon in df['forecast_horizon_days'].unique():
            horizon_df = df[df['forecast_horizon_days'] == horizon]
            accuracy_by_horizon[f'{horizon}d'] = {
                'mape': horizon_df['percentage_error'].mean(),
                'hit_rate': (horizon_df['within_confidence_interval'].sum() / len(horizon_df)) * 100,
                'count': len(horizon_df)
            }

        # Trend analysis (improving or declining)
        if len(df) >= 5:
            # Split into first half vs second half
            mid_point = len(df) // 2
            first_half_mape = df.iloc[:mid_point]['percentage_error'].mean()
            second_half_mape = df.iloc[mid_point:]['percentage_error'].mean()

            if second_half_mape < first_half_mape:
                trend = 'improving'
                trend_change = first_half_mape - second_half_mape
            else:
                trend = 'declining'
                trend_change = second_half_mape - first_half_mape
        else:
            trend = 'stable'
            trend_change = 0

        return {
            'has_data': True,
            'mape': mape,
            'hit_rate': hit_rate,
            'bias': bias,
            'bias_pct': bias_pct,
            'accuracy_by_horizon': accuracy_by_horizon,
            'trend': trend,
            'trend_change': trend_change,
            'total_forecasts': len(df),
            'grade': 'A' if mape < 15 else 'B' if mape < 25 else 'C' if mape < 35 else 'D',
            'interpretation': f"Forecast accuracy: {100 - mape:.0f}% (±{mape:.0f}% average error)"
        }

    except Exception as e:
        print(f"⚠️ Error calculating forecast accuracy: {str(e)}")
        return {'has_data': False, 'error': str(e)}
