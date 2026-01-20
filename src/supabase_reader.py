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
from datetime import date, timedelta
from supabase import Client


# ============================================
# SUPABASE SINGLETON - Reuses single connection
# ============================================

_supabase_client: Optional[Client] = None


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
        "asin": "asin"
    }
    
    # Rename columns that exist
    rename_map = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # CRITICAL: Convert numeric columns to proper dtypes
    # This prevents 'cannot use method nlargest with dtype object' errors
    numeric_columns = [
        "weekly_sales_filled", "revenue_proxy", "filled_price", "buy_box_price",
        "amazon_price", "new_fba_price", "sales_rank_filled", "amazon_bb_share",
        "estimated_units", "rating", "review_count", "new_offer_count"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure required columns exist with proper values
    if "weekly_sales_filled" not in df.columns or df["weekly_sales_filled"].sum() == 0:
        if "filled_price" in df.columns and "estimated_units" in df.columns:
            df["weekly_sales_filled"] = pd.to_numeric(df["estimated_units"], errors='coerce').fillna(0) * \
                                        pd.to_numeric(df["filled_price"], errors='coerce').fillna(0)
        else:
            df["weekly_sales_filled"] = 0.0

    # Ensure weekly_sales_filled is numeric
    df["weekly_sales_filled"] = pd.to_numeric(df["weekly_sales_filled"], errors='coerce').fillna(0)

    # CRITICAL: Add revenue_proxy as alias for weekly_sales_filled
    # The dashboard expects 'revenue_proxy' for market share calculations
    df["revenue_proxy"] = df["weekly_sales_filled"].copy()

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
    """
    df, stats = load_project_data(project_asins)
    
    if df.empty:
        return df, {"total_revenue": 0, "your_revenue": 0, "competitor_revenue": 0}
    
    # Calculate market metrics
    total_revenue = df["weekly_sales_filled"].sum() if "weekly_sales_filled" in df.columns else 0
    
    market_stats = {
        "total_revenue": total_revenue,
        "total_products": len(df),
        "source": stats.get("source", "unknown")
    }
    
    # Split by brand if seed_brand provided
    if seed_brand and "brand" in df.columns:
        df["is_yours"] = df["brand"].str.lower().str.contains(seed_brand.lower(), na=False)
        your_df = df[df["is_yours"]]
        competitor_df = df[~df["is_yours"]]
        
        market_stats["your_revenue"] = your_df["weekly_sales_filled"].sum() if not your_df.empty else 0
        market_stats["competitor_revenue"] = competitor_df["weekly_sales_filled"].sum() if not competitor_df.empty else 0
        market_stats["your_product_count"] = len(your_df)
        market_stats["competitor_product_count"] = len(competitor_df)
        
        if total_revenue > 0:
            market_stats["market_share"] = (market_stats["your_revenue"] / total_revenue) * 100
    
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
            revenue_val = _get_first_valid(
                row_dict,
                ["weekly_sales_filled", "revenue_proxy", "estimated_weekly_revenue"],
                _safe_float
            )
            
            # Extract units with proper fallback
            units_val = _get_first_valid(
                row_dict,
                ["estimated_units", "monthly_units"],
                _safe_int
            )
            
            # If we have price and units but no revenue, calculate it
            if revenue_val is None and price_val and units_val:
                revenue_val = price_val * units_val / 4  # Weekly from monthly

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
                "fetched_at": datetime.now().isoformat()
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
            try:
                supabase.table("product_snapshots").upsert(
                    chunk,
                    on_conflict="asin,snapshot_date"
                ).execute()
                total_cached += len(chunk)
            except Exception as e:
                # Log but don't fail - caching is optional
                st.warning(f"Cache batch failed: {e}")
        
        return total_cached
        
    except Exception as e:
        # Caching failed - not critical, just log it
        st.caption(f"⚠️ Cache write failed: {e}")
        return 0


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
        column_map = {
            "datetime": "week_start",
            "price": "filled_price",
            "bsr": "sales_rank_filled",
            "revenue": "weekly_sales_filled",
            "units": "estimated_units"
        }

        # Rename columns that exist
        rename_map = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Ensure required columns exist
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"])

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

        # Normalize column names
        column_map = {
            "datetime": "week_start",
            "price": "filled_price",
            "bsr": "sales_rank_filled",
            "revenue": "weekly_sales_filled",
            "units": "estimated_units"
        }

        rename_map = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"])

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

                # Add benchmarks to stats
                stats['network_intelligence'] = {
                    'category_id': category_id,
                    'median_price': benchmarks.get('median_price'),
                    'median_bsr': benchmarks.get('median_bsr'),
                    'median_review_count': benchmarks.get('median_review_count'),
                    'median_rating': benchmarks.get('median_rating'),
                    'total_asins_tracked': benchmarks.get('total_asins_tracked', 0),
                    'data_quality': benchmarks.get('data_quality', 'LOW'),
                    'snapshot_date': benchmarks.get('snapshot_date')
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
