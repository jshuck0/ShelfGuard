"""
ShelfGuard Two-Phase Discovery Architecture
============================================
Solves the "Universe Definition" problem with a strategic two-phase approach.

Phase 1 (Seed Discovery):
    - Lightweight keyword search (25-50 results)
    - User selects "Seed Product" to define market focus
    - Extract category breadcrumb for Phase 2

Phase 2 (Market Mapping):
    - Fetch products from seed's category (100 ASINs)
    - Fetch 90 days of historical price/BSR data for those ASINs
    - Build weekly tables using same methodology as main dashboard
    - Aggregate to create snapshot with actual historical revenue
    - Result: Clean denominator for market share calculations based on real data

FIX 1.3: Added database cache checks before Keepa API calls to reduce API costs.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import keepa
import requests
import os
import time
from datetime import datetime, timedelta
from openai import OpenAI
import json
import hashlib

# Import scrapers for building weekly tables
from scrapers.keepa_client import build_keepa_weekly_table
from src.backfill import fetch_90day_history

# Import family harvester for variation-aware discovery
from src.family_harvester import (
    harvest_product_families,
    harvest_to_seed_dataframe,
    HarvestResult
)


# ========================================
# GLOBAL CIRCUIT BREAKER for API Calls
# ========================================
# Prevents infinite loops from draining Keepa tokens
# Tracks API calls per session and enforces hard limits

_API_CALL_TRACKER = {
    "session_id": None,
    "call_count": 0,
    "last_reset": None,
    "max_calls_per_minute": 100,  # Keepa limit
    "max_calls_per_session": 200,  # Safety limit (~3-5 searches before circuit trips)
}

def _check_circuit_breaker() -> bool:
    """
    Check if API circuit breaker should trip.

    Returns:
        True if API call is allowed, False if circuit is open (rate limit hit)
    """
    global _API_CALL_TRACKER

    # Initialize session tracking
    current_session = st.session_state.get("session_id")
    if current_session is None:
        # Generate unique session ID
        current_session = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        st.session_state["session_id"] = current_session

    # Reset tracker if new session
    if _API_CALL_TRACKER["session_id"] != current_session:
        _API_CALL_TRACKER.update({
            "session_id": current_session,
            "call_count": 0,
            "last_reset": time.time()
        })

    # Check session limit
    if _API_CALL_TRACKER["call_count"] >= _API_CALL_TRACKER["max_calls_per_session"]:
        st.error(
            f"🚨 **CIRCUIT BREAKER TRIPPED**\n\n"
            f"API call limit reached ({_API_CALL_TRACKER['max_calls_per_session']} calls/session).\n\n"
            f"This prevents infinite loops from draining your Keepa tokens.\n\n"
            f"**To continue:** Refresh the page or wait 60 seconds."
        )
        return False

    # Check per-minute rate limit
    if _API_CALL_TRACKER["last_reset"]:
        elapsed = time.time() - _API_CALL_TRACKER["last_reset"]
        if elapsed < 60:
            rate = _API_CALL_TRACKER["call_count"] / (elapsed / 60)
            if rate > _API_CALL_TRACKER["max_calls_per_minute"]:
                wait_time = int(60 - elapsed)
                st.warning(
                    f"⚠️ **Rate Limit Protection**\n\n"
                    f"Too many API calls per minute ({int(rate)}/min > {_API_CALL_TRACKER['max_calls_per_minute']}/min).\n\n"
                    f"Waiting {wait_time}s before next request..."
                )
                time.sleep(wait_time)
                # Reset counter after waiting
                _API_CALL_TRACKER["call_count"] = 0
                _API_CALL_TRACKER["last_reset"] = time.time()

    # Increment counter
    _API_CALL_TRACKER["call_count"] += 1

    return True


def _record_api_call(call_type: str = "query"):
    """
    Record an API call for circuit breaker tracking.

    Args:
        call_type: Type of API call (query, product_finder, etc.)
    """
    if not _check_circuit_breaker():
        raise RuntimeError("Circuit breaker tripped - API call limit reached")


# ========================================
# HELPER: SAFE KEEPA CSV VALUE EXTRACTION
# ========================================

def _safe_csv_value(arr, default=0):
    """
    Safely extract the last numeric value from a Keepa CSV array.
    Handles edge cases where Keepa returns nested lists instead of flat arrays.
    """
    if not arr:
        return default
    try:
        val = arr[-1]
        # Handle nested lists (rare Keepa edge case)
        while isinstance(val, (list, tuple)) and len(val) > 0:
            val = val[-1]
        # Ensure we have a numeric value
        if val is None or val == -1:
            return default
        if isinstance(val, (int, float)):
            return val
        return default
    except (IndexError, TypeError):
        return default


def _safe_numeric(val, default=0):
    """
    Safely extract a numeric value from Keepa data.
    Handles edge cases where Keepa returns lists, None, or other non-numeric types.
    """
    if val is None:
        return default
    # If it's already a number, return it
    if isinstance(val, (int, float)):
        return val
    # If it's a list/tuple, try to get the last numeric value
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return default
        return _safe_numeric(val[-1], default)
    # Try to convert string to number
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _scalarize_df_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Scalarize list/array values in DataFrame columns to prevent comparison errors.
    
    Some Keepa metrics are returned as lists (e.g., sellerIds). When aggregated or 
    used directly, these cause "'>' not supported between instances of 'list' and 'int'" errors.
    
    Args:
        df: DataFrame to process
        columns: List of column names to scalarize. If None, process all numeric-intended columns.
    
    Returns:
        DataFrame with scalar values in specified columns
    """
    if df.empty:
        return df
    
    # Default columns that should be numeric
    if columns is None:
        columns = ['bsr', 'price', 'monthly_units', 'revenue_proxy', 'review_count', 
                   'rating', 'new_offer_count', 'amazon_bb_share', 'sales_rank', 
                   'filled_price', 'sales_rank_filled']
    
    def scalarize_value(val):
        """Convert list/array to scalar."""
        if val is None:
            return 0
        if isinstance(val, (list, tuple)):
            return val[-1] if len(val) > 0 else 0
        if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
            try:
                arr = list(val)
                return arr[-1] if len(arr) > 0 else 0
            except (TypeError, IndexError):
                return 0
        return val
    
    for col in columns:
        if col in df.columns:
            # Apply scalarization to all values - some rows may have lists even if first doesn't
            df[col] = df[col].apply(scalarize_value)
            # Ensure column is numeric after scalarization
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


# ===========================================================================
# VARIATION DEDUPLICATION: Prevent revenue overcounting for child variations
# ===========================================================================
# When products share the same parent_asin, they are variations of ONE listing.
# They all inherit the parent's BSR, so revenue would be counted N times if
# we just sum all products. Fix: divide revenue by sibling count.
#
# Example: ALOHA has 22 variations all sharing parent B0FP2N64VD with BSR=25
# Without fix: 22 products × $72k each = $1.5M (22x overcounted)
# With fix: $72k / 22 per product = $72k total (correct)
# ===========================================================================

def _calculate_adjusted_revenue(products: list, revenue_key: str = "revenue_proxy") -> float:
    """
    Calculate total revenue with variation deduplication.
    
    Products that share a parent_asin are variations of the same listing.
    They share the same BSR/revenue, so we divide by sibling count.
    
    Args:
        products: List of product dicts with 'parent_asin' and revenue_key
        revenue_key: Key for revenue value (default 'revenue_proxy')
    
    Returns:
        Adjusted total revenue (float)
    """
    if not products:
        return 0.0
    
    # Count siblings per parent_asin
    parent_counts = {}
    for p in products:
        parent = p.get("parent_asin", "")
        if parent:
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
    
    # Sum revenue with adjustment
    total = 0.0
    for p in products:
        revenue = p.get(revenue_key, 0) or 0
        parent = p.get("parent_asin", "")
        if parent and parent in parent_counts:
            # Divide revenue by number of siblings to avoid overcounting
            total += revenue / parent_counts[parent]
        else:
            # No parent - count full revenue
            total += revenue
    
    return total


def _apply_variation_adjustment_to_df(df: pd.DataFrame, revenue_col: str = "weekly_revenue") -> pd.DataFrame:
    """
    Apply variation deduplication to a DataFrame for all BSR-derived metrics.

    When products share a parent_asin, they are variations of ONE listing.
    They all inherit the parent's BSR, so all BSR-derived metrics (revenue, units)
    would be counted N times if we just sum. Fix: divide by sibling count.

    Args:
        df: DataFrame with 'parent_asin' and revenue/units columns
        revenue_col: Name of revenue column to adjust (default: weekly_revenue)

    Returns:
        DataFrame with adjusted columns and '_sibling_count' added
    """
    WEEKS_PER_MONTH = 4.33

    if df.empty:
        df["weekly_revenue_adjusted"] = 0.0
        df["monthly_revenue_adjusted"] = 0.0
        df["revenue_proxy_adjusted"] = 0.0  # Legacy
        df["monthly_units_adjusted"] = 0.0
        df["_sibling_count"] = 1
        return df

    if "parent_asin" in df.columns:
        # Count siblings per parent ASIN (products with same parent)
        parent_counts = df[df["parent_asin"].notna() & (df["parent_asin"] != "")].groupby("parent_asin").size()

        # Create sibling count column (default 1 for products without parent)
        df["_sibling_count"] = df["parent_asin"].map(parent_counts).fillna(1).astype(int)
        df.loc[df["parent_asin"].isna() | (df["parent_asin"] == ""), "_sibling_count"] = 1

        # === STANDARDIZED REVENUE ADJUSTMENTS (2026-01-30) ===
        # Adjust weekly_revenue (base unit)
        if "weekly_revenue" in df.columns:
            df["weekly_revenue_adjusted"] = df["weekly_revenue"] / df["_sibling_count"]
            df["monthly_revenue_adjusted"] = df["weekly_revenue_adjusted"] * WEEKS_PER_MONTH
        elif revenue_col in df.columns:
            # Fallback to provided column (for legacy code paths)
            df["weekly_revenue_adjusted"] = df[revenue_col] / df["_sibling_count"]
            df["monthly_revenue_adjusted"] = df["weekly_revenue_adjusted"] * WEEKS_PER_MONTH
        else:
            df["weekly_revenue_adjusted"] = 0.0
            df["monthly_revenue_adjusted"] = 0.0

        # Legacy alias
        df["revenue_proxy_adjusted"] = df["weekly_revenue_adjusted"].copy()

        # Also adjust units if present (also BSR-derived)
        if "monthly_units" in df.columns:
            df["monthly_units_adjusted"] = df["monthly_units"] / df["_sibling_count"]
        elif "weekly_units" in df.columns:
            df["weekly_units_adjusted"] = df["weekly_units"] / df["_sibling_count"]
            df["monthly_units_adjusted"] = df["weekly_units_adjusted"] * WEEKS_PER_MONTH
        else:
            df["monthly_units_adjusted"] = 0.0

        # Adjust weekly sales if present (for weekly data) - legacy
        if "weekly_sales_filled" in df.columns:
            df["weekly_sales_adjusted"] = df["weekly_sales_filled"] / df["_sibling_count"]
        if "estimated_units" in df.columns:
            df["estimated_units_adjusted"] = df["estimated_units"] / df["_sibling_count"]
    else:
        df["weekly_revenue_adjusted"] = df.get("weekly_revenue", df.get(revenue_col, 0))
        df["monthly_revenue_adjusted"] = df["weekly_revenue_adjusted"] * WEEKS_PER_MONTH
        df["revenue_proxy_adjusted"] = df["weekly_revenue_adjusted"]  # Legacy
        df["monthly_units_adjusted"] = df.get("monthly_units", 0)
        df["_sibling_count"] = 1

    return df


# ========================================
# FIX 1.3: DATABASE CACHE FOR API CALLS
# ========================================

def _get_search_cache_key(keyword: str, domain: str, category_filter: Optional[int] = None) -> str:
    """Generate a unique cache key for a search query."""
    key_str = f"search_{keyword.lower().strip()}_{domain}_{category_filter or 'all'}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _check_search_cache(keyword: str, domain: str, category_filter: Optional[int] = None, max_age_hours: int = 6) -> Optional[pd.DataFrame]:
    """
    Check if we have recent cached search results in the database.

    FIX 1.3: This reduces Keepa API calls by 50-80% for repeated searches.

    Args:
        keyword: Search keyword
        domain: Amazon domain
        category_filter: Optional category filter
        max_age_hours: Maximum cache age in hours (default 6)

    Returns:
        Cached DataFrame if found and fresh, None otherwise
    """
    try:
        # First check session state (fastest)
        cache_key = _get_search_cache_key(keyword, domain, category_filter)
        session_key = f"search_cache_{cache_key}"
        time_key = f"{session_key}_time"

        if session_key in st.session_state and time_key in st.session_state:
            cache_time = st.session_state[time_key]
            if isinstance(cache_time, datetime):
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                if age_hours < max_age_hours:
                    st.caption(f"⚡ Using cached search results ({age_hours:.1f}h old)")
                    return st.session_state[session_key]

        # Then check Supabase for cross-session cache
        try:
            from src.supabase_reader import create_supabase_client
            supabase = create_supabase_client()

            # Check if we have recent snapshots for products matching this search
            # This is a lightweight check - we just see if we have data
            cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

            # Query for recent snapshots that might match this search
            # We use title search as a proxy for the keyword
            result = supabase.table("product_snapshots").select(
                "asin, title, brand, sales_rank, buy_box_price, main_image, category_id"
            ).ilike("title", f"%{keyword}%").gte(
                "fetched_at", cutoff_time
            ).order("sales_rank", desc=False).limit(50).execute()

            if result.data and len(result.data) >= 10:
                # We have enough cached products - build DataFrame
                df = pd.DataFrame(result.data)

                # Rename columns to match expected format
                df = df.rename(columns={
                    "sales_rank": "bsr",
                    "buy_box_price": "price"
                })

                # Add placeholder columns
                df["category_path"] = "Cached"
                df["leaf_category_id"] = df.get("category_id", 0)
                df["category_tree_ids"] = [[]]
                df["category_tree_names"] = [[]]

                # Store in session state for faster subsequent access
                st.session_state[session_key] = df
                st.session_state[time_key] = datetime.now()

                st.caption(f"⚡ Found {len(df)} cached products matching '{keyword}'")
                return df

        except Exception as e:
            # Database check failed - continue to API
            pass

        return None

    except Exception as e:
        return None


def _store_search_in_cache(keyword: str, domain: str, df: pd.DataFrame, category_filter: Optional[int] = None) -> None:
    """Store search results in session state cache."""
    try:
        cache_key = _get_search_cache_key(keyword, domain, category_filter)
        session_key = f"search_cache_{cache_key}"
        time_key = f"{session_key}_time"

        st.session_state[session_key] = df
        st.session_state[time_key] = datetime.now()
    except Exception:
        pass  # Caching failure is not critical


# ========================================
# ENHANCEMENT 2.2: LLM RESPONSE CACHING
# ========================================

def _get_llm_cache_key(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Generate a unique cache key for an LLM call."""
    key_str = f"llm_{model}_{prompt}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _check_llm_cache(prompt: str, model: str = "gpt-4o-mini", max_age_hours: int = 24) -> Optional[str]:
    """
    Check if we have a cached LLM response.

    ENHANCEMENT 2.2: This reduces OpenAI API costs by 60-90% for repeated queries.

    Args:
        prompt: The prompt text
        model: Model name
        max_age_hours: Maximum cache age in hours (default 24)

    Returns:
        Cached response if found and fresh, None otherwise
    """
    try:
        cache_key = _get_llm_cache_key(prompt, model)
        session_key = f"llm_cache_{cache_key}"
        time_key = f"{session_key}_time"

        # Check session state first (fastest)
        if session_key in st.session_state and time_key in st.session_state:
            cache_time = st.session_state[time_key]
            if isinstance(cache_time, datetime):
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                if age_hours < max_age_hours:
                    st.caption(f"⚡ Using cached LLM response ({age_hours:.1f}h old)")
                    return st.session_state[session_key]

        # Try Supabase for persistent cache
        try:
            from src.supabase_reader import create_supabase_client
            supabase = create_supabase_client()

            cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

            result = supabase.table("llm_cache").select("response").eq(
                "cache_key", cache_key
            ).gte("created_at", cutoff_time).limit(1).execute()

            if result.data and len(result.data) > 0:
                response = result.data[0].get("response")
                if response:
                    # Store in session for faster subsequent access
                    st.session_state[session_key] = response
                    st.session_state[time_key] = datetime.now()
                    st.caption(f"⚡ Using cached LLM response from database")
                    return response

        except Exception:
            pass  # Database not available or table doesn't exist

        return None

    except Exception:
        return None


def _store_llm_in_cache(prompt: str, response: str, model: str = "gpt-4o-mini") -> None:
    """Store LLM response in cache."""
    try:
        cache_key = _get_llm_cache_key(prompt, model)
        session_key = f"llm_cache_{cache_key}"
        time_key = f"{session_key}_time"

        # Store in session state
        st.session_state[session_key] = response
        st.session_state[time_key] = datetime.now()

        # Try to store in Supabase for persistence
        try:
            from src.supabase_reader import create_supabase_client
            supabase = create_supabase_client()

            supabase.table("llm_cache").upsert({
                "cache_key": cache_key,
                "prompt_hash": cache_key,  # Using hash instead of full prompt to save space
                "response": response,
                "model": model,
                "created_at": datetime.now().isoformat()
            }, on_conflict="cache_key").execute()

        except Exception:
            pass  # Database storage failed, session cache still works

    except Exception:
        pass  # Caching failure is not critical


def _check_category_cache(category_id: int, max_age_hours: int = 12) -> Optional[Tuple[pd.DataFrame, Dict]]:
    """
    Check if we have recent cached category data in the database.

    FIX 1.3: This reduces Keepa API calls for Phase 2 market mapping.

    Args:
        category_id: Category ID to check
        max_age_hours: Maximum cache age in hours (default 12)

    Returns:
        Tuple of (DataFrame, market_stats) if found and fresh, None otherwise
    """
    try:
        from src.supabase_reader import create_supabase_client
        supabase = create_supabase_client()

        cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        # Check for recent snapshots in this category
        result = supabase.table("product_snapshots").select(
            "asin, title, brand, parent_asin, sales_rank, buy_box_price, estimated_weekly_revenue, "
            "estimated_units, review_count, rating, main_image, category_id"
        ).eq("category_id", category_id).gte(
            "fetched_at", cutoff_time
        ).order("sales_rank", desc=False).limit(100).execute()

        if result.data and len(result.data) >= 50:
            df = pd.DataFrame(result.data)

            # Rename to expected format
            df = df.rename(columns={
                "sales_rank": "bsr",
                "buy_box_price": "price",
                "estimated_weekly_revenue": "revenue_proxy",
                "estimated_units": "monthly_units"
            })

            # Calculate revenue_proxy if missing
            if 'revenue_proxy' not in df.columns or df['revenue_proxy'].isna().all():
                if 'monthly_units' in df.columns and 'price' in df.columns:
                    df['revenue_proxy'] = df['monthly_units'] * df['price']
                else:
                    df['revenue_proxy'] = 0

            # Add data_weeks for AI confidence (cached data = assume 4 weeks)
            df['data_weeks'] = 4

            # Ensure weekly_revenue exists
            if "weekly_revenue" not in df.columns:
                if "revenue_proxy" in df.columns:
                    df["weekly_revenue"] = df["revenue_proxy"]
                else:
                    df["weekly_revenue"] = 0
            df["monthly_revenue"] = df["weekly_revenue"] * 4.33

            # Apply variation deduplication to prevent revenue overcounting
            df = _apply_variation_adjustment_to_df(df, "weekly_revenue")
            adjusted_weekly_revenue = df["weekly_revenue_adjusted"].sum()
            adjusted_monthly_revenue = df["monthly_revenue_adjusted"].sum()

            market_stats = {
                "total_products": len(df),
                "total_category_revenue": adjusted_weekly_revenue,  # Weekly (base unit)
                "weekly_revenue": adjusted_weekly_revenue,
                "monthly_revenue": adjusted_monthly_revenue,
                "category_id": category_id,
                "validated_products": len(df),
                "validated_revenue": adjusted_weekly_revenue,
                "source": "cache",
                "df_weekly": pd.DataFrame()  # No weekly data from cache
            }

            st.caption(f"⚡ Found {len(df)} cached products for category {category_id} (${adjusted_weekly_revenue:,.0f}/wk)")
            return df, market_stats

        return None

    except Exception as e:
        return None


def get_keepa_api_key() -> Optional[str]:
    """Get Keepa API key from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets'):
            # Try nested structure: [keepa] api_key = "..."
            key = st.secrets.get("keepa", {}).get("api_key")
            if key:
                return key
            
            # Try flat structure: keepa_api_key = "..."
            key = st.secrets.get("keepa_api_key")
            if key:
                return key
    except Exception:
        pass
    
    # Fall back to environment variables
    return os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client from secrets or environment."""
    try:
        return OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
    except:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None


# ========================================
# FAMILY HARVESTER TOGGLE
# ========================================

# Set this to True to enable variation-aware discovery
# This will fetch complete product families instead of naive keyword matches
ENABLE_FAMILY_HARVESTER = True


@st.cache_data(ttl=3600, hash_funcs={type(None): lambda x: None})
def phase1_seed_discovery(
    keyword: str,
    limit: int = 50,
    domain: str = "US",
    category_filter: Optional[int] = None,
    check_cache: bool = True,
    use_family_harvester: bool = None,  # None = use global toggle
    search_mode: str = "keyword",  # "keyword" or "brand"
    conquest_mode: bool = False,   # NEW: Find Amazon vulnerabilities
    quality_filter: bool = False   # NEW: Ensure basic quality (reviews/variations)
) -> pd.DataFrame:
    """
    Phase 1: Lightweight search to find seed products.

    FIX 1.3: Now checks database cache before making Keepa API calls.
    This reduces API costs by 50-80% for repeated searches.
    
    NEW: When use_family_harvester=True, uses intelligent variation-aware
    discovery that fetches complete product families instead of naive keyword matches.
    
    NEW (Audit): Added 'conquest_mode' and 'quality_filter' additives.

    Args:
        keyword: User's search term
        limit: Max results (25-50 recommended)
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)
        category_filter: Optional category ID to restrict search (e.g., 16310101 for Grocery)
        check_cache: Whether to check database cache first (default True)
        use_family_harvester: Use variation-aware discovery (None = use global toggle)
        conquest_mode: Filter for Amazon supply instability (OOS > 2)
        quality_filter: Filter for established products (Reviews > 50)

    Returns:
        DataFrame with [asin, title, brand, category_id, category_path, price, bsr]
    """
    # Determine if we should use family harvester
    use_families = use_family_harvester if use_family_harvester is not None else ENABLE_FAMILY_HARVESTER
    
    # NEW: If family harvester is enabled, use that instead
    if use_families:
        try:
            if search_mode == "brand":
                st.info("🏷️ Using Brand Search (exact brand name matching)")
            else:
                st.info("🧬 Using Family Harvester (variation-aware discovery)")
            return harvest_to_seed_dataframe(
                keyword=keyword,
                limit=limit,
                domain=domain,
                category_filter=category_filter,
                search_mode=search_mode
            )
        except Exception as e:
            # Store error in session state so it persists after rerun
            error_msg = f"Family Harvester failed: {type(e).__name__}: {str(e)}"
            st.session_state["last_harvester_error"] = error_msg
            st.error(f"⚠️ {error_msg}\n\nFalling back to naive search...")
            import traceback
            st.code(traceback.format_exc())
            # Fall through to legacy logic
    
    # FIX 1.3: Check database cache first to avoid unnecessary API calls
    if check_cache:
        cached_df = _check_search_cache(keyword, domain, category_filter, max_age_hours=6)
        if cached_df is not None and not cached_df.empty:
            return cached_df.head(limit)

    api_key = get_keepa_api_key()
    if not api_key:
        raise ValueError("KEEPA_API_KEY not found")

    # Use direct HTTP API (Python keepa library has issues with product_finder)
    try:
        # Build query JSON
        query_json = {
            "title": keyword,
            "perPage": max(50, limit),  # Minimum 50 per Keepa API docs
            "page": 0,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
            # CRITICAL: Sort by Sales Rank ascending (lower rank = better seller)
            # Without this, Keepa returns by "relevance" which misses high-volume hero ASINs
            "sort": [["current_SALES", "asc"]]
        }

        # === ADDITIVE FILTERS (Audit Upgrade) ===
        if conquest_mode:
            st.caption("⚔️ CONQUEST MODE: Filtering for Amazon 1P Instability")
            # Look for items where Amazon has been OOS at least twice in 30 days
            query_json["outOfStockCountAmazon30_gte"] = 2
            
        if quality_filter:
            st.caption("✨ QUALITY FILTER: Ensuring established products")
            query_json["hasReviews"] = True
            query_json["variationCount_gte"] = 1 # Must correspond to a product vs isolated sku

        # Add category filter if provided (category-first mode)
        if category_filter:
            query_json["rootCategory"] = [category_filter]
            st.info(f"🎯 Searching within category {category_filter}")

        # Convert domain string to domain ID
        domain_map = {"US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12}
        domain_id = domain_map.get(domain, 1)

        # Make HTTP POST request to Keepa API
        url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}&stats=0"
        response = requests.post(url, json=query_json, timeout=30)

        if response.status_code != 200:
            st.error(f"Keepa API error: {response.status_code} - {response.text}")
            return pd.DataFrame()

        result = response.json()
        asins = result.get("asinList", [])

        if not asins:
            st.warning(f"⚠️ No products found for '{keyword}'")
            return pd.DataFrame()

        # Limit to requested amount
        asins = asins[:limit]

        # Fetch full product data using keepa library
        # Process in smaller batches with retry logic to avoid timeout (keepa library has 10s default timeout)
        api = keepa.Keepa(api_key)
        products = []
        keepa_batch_size = 20  # Smaller batches to prevent timeout
        
        for i in range(0, len(asins), keepa_batch_size):
            batch_asins = asins[i:i + keepa_batch_size]
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    batch_products = api.query(batch_asins, stats=30, rating=True)
                    products.extend(batch_products)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timeout" in error_msg or "read timeout" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                            time.sleep(wait_time)
                            continue
                        else:
                            st.warning(f"⚠️ Timeout after {max_retries} attempts for batch {i//keepa_batch_size + 1}. Skipping batch.")
                            break  # Give up on this batch
                    else:
                        # Non-timeout error, break immediately
                        st.warning(f"⚠️ Error fetching batch {i//keepa_batch_size + 1}: {str(e)}")
                        break
            
            # Small delay between batches to be polite to API
            if i + keepa_batch_size < len(asins):
                time.sleep(0.5)

        records = []
        for product in products:
            asin = product.get("asin", "UNK")
            title = product.get("title", "Unknown")

            # Extract category info
            category_tree = product.get("categoryTree", [])
            root_category = product.get("rootCategory", 0)
            
            # Extract the most specific (leaf) category ID from categoryTree
            # Leaf node ensures high relevance and "purity" of data
            # Future: Allow manual expansion to parent/sibling nodes for benchmarking vs broader analysis
            leaf_category_id = root_category  # Default to root if no tree
            if category_tree and len(category_tree) > 0:
                # Last node in tree is the leaf (most specific) category
                leaf_category = category_tree[-1]
                leaf_category_id = leaf_category.get("catId", root_category)

            # Build category path (breadcrumb)
            category_path = " > ".join([cat.get("name", "") for cat in category_tree]) if category_tree else "Unknown"

            # Get price (using safe extraction for nested lists)
            csv = product.get("csv", [])
            price_cents = _safe_csv_value(csv[18] if csv and len(csv) > 18 else None, 0)
            price = price_cents / 100.0 if price_cents > 0 else 0

            # Get BSR (using safe extraction)
            bsr = _safe_csv_value(csv[3] if csv and len(csv) > 3 else None, 0)

            # Extract brand from Keepa data (use brand field directly, fallback to title)
            brand = product.get("brand", "")
            if not brand:
                brand = title.split()[0] if title else "Unknown"

            # Extract all category IDs from the tree (for progressive fallback)
            category_tree_ids = [cat.get("catId", 0) for cat in category_tree] if category_tree else [root_category]
            
            # Also store category names for building effective_category_path later
            category_tree_names = [cat.get("name", "") for cat in category_tree] if category_tree else ["Unknown"]
            
            records.append({
                "asin": asin,
                "title": title,
                "brand": brand,
                "category_id": root_category,  # Root category for backward compatibility
                "leaf_category_id": leaf_category_id,  # Leaf node for high relevance/purity
                "category_tree_ids": category_tree_ids,  # Full tree for progressive fallback
                "category_tree_names": category_tree_names,  # Names for building paths
                "category_path": category_path,
                "price": price,
                "bsr": bsr
            })

        df = pd.DataFrame(records)
        df = df.sort_values("bsr").reset_index(drop=True)  # Sort by best sellers

        # FIX 1.3: Store in session cache for faster subsequent access
        _store_search_in_cache(keyword, domain, df, category_filter)

        return df

    except Exception as e:
        st.error(f"Phase 1 Discovery Error: {str(e)}")
        return pd.DataFrame()


def phase1_brand_focused_discovery(
    brand_name: str,
    limit: int = 100,
    domain: str = "US",
    category_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Brand-Focused Discovery: Get ALL products from a specific brand.
    
    This is optimized for scenarios like "I want all RXBAR products"
    where the user knows the exact brand they want to analyze.
    
    Uses Keepa's brand filter for precise matching, then fetches
    complete product families for each result.
    
    Args:
        brand_name: Exact brand name (e.g., "RXBAR", "Poppi", "Olipop")
        limit: Maximum ASINs to return
        domain: Amazon marketplace
        category_filter: Optional category to restrict search
        
    Returns:
        DataFrame with all products from the brand, prioritized by BSR
    """
    from src.family_harvester import harvest_product_families
    
    st.info(f"🎯 Brand-focused discovery for '{brand_name}'")
    
    try:
        result = harvest_product_families(
            keyword=brand_name,
            max_asins=limit,
            domain=domain,
            category_filter=category_filter,
            brand_filter=brand_name,  # KEY: Use brand filter for exact matching
            seed_limit=20,  # More seeds since we're filtering to one brand
            expand_variations=True,
            filter_children=True
        )
        
        if not result.families:
            st.warning(f"No products found for brand '{brand_name}'")
            return pd.DataFrame()
        
        df = result.to_dataframe()
        
        # Add stats
        st.success(
            f"✅ Found {len(df)} products across {len(result.families)} product lines "
            f"for brand '{brand_name}'"
        )
        
        return df
        
    except Exception as e:
        st.error(f"Brand discovery failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def phase2_category_market_mapping(
    category_id: int,
    seed_product_title: str,
    seed_asin: Optional[str] = None,
    target_revenue_pct: float = 80.0,
    max_products: int = 500,
    batch_size: int = 100,
    domain: str = "US",
    leaf_category_id: Optional[int] = None,
    category_path: Optional[str] = None,
    category_tree_ids: Optional[tuple] = None,
    min_products: int = 10,
    check_cache: bool = False,
    brand_filter: Optional[str] = None,
    target_brand: Optional[str] = None,  # NEW: Fetch ALL products from this brand first
    mvp_mode: bool = False,              # MVP: strict leaf membership enforcement + capped fallback
    arena_size: int = 300,              # mvp_mode only: total ASINs in final arena
    min_competitors: int = 150,          # mvp_mode only: guaranteed competitor slots
    brand_cap: Optional[int] = None,    # mvp_mode only: max brand ASINs; defaults to arena_size - min_competitors
) -> Tuple[pd.DataFrame, Dict]:
    """
    Phase 2: Map competitive market with brand-first fetching.

    Flow:
    1. Progressive category fallback: Start with leaf, walk up until >= min_products found
    2. Fetch ALL products from target_brand in that category
    3. Fill remaining slots with competitors
    4. Combine into market snapshot

    Args:
        category_id: Root category ID from seed product
        seed_product_title: Title of seed product (for LLM context)
        seed_asin: ASIN of seed product (will be force-included in results)
        target_revenue_pct: Stop when this % of revenue is captured (default 80%)
        max_products: Safety limit to prevent runaway fetching
        batch_size: How many products to fetch per iteration
        domain: Amazon marketplace
        leaf_category_id: Most specific subcategory ID (leaf node)
        category_tree_ids: Full list of category IDs from root to leaf
        min_products: Minimum products needed before walking up hierarchy (default 10)
        check_cache: Whether to check database cache first
        target_brand: Brand name - fetch ALL products from this brand first, then competitors

    Returns:
        Tuple of (validated_df, market_stats)
    """
    # FIX 1.3: Optional category cache check
    effective_category = leaf_category_id if leaf_category_id else category_id
    if check_cache:
        cached_result = _check_category_cache(effective_category, max_age_hours=12)
        if cached_result is not None:
            df, stats = cached_result
            # Ensure seed ASIN is included if provided
            if seed_asin and seed_asin not in df['asin'].values:
                st.warning(f"⚠️ Seed ASIN {seed_asin} not in cache - will need fresh fetch")
            else:
                return cached_result

    api_key = get_keepa_api_key()
    if not api_key:
        raise ValueError("KEEPA_API_KEY not found")

    # Display category info with debugging
    st.info(f"🎯 Fetching 100 ASINs from category: {category_id}")
    if category_path:
        st.caption(f"📂 Category: {category_path}")
    
    # Debug: Show what category filters will be used
    with st.expander("🔍 Debug: Category Filtering Parameters", expanded=False):
        st.write(f"**Root Category ID:** {category_id}")
        st.write(f"**Leaf Category ID:** {leaf_category_id}")
        st.write(f"**Category Path:** {category_path}")
        st.write(f"**Category Tree IDs:** {category_tree_ids}")
        st.write(f"**Seed ASIN:** {seed_asin}")
        if leaf_category_id:
            st.success(f"✅ **API Query**: Using categories_include=[{leaf_category_id}] + rootCategory=[{category_id}]")
            st.info("Using categories_include for precise subcategory filtering (per Keepa API docs)")
        else:
            st.warning(f"⚠️ No leaf category - using rootCategory=[{category_id}] only")

    # Fetch products from the category in batches
    # Target: 200 products with valid price/BSR (100 brand + 100 competitors)
    # Fetch extra ASINs to account for ones without price/BSR data (~40-50% may be filtered)
    all_products = []
    page = 0
    cumulative_revenue = 0
    if mvp_mode:
        _brand_cap = brand_cap if brand_cap is not None else (arena_size - min_competitors)
        target_valid_products = arena_size        # replaces hardcoded 200
        _min_competitors = min_competitors
    else:
        MAX_BRAND_ASINS_LEGACY = 100             # used below in brand fetch cap
        _brand_cap = MAX_BRAND_ASINS_LEGACY
        target_valid_products = 200             # existing value, unchanged
        _min_competitors = 0                    # no floor enforced in legacy mode

    # Convert domain string to domain ID
    domain_map = {"US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12}
    domain_id = domain_map.get(domain, 1)

    # Initialize Keepa for query() calls
    api = keepa.Keepa(api_key)

    # ========== PROGRESSIVE CATEGORY FALLBACK ==========
    # Start with the most specific (leaf) category
    # If it has < min_products, walk back up the category tree
    # This ensures we get relevant products without going too broad
    
    effective_category_id = leaf_category_id if leaf_category_id else category_id
    use_categories_include = leaf_category_id is not None
    effective_category_path = category_path or ""
    
    # Build ordered list of category IDs from leaf → root
    # category_tree_ids is (leaf, ..., mid, ..., root) but we need to test from leaf first
    category_levels = []
    if category_tree_ids:
        # category_tree_ids comes from Keepa in root→leaf order, so reverse it
        category_levels = list(reversed(category_tree_ids))
    elif leaf_category_id:
        category_levels = [leaf_category_id, category_id]
    else:
        category_levels = [category_id]
    
    # Also get category names for display (parallel to IDs)
    category_level_names = []
    if category_path:
        # category_path is "Root > Mid > Leaf", split and reverse
        names = [n.strip() for n in category_path.split(">")]
        category_level_names = list(reversed(names))  # Now leaf → root
    
    # ========== TEST EACH CATEGORY LEVEL ==========
    # Quick count query to find the first level with enough products.
    # MVP mode: caps at leaf (level 0) + one-step parent (level 1). Never falls to root-only.
    st.caption("🔍 Finding optimal category depth...")

    # Observability counters (populated below; used in market_stats)
    excluded_off_leaf_count = 0
    _mvp_search_scope = "leaf"  # "leaf" | "parent" (root never used in mvp_mode)

    for level_idx, test_cat_id in enumerate(category_levels):
        # MVP mode: stop after testing leaf (0) and immediate parent (1)
        if mvp_mode and level_idx >= 2:
            st.warning(
                f"⚠️ MVP mode: capping category search at 1 level above leaf. "
                f"Using search category {effective_category_id}. "
                f"Arena may be smaller than target — leaf membership enforced post-fetch."
            )
            break

        # Root category detected
        if test_cat_id == category_id and level_idx > 0:
            if mvp_mode:
                # MVP: never drop to root-only; keep the subcategory from the last iteration
                st.warning(
                    f"⚠️ MVP mode: reached root — NOT falling back to root-only query. "
                    f"Keeping search category {effective_category_id} with categories_include active."
                )
                break
            # Legacy: fall back to root with no categories_include
            st.caption(f"  → Falling back to root category")
            effective_category_id = category_id
            use_categories_include = False
            effective_category_path = category_level_names[-1] if category_level_names else "Root Category"
            break

        # Build a quick count query for this category level
        count_query = {
            "perPage": min_products,  # Just need to know if >= min_products exist
            "page": 0,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
        }

        # Add category filter
        if test_cat_id != category_id:
            count_query["categories_include"] = [int(test_cat_id)]
            count_query["rootCategory"] = [int(category_id)]
        else:
            count_query["rootCategory"] = [int(category_id)]

        # Add brand filter if provided
        if brand_filter:
            count_query["brand"] = [brand_filter]

        try:
            url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}&stats=0"
            response = requests.post(url, json=count_query, timeout=15)

            if response.status_code == 200:
                result = response.json()
                product_count = len(result.get("asinList", []))

                level_name = category_level_names[level_idx] if level_idx < len(category_level_names) else f"Level {level_idx}"

                if product_count >= min_products:
                    st.caption(f"  ✅ '{level_name}' has {product_count}+ products → using this category")
                    effective_category_id = test_cat_id
                    use_categories_include = test_cat_id != category_id
                    effective_category_path = " > ".join(list(reversed(category_level_names[:level_idx+1]))) if category_level_names else category_path
                    _mvp_search_scope = "leaf" if level_idx == 0 else "parent"
                    break
                else:
                    st.caption(f"  ⚠️ '{level_name}' has only {product_count} products → trying broader category...")
                    # Continue to next (broader) category level
            else:
                # API error, just use this level and continue
                st.caption(f"  ⚠️ Could not count products in category {test_cat_id}, using it anyway")
                effective_category_id = test_cat_id
                use_categories_include = test_cat_id != category_id
                _mvp_search_scope = "leaf" if level_idx == 0 else "parent"
                break

        except Exception as e:
            # Timeout or other error, just use current level
            st.caption(f"  ⚠️ Category test failed: {str(e)[:50]}, using current level")
            effective_category_id = test_cat_id
            use_categories_include = test_cat_id != category_id
            _mvp_search_scope = "leaf" if level_idx == 0 else "parent"
            break
    
    # Show final category being used
    st.info(f"📂 Fetching from: **{effective_category_path}** (ID: {effective_category_id})")
    
    # ========== BRAND-FIRST FETCHING ==========
    # If target_brand is provided, fetch ALL products from that brand first
    # Then fill remaining slots with competitors
    brand_products = []
    brand_asins_fetched = set()
    
    if target_brand:
        st.info(f"🎯 Step 1: Fetching ALL **{target_brand}** products...")
        
        # Query for all products from target brand in this category
        brand_query = {
            "brand": [target_brand],
            "perPage": 100,
            "page": 0,
            "current_SALES_gte": 1,
            "current_SALES_lte": 200000,
            # CRITICAL: Sort by Sales Rank ascending (lower rank = better seller)
            # Ensures we capture hero ASINs (bulk packs, 1P listings) before hitting caps
            "sort": [["current_SALES", "asc"]]
        }
        
        # Add category filter using the effective category from progressive fallback
        if use_categories_include:
            brand_query["categories_include"] = [int(effective_category_id)]
            brand_query["rootCategory"] = [int(category_id)]
        else:
            brand_query["rootCategory"] = [int(category_id)]
        
        try:
            url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}&stats=0"
            response = requests.post(url, json=brand_query, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                brand_asins = result.get("asinList", [])
                
                if brand_asins:
                    # CRITICAL FIX: Hard cap on brand products to prevent runaway fetching
                    # Keepa returns ALL matching ASINs (can be 24,000+ for large brands like Purina)
                    # In mvp_mode: _brand_cap = arena_size - min_competitors (guarantees competitor slots)
                    original_count = len(brand_asins)
                    if original_count > _brand_cap:
                        st.warning(f"⚠️ Found {original_count} **{target_brand}** products - limiting to top {_brand_cap} by Sales Rank")
                        brand_asins = brand_asins[:_brand_cap]  # Take best sellers (sorted by current_SALES asc)
                    else:
                        st.success(f"✅ Found {len(brand_asins)} **{target_brand}** products")
                    
                    # Fetch full product data for brand ASINs
                    # Rate limiting: Add delays between batches to avoid hitting Keepa limits
                    for i in range(0, len(brand_asins), 20):
                        batch_asins = brand_asins[i:i + 20]
                        max_retries = 3
                        retry_delay = 3
                        batch_products = []
                        
                        for attempt in range(max_retries):
                            try:
                                batch_products = api.query(batch_asins, stats=90, rating=True)
                                break  # Success
                            except Exception as e:
                                error_msg = str(e).lower()
                                if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                                    if attempt < max_retries - 1:
                                        wait_time = retry_delay * (2 ** attempt)  # Exponential: 3s, 6s, 12s
                                        st.warning(f"⚠️ Rate limited on brand batch {i//20 + 1}. Waiting {wait_time}s...")
                                        time.sleep(wait_time)
                                        continue
                                    else:
                                        st.error(f"❌ Rate limited after {max_retries} attempts. Saving {len(brand_products)} products collected so far.")
                                        break  # Save what we have
                                elif "timeout" in error_msg:
                                    if attempt < max_retries - 1:
                                        wait_time = retry_delay * (attempt + 1)
                                        st.caption(f"⚠️ Timeout on brand batch {i//20 + 1}, retrying in {wait_time}s...")
                                        time.sleep(wait_time)
                                        continue
                                    else:
                                        st.warning(f"⚠️ Timeout after {max_retries} attempts. Skipping batch.")
                                        break
                                else:
                                    st.warning(f"⚠️ Error fetching brand batch {i//20 + 1}: {str(e)[:50]}")
                                    break
                        
                        if not batch_products:
                            continue  # Skip this batch if we couldn't fetch it
                        
                        for product in batch_products:
                                asin = product.get("asin")
                                if not asin:
                                    continue
                                
                                # Skip variation parents (no data)
                                if product.get("productType") == 5:
                                    continue

                                # MVP mode: enforce leaf membership for brand products
                                if mvp_mode and leaf_category_id:
                                    product_cats = product.get("categories", [])
                                    if int(leaf_category_id) not in [int(c) for c in product_cats]:
                                        excluded_off_leaf_count += 1
                                        continue

                                # Extract product data (using safe extraction to handle nested lists)
                                csv = product.get("csv", [])
                                stats = product.get("stats", {})

                                # Price with multi-level fallback (same as competitor fetch)
                                price_cents = _safe_csv_value(csv[18] if csv and len(csv) > 18 else None, 0)
                                if price_cents <= 0:
                                    price_cents = _safe_csv_value(csv[0] if csv and len(csv) > 0 else None, 0)
                                if price_cents <= 0:
                                    price_cents = _safe_csv_value(csv[10] if csv and len(csv) > 10 else None, 0)

                                # FALLBACK: Use Keepa avg stats if current price unavailable
                                if price_cents <= 0 and stats:
                                    avg30 = stats.get("avg30", []) if isinstance(stats.get("avg30"), list) else []
                                    avg90 = stats.get("avg90", []) if isinstance(stats.get("avg90"), list) else []
                                    for idx in [18, 0, 1, 10]:
                                        if avg30 and len(avg30) > idx and avg30[idx] and avg30[idx] > 0:
                                            price_cents = avg30[idx]
                                            break
                                        if avg90 and len(avg90) > idx and avg90[idx] and avg90[idx] > 0:
                                            price_cents = avg90[idx]
                                            break

                                price = price_cents / 100.0 if price_cents > 0 else 0

                                # BSR with fallback to avg stats
                                bsr_value = _safe_csv_value(csv[3] if csv and len(csv) > 3 else None, 0)
                                if bsr_value <= 0 and stats:
                                    avg30 = stats.get("avg30", []) if isinstance(stats.get("avg30"), list) else []
                                    avg90 = stats.get("avg90", []) if isinstance(stats.get("avg90"), list) else []
                                    if avg30 and len(avg30) > 3 and avg30[3] and avg30[3] > 0:
                                        bsr_value = avg30[3]
                                    elif avg90 and len(avg90) > 3 and avg90[3] and avg90[3] > 0:
                                        bsr_value = avg90[3]

                                bsr = bsr_value if bsr_value > 0 else None

                                monthly_units = 0
                                if bsr and bsr > 0:
                                    monthly_units = 145000.0 * (bsr ** -0.9)

                                revenue = monthly_units * price
                                
                                title = product.get("title", "")
                                brand = product.get("brand", target_brand)
                                
                                image_urls = product.get("imagesCSV", "").split(",") if product.get("imagesCSV") else []
                                main_image = f"https://images-na.ssl-images-amazon.com/images/I/{image_urls[0]}" if image_urls else ""
                                
                                # === EXTRACT COMPETITIVE METRICS FROM KEEPA (brand products) ===
                                stats = product.get("stats", {})
                                
                                # Out of Stock Percentage (90 days) - use safe extraction
                                oos_90 = _safe_numeric(stats.get("outOfStockPercentage90") if stats else None, 0.0)
                                if oos_90 > 1:  # Normalize if > 1 (Keepa sometimes returns as percentage)
                                    oos_90 = oos_90 / 100
                                
                                # Buy Box Stats (30/90 day) - Amazon's % ownership
                                # Try 30-day first, then 90-day, then assume 50% if unavailable
                                bb_stats_30 = _safe_numeric(stats.get("buyBoxStatsAmazon30") if stats else None, -1)
                                bb_stats_90 = _safe_numeric(stats.get("buyBoxStatsAmazon90") if stats else None, -1)
                                if bb_stats_30 >= 0:
                                    amazon_bb_share = bb_stats_30 / 100.0 if bb_stats_30 > 1 else bb_stats_30
                                elif bb_stats_90 >= 0:
                                    amazon_bb_share = bb_stats_90 / 100.0 if bb_stats_90 > 1 else bb_stats_90
                                else:
                                    amazon_bb_share = 0.5  # Default to 50% if no data
                                
                                # Seller Count (index 11) - using safe extraction
                                new_offer_count = _safe_csv_value(csv[11] if csv and len(csv) > 11 else None, 1)
                                new_offer_count = max(1, new_offer_count)
                                
                                # Review Count (index 17 in Keepa Python lib)
                                review_count = _safe_csv_value(csv[17] if csv and len(csv) > 17 else None, 0)

                                # Rating (index 16 in Keepa Python lib) - stored as 10x (45 = 4.5 stars)
                                rating_raw = _safe_csv_value(csv[16] if csv and len(csv) > 16 else None, 0)
                                rating = rating_raw / 10.0 if rating_raw > 0 else 0.0

                                # Parent ASIN - critical for variation deduplication
                                parent_asin = product.get("parentAsin", "")
                                
                                # === NEW CRITICAL METRICS (2026-01-21) ===
                                # Amazon's actual monthly sold estimate
                                monthly_sold = _safe_numeric(product.get("monthlySold"), 0)
                                
                                # Pack size for per-unit normalization
                                number_of_items = _safe_numeric(product.get("numberOfItems"), 1)
                                number_of_items = max(1, int(number_of_items))
                                
                                # Buy Box ownership flags
                                buybox_is_amazon = product.get("buyBoxIsAmazon", None)
                                buybox_is_fba = product.get("buyBoxIsFBA", None)
                                buybox_is_backorder = product.get("buyBoxIsBackorder", False) or False
                                
                                # Seller IDs for true seller count
                                seller_ids = product.get("sellerIds", []) or []
                                seller_count = len(seller_ids)
                                has_amazon_seller = "ATVPDKIKX0DER" in seller_ids if seller_ids else False
                                
                                # OOS counts (more actionable than %)
                                oos_count_30 = _safe_numeric(stats.get("outOfStockCountAmazon30") if stats else None, 0)
                                oos_count_90 = _safe_numeric(stats.get("outOfStockCountAmazon90") if stats else None, 0)
                                
                                # Pre-calculated velocity from Keepa
                                delta_pct_30 = stats.get("deltaPercent30", []) if stats else []
                                delta_pct_90 = stats.get("deltaPercent90", []) if stats else []
                                velocity_30d = delta_pct_30[3] if delta_pct_30 and len(delta_pct_30) > 3 else None
                                velocity_90d = delta_pct_90[3] if delta_pct_90 and len(delta_pct_90) > 3 else None
                                
                                # Buy Box seller count and top seller share
                                bb_seller_count_30 = _safe_numeric(stats.get("buyBoxStatsSellerCount30") if stats else None, 0)
                                bb_top_seller_30 = _safe_numeric(stats.get("buyBoxStatsTopSeller30") if stats else None, 0)
                                if bb_top_seller_30 > 1:
                                    bb_top_seller_30 = bb_top_seller_30 / 100.0
                                
                                # Subscribe & Save
                                is_sns = product.get("isSNS", False) or False
                                
                                # Use monthlySold if available, otherwise keep BSR-based estimate
                                if monthly_sold > 0:
                                    monthly_units = monthly_sold
                                    revenue = monthly_units * price
                                    units_source = "amazon_monthly_sold"
                                else:
                                    units_source = "bsr_formula"

                                brand_products.append({
                                    "asin": asin,
                                    "title": title,
                                    "brand": brand,
                                    "price": price,
                                    "monthly_units": monthly_units,
                                    "revenue_proxy": revenue,
                                    "bsr": bsr,
                                    "main_image": main_image,
                                    "parent_asin": parent_asin,  # For variation deduplication
                                    "_is_target_brand": True,
                                    # Competitive metrics
                                    "outOfStockPercentage90": oos_90,
                                    "amazon_bb_share": amazon_bb_share,
                                    "new_offer_count": new_offer_count,
                                    "review_count": review_count,
                                    "rating": rating,
                                    # NEW CRITICAL METRICS (2026-01-21)
                                    "monthly_sold": monthly_sold,
                                    "number_of_items": number_of_items,
                                    "price_per_unit": price / number_of_items if number_of_items > 0 else price,
                                    "buybox_is_amazon": buybox_is_amazon,
                                    "buybox_is_fba": buybox_is_fba,
                                    "buybox_is_backorder": buybox_is_backorder,
                                    "seller_count": seller_count,
                                    "has_amazon_seller": has_amazon_seller,
                                    "oos_count_amazon_30": int(oos_count_30),
                                    "oos_count_amazon_90": int(oos_count_90),
                                    "velocity_30d": velocity_30d,
                                    "velocity_90d": velocity_90d,
                                    "bb_seller_count_30": int(bb_seller_count_30),
                                    "bb_top_seller_30": bb_top_seller_30,
                                    "is_sns": is_sns,
                                    "units_source": units_source,
                                })
                                brand_asins_fetched.add(asin)
                        
                        # Rate limiting: Longer delay between batches to avoid hitting Keepa limits
                        # Keepa allows ~100 requests/minute, so we space out batches
                        if i + 20 < len(brand_asins):
                            time.sleep(1.5)  # Increased from 0.5s to 1.5s
                    
                    st.caption(f"📊 Fetched {len(brand_products)} {target_brand} products with valid data")
                else:
                    st.warning(f"⚠️ No {target_brand} products found in this category")
            else:
                st.warning(f"⚠️ Brand query failed: {response.status_code}")
        except Exception as e:
            st.warning(f"⚠️ Brand fetch failed: {str(e)[:50]}")
        
        # Calculate remaining slots for competitors
        remaining_slots = target_valid_products - len(brand_products)
        if remaining_slots > 0:
            st.info(f"🎯 Step 2: Fetching {remaining_slots} competitor products...")
        
        # Pre-populate all_products with brand products
        all_products = brand_products.copy()

    # Track initial brand count so we can compute valid competitor count in the loop
    _initial_brand_count = len(all_products)

    # ========== MAIN FETCH LOOP (Competitors) ==========
    max_pages = 10  # Safety limit (10 pages x 100 = 1000 products max)
    
    while len(all_products) < target_valid_products and page < max_pages:
        # Build query for this category (use direct HTTP API)
        query_json = {
            "perPage": max(50, batch_size),  # Minimum 50
            "page": page,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
            # CRITICAL: Sort by Sales Rank ascending (lower rank = better seller)
            # Ensures we get top-selling competitors first, capturing 80-90% of market revenue
            "sort": [["current_SALES", "asc"]]
        }
        
        # Add category filter based on our progressive fallback decision
        if use_categories_include:
            # Use categories_include for precise subcategory filtering
            query_json["categories_include"] = [int(effective_category_id)]
            query_json["rootCategory"] = [int(category_id)]
        else:
            # Using root category only
            query_json["rootCategory"] = [int(category_id)]
        
        # NEW: Add brand filter to API query when in brand-focused mode
        # This filters at the API level, not post-fetch, preventing wrong products
        if brand_filter:
            query_json["brand"] = [brand_filter]
            if page == 0:
                st.info(f"🎯 Brand filter active: Only fetching '{brand_filter}' products")
        
        if page == 0:
            # Debug: Show exact query being sent
            with st.expander("🔍 Debug: Keepa Query JSON", expanded=False):
                st.json(query_json)
            st.caption(f"🔍 Querying with filter: {list(query_json.keys())}...")

        try:
            # CIRCUIT BREAKER: Check if we can make API call
            _record_api_call("product_finder")

            # Make HTTP POST request with retry logic for rate limits
            url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}&stats=0"

            max_retries = 3
            retry_delay = 5  # Start with 5 seconds
            response = None

            for attempt in range(max_retries):
                response = requests.post(url, json=query_json, timeout=30)
                
                if response.status_code == 200:
                    break  # Success
                elif response.status_code == 429:  # Rate limited
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                        st.warning(f"⚠️ Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error(f"❌ Rate limited after {max_retries} attempts. Saving {len(all_products)} products collected so far.")
                        st.info("💡 **Tip:** Keepa API allows ~100 requests/minute. Your data has been saved - you can continue later.")
                        break  # Give up after max retries, but data collected so far will be saved
                else:
                    # Other error codes - don't retry
                    st.warning(f"Keepa API error on page {page}: {response.status_code} - {response.text[:200]}")
                    break
            
            if not response or response.status_code != 200:
                break  # Exit loop if request failed

            result = response.json()
            asins = result.get("asinList", [])

            if not asins:
                break  # No more products in category
            
            # Debug: Show first page ASINs
            if page == 0:
                with st.expander(f"🔍 Debug: First {min(10, len(asins))} ASINs returned by Keepa query", expanded=False):
                    st.write(f"Total ASINs in this batch: {len(asins)}")
                    st.write(f"First 10 ASINs: {asins[:10]}")

            # Fetch product data with stats using keepa library
            # Process in smaller batches to avoid timeout (keepa library has 10s default timeout)
            products = []
            keepa_batch_size = 20  # Smaller batches to prevent timeout
            
            for i in range(0, len(asins), keepa_batch_size):
                batch_asins = asins[i:i + keepa_batch_size]
                max_retries = 3
                retry_delay = 3
                batch_success = False
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        # CIRCUIT BREAKER: Check before each batch
                        _record_api_call("batch_query")

                        batch_products = api.query(batch_asins, stats=90, rating=True)
                        products.extend(batch_products)
                        batch_success = True
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_error = e
                        error_msg = str(e).lower()
                        if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential: 3s, 6s, 12s
                                st.warning(f"⚠️ Rate limited on batch {i//keepa_batch_size + 1}. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                            else:
                                st.error(f"❌ Rate limited after {max_retries} attempts. Saving {len(products)} products collected so far.")
                                # Break out of both loops - we've hit the rate limit
                                break
                        elif "timeout" in error_msg or "read timeout" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                                st.caption(f"⚠️ Timeout on batch {i//keepa_batch_size + 1}, retrying in {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                            else:
                                st.warning(f"⚠️ Timeout after {max_retries} attempts for batch {i//keepa_batch_size + 1}. Skipping batch.")
                                break  # Give up on this batch
                        else:
                            # Non-timeout error, break immediately
                            st.warning(f"⚠️ Error fetching batch {i//keepa_batch_size + 1}: {str(e)[:50]}")
                            break
                
                # If we hit a rate limit, stop fetching more batches
                if not batch_success and last_error and "rate" in str(last_error).lower():
                    st.warning(f"⚠️ Stopping competitor fetch due to rate limit. Collected {len(all_products)} total products.")
                    break
                
                # Rate limiting: Longer delay between batches to avoid hitting Keepa limits
                # Keepa allows ~100 requests/minute, so we space out batches
                if i + keepa_batch_size < len(asins):
                    time.sleep(1.5)  # Increased from 0.5s to 1.5s

            # Track stats for debugging (we keep all products now, just track data quality)
            missing_price_bsr_count = 0
            
            # Debug: Check what categories the returned products actually belong to
            if page == 0 and products:
                root_categories = {}
                for p in products[:20]:  # Check first 20 products
                    rc = p.get("rootCategory", "Unknown")
                    root_categories[rc] = root_categories.get(rc, 0) + 1
                
                with st.expander("🔍 Debug: Root categories of fetched products", expanded=False):
                    st.write(f"Expected category ID: {leaf_category_id or category_id}")
                    st.write(f"Root categories found in first 20 products:")
                    for rc, count in sorted(root_categories.items(), key=lambda x: -x[1]):
                        match = "✅" if rc == category_id else "❌"
                        st.write(f"  {match} Category {rc}: {count} products")
                    
                    # Show first 3 product titles for context
                    st.write("First 3 product titles:")
                    for p in products[:3]:
                        st.write(f"  - {p.get('title', 'Unknown')[:60]}...")
            
            # Collect products 
            # Since we're using categories_include at the API level, 
            # we trust Keepa's filtering and only do light validation
            all_valid_products = []
            products_filtered_by_category = 0
            
            for product in products:
                asin = product.get("asin")
                
                # Skip ASINs already fetched in brand-first pass
                if asin and asin in brand_asins_fetched:
                    continue
                
                # CATEGORY VALIDATION
                # MVP mode: enforce leaf membership — product.categories must contain leaf_id.
                # Legacy mode: root-only sanity check (previous behavior).
                product_root_category = product.get("rootCategory", 0)
                if mvp_mode and leaf_category_id:
                    product_cats = product.get("categories", []) or []
                    try:
                        cat_ids = {int(c) for c in product_cats if c is not None}
                        is_valid_category = int(leaf_category_id) in cat_ids
                    except (TypeError, ValueError):
                        is_valid_category = False
                    if not is_valid_category:
                        excluded_off_leaf_count += 1
                else:
                    is_valid_category = product_root_category == category_id

                if not is_valid_category:
                    products_filtered_by_category += 1
                    continue  # Skip off-category products
                
                # Calculate revenue proxy (using safe extraction to handle nested lists)
                csv = product.get("csv", [])
                stats = product.get("stats", {})

                # Get price with multi-level fallback:
                # 1. Current Buy Box (csv[18])
                # 2. Current Amazon (csv[0])
                # 3. Current New FBA (csv[10])
                # 4. 30-day avg price from stats (avg30)
                # 5. 90-day avg price from stats (avg90)
                price_cents = _safe_csv_value(csv[18] if csv and len(csv) > 18 else None, 0)
                if price_cents <= 0:
                    price_cents = _safe_csv_value(csv[0] if csv and len(csv) > 0 else None, 0)
                if price_cents <= 0:
                    price_cents = _safe_csv_value(csv[10] if csv and len(csv) > 10 else None, 0)

                # FALLBACK: Use Keepa avg stats if current price unavailable
                if price_cents <= 0 and stats:
                    # Try avg30 for Amazon/New prices
                    avg30 = stats.get("avg30", []) if isinstance(stats.get("avg30"), list) else []
                    avg90 = stats.get("avg90", []) if isinstance(stats.get("avg90"), list) else []

                    # Keepa stats array indices: 0=Amazon, 1=New, 10=NewFBA, 18=BuyBox
                    # Try multiple price types from avg stats
                    for idx in [18, 0, 1, 10]:  # BuyBox, Amazon, New, NewFBA
                        if avg30 and len(avg30) > idx and avg30[idx] and avg30[idx] > 0:
                            price_cents = avg30[idx]
                            break
                        if avg90 and len(avg90) > idx and avg90[idx] and avg90[idx] > 0:
                            price_cents = avg90[idx]
                            break

                price = price_cents / 100.0 if price_cents > 0 else 0

                # Get BSR with fallback to avg stats
                bsr_value = _safe_csv_value(csv[3] if csv and len(csv) > 3 else None, 0)

                # FALLBACK: Use avg30/avg90 BSR from stats
                if bsr_value <= 0 and stats:
                    avg30 = stats.get("avg30", []) if isinstance(stats.get("avg30"), list) else []
                    avg90 = stats.get("avg90", []) if isinstance(stats.get("avg90"), list) else []
                    # BSR is at index 3 in stats arrays
                    if avg30 and len(avg30) > 3 and avg30[3] and avg30[3] > 0:
                        bsr_value = avg30[3]
                    elif avg90 and len(avg90) > 3 and avg90[3] and avg90[3] > 0:
                        bsr_value = avg90[3]

                bsr = bsr_value if bsr_value > 0 else None

                # Calculate monthly units from BSR using power law formula
                # Formula: monthly_units = 145000 * (BSR ^ -0.9)
                # This is calibrated for Grocery velocity (from keepa_client.py)
                monthly_units = 0
                if bsr and bsr > 0:
                    monthly_units = 145000.0 * (bsr ** -0.9)

                # Calculate revenue (will be 0 if price or monthly_units is 0/missing)
                # Keep products even with zero revenue - similar to weekly pipeline
                revenue = monthly_units * price
                
                # Track stats but don't filter (keep all products like weekly pipeline)
                if price == 0 or bsr is None or bsr == 0:
                    missing_price_bsr_count += 1

                # Extract brand from Keepa data (use brand field directly, fallback to title)
                title = product.get("title", "")
                brand = product.get("brand", "")
                if not brand:
                    # Fallback: extract first word from title
                    brand = title.split()[0] if title else "Unknown"

                # Get product image
                image_urls = product.get("imagesCSV", "").split(",") if product.get("imagesCSV") else []
                main_image = f"https://images-na.ssl-images-amazon.com/images/I/{image_urls[0]}" if image_urls else ""
                
                # === EXTRACT COMPETITIVE METRICS FROM KEEPA ===
                # These power the Competitive Intelligence panel in the dashboard
                stats = product.get("stats", {})
                
                # Out of Stock Percentage (90 days) - from stats (use safe extraction)
                oos_90 = _safe_numeric(stats.get("outOfStockPercentage90") if stats else None, 0.0)
                if oos_90 > 1:  # Normalize if > 1 (Keepa sometimes returns as percentage)
                    oos_90 = oos_90 / 100
                
                # Buy Box Stats (30/90 day) - Amazon's % ownership (or top seller's)
                bb_stats_30 = _safe_numeric(stats.get("buyBoxStatsAmazon30") if stats else None, -1)
                bb_stats_90 = _safe_numeric(stats.get("buyBoxStatsAmazon90") if stats else None, -1)
                if bb_stats_30 >= 0:
                    amazon_bb_share = bb_stats_30 / 100.0 if bb_stats_30 > 1 else bb_stats_30
                elif bb_stats_90 >= 0:
                    amazon_bb_share = bb_stats_90 / 100.0 if bb_stats_90 > 1 else bb_stats_90
                else:
                    amazon_bb_share = 0.5  # Default to 50% if no data
                
                # Seller Count (index 11) - using safe extraction
                new_offer_count = _safe_csv_value(csv[11] if csv and len(csv) > 11 else None, 0)
                if new_offer_count <= 0 and stats and "current" in stats:
                    current_stats = stats.get("current", {})
                    if isinstance(current_stats, dict):
                        new_offer_count = _safe_numeric(current_stats.get("COUNT_NEW"), 1)
                new_offer_count = max(1, int(new_offer_count))  # At least 1 seller
                
                # Review Count (index 17 in Keepa Python lib)
                review_count = _safe_csv_value(csv[17] if csv and len(csv) > 17 else None, 0)
                if review_count <= 0 and stats and "current" in stats:
                    current_stats = stats.get("current", {})
                    if isinstance(current_stats, dict):
                        review_count = _safe_numeric(current_stats.get("COUNT_REVIEWS"), 0)

                # Rating (index 16 in Keepa Python lib) - Keepa stores as 10x
                rating_raw = _safe_csv_value(csv[16] if csv and len(csv) > 16 else None, 0)
                if rating_raw <= 0 and stats and "current" in stats:
                    current_stats = stats.get("current", {})
                    if isinstance(current_stats, dict):
                        rating_raw = _safe_numeric(current_stats.get("RATING"), 0)
                rating = rating_raw / 10.0 if rating_raw > 0 else 0.0

                # Parent ASIN - critical for variation deduplication
                # When multiple products share a parent, they share BSR and revenue should not be double-counted
                parent_asin = product.get("parentAsin", "")
                
                # === NEW CRITICAL METRICS (2026-01-21) ===
                # Amazon's actual monthly sold estimate
                monthly_sold = _safe_numeric(product.get("monthlySold"), 0)
                
                # Pack size for per-unit normalization
                number_of_items = _safe_numeric(product.get("numberOfItems"), 1)
                number_of_items = max(1, int(number_of_items))
                
                # Buy Box ownership flags
                buybox_is_amazon = product.get("buyBoxIsAmazon", None)
                buybox_is_fba = product.get("buyBoxIsFBA", None)
                buybox_is_backorder = product.get("buyBoxIsBackorder", False) or False
                
                # Seller IDs for true seller count
                seller_ids = product.get("sellerIds", []) or []
                seller_count = len(seller_ids)
                has_amazon_seller = "ATVPDKIKX0DER" in seller_ids if seller_ids else False
                
                # OOS counts (more actionable than %)
                oos_count_30 = _safe_numeric(stats.get("outOfStockCountAmazon30") if stats else None, 0)
                oos_count_90 = _safe_numeric(stats.get("outOfStockCountAmazon90") if stats else None, 0)
                
                # Pre-calculated velocity from Keepa
                delta_pct_30 = stats.get("deltaPercent30", []) if stats else []
                delta_pct_90 = stats.get("deltaPercent90", []) if stats else []
                velocity_30d = delta_pct_30[3] if delta_pct_30 and len(delta_pct_30) > 3 else None
                velocity_90d = delta_pct_90[3] if delta_pct_90 and len(delta_pct_90) > 3 else None
                
                # Buy Box seller count and top seller share
                bb_seller_count_30 = _safe_numeric(stats.get("buyBoxStatsSellerCount30") if stats else None, 0)
                bb_top_seller_30 = _safe_numeric(stats.get("buyBoxStatsTopSeller30") if stats else None, 0)
                if bb_top_seller_30 > 1:
                    bb_top_seller_30 = bb_top_seller_30 / 100.0
                
                # Subscribe & Save
                is_sns = product.get("isSNS", False) or False
                
                # Use monthlySold if available, otherwise keep BSR-based estimate
                units_source = "bsr_formula"
                if monthly_sold > 0:
                    monthly_units = monthly_sold
                    revenue = monthly_units * price
                    units_source = "amazon_monthly_sold"

                product_data = {
                    "asin": product.get("asin"),
                    "title": title,
                    "brand": brand,
                    "price": price,
                    "monthly_units": monthly_units,
                    "revenue_proxy": revenue,
                    "bsr": bsr,
                    "main_image": main_image,
                    "parent_asin": parent_asin,
                    # Competitive metrics
                    "outOfStockPercentage90": oos_90,
                    "amazon_bb_share": amazon_bb_share,
                    "new_offer_count": new_offer_count,
                    "review_count": review_count,
                    "rating": rating,
                    # NEW CRITICAL METRICS (2026-01-21)
                    "monthly_sold": monthly_sold,
                    "number_of_items": number_of_items,
                    "price_per_unit": price / number_of_items if number_of_items > 0 else price,
                    "buybox_is_amazon": buybox_is_amazon,
                    "buybox_is_fba": buybox_is_fba,
                    "buybox_is_backorder": buybox_is_backorder,
                    "seller_count": seller_count,
                    "has_amazon_seller": has_amazon_seller,
                    "oos_count_amazon_30": int(oos_count_30),
                    "oos_count_amazon_90": int(oos_count_90),
                    "velocity_30d": velocity_30d,
                    "velocity_90d": velocity_90d,
                    "bb_seller_count_30": int(bb_seller_count_30),
                    "bb_top_seller_30": bb_top_seller_30,
                    "is_sns": is_sns,
                    "units_source": units_source,
                }

                # Track all valid products
                all_valid_products.append(product_data)
            
            # Show data quality stats on first page
            if page == 0 and len(products) > 0:
                total_fetched = len(products)
                products_with_revenue = sum(1 for p in all_valid_products if p.get("revenue_proxy", 0) > 0)
                
                if products_filtered_by_category > 0:
                    st.warning(
                        f"⚠️ **Category Mismatch**: Keepa returned {products_filtered_by_category}/{total_fetched} products "
                        f"from wrong categories. These were filtered out."
                    )
                
                st.caption(
                    f"📊 Data quality: {products_with_revenue}/{len(all_valid_products)} products have sales data "
                    f"({missing_price_bsr_count} missing price/BSR, revenue set to $0)"
                )
            
            # Add all valid products (will stop when we have 100 total)
            all_products.extend(all_valid_products)
            
            # Sort by BSR (ascending = best sellers first) to ensure we're processing top products
            # This ensures we capture the highest revenue products first
            all_products_sorted_by_bsr = sorted(all_products, key=lambda x: x["bsr"] if x.get("bsr") and x["bsr"] > 0 else float('inf'))
            
            # Calculate revenue before this batch (for diminishing returns check)
            revenue_before_batch = cumulative_revenue
            
            # Calculate cumulative revenue from products sorted by BSR (best sellers first)
            cumulative_revenue = sum(p["revenue_proxy"] for p in all_products_sorted_by_bsr)
            
            # Revenue added by this batch
            batch_revenue = cumulative_revenue - revenue_before_batch
            
            # Early termination: If revenue is $0 after first batch, something is wrong with data
            if page == 0 and len(all_products) >= batch_size and cumulative_revenue == 0:
                st.warning(
                    f"⚠️ **Data Issue Detected**: Fetched {len(all_products)} products but all have $0 revenue.\n\n"
                    f"This usually means:\n"
                    f"- Products have no sales data (BSR = 0 or invalid) AND no valid price\n"
                    f"- Price data is missing\n"
                    f"- Category may have many inactive/discontinued products\n\n"
                    f"Stopping fetch to prevent wasting API tokens. Please try a different category or search term."
                )
                break

            # Stop condition: arena cap OR competitor floor (mvp_mode), or total cap (legacy)
            _valid_comp_count = len(all_products) - _initial_brand_count
            _hit_arena_cap = len(all_products) >= (arena_size if mvp_mode else 100)
            _hit_comp_floor = mvp_mode and _valid_comp_count >= _min_competitors
            if _hit_arena_cap or _hit_comp_floor:
                st.success(
                    f"✅ **{len(all_products)} ASINs selected** "
                    f"({_initial_brand_count} brand + {_valid_comp_count} competitors) | "
                    f"Raw revenue: ${cumulative_revenue:,.0f}"
                )
                break

            page += 1
            st.caption(
                f"Page {page}: Fetched {len(all_products)} valid products | "
                f"Batch revenue: ${batch_revenue:,.0f} | "
                f"Cumulative (raw): ${cumulative_revenue:,.0f}"
            )
            
            # Rate limit protection between pages
            # Keepa allows ~100 requests/minute, so we add a longer delay between pages
            # to avoid hitting rate limits when fetching large batches
            time.sleep(2)  # Increased from 1s to 2s for better rate limit protection

        except Exception as e:
            st.warning(f"Error fetching page {page}: {str(e)}")
            break
    
    # Check if we hit max pages without enough products
    if len(all_products) < target_valid_products and page >= max_pages:
        st.warning(
            f"⚠️ **Reached page limit ({max_pages} pages)** with only {len(all_products)}/{target_valid_products} valid products.\n\n"
            f"This may indicate:\n"
            f"- Keepa's category filter is not working correctly\n"
            f"- The category has fewer products than expected\n"
            f"- Most products in the category are in subcategories\n\n"
            f"Continuing with {len(all_products)} products."
        )

    # ========== ENSURE SEED ASIN IS INCLUDED ==========
    # CRITICAL: The seed ASIN must always be in the results for brand identification
    if seed_asin:
        # Check if seed ASIN is already in the results
        seed_asin_found = any(p.get("asin") == seed_asin for p in all_products)

        if not seed_asin_found:
            st.warning(f"⚠️ Seed ASIN {seed_asin} not found in category results - fetching explicitly...")
            try:
                # Fetch the seed product explicitly
                seed_products = api.query([seed_asin], stats=90, rating=True)
                if seed_products and len(seed_products) > 0:
                    seed_product_data = seed_products[0]

                    # Extract data using safe helper (handles nested lists)
                    csv = seed_product_data.get("csv", [])
                    price_cents = _safe_csv_value(csv[18] if csv and len(csv) > 18 else None, 0)
                    price = price_cents / 100.0 if price_cents > 0 else 0

                    bsr_value = _safe_csv_value(csv[3] if csv and len(csv) > 3 else None, 0)
                    bsr = bsr_value if bsr_value > 0 else None

                    monthly_units = 0
                    if bsr and bsr > 0:
                        monthly_units = 145000.0 * (bsr ** -0.9)

                    revenue = monthly_units * price

                    title = seed_product_data.get("title", "")
                    brand = seed_product_data.get("brand", "")
                    if not brand:
                        brand = title.split()[0] if title else "Unknown"

                    image_urls = seed_product_data.get("imagesCSV", "").split(",") if seed_product_data.get("imagesCSV") else []
                    main_image = f"https://images-na.ssl-images-amazon.com/images/I/{image_urls[0]}" if image_urls else ""

                    # Add to beginning of list (high priority)
                    all_products.insert(0, {
                        "asin": seed_asin,
                        "title": title,
                        "brand": brand,
                        "price": price,
                        "monthly_units": monthly_units,
                        "revenue_proxy": revenue,
                        "bsr": bsr,
                        "main_image": main_image
                    })
                    st.success(f"✅ Added seed ASIN {seed_asin} to results")
            except Exception as e:
                st.error(f"❌ Failed to fetch seed ASIN {seed_asin}: {str(e)}")

    # Remove duplicates (by ASIN) and internal tracking fields
    seen_asins = set()
    unique_products = []
    for p in all_products:
        p.pop("_matched_level", None)
        asin = p.get("asin")
        if asin and asin not in seen_asins:
            seen_asins.add(asin)
            unique_products.append(p)
    
    if not unique_products:
        return pd.DataFrame(), {}
    
    # Extract just the ASINs for historical fetch
    asin_list = [p.get("asin") for p in unique_products if p.get("asin")]
    
    # ========== FETCH 90-DAY HISTORICAL DATA ==========
    # This is the key change: instead of using current price/BSR estimates,
    # we fetch 90 days of actual historical data and build weekly tables
    st.caption(f"📊 Fetching 90 days of historical data for {len(asin_list)} ASINs...")
    
    try:
        # Fetch full 90-day history for all discovered ASINs
        historical_products = fetch_90day_history(asin_list, domain=1, days=90)
        
        if not historical_products:
            st.warning("⚠️ Could not fetch historical data - falling back to current estimates")
            # Fallback to old method if historical fetch fails
            WEEKS_PER_MONTH = 4.33
            df = pd.DataFrame(unique_products)
            df = _scalarize_df_columns(df)  # Fix: Scalarize before numeric operations
            df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
            df["weekly_units"] = df["monthly_units"] / WEEKS_PER_MONTH
            df["weekly_revenue"] = df["weekly_units"] * df["price"]
            df["monthly_revenue"] = df["weekly_revenue"] * WEEKS_PER_MONTH
            df["revenue_proxy"] = df["weekly_revenue"]  # Legacy alias
            df["data_weeks"] = 4  # Fallback: assume 4 weeks for AI confidence
            # Apply variation deduplication
            df = _apply_variation_adjustment_to_df(df, "weekly_revenue")
            adjusted_weekly_revenue = df["weekly_revenue_adjusted"].sum()
            adjusted_monthly_revenue = df["monthly_revenue_adjusted"].sum()
            df = df.sort_values(["bsr", "weekly_revenue"], ascending=[True, False]).reset_index(drop=True)
            brand_product_count = len(brand_products) if target_brand else 0
            market_stats = {
                "total_products": len(df),
                "total_category_revenue": adjusted_weekly_revenue,
                "weekly_revenue": adjusted_weekly_revenue,
                "monthly_revenue": adjusted_monthly_revenue,
                "category_id": category_id,
                "effective_category_id": effective_category_id,
                "effective_category_path": effective_category_path,
                "use_categories_include": use_categories_include,
                "validated_products": len(df),
                "validated_revenue": adjusted_weekly_revenue,
                "df_weekly": pd.DataFrame(),
                "target_brand": target_brand,
                "brand_product_count": brand_product_count,
                "competitor_count": len(df) - brand_product_count
            }
            return df, market_stats
        
        st.caption(f"✅ Received historical data for {len(historical_products)} products")
        
        # Build weekly table using the same methodology as the main dashboard
        # This creates: week_start, asin, title, sales_rank_filled, filled_price, weekly_sales_filled, etc.
        # Pass window_start to limit to last 90 days (not 2+ years of history)
        window_start_90d = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        df_weekly = build_keepa_weekly_table(historical_products, window_start=window_start_90d)
        
        if df_weekly.empty:
            st.warning("⚠️ Could not build weekly table - falling back to current estimates")
            WEEKS_PER_MONTH = 4.33
            df = pd.DataFrame(unique_products)
            df = _scalarize_df_columns(df)  # Fix: Scalarize before numeric operations
            df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
            df["weekly_units"] = df["monthly_units"] / WEEKS_PER_MONTH
            df["weekly_revenue"] = df["weekly_units"] * df["price"]
            df["monthly_revenue"] = df["weekly_revenue"] * WEEKS_PER_MONTH
            df["revenue_proxy"] = df["weekly_revenue"]  # Legacy alias
            df["data_weeks"] = 4  # Fallback: assume 4 weeks for AI confidence
            # Apply variation deduplication
            df = _apply_variation_adjustment_to_df(df, "weekly_revenue")
            adjusted_weekly_revenue = df["weekly_revenue_adjusted"].sum()
            adjusted_monthly_revenue = df["monthly_revenue_adjusted"].sum()
            df = df.sort_values(["bsr", "weekly_revenue"], ascending=[True, False]).reset_index(drop=True)
            brand_product_count = len(brand_products) if target_brand else 0
            market_stats = {
                "total_products": len(df),
                "total_category_revenue": adjusted_weekly_revenue,
                "weekly_revenue": adjusted_weekly_revenue,
                "monthly_revenue": adjusted_monthly_revenue,
                "category_id": category_id,
                "effective_category_id": effective_category_id,
                "effective_category_path": effective_category_path,
                "use_categories_include": use_categories_include,
                "validated_products": len(df),
                "validated_revenue": adjusted_weekly_revenue,
                "df_weekly": pd.DataFrame(),
                "target_brand": target_brand,
                "brand_product_count": brand_product_count,
                "competitor_count": len(df) - brand_product_count
            }
            return df, market_stats
        
        st.caption(f"📈 Built weekly table with {len(df_weekly)} rows across {df_weekly['asin'].nunique()} ASINs")

        # Initialize seed debug tracking
        if seed_asin:
            st.session_state["seed_debug_trail"] = []
            st.session_state["seed_debug_trail"].append(f"Starting seed ASIN tracking: {seed_asin}")
            st.session_state["seed_debug_trail"].append(f"Weekly table has {df_weekly['asin'].nunique()} unique ASINs")

        # CRITICAL FIX: If seed ASIN was filtered out (no historical data), add it back
        # This happens with new products or products with sparse data
        if seed_asin and seed_asin not in df_weekly['asin'].values:
            st.warning(f"⚠️ Seed ASIN {seed_asin} missing from weekly table (no historical data) - adding with current values")

            # Find the seed product in unique_products (it was explicitly added earlier)
            seed_product = next((p for p in unique_products if p.get("asin") == seed_asin), None)

            if seed_product:
                # Create a seed row matching df_weekly schema
                current_week = pd.Timestamp.now().normalize() - pd.Timedelta(days=pd.Timestamp.now().weekday())

                # Start with all columns from df_weekly, filled with NaN
                seed_row = {col: np.nan for col in df_weekly.columns}

                # Override with known values
                seed_row.update({
                    "week_start": current_week,
                    "asin": seed_asin,
                    "title": seed_product.get("title", ""),
                    "brand": seed_product.get("brand", ""),
                    "filled_price": seed_product.get("price", 0),
                    "sales_rank_filled": seed_product.get("bsr", 50000),
                    "weekly_sales_filled": seed_product.get("revenue_proxy", 0) / 4.33,  # Monthly → weekly
                    "estimated_units": seed_product.get("monthly_units", 0) / 4.33,  # Monthly → weekly
                    "main_image": seed_product.get("main_image", "")
                })

                # Add the row to df_weekly
                df_weekly = pd.concat([df_weekly, pd.DataFrame([seed_row])], ignore_index=True)
                st.success(f"✅ Added seed ASIN {seed_asin} to weekly table")
                st.session_state["seed_debug_trail"].append(f"✅ Added seed ASIN to df_weekly (now {len(df_weekly)} rows, {df_weekly['asin'].nunique()} unique ASINs)")

        # ========== AGGREGATE WEEKLY DATA INTO SNAPSHOT ==========
        # For each ASIN, calculate:
        # - avg_price: Average price over 3 months
        # - avg_bsr: Average BSR over 3 months
        # - total_weekly_sales: Sum of weekly_sales_filled over 3 months
        # - monthly_revenue: total_weekly_sales / 3 (for monthly estimate)
        
        # Count actual weeks of data per ASIN for accurate monthly calculation
        weeks_per_asin = df_weekly.groupby("asin")["week_start"].count().reset_index()
        weeks_per_asin.columns = ["asin", "weeks_count"]
        
        # Build aggregation dict dynamically based on available columns
        agg_dict = {
            "filled_price": "mean",                    # Average price over available weeks
            "sales_rank_filled": "mean",               # Average BSR over available weeks
            "weekly_sales_filled": "sum",              # Total sales over available weeks
            "estimated_units": "sum",                  # Total units over available weeks
            "title": "first",                          # Keep title
            "brand": "first",                          # Keep brand
            "main_image": "first",                     # Keep image
        }
        
        # Add critical AI metrics if available (needed for intelligent recommendations)
        if "amazon_bb_share" in df_weekly.columns:
            agg_dict["amazon_bb_share"] = "mean"       # Average Buy Box share
        if "review_count" in df_weekly.columns:
            agg_dict["review_count"] = "last"          # Most recent review count
        if "rating" in df_weekly.columns:
            agg_dict["rating"] = "mean"                # Average rating
        if "new_offer_count" in df_weekly.columns:
            agg_dict["new_offer_count"] = "last"       # Most recent seller count
        if "buy_box_switches" in df_weekly.columns:
            agg_dict["buy_box_switches"] = "sum"       # Total Buy Box switches
        if "parent_asin" in df_weekly.columns:
            agg_dict["parent_asin"] = "first"          # Keep parent ASIN
        
        # Only include columns that exist in the DataFrame
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_weekly.columns}
        
        asin_summary = df_weekly.groupby("asin").agg(agg_dict).reset_index()

        # Debug: Check if seed ASIN survived aggregation
        if seed_asin:
            if seed_asin in asin_summary['asin'].values:
                msg = f"✅ Checkpoint 1: Seed ASIN in asin_summary ({len(asin_summary)} products)"
                st.caption(msg)
                st.session_state["seed_debug_trail"].append(msg)
            else:
                msg = f"❌ LOST at aggregation: {len(df_weekly)} rows → {len(asin_summary)} products (seed missing)"
                st.error(msg)
                st.session_state["seed_debug_trail"].append(msg)
                st.session_state["seed_debug_trail"].append(f"   df_weekly has seed: {seed_asin in df_weekly['asin'].values}")
                st.session_state["seed_debug_trail"].append(f"   asin_summary first 10: {asin_summary['asin'].tolist()[:10]}")

        # Merge week counts
        asin_summary = asin_summary.merge(weeks_per_asin, on="asin", how="left")
        
        # Rename columns for snapshot
        asin_summary = asin_summary.rename(columns={
            "filled_price": "price",
            "sales_rank_filled": "bsr",
            "weekly_sales_filled": "total_90d_revenue",
            "estimated_units": "total_90d_units"
        })
        
        # === STANDARDIZED REVENUE CALCULATION (2026-01-30) ===
        # Base unit: WEEKLY (weekly_revenue is the source of truth)
        # Monthly is always calculated as weekly * 4.33
        WEEKS_PER_MONTH = 4.33

        # Calculate averages using actual weeks of data
        asin_summary["weekly_revenue"] = asin_summary["total_90d_revenue"] / asin_summary["weeks_count"].clip(lower=1)
        asin_summary["weekly_units"] = asin_summary["total_90d_units"] / asin_summary["weeks_count"].clip(lower=1)

        # Derived monthly values
        asin_summary["monthly_revenue"] = asin_summary["weekly_revenue"] * WEEKS_PER_MONTH
        asin_summary["monthly_units"] = asin_summary["weekly_units"] * WEEKS_PER_MONTH

        # Legacy aliases (to be deprecated)
        asin_summary["avg_weekly_revenue"] = asin_summary["weekly_revenue"]
        asin_summary["revenue_proxy"] = asin_summary["weekly_revenue"]  # Now weekly, not monthly!

        # Debug: Show revenue calculation stats
        avg_weeks = asin_summary["weeks_count"].mean()
        total_weekly_rev = asin_summary["weekly_revenue"].sum()
        total_monthly_rev = asin_summary["monthly_revenue"].sum()
        avg_product_weekly = asin_summary["weekly_revenue"].mean()
        st.caption(f"📊 Revenue calc: {len(asin_summary)} products, avg {avg_weeks:.1f} weeks data, ${total_weekly_rev:,.0f}/wk (${total_monthly_rev:,.0f}/mo), avg ${avg_product_weekly:,.0f}/wk per product")
        
        # FIX: Convert any list values to scalars before comparison
        # Some metrics (like sellerIds) might be aggregated as lists
        def scalarize(value):
            """Convert list/numpy array to scalar, return value as-is if already scalar."""
            import numpy as np
            if isinstance(value, (list, tuple)):
                # Take first element if list, or return 0 if empty
                return value[0] if len(value) > 0 else 0
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                # Handle numpy arrays and other iterables
                try:
                    return value[0] if len(value) > 0 else 0
                except (TypeError, IndexError):
                    return 0
            return value
        
        # Apply scalarization to price and bsr columns if they contain lists
        if "price" in asin_summary.columns:
            asin_summary["price"] = asin_summary["price"].apply(scalarize)
        if "bsr" in asin_summary.columns:
            asin_summary["bsr"] = asin_summary["bsr"].apply(scalarize)
        
        # Fill any NaN values with median from products that have data
        valid_prices = asin_summary[asin_summary["price"] > 0]["price"]
        valid_bsrs = asin_summary[(asin_summary["bsr"].notna()) & (asin_summary["bsr"] > 0)]["bsr"]
        
        median_price = valid_prices.median() if len(valid_prices) > 0 else 15.0
        median_bsr = valid_bsrs.median() if len(valid_bsrs) > 0 else 50000
        
        missing_price_count = asin_summary["price"].isna().sum() + (asin_summary["price"] == 0).sum()
        missing_bsr_count = asin_summary["bsr"].isna().sum() + (asin_summary["bsr"] == 0).sum()
        
        asin_summary["price"] = asin_summary["price"].fillna(median_price)
        asin_summary.loc[asin_summary["price"] == 0, "price"] = median_price
        asin_summary["bsr"] = asin_summary["bsr"].fillna(median_bsr)
        asin_summary.loc[asin_summary["bsr"] == 0, "bsr"] = median_bsr
        
        # Recalculate revenue for products that were missing data
        # Use BSR formula for products without weekly sales data
        missing_revenue_mask = asin_summary["weekly_revenue"].isna() | (asin_summary["weekly_revenue"] == 0)
        if missing_revenue_mask.any():
            # BSR formula gives monthly units
            asin_summary.loc[missing_revenue_mask, "monthly_units"] = (
                145000.0 * (asin_summary.loc[missing_revenue_mask, "bsr"].clip(lower=1) ** -0.9)
            )
            asin_summary.loc[missing_revenue_mask, "weekly_units"] = (
                asin_summary.loc[missing_revenue_mask, "monthly_units"] / WEEKS_PER_MONTH
            )
            # Weekly revenue = weekly units * price
            asin_summary.loc[missing_revenue_mask, "weekly_revenue"] = (
                asin_summary.loc[missing_revenue_mask, "weekly_units"] *
                asin_summary.loc[missing_revenue_mask, "price"]
            )
            asin_summary.loc[missing_revenue_mask, "monthly_revenue"] = (
                asin_summary.loc[missing_revenue_mask, "weekly_revenue"] * WEEKS_PER_MONTH
            )
            # Update legacy alias
            asin_summary.loc[missing_revenue_mask, "revenue_proxy"] = (
                asin_summary.loc[missing_revenue_mask, "weekly_revenue"]
            )

        if missing_price_count > 0 or missing_bsr_count > 0:
            st.caption(
                f"📊 Filled {int(missing_price_count)} missing prices (→ ${median_price:.2f}) "
                f"and {int(missing_bsr_count)} missing BSRs (→ {int(median_bsr):,})"
            )

        # Sort by BSR (best sellers first), then by weekly revenue
        df = asin_summary.sort_values(["bsr", "weekly_revenue"], ascending=[True, False]).reset_index(drop=True)

        # CRITICAL: Rename weeks_count to data_weeks for AI confidence calculation
        # Without this, all products get 40% confidence (assumes 0 weeks of data)
        if 'weeks_count' in df.columns:
            df['data_weeks'] = df['weeks_count']

        # Apply variation deduplication to prevent revenue overcounting
        df = _apply_variation_adjustment_to_df(df, "weekly_revenue")
        adjusted_weekly_revenue = df["weekly_revenue_adjusted"].sum()
        adjusted_monthly_revenue = df["monthly_revenue_adjusted"].sum()

        # Debug: Check if seed ASIN survived variation adjustment
        if seed_asin:
            if seed_asin in df['asin'].values:
                msg = f"✅ Checkpoint 2: Seed ASIN in final df ({len(df)} products)"
                st.caption(msg)
                st.session_state["seed_debug_trail"].append(msg)
            else:
                msg = f"❌ LOST during variation adjustment"
                st.error(msg)
                st.session_state["seed_debug_trail"].append(msg)

        # Calculate market stats
        brand_product_count = len(brand_products) if target_brand else 0
        competitor_count = len(df) - brand_product_count

        market_stats = {
            "total_products": len(df),
            "total_category_revenue": adjusted_weekly_revenue,  # Weekly (base unit)
            "weekly_revenue": adjusted_weekly_revenue,  # Explicit weekly
            "monthly_revenue": adjusted_monthly_revenue,  # Explicit monthly
            "category_id": category_id,
            "effective_category_id": effective_category_id,
            "effective_category_path": effective_category_path,
            "use_categories_include": use_categories_include,
            "validated_products": len(df),
            "validated_revenue": adjusted_weekly_revenue,  # Weekly (base unit)
            "df_weekly": df_weekly,
            "time_period": "90 days (weekly aggregated)",
            "target_brand": target_brand,
            "brand_product_count": brand_product_count,
            "competitor_count": competitor_count,
            # MVP observability
            "leaf_category_id": leaf_category_id,
            "mvp_search_scope": _mvp_search_scope,
            "excluded_off_leaf_count": excluded_off_leaf_count if mvp_mode else None,
            "mvp_mode": mvp_mode,
            # Arena sizing (new — populated in mvp_mode)
            "brand_selected_count": brand_product_count,
            "competitor_selected_count": competitor_count,
            "selected_asins_count": len(df),
            "arena_size_target": arena_size if mvp_mode else None,
            "coverage_note": "estimated within scanned universe" if mvp_mode else None,
        }

        if target_brand:
            st.success(
                f"✅ **Market Snapshot Complete**\n\n"
                f"**{target_brand}**: {brand_product_count} products | "
                f"**Competitors**: {competitor_count} products\n\n"
                f"**Est. Weekly Revenue: ${adjusted_weekly_revenue:,.0f}/wk** (${adjusted_monthly_revenue:,.0f}/mo)\n"
                f"_(Projected from avg weekly sales over last 90 days)_"
            )
        else:
            st.success(
                f"✅ **Market Snapshot Complete**\n\n"
                f"Products: **{len(df)}** | Est. Weekly Revenue: **${adjusted_weekly_revenue:,.0f}/wk** (${adjusted_monthly_revenue:,.0f}/mo)\n"
                f"_(Projected from avg weekly sales over last 90 days)_"
            )

        # FINAL CHECK: Verify seed ASIN is in results
        if seed_asin and seed_asin not in df['asin'].values:
            st.error(f"❌ **Critical Error**: Seed ASIN {seed_asin} not found in market data.")
            st.info("This usually means the product has no historical sales data or was filtered during processing.")

        return df, market_stats
        
    except Exception as e:
        st.warning(f"⚠️ Error fetching historical data: {str(e)} - falling back to current estimates")
        # Fallback to old method
        df = pd.DataFrame(unique_products)
        if df.empty:
            return df, {}
        WEEKS_PER_MONTH = 4.33
        df = _scalarize_df_columns(df)  # Fix: Scalarize before numeric operations
        df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
        df["weekly_units"] = df["monthly_units"] / WEEKS_PER_MONTH
        df["weekly_revenue"] = df["weekly_units"] * df["price"]
        df["monthly_revenue"] = df["weekly_revenue"] * WEEKS_PER_MONTH
        df["revenue_proxy"] = df["weekly_revenue"]  # Legacy alias
        df["data_weeks"] = 4  # Fallback: assume 4 weeks for AI confidence
        # Apply variation deduplication
        df = _apply_variation_adjustment_to_df(df, "weekly_revenue")
        adjusted_weekly_revenue = df["weekly_revenue_adjusted"].sum()
        adjusted_monthly_revenue = df["monthly_revenue_adjusted"].sum()
        df = df.sort_values(["bsr", "weekly_revenue"], ascending=[True, False]).reset_index(drop=True)

        brand_product_count = len(brand_products) if target_brand else 0
        competitor_count = len(df) - brand_product_count

        market_stats = {
            "total_products": len(df),
            "total_category_revenue": adjusted_weekly_revenue,
            "weekly_revenue": adjusted_weekly_revenue,
            "monthly_revenue": adjusted_monthly_revenue,
            "category_id": category_id,
            "effective_category_id": effective_category_id,
            "effective_category_path": effective_category_path,
            "use_categories_include": use_categories_include,
            "validated_products": len(df),
            "validated_revenue": adjusted_weekly_revenue,
            "df_weekly": pd.DataFrame(),
            "target_brand": target_brand,
            "brand_product_count": brand_product_count,
            "competitor_count": competitor_count
        }
        return df, market_stats


def llm_validate_competitive_set(
    df: pd.DataFrame,
    seed_product_title: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Use LLM to filter out non-competitive products (accessories, bundles, etc.)

    ENHANCEMENT 2.2: Now supports LLM response caching to reduce API costs.

    Args:
        df: DataFrame of products from category
        seed_product_title: Title of seed product for context
        use_cache: Whether to use LLM response cache (default True)

    Returns:
        Filtered DataFrame with only relevant competitors
    """
    client = get_openai_client()

    if not client or df.empty:
        return df  # Skip validation if no LLM

    # Skip LLM validation if we have very few products (< 10)
    # Category filtering already ensures relevance, and LLM might be too strict with small sets
    if len(df) < 10:
        st.info(f"ℹ️ Skipping LLM validation for {len(df)} products (category filtering is sufficient)")
        return df

    # Take top 100 by revenue for validation (LLM token limits)
    top_products = df.head(100)

    # Build prompt with titles
    titles_list = "\n".join([f"{i+1}. {row['title']}" for i, row in top_products.iterrows()])

    prompt = f"""You are validating a competitive set for market analysis.

Seed Product: "{seed_product_title}"

Below are 100 products from the same Amazon category. Your task:
1. Identify which products are DIRECT COMPETITORS to the seed product
2. REMOVE products that are:
   - Accessories (e.g., sprayers, holders, dispensers)
   - Bundles or multi-packs (unless seed is also a bundle)
   - Unrelated items miscategorized
   - Replacement parts

Return a JSON array of product numbers (1-100) that are VALID competitors:
{{"valid_products": [1, 2, 3, ...]}}

Products:
{titles_list}
"""

    try:
        # ENHANCEMENT 2.2: Check cache first
        content = None
        if use_cache:
            content = _check_llm_cache(prompt, model="gpt-4o-mini", max_age_hours=24)

        if content is None:
            # Cache miss - call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a market intelligence expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            # Get response content
            content = response.choices[0].message.content.strip()

            # Cache the response for future use
            if use_cache:
                _store_llm_in_cache(prompt, content, model="gpt-4o-mini")

        # Remove markdown code blocks if present (```json ... ```)
        if content.startswith("```"):
            # Extract content between code fences
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        result = json.loads(content)
        valid_indices = [i - 1 for i in result.get("valid_products", [])]  # Convert to 0-indexed

        # Filter DataFrame
        df_validated = top_products.iloc[valid_indices].copy()

        # Add back any products from the tail that weren't validated (they're likely valid too)
        if len(df) > 100:
            df_validated = pd.concat([df_validated, df.iloc[100:]], ignore_index=True)

        st.success(f"🧹 LLM validated {len(df_validated)}/{len(df)} products as competitive")

        return df_validated

    except Exception as e:
        st.warning(f"⚠️ LLM validation skipped: {str(e)}. Using all products.")
        return df


def calculate_brand_market_share(
    df_market: pd.DataFrame,
    brand_name: str
) -> Dict:
    """
    Calculate market share for a specific brand.

    Args:
        df_market: Validated market DataFrame
        brand_name: Brand to calculate share for

    Returns:
        Dict with market share metrics
    """
    total_revenue = df_market["revenue_proxy"].sum()
    brand_df = df_market[df_market["title"].str.contains(brand_name, case=False, na=False)]
    brand_revenue = brand_df["revenue_proxy"].sum()

    if total_revenue == 0:
        market_share = 0
    else:
        market_share = (brand_revenue / total_revenue) * 100

    return {
        "brand": brand_name,
        "brand_revenue": brand_revenue,
        "total_market_revenue": total_revenue,
        "market_share_pct": market_share,
        "brand_product_count": len(brand_df),
        "total_product_count": len(df_market)
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_detailed_weekly_data(asins: tuple, days: int = 90) -> pd.DataFrame:
    """
    Fetch 3 months of detailed weekly data for discovered ASINs.
    
    This produces the same format as the Current Dashboard uses,
    enabling full analysis through the engine.
    
    Args:
        asins: Tuple of ASINs from Phase 2 discovery (tuple for caching)
        days: Number of days of history (default: 90 = 3 months)
    
    Returns:
        DataFrame with weekly data in the same format as keepa_weekly_rows
    """
    from scrapers.keepa_client import build_keepa_weekly_table
    from src.backfill import fetch_90day_history
    
    # Convert tuple to list for fetch_90day_history
    asin_list = list(asins)
    
    st.caption(f"📊 Fetching {days} days of detailed data for {len(asin_list)} ASINs...")
    
    # Fetch historical data using the shared backfill function
    try:
        all_products = fetch_90day_history(asin_list, domain=1, days=days)
    except ValueError as e:
        # KEEPA_API_KEY not found
        st.warning(f"⚠️ {str(e)}")
        return pd.DataFrame()
    
    if not all_products:
        st.warning("⚠️ No detailed data fetched")
        return pd.DataFrame()
    
    # Build weekly table using the same function as the scraper
    # Pass window_start to limit to last 90 days
    window_start_90d = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    df_weekly = build_keepa_weekly_table(all_products, window_start=window_start_90d)
    
    if df_weekly.empty:
        st.warning("⚠️ Could not build weekly table from Keepa data")
        return pd.DataFrame()
    
    # Mark all products as "tracked" (equivalent to is_starbucks for user's market)
    df_weekly["is_starbucks"] = 1  # All discovered products are "tracked"
    
    # Ensure required columns exist
    required_cols = [
        "week_start", "asin", "title", "weekly_sales_filled", 
        "sales_rank_filled", "filled_price", "estimated_units"
    ]
    
    for col in required_cols:
        if col not in df_weekly.columns:
            if col == "title":
                df_weekly[col] = df_weekly.get("asin", "Unknown")
            else:
                df_weekly[col] = 0
    
    st.caption(f"✅ Fetched {len(df_weekly)} weekly data points for {df_weekly['asin'].nunique()} ASINs")

    return df_weekly


# ========================================
# INTELLIGENCE PIPELINE INTEGRATION
# ========================================

def generate_strategic_intelligence(
    df_market_snapshot: pd.DataFrame,
    df_weekly: pd.DataFrame,
    portfolio_asins: List[str],
    category_context: Dict[str, Any],
    enable_network_accumulation: bool = True
) -> List:
    """
    Generate strategic intelligence for portfolio ASINs using the new unified pipeline.

    This function should be called AFTER phase2_category_market_mapping() completes.

    Args:
        df_market_snapshot: Market snapshot from Phase 2 (100 ASINs)
        df_weekly: Weekly historical data from Phase 2
        portfolio_asins: List of user's tracked ASINs to analyze
        category_context: Dict with category_id, category_name, category_path
        enable_network_accumulation: Whether to accumulate data for network effect (default True)

    Returns:
        List of UnifiedIntelligence objects with insights
    """
    from src.intelligence_pipeline import IntelligencePipeline
    from supabase import create_client
    from apps.synthetic_intel import enrich_synthetic_financials
    import os

    st.info("🧠 Generating strategic intelligence...")

    # Get Supabase credentials
    try:
        supabase_url = st.secrets.get("supabase", {}).get("url") or os.getenv("SUPABASE_URL")
        supabase_key = st.secrets.get("supabase", {}).get("service_key") or os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase credentials not found. Intelligence pipeline requires Supabase to store insights.")
            st.caption("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env or Streamlit secrets")
            return []
    except Exception as e:
        st.warning(f"⚠️ Could not load Supabase credentials: {str(e)}")
        return []

    # Get OpenAI API key
    try:
        openai_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.warning("⚠️ OpenAI API key not found. Intelligence pipeline requires OpenAI for insights.")
            st.caption("Set OPENAI_API_KEY in .env or Streamlit secrets")
            return []
    except Exception as e:
        st.warning(f"⚠️ Could not load OpenAI credentials: {str(e)}")
        return []

    try:
        # Initialize Supabase client
        supabase = create_client(supabase_url, supabase_key)

        # Initialize intelligence pipeline
        pipeline = IntelligencePipeline(
            supabase=supabase,
            openai_api_key=openai_key,
            enable_data_accumulation=enable_network_accumulation
        )

        # STEP 1: Accumulate market data for network effect (optional but recommended)
        if enable_network_accumulation and not df_market_snapshot.empty:
            with st.spinner("📊 Accumulating market data for network intelligence..."):
                # Enrich with synthetic financials first
                df_enriched = enrich_synthetic_financials(df_market_snapshot.copy())

                pipeline.accumulate_market_data(
                    market_snapshot=df_enriched,
                    category_id=category_context.get('category_id'),
                    category_name=category_context.get('category_name', 'Unknown'),
                    category_tree=category_context.get('category_path', '').split(' > ') if category_context.get('category_path') else []
                )
                st.success("✅ Market data accumulated for network intelligence")

        # STEP 2: Prepare market data for portfolio ASINs
        if not portfolio_asins or len(portfolio_asins) == 0:
            st.info("ℹ️ No portfolio ASINs specified. Skipping intelligence generation.")
            st.caption("Specify ASINs to analyze by passing portfolio_asins parameter")
            return []

        with st.spinner(f"🔍 Preparing data for {len(portfolio_asins)} portfolio ASINs..."):
            market_data = _prepare_market_data_for_pipeline(
                portfolio_asins=portfolio_asins,
                df_market_snapshot=df_market_snapshot,
                df_weekly=df_weekly
            )

        # STEP 3: Generate unified intelligence
        with st.spinner(f"💡 Generating insights for {len(portfolio_asins)} ASINs..."):
            intelligence_results = pipeline.generate_portfolio_intelligence(
                portfolio_asins=portfolio_asins,
                market_data=market_data,
                category_context=category_context
            )

        if intelligence_results:
            st.success(f"✅ Generated {len(intelligence_results)} strategic insights")

            # Show summary
            critical_count = sum(1 for i in intelligence_results if i.product_status.priority == 100)
            opportunity_count = sum(1 for i in intelligence_results if i.product_status.priority == 75)

            if critical_count > 0:
                st.error(f"🚨 {critical_count} CRITICAL alerts requiring immediate attention")
            if opportunity_count > 0:
                st.info(f"💡 {opportunity_count} opportunities identified")

        return intelligence_results

    except Exception as e:
        st.error(f"❌ Intelligence pipeline error: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details", expanded=False):
            st.code(traceback.format_exc())
        return []


def _prepare_market_data_for_pipeline(
    portfolio_asins: List[str],
    df_market_snapshot: pd.DataFrame,
    df_weekly: pd.DataFrame
) -> Dict[str, Any]:
    """
    Transform Phase 2 data into format expected by intelligence pipeline.

    Args:
        portfolio_asins: List of ASINs to analyze
        df_market_snapshot: Market snapshot DataFrame
        df_weekly: Weekly historical DataFrame

    Returns:
        Dict mapping ASIN → {historical, competitors, current_metrics}
    """
    market_data = {}

    for asin in portfolio_asins:
        # Get competitor data (all other ASINs in category)
        competitor_data = df_market_snapshot[df_market_snapshot['asin'] != asin].copy()

        # Get historical data for this ASIN
        historical_df = df_weekly[df_weekly['asin'] == asin].copy() if not df_weekly.empty else pd.DataFrame()

        # Reshape historical data for trigger detection (needs time series format)
        if not historical_df.empty:
            # Sort by date
            historical_df = historical_df.sort_values('week_start')

            # Rename columns to match trigger detection expectations
            historical_df = historical_df.rename(columns={
                'filled_price': 'price',
                'sales_rank_filled': 'bsr',
                'weekly_sales_filled': 'revenue'
            })

        # Get current metrics from snapshot
        current_row = df_market_snapshot[df_market_snapshot['asin'] == asin]

        if current_row.empty:
            # ASIN not in market snapshot - skip
            continue

        current_metrics = {
            'price': float(current_row['price'].iloc[0]) if 'price' in current_row else 0,
            'bsr': float(current_row['bsr'].iloc[0]) if 'bsr' in current_row and pd.notna(current_row['bsr'].iloc[0]) else 0,
            'review_count': int(current_row.get('review_count', {}).iloc[0]) if 'review_count' in current_row else 0,
            'rating': float(current_row.get('rating', 0).iloc[0]) if 'rating' in current_row else 0,
            'buybox_share': 1.0,  # Default assumption for own products
            'inventory': 0,  # Not available in Phase 2 data
            'brand': str(current_row['brand'].iloc[0]) if 'brand' in current_row else 'Unknown',
            'category_root': str(current_row.get('category_path', '').iloc[0].split(' > ')[0]) if 'category_path' in current_row else '',
            'estimated_monthly_sales': float(current_row['monthly_units'].iloc[0]) if 'monthly_units' in current_row else 0,
            'estimated_monthly_revenue': float(current_row['revenue_proxy'].iloc[0]) if 'revenue_proxy' in current_row else 0
        }

        market_data[asin] = {
            'historical': historical_df,
            'competitors': competitor_data,
            'current_metrics': current_metrics
        }

    return market_data
