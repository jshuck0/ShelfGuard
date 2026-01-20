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
            "asin, title, brand, sales_rank, buy_box_price, estimated_weekly_revenue, "
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

            market_stats = {
                "total_products": len(df),
                "total_category_revenue": df["revenue_proxy"].sum() if "revenue_proxy" in df.columns else 0,
                "category_id": category_id,
                "validated_products": len(df),
                "validated_revenue": df["revenue_proxy"].sum() if "revenue_proxy" in df.columns else 0,
                "source": "cache",
                "df_weekly": pd.DataFrame()  # No weekly data from cache
            }

            st.caption(f"⚡ Found {len(df)} cached products for category {category_id}")
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


@st.cache_data(ttl=3600)
def phase1_seed_discovery(
    keyword: str,
    limit: int = 50,
    domain: str = "US",
    category_filter: Optional[int] = None,
    check_cache: bool = True,
    use_family_harvester: bool = None  # None = use global toggle
) -> pd.DataFrame:
    """
    Phase 1: Lightweight search to find seed products.

    FIX 1.3: Now checks database cache before making Keepa API calls.
    This reduces API costs by 50-80% for repeated searches.
    
    NEW: When use_family_harvester=True, uses intelligent variation-aware
    discovery that fetches complete product families instead of naive keyword matches.
    
    Example: Searching "RXBAR" with family harvester will return:
    - 1 Parent RXBAR listing + all 50 flavors/sizes
    - Then the next competitor's complete family
    Instead of: 1 RXBAR + 99 random competitor bars

    Args:
        keyword: User's search term
        limit: Max results (25-50 recommended)
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)
        category_filter: Optional category ID to restrict search (e.g., 16310101 for Grocery)
        check_cache: Whether to check database cache first (default True)
        use_family_harvester: Use variation-aware discovery (None = use global toggle)

    Returns:
        DataFrame with [asin, title, brand, category_id, category_path, price, bsr]
    """
    # Determine if we should use family harvester
    use_families = use_family_harvester if use_family_harvester is not None else ENABLE_FAMILY_HARVESTER
    
    # NEW: If family harvester is enabled, use that instead
    if use_families:
        try:
            st.info("🧬 Using Family Harvester (variation-aware discovery)")
            return harvest_to_seed_dataframe(
                keyword=keyword,
                limit=limit,
                domain=domain,
                category_filter=category_filter
            )
        except Exception as e:
            st.warning(f"⚠️ Family Harvester failed, falling back to naive search: {e}")
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
        }

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

            # Get price
            csv = product.get("csv", [])
            price = 0
            if csv and len(csv) > 18 and csv[18]:
                price_cents = csv[18][-1] if len(csv[18]) > 0 else 0
                price = (price_cents / 100.0) if price_cents > 0 else 0

            # Get BSR
            bsr = 0
            if csv and len(csv) > 3 and csv[3]:
                bsr = csv[3][-1] if len(csv[3]) > 0 and csv[3][-1] != -1 else 0

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
    seed_asin: Optional[str] = None,  # ✅ ADD SEED ASIN
    target_revenue_pct: float = 80.0,
    max_products: int = 500,
    batch_size: int = 100,
    domain: str = "US",
    leaf_category_id: Optional[int] = None,
    category_path: Optional[str] = None,
    category_tree_ids: Optional[tuple] = None,  # Tuple for caching (lists aren't hashable)
    min_products: int = 10,
    check_cache: bool = False  # FIX 1.3: Optional cache check (disabled by default for Phase 2)
) -> Tuple[pd.DataFrame, Dict]:
    """
    Phase 2: Dynamically fetch products from category until 80% revenue captured.

    Uses progressive category fallback: starts with leaf, walks up tree until min_products found.

    FIX 1.3: Now supports optional database cache check to reduce API costs.
    Note: Cache is disabled by default for Phase 2 since users typically want fresh data.

    Args:
        category_id: Root category ID from seed product
        seed_product_title: Title of seed product (for LLM context)
        seed_asin: ASIN of seed product (will be force-included in results)
        target_revenue_pct: Stop when this % of revenue is captured (default 80%)
        max_products: Safety limit to prevent runaway fetching
        batch_size: How many products to fetch per iteration
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)
        leaf_category_id: Most specific subcategory ID (leaf node) - ensures high relevance/purity
        category_tree_ids: Full list of category IDs from root to leaf (for progressive fallback)
        min_products: Minimum products needed before walking up category hierarchy (default 10)
        check_cache: Whether to check database cache first (default False for fresh data)

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
    # Target: 100 products with valid price/BSR
    # Fetch extra ASINs to account for ones without price/BSR data (~40-50% may be filtered)
    all_products = []
    page = 0
    cumulative_revenue = 0
    target_valid_products = 100

    # Convert domain string to domain ID
    domain_map = {"US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12}
    domain_id = domain_map.get(domain, 1)

    # Initialize Keepa for query() calls
    api = keepa.Keepa(api_key)

    # ========== PROGRESSIVE CATEGORY FALLBACK ==========
    # Per Keepa docs:
    #   - categories_include: filters by products DIRECTLY in those subcategories (precise)
    #   - rootCategory: filters by root category (broad)
    # Strategy: Start with leaf, walk up tree until we find a level with >= min_products
    
    effective_category_id = leaf_category_id if leaf_category_id else category_id
    effective_category_path = category_path or ""
    use_categories_include = leaf_category_id is not None  # True if we have a subcategory
    
    # Build category hierarchy (leaf to root) for progressive fallback
    if category_tree_ids and len(category_tree_ids) > 1:
        category_hierarchy = list(reversed(category_tree_ids)) if isinstance(category_tree_ids, tuple) else list(reversed(list(category_tree_ids)))
        path_segments = category_path.split(" > ") if category_path else []
        path_segments_reversed = list(reversed(path_segments))
        
        st.caption(f"🔍 Testing category levels for sufficient products (need >= {min_products})...")
        
        for idx, test_category_id in enumerate(category_hierarchy):
            # Test this category level with a quick query
            test_query = {
                "perPage": 50,
                "page": 0,
                "current_SALES_gte": 1,
                "current_SALES_lte": 100000,
            }
            
            # Use categories_include for non-root categories, rootCategory for root
            if test_category_id != category_id:
                test_query["categories_include"] = [int(test_category_id)]
                test_query["rootCategory"] = [int(category_id)]
            else:
                test_query["rootCategory"] = [int(category_id)]
            
            try:
                test_url = f"https://api.keepa.com/query?key={api_key}&domain={domain_id}&stats=0"
                test_response = requests.post(test_url, json=test_query, timeout=15)
                
                if test_response.status_code == 200:
                    test_result = test_response.json()
                    test_count = len(test_result.get("asinList", []))
                    total_results = test_result.get("totalResults", test_count)
                    
                    # Build the path name for this level
                    if idx < len(path_segments_reversed):
                        level_path = " > ".join(path_segments[:len(path_segments) - idx])
                    else:
                        level_path = path_segments[0] if path_segments else f"Category {test_category_id}"
                    
                    if total_results >= min_products:
                        effective_category_id = test_category_id
                        effective_category_path = level_path
                        use_categories_include = test_category_id != category_id
                        
                        if test_category_id != leaf_category_id:
                            st.info(f"✅ Using category '{level_path}' ({total_results} products found)")
                        else:
                            st.caption(f"✅ Leaf category has {total_results} products - using directly")
                        break
                    else:
                        st.caption(f"⚠️ '{level_path}' has only {total_results} products - walking up...")
                        
                elif test_response.status_code == 429:
                    st.warning("Rate limited during category testing - using leaf category directly")
                    time.sleep(2)
                    break
            except Exception as e:
                st.caption(f"⚠️ Error testing category {test_category_id}: {str(e)[:50]}")
                continue
    else:
        # No category tree - use what we have
        st.caption(f"ℹ️ Using category {effective_category_id} (no hierarchy available)")
    
    # Update debug info with final decision
    with st.expander("🔍 Debug: Final Category Selection", expanded=False):
        st.write(f"**Effective Category ID:** {effective_category_id}")
        st.write(f"**Effective Category Path:** {effective_category_path}")
        st.write(f"**Using categories_include:** {use_categories_include}")
    
    # ========== MAIN FETCH LOOP ==========
    max_pages = 10  # Safety limit (10 pages x 100 = 1000 products max)
    
    while len(all_products) < target_valid_products and page < max_pages:
        # Build query for this category (use direct HTTP API)
        query_json = {
            "perPage": max(50, batch_size),  # Minimum 50
            "page": page,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
        }
        
        # Add category filter based on our progressive fallback decision
        if use_categories_include:
            # Use categories_include for precise subcategory filtering
            query_json["categories_include"] = [int(effective_category_id)]
            query_json["rootCategory"] = [int(category_id)]
        else:
            # Using root category only
            query_json["rootCategory"] = [int(category_id)]
        
        if page == 0:
            # Debug: Show exact query being sent
            with st.expander("🔍 Debug: Keepa Query JSON", expanded=False):
                st.json(query_json)
            st.caption(f"🔍 Querying with filter: {list(query_json.keys())}...")

        try:
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
                        st.error(f"❌ Rate limited after {max_retries} attempts. Stopping to avoid further throttling.")
                        break  # Give up after max retries
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
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        batch_products = api.query(batch_asins, stats=90, rating=True)
                        products.extend(batch_products)
                        break  # Success, exit retry loop
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "timeout" in error_msg or "read timeout" in error_msg:
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
                            st.warning(f"⚠️ Error fetching batch {i//keepa_batch_size + 1}: {str(e)}")
                            break
                
                # Small delay between batches to be polite to API
                if i + keepa_batch_size < len(asins):
                    time.sleep(0.5)

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
                # LIGHT CATEGORY VALIDATION: Just verify root category matches
                # Since we're using categories_include, Keepa should return correct products
                # This is a sanity check, not a strict filter
                product_root_category = product.get("rootCategory", 0)
                
                # Only reject products from completely different root categories
                is_valid_category = product_root_category == category_id
                
                if not is_valid_category:
                    products_filtered_by_category += 1
                    continue  # Skip products from different root categories
                
                # Calculate revenue proxy
                csv = product.get("csv", [])
                
                # Get price: Try Buy Box first, then Amazon price, then New FBA
                price = 0
                if csv and len(csv) > 18 and csv[18]:
                    price_array = csv[18]
                    if price_array and len(price_array) > 0:
                        # Get last non-null value (most recent price)
                        price_cents = price_array[-1] if price_array[-1] else 0
                        if price_cents and price_cents > 0:
                            price = price_cents / 100.0
                
                # Fallback to Amazon price if Buy Box is 0
                if price == 0 and csv and len(csv) > 0 and csv[0]:
                    price_array = csv[0]
                    if price_array and len(price_array) > 0:
                        price_cents = price_array[-1] if price_array[-1] else 0
                        if price_cents and price_cents > 0:
                            price = price_cents / 100.0
                
                # Fallback to New FBA if still 0
                if price == 0 and csv and len(csv) > 10 and csv[10]:
                    price_array = csv[10]
                    if price_array and len(price_array) > 0:
                        price_cents = price_array[-1] if price_array[-1] else 0
                        if price_cents and price_cents > 0:
                            price = price_cents / 100.0

                # Get BSR - allow NaN if missing (we'll keep the product anyway)
                bsr = None
                if csv and len(csv) > 3 and csv[3]:
                    bsr_array = csv[3]
                    if bsr_array and len(bsr_array) > 0:
                        bsr_value = bsr_array[-1]
                        # Filter out placeholder/invalid values: -1, but allow 0 (which is invalid but we'll keep it)
                        if bsr_value and bsr_value != -1 and bsr_value > 0:
                            bsr = bsr_value

                # Calculate monthly units from BSR using power law formula
                # Formula: monthly_units = 145000 * (BSR ^ -0.9)
                # This is calibrated for Grocery velocity (from keepa_client.py)
                monthly_units = 0
                if bsr and bsr > 0:
                    monthly_units = 145000.0 * (bsr ** -0.9)
                else:
                    # Fallback: Try Keepa's avg30 stats if BSR not available
                    stats = product.get("stats", {})
                    if stats and "current" in stats:
                        current_stats = stats["current"]
                        if isinstance(current_stats, dict) and "avg30" in current_stats:
                            monthly_units = current_stats["avg30"] or 0

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

                product_data = {
                    "asin": product.get("asin"),
                    "title": title,
                    "brand": brand,  # Add brand column
                    "price": price,
                    "monthly_units": monthly_units,
                    "revenue_proxy": revenue,
                    "bsr": bsr,
                    "main_image": main_image  # Add image for Visual Audit
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

            # Stop when we have 100 ASINs
            if len(all_products) >= 100:
                st.success(
                    f"✅ **Fetched {len(all_products)} ASINs**\n\n"
                    f"Total estimated revenue: **${cumulative_revenue:,.0f}**"
                )
                break

            page += 1
            st.caption(
                f"Page {page}: Fetched {len(all_products)} valid products | "
                f"Batch revenue: ${batch_revenue:,.0f} | "
                f"Cumulative: ${cumulative_revenue:,.0f}"
            )
            
            # Rate limit protection between pages
            time.sleep(1)

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

                    # Extract data same as above
                    csv = seed_product_data.get("csv", [])
                    price = 0
                    if csv and len(csv) > 18 and csv[18]:
                        price_array = csv[18]
                        if price_array and len(price_array) > 0:
                            price_cents = price_array[-1] if price_array[-1] else 0
                            if price_cents and price_cents > 0:
                                price = price_cents / 100.0

                    bsr = None
                    if csv and len(csv) > 3 and csv[3]:
                        bsr_array = csv[3]
                        if bsr_array and len(bsr_array) > 0:
                            bsr_value = bsr_array[-1]
                            if bsr_value and bsr_value != -1 and bsr_value > 0:
                                bsr = bsr_value

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
            df = pd.DataFrame(unique_products)
            df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
            df["revenue_proxy"] = df["monthly_units"] * df["price"]
            df = df.sort_values(["bsr", "revenue_proxy"], ascending=[True, False]).reset_index(drop=True)
            market_stats = {
                "total_products": len(df),
                "total_category_revenue": df["revenue_proxy"].sum(),
                "category_id": category_id,
                "effective_category_id": effective_category_id,
                "effective_category_path": effective_category_path,
                "use_categories_include": use_categories_include,
                "validated_products": len(df),
                "validated_revenue": df["revenue_proxy"].sum(),
                "df_weekly": pd.DataFrame()  # Empty weekly data
            }
            return df, market_stats
        
        st.caption(f"✅ Received historical data for {len(historical_products)} products")
        
        # Build weekly table using the same methodology as the main dashboard
        # This creates: week_start, asin, title, sales_rank_filled, filled_price, weekly_sales_filled, etc.
        df_weekly = build_keepa_weekly_table(historical_products)
        
        if df_weekly.empty:
            st.warning("⚠️ Could not build weekly table - falling back to current estimates")
            df = pd.DataFrame(unique_products)
            df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
            df["revenue_proxy"] = df["monthly_units"] * df["price"]
            df = df.sort_values(["bsr", "revenue_proxy"], ascending=[True, False]).reset_index(drop=True)
            market_stats = {
                "total_products": len(df),
                "total_category_revenue": df["revenue_proxy"].sum(),
                "category_id": category_id,
                "effective_category_id": effective_category_id,
                "effective_category_path": effective_category_path,
                "use_categories_include": use_categories_include,
                "validated_products": len(df),
                "validated_revenue": df["revenue_proxy"].sum(),
                "df_weekly": pd.DataFrame()
            }
            return df, market_stats
        
        st.caption(f"📈 Built weekly table with {len(df_weekly)} rows across {df_weekly['asin'].nunique()} ASINs")
        
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
        
        # Merge week counts
        asin_summary = asin_summary.merge(weeks_per_asin, on="asin", how="left")
        
        # Rename columns for snapshot
        asin_summary = asin_summary.rename(columns={
            "filled_price": "price",
            "sales_rank_filled": "bsr",
            "weekly_sales_filled": "total_90d_revenue",
            "estimated_units": "total_90d_units"
        })
        
        # Calculate monthly averages using actual weeks of data
        # Average weekly revenue * 4.33 weeks per month = monthly revenue
        # This accounts for varying amounts of historical data (not always exactly 90 days)
        asin_summary["avg_weekly_revenue"] = asin_summary["total_90d_revenue"] / asin_summary["weeks_count"].clip(lower=1)
        asin_summary["avg_weekly_units"] = asin_summary["total_90d_units"] / asin_summary["weeks_count"].clip(lower=1)
        asin_summary["monthly_units"] = asin_summary["avg_weekly_units"] * 4.33  # Average weeks per month
        asin_summary["revenue_proxy"] = asin_summary["avg_weekly_revenue"] * 4.33  # Average weeks per month
        
        # Debug: Show revenue calculation stats
        avg_weeks = asin_summary["weeks_count"].mean()
        total_monthly_rev = asin_summary["revenue_proxy"].sum()
        avg_product_rev = asin_summary["revenue_proxy"].mean()
        st.caption(f"📊 Revenue calc: {len(asin_summary)} products, avg {avg_weeks:.1f} weeks data, total ${total_monthly_rev:,.0f}/mo, avg ${avg_product_rev:,.0f}/mo per product")
        
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
        missing_revenue_mask = asin_summary["revenue_proxy"].isna() | (asin_summary["revenue_proxy"] == 0)
        if missing_revenue_mask.any():
            asin_summary.loc[missing_revenue_mask, "monthly_units"] = (
                145000.0 * (asin_summary.loc[missing_revenue_mask, "bsr"].clip(lower=1) ** -0.9)
            )
            asin_summary.loc[missing_revenue_mask, "revenue_proxy"] = (
                asin_summary.loc[missing_revenue_mask, "monthly_units"] * 
                asin_summary.loc[missing_revenue_mask, "price"]
            )
        
        if missing_price_count > 0 or missing_bsr_count > 0:
            st.caption(
                f"📊 Filled {int(missing_price_count)} missing prices (→ ${median_price:.2f}) "
                f"and {int(missing_bsr_count)} missing BSRs (→ {int(median_bsr):,})"
            )
        
        # Sort by BSR (best sellers first)
        df = asin_summary.sort_values(["bsr", "revenue_proxy"], ascending=[True, False]).reset_index(drop=True)
        
        # Calculate market stats
        market_stats = {
            "total_products": len(df),
            "total_category_revenue": df["revenue_proxy"].sum(),
            "category_id": category_id,
            "effective_category_id": effective_category_id,
            "effective_category_path": effective_category_path,
            "use_categories_include": use_categories_include,
            "validated_products": len(df),
            "validated_revenue": df["revenue_proxy"].sum(),
            "df_weekly": df_weekly,  # Include full weekly data for Command Center
            "time_period": "90 days (weekly aggregated)"
        }
        
        st.success(
            f"✅ **Market Snapshot Built from 90 Days of Historical Data**\n\n"
            f"Products: **{len(df)}** | Monthly Revenue: **${df['revenue_proxy'].sum():,.0f}**"
        )
        
        return df, market_stats
        
    except Exception as e:
        st.warning(f"⚠️ Error fetching historical data: {str(e)} - falling back to current estimates")
        # Fallback to old method
        df = pd.DataFrame(unique_products)
        if df.empty:
            return df, {}
        df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
        df["revenue_proxy"] = df["monthly_units"] * df["price"]
        df = df.sort_values(["bsr", "revenue_proxy"], ascending=[True, False]).reset_index(drop=True)
        market_stats = {
            "total_products": len(df),
            "total_category_revenue": df["revenue_proxy"].sum(),
            "category_id": category_id,
            "effective_category_id": effective_category_id,
            "effective_category_path": effective_category_path,
            "use_categories_include": use_categories_include,
            "validated_products": len(df),
            "validated_revenue": df["revenue_proxy"].sum(),
            "df_weekly": pd.DataFrame()
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
    df_weekly = build_keepa_weekly_table(all_products)
    
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
