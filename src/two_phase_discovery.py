"""
ShelfGuard Two-Phase Discovery Architecture
============================================
Solves the "Universe Definition" problem with a strategic two-phase approach.

Phase 1 (Seed Discovery):
    - Lightweight keyword search (25-50 results)
    - User selects "Seed Product" to define market focus
    - Extract category breadcrumb for Phase 2

Phase 2 (Market Mapping):
    - Fetch products from seed's category
    - DYNAMIC: Keep fetching until 80% of category revenue captured
    - LLM validates competitive relevance
    - Result: Clean denominator for market share calculations
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import keepa
import requests
import os
import time
from openai import OpenAI
import json


KEEPA_API_KEY = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client from secrets or environment."""
    try:
        return OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
    except:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None


@st.cache_data(ttl=3600)
def phase1_seed_discovery(
    keyword: str,
    limit: int = 50,
    domain: str = "US",
    category_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Phase 1: Lightweight search to find seed products.

    Args:
        keyword: User's search term
        limit: Max results (25-50 recommended)
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)
        category_filter: Optional category ID to restrict search (e.g., 16310101 for Grocery)

    Returns:
        DataFrame with [asin, title, brand, category_id, category_path, price, bsr]
    """
    if not KEEPA_API_KEY:
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
        url = f"https://api.keepa.com/query?key={KEEPA_API_KEY}&domain={domain_id}&stats=0"
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
        api = keepa.Keepa(KEEPA_API_KEY)
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

            # Extract brand (first word of title)
            brand = title.split()[0] if title else "Unknown"

            # Extract all category IDs from the tree (for progressive fallback)
            category_tree_ids = [cat.get("catId", 0) for cat in category_tree] if category_tree else [root_category]
            
            records.append({
                "asin": asin,
                "title": title,
                "brand": brand,
                "category_id": root_category,  # Root category for backward compatibility
                "leaf_category_id": leaf_category_id,  # Leaf node for high relevance/purity
                "category_tree_ids": category_tree_ids,  # Full tree for progressive fallback
                "category_path": category_path,
                "price": price,
                "bsr": bsr
            })

        df = pd.DataFrame(records)
        return df.sort_values("bsr").reset_index(drop=True)  # Sort by best sellers

    except Exception as e:
        st.error(f"Phase 1 Discovery Error: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def phase2_category_market_mapping(
    category_id: int,
    seed_product_title: str,
    target_revenue_pct: float = 80.0,
    max_products: int = 500,
    batch_size: int = 100,
    domain: str = "US",
    leaf_category_id: Optional[int] = None,
    category_path: Optional[str] = None,
    category_tree_ids: Optional[tuple] = None,  # Tuple for caching (lists aren't hashable)
    min_products: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Phase 2: Dynamically fetch products from category until 80% revenue captured.
    
    Uses progressive category fallback: starts with leaf, walks up tree until min_products found.

    Args:
        category_id: Root category ID from seed product
        seed_product_title: Title of seed product (for LLM context)
        target_revenue_pct: Stop when this % of revenue is captured (default 80%)
        max_products: Safety limit to prevent runaway fetching
        batch_size: How many products to fetch per iteration
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)
        leaf_category_id: Most specific subcategory ID (leaf node) - ensures high relevance/purity
        category_tree_ids: Full list of category IDs from root to leaf (for progressive fallback)
        min_products: Minimum products needed before walking up category hierarchy (default 10)

    Returns:
        Tuple of (validated_df, market_stats)
    """
    if not KEEPA_API_KEY:
        raise ValueError("KEEPA_API_KEY not found")

    # Display category info
    st.info(f"🎯 Fetching 100 ASINs from category: {category_id}")
    if category_path:
        st.caption(f"📂 Category: {category_path}")

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
    api = keepa.Keepa(KEEPA_API_KEY)

    # Continue fetching until we have 100 valid products (with price/BSR)
    while len(all_products) < target_valid_products:
        # Build query for this category (use direct HTTP API)
        # Always query by rootCategory (reliable) - we'll post-filter by leaf if needed
        query_json = {
            "perPage": max(50, batch_size),  # Minimum 50
            "page": page,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
            "rootCategory": [int(category_id)],  # Always use root category for API query
        }
        
        if page == 0:
            st.caption(f"🔍 Querying category {category_id}...")

        try:
            # Make HTTP POST request with retry logic for rate limits
            url = f"https://api.keepa.com/query?key={KEEPA_API_KEY}&domain={domain_id}&stats=0"
            
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
            
            # Simplified: Just collect all valid products (no category filtering)
            # We already queried by root category, so products are relevant
            all_valid_products = []
            
            for product in products:
                
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

                product_data = {
                    "asin": product.get("asin"),
                    "title": product.get("title", ""),
                    "price": price,
                    "monthly_units": monthly_units,
                    "revenue_proxy": revenue,
                    "bsr": bsr
                }
                
                # Track all valid products
                all_valid_products.append(product_data)
            
            # Show data quality stats on first page
            if page == 0 and len(products) > 0:
                total_fetched = len(products)
                products_with_revenue = sum(1 for p in all_valid_products if p.get("revenue_proxy", 0) > 0)
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
                f"Fetched {len(all_products)} products | "
                f"Batch revenue: ${batch_revenue:,.0f} | "
                f"Cumulative: ${cumulative_revenue:,.0f}"
            )

        except Exception as e:
            st.warning(f"Error fetching page {page}: {str(e)}")
            break

    # Remove duplicates (by ASIN) and internal tracking fields
    seen_asins = set()
    unique_products = []
    for p in all_products:
        p.pop("_matched_level", None)
        asin = p.get("asin")
        if asin and asin not in seen_asins:
            seen_asins.add(asin)
            unique_products.append(p)
    
    df = pd.DataFrame(unique_products)

    if df.empty:
        return df, {}

    # ========== SYNTHETIC DATA FILLING (like weekly pipeline) ==========
    # Fill missing price/BSR using median from products that have data
    # This ensures all 100 ASINs have usable price/BSR values
    
    # Calculate medians from products with valid data
    valid_prices = df[df["price"] > 0]["price"]
    valid_bsrs = df[(df["bsr"].notna()) & (df["bsr"] > 0)]["bsr"]
    
    median_price = valid_prices.median() if len(valid_prices) > 0 else 15.0  # Default $15 if no data
    median_bsr = valid_bsrs.median() if len(valid_bsrs) > 0 else 50000  # Default BSR 50k if no data
    
    # Count products needing fill
    missing_price_count = (df["price"] == 0).sum()
    missing_bsr_count = (df["bsr"].isna() | (df["bsr"] == 0)).sum()
    
    # Fill missing prices with median
    df.loc[df["price"] == 0, "price"] = median_price
    
    # Fill missing BSR with median
    df.loc[df["bsr"].isna() | (df["bsr"] == 0), "bsr"] = median_bsr
    
    # Recalculate monthly_units and revenue for filled products
    # Formula: monthly_units = 145000 * (BSR ^ -0.9)
    df["monthly_units"] = 145000.0 * (df["bsr"].clip(lower=1) ** -0.9)
    df["revenue_proxy"] = df["monthly_units"] * df["price"]
    
    if missing_price_count > 0 or missing_bsr_count > 0:
        st.caption(
            f"📊 Filled {missing_price_count} missing prices (→ ${median_price:.2f}) "
            f"and {missing_bsr_count} missing BSRs (→ {int(median_bsr):,})"
        )
    
    # Sort by BSR (best sellers first) to maintain order, then by revenue as secondary
    # This ensures we show products in order of market importance
    df = df.sort_values(["bsr", "revenue_proxy"], ascending=[True, False]).reset_index(drop=True)

    # Calculate market stats
    market_stats = {
        "total_products": len(df),
        "total_category_revenue": df["revenue_proxy"].sum(),
        "category_id": category_id
    }

    # Phase 2.5: LLM Validation (disabled - category filtering is sufficient)
    # LLM validation was too strict and rejected valid products
    # Category + price/BSR filtering already ensures relevance
    df_validated = df  # Skip LLM validation

    market_stats["validated_products"] = len(df_validated)
    market_stats["validated_revenue"] = df_validated["revenue_proxy"].sum()

    return df_validated, market_stats


def llm_validate_competitive_set(
    df: pd.DataFrame,
    seed_product_title: str
) -> pd.DataFrame:
    """
    Use LLM to filter out non-competitive products (accessories, bundles, etc.)

    Args:
        df: DataFrame of products from category
        seed_product_title: Title of seed product for context

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
