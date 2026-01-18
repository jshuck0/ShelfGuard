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
    - DYNAMIC: Keep fetching until 90% of category revenue captured
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
        api = keepa.Keepa(KEEPA_API_KEY)
        products = api.query(asins, stats=30, rating=True)

        records = []
        for product in products:
            asin = product.get("asin", "UNK")
            title = product.get("title", "Unknown")

            # Extract category info
            category_tree = product.get("categoryTree", [])
            root_category = product.get("rootCategory", 0)

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

            records.append({
                "asin": asin,
                "title": title,
                "brand": brand,
                "category_id": root_category,
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
    target_revenue_pct: float = 90.0,
    max_products: int = 500,
    batch_size: int = 100,
    domain: str = "US"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Phase 2: Dynamically fetch products from category until 90% revenue captured.

    Args:
        category_id: Root category ID from seed product
        seed_product_title: Title of seed product (for LLM context)
        target_revenue_pct: Stop when this % of revenue is captured (default 90%)
        max_products: Safety limit to prevent runaway fetching
        batch_size: How many products to fetch per iteration
        domain: Amazon marketplace ('US', 'GB', 'DE', etc.)

    Returns:
        Tuple of (validated_df, market_stats)
    """
    if not KEEPA_API_KEY:
        raise ValueError("KEEPA_API_KEY not found")

    # Fetch products from the category in batches
    all_products = []
    page = 0
    cumulative_revenue = 0
    total_revenue_estimate = None

    st.info(f"🎯 Mapping market for category ID: {category_id}")

    # Convert domain string to domain ID
    domain_map = {"US": 1, "GB": 2, "DE": 3, "FR": 4, "JP": 5, "CA": 6, "IT": 8, "ES": 9, "IN": 10, "MX": 11, "BR": 12}
    domain_id = domain_map.get(domain, 1)

    # Initialize Keepa for query() calls
    api = keepa.Keepa(KEEPA_API_KEY)

    while len(all_products) < max_products:
        # Build query for this category (use direct HTTP API)
        query_json = {
            "rootCategory": [int(category_id)],  # Note: must be array, convert numpy int64 to Python int
            "perPage": max(50, batch_size),  # Minimum 50
            "page": page,
            "current_SALES_gte": 1,
            "current_SALES_lte": 100000,
        }

        try:
            # Make HTTP POST request
            url = f"https://api.keepa.com/query?key={KEEPA_API_KEY}&domain={domain_id}&stats=0"
            response = requests.post(url, json=query_json, timeout=30)

            if response.status_code != 200:
                st.warning(f"Keepa API error on page {page}: {response.status_code}")
                break

            result = response.json()
            asins = result.get("asinList", [])

            if not asins:
                break  # No more products in category

            # Fetch product data with stats using keepa library
            products = api.query(asins, stats=90, rating=True)

            for product in products:
                # Calculate revenue proxy
                csv = product.get("csv", [])
                price = 0
                if csv and len(csv) > 18 and csv[18]:
                    price_cents = csv[18][-1] if len(csv[18]) > 0 else 0
                    price = (price_cents / 100.0) if price_cents > 0 else 0

                # Get BSR
                bsr = 0
                if csv and len(csv) > 3 and csv[3]:
                    bsr = csv[3][-1] if len(csv[3]) > 0 and csv[3][-1] != -1 else 0

                # Calculate monthly units from BSR using power law formula
                # Formula: monthly_units = 145000 * (BSR ^ -0.9)
                # This is calibrated for Grocery velocity (from keepa_client.py)
                if bsr > 0:
                    monthly_units = 145000.0 * (bsr ** -0.9)
                else:
                    # Fallback to Keepa's avg30 if BSR not available
                    monthly_units = 0
                    stats = product.get("stats", {})
                    if stats and "current" in stats:
                        current_stats = stats["current"]
                        if isinstance(current_stats, dict) and "avg30" in current_stats:
                            monthly_units = current_stats["avg30"] or 0

                revenue = monthly_units * price

                all_products.append({
                    "asin": product.get("asin"),
                    "title": product.get("title", ""),
                    "price": price,
                    "monthly_units": monthly_units,
                    "revenue_proxy": revenue,
                    "bsr": bsr
                })

            # Sort by revenue descending
            all_products_sorted = sorted(all_products, key=lambda x: x["revenue_proxy"], reverse=True)

            # Calculate cumulative revenue
            cumulative_revenue = sum(p["revenue_proxy"] for p in all_products_sorted)

            # Estimate total category revenue (assume we've seen enough to extrapolate)
            if page == 0 and len(all_products) >= batch_size:
                # First batch - estimate total market size
                avg_revenue_per_product = cumulative_revenue / len(all_products)
                # Rough estimate: assume 1000 products in category
                total_revenue_estimate = avg_revenue_per_product * 1000

            # Check if we've hit 90% of estimated revenue
            if total_revenue_estimate and cumulative_revenue >= (total_revenue_estimate * (target_revenue_pct / 100)):
                st.success(f"✅ Captured {target_revenue_pct}% of category revenue with {len(all_products)} products")
                break

            page += 1
            st.caption(f"Fetched {len(all_products)} products, cumulative revenue: ${cumulative_revenue:,.0f}")

        except Exception as e:
            st.warning(f"Error fetching page {page}: {str(e)}")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_products)

    if df.empty:
        return df, {}

    # Sort by revenue
    df = df.sort_values("revenue_proxy", ascending=False).reset_index(drop=True)

    # Calculate market stats
    market_stats = {
        "total_products": len(df),
        "total_category_revenue": df["revenue_proxy"].sum(),
        "category_id": category_id
    }

    # Phase 2.5: LLM Validation (remove non-competitors)
    df_validated = llm_validate_competitive_set(df, seed_product_title)

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
