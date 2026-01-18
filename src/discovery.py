"""
ShelfGuard Discovery Engine
============================
Phase 1: Search-to-Snapshot with 90% Revenue Pruning

This module implements the Market MRI logic:
1. Search Keepa for top 500 ASINs by category/brand
2. Calculate Revenue Proxy (Units × Price)
3. Prune to the minimum set that captures 90% of total revenue
4. Return pruned market snapshot for visualization
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import json
from typing import Dict, List, Tuple, Optional
import keepa

# Keepa API Configuration
KEEPA_API_KEY = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_asins_from_keepa(asins: List[str]) -> pd.DataFrame:
    """
    Fetch product data for specific ASINs using the Keepa Python library.

    Args:
        asins: List of ASIN strings

    Returns:
        DataFrame with [asin, title, price, monthly_units, revenue_proxy, bsr, main_image]
    """
    if not KEEPA_API_KEY:
        raise ValueError("KEEPA_API_KEY not found in environment variables")

    # Initialize Keepa API
    api = keepa.Keepa(KEEPA_API_KEY)

    # Fetch products (handles chunking automatically)
    try:
        products = api.query(asins, stats=90, rating=True)
    except Exception as e:
        st.error(f"Keepa API error: {str(e)}")
        return pd.DataFrame()

    records = []

    for product in products:
        asin = product.get("asin", "UNK")
        title = product.get("title", "Unknown Product")

        # Get current price (Buy Box > Amazon > New FBA)
        price = 0
        csv = product.get("csv")
        if csv and len(csv) > 18 and csv[18]:  # Buy Box (index 18)
            price_cents = csv[18][-1] if csv[18] and len(csv[18]) > 0 else 0
            price = (price_cents / 100.0) if price_cents and price_cents > 0 else 0

        if price == 0 and csv and len(csv) > 0 and csv[0]:  # Amazon (index 0)
            price_cents = csv[0][-1] if csv[0] and len(csv[0]) > 0 else 0
            price = (price_cents / 100.0) if price_cents and price_cents > 0 else 0

        # Get current BSR (Sales Rank index 3)
        bsr = 0
        if csv and len(csv) > 3 and csv[3]:
            bsr = csv[3][-1] if csv[3] and len(csv[3]) > 0 and csv[3][-1] != -1 else 0

        # Get monthly units from stats
        monthly_sold = 0
        stats = product.get("stats")
        if stats and "current" in stats:
            current_stats = stats["current"]
            if isinstance(current_stats, dict) and "avg30" in current_stats:
                monthly_sold = current_stats["avg30"] or 0

        # Calculate revenue proxy
        revenue_proxy = monthly_sold * price

        # Get main image
        images_csv = product.get("imagesCSV", "")
        main_image = None
        if images_csv:
            images = images_csv.split(",")
            if images:
                main_image = f"https://m.media-amazon.com/images/I/{images[0]}"

        records.append({
            "asin": asin,
            "title": title,
            "price": price,
            "monthly_units": monthly_sold,
            "revenue_proxy": revenue_proxy,
            "bsr": bsr,
            "main_image": main_image
        })

    df = pd.DataFrame(records)
    return df.sort_values("revenue_proxy", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=86400)  # Cache for 24 hours to conserve API tokens
def search_keepa_market(
    query: str,
    search_type: str = "brand",  # "brand" or "category"
    limit: int = 500,
    domain: int = 1  # 1 = US marketplace
) -> List[Dict]:
    """
    Search Keepa Product Finder API for top products.

    Args:
        query: Brand name (e.g., "Starbucks") or category ID
        search_type: "brand" or "category"
        limit: Max products to fetch (default 500 for the 90% rule)
        domain: Amazon domain (1 = US, 2 = UK, etc.)

    Returns:
        List of product dictionaries with ASIN, price, BSR, sales data

    Performance: Cached for 24h to avoid redundant API calls
    """
    if not KEEPA_API_KEY:
        raise ValueError("KEEPA_API_KEY not found in environment variables")

    # Build Keepa Product Finder request
    # Keepa uses /query endpoint for product searches, not /product
    if search_type == "brand":
        # Search by brand name
        params = {
            "key": KEEPA_API_KEY,
            "domain": domain,
            "selection": json.dumps({
                "brand": [query],
                "current_SALES": {
                    "min": 0,
                    "max": 100000  # Top 100k products
                }
            }),
            "page": 0,
            "perPage": limit,
            "stats": 90,  # Request 90-day stats
        }
    else:  # category search
        params = {
            "key": KEEPA_API_KEY,
            "domain": domain,
            "selection": json.dumps({
                "categoryTree": [query],
                "current_SALES": {
                    "min": 0,
                    "max": 100000
                }
            }),
            "page": 0,
            "perPage": limit,
            "stats": 90,
        }

    try:
        response = requests.get(
            f"{KEEPA_BASE_URL}/query",  # Use /query endpoint
            params=params,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        # Debug logging
        if "products" not in data or not data["products"]:
            st.warning(f"⚠️ Keepa returned no products for '{query}'. Try a different search term.")
            st.caption(f"Debug: API response keys: {list(data.keys())}")

        return data.get("products", {}).get("asinList", [])

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Keepa API Error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return []


def calculate_revenue_proxy(products: List[Dict]) -> pd.DataFrame:
    """
    Calculate Revenue Proxy for each product.

    Formula: Revenue Proxy = Monthly Units Sold × Current Price

    Uses Keepa's "stats.current" object which contains:
    - avg30: Bought in the past month (units)
    - current price data

    Returns: DataFrame with columns [asin, title, price, units, revenue_proxy, bsr]
    """
    records = []

    for product in products:
        asin = product.get("asin", "UNK")
        title = product.get("title", "Unknown Product")

        # Extract current stats
        stats = product.get("stats", {}).get("current", [])
        if not stats or len(stats) < 1:
            continue

        current_stats = stats[0] if isinstance(stats, list) else stats

        # Monthly units sold (Keepa field: avg30)
        monthly_sold = current_stats.get("avg30")
        if not monthly_sold or monthly_sold <= 0:
            monthly_sold = 0

        # Current price (try multiple fields in priority order)
        # Check: Buy Box > Amazon > New FBA > New
        price_cents = None
        csv = product.get("csv", {})

        # Try buy box first (index 18)
        if isinstance(csv, list) and len(csv) > 18:
            bb_data = csv[18]
            if bb_data and len(bb_data) > 1:
                price_cents = bb_data[-1]  # Last value = current

        # Fallback to Amazon price (index 0)
        if not price_cents or price_cents <= 0:
            if isinstance(csv, list) and len(csv) > 0:
                amz_data = csv[0]
                if amz_data and len(amz_data) > 1:
                    price_cents = amz_data[-1]

        # Fallback to New FBA (index 10)
        if not price_cents or price_cents <= 0:
            if isinstance(csv, list) and len(csv) > 10:
                fba_data = csv[10]
                if fba_data and len(fba_data) > 1:
                    price_cents = fba_data[-1]

        # Convert cents to dollars
        price = (price_cents / 100.0) if price_cents and price_cents > 0 else 0

        # Sales Rank (index 3)
        bsr = 0
        if isinstance(csv, list) and len(csv) > 3:
            bsr_data = csv[3]
            if bsr_data and len(bsr_data) > 1:
                bsr = bsr_data[-1] if bsr_data[-1] != -1 else 0

        # Calculate revenue proxy
        revenue_proxy = monthly_sold * price

        # Get image
        images_csv = product.get("imagesCSV", "")
        main_image = None
        if images_csv:
            images = images_csv.split(",")
            if images:
                main_image = f"https://m.media-amazon.com/images/I/{images[0]}"

        records.append({
            "asin": asin,
            "title": title,
            "price": price,
            "monthly_units": monthly_sold,
            "revenue_proxy": revenue_proxy,
            "bsr": bsr,
            "main_image": main_image
        })

    df = pd.DataFrame(records)
    return df.sort_values("revenue_proxy", ascending=False).reset_index(drop=True)


def prune_to_90_percent(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    The 90% Revenue Logic (The "Pruning" Algorithm).

    Algorithm:
    1. Sort products by revenue_proxy descending
    2. Calculate cumulative revenue percentage
    3. Find the cutoff index where cumsum reaches 90%
    4. Return only the ASINs that make up the first 90% of revenue

    Returns:
        - pruned_df: DataFrame with only the top revenue-driving ASINs
        - stats: Dict with metrics about the pruning
    """
    if df.empty or "revenue_proxy" not in df.columns:
        return df, {}

    # Ensure sorted by revenue descending
    df = df.sort_values("revenue_proxy", ascending=False).reset_index(drop=True)

    # Calculate cumulative revenue percentage
    total_revenue = df["revenue_proxy"].sum()
    df["cumulative_revenue"] = df["revenue_proxy"].cumsum()
    df["cumulative_pct"] = (df["cumulative_revenue"] / total_revenue) * 100

    # Find 90% cutoff index
    cutoff_idx = df[df["cumulative_pct"] >= 90.0].index.min()

    if pd.isna(cutoff_idx):
        cutoff_idx = len(df) - 1  # Fallback: take all if calculation fails

    # Prune
    pruned_df = df.iloc[:cutoff_idx + 1].copy()

    # Generate stats
    stats = {
        "total_asins": len(df),
        "pruned_asins": len(pruned_df),
        "pruned_pct": (len(pruned_df) / len(df)) * 100,
        "revenue_captured_pct": pruned_df["cumulative_pct"].iloc[-1],
        "total_revenue_proxy": total_revenue,
        "pruned_revenue_proxy": pruned_df["revenue_proxy"].sum()
    }

    return pruned_df, stats


def execute_market_discovery(
    query: str,
    search_type: str = "brand",
    limit: int = 500
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full discovery pipeline: Search → Calculate → Prune.

    Returns:
        - market_snapshot: Pruned DataFrame ready for visualization
        - stats: Metrics about the discovery (for UI display)
    """
    # Step 1: Search Keepa
    products = search_keepa_market(query, search_type, limit)

    if not products:
        return pd.DataFrame(), {"error": "No products found"}

    # Step 2: Calculate Revenue Proxy
    df_full = calculate_revenue_proxy(products)

    # Step 3: Prune to 90%
    market_snapshot, stats = prune_to_90_percent(df_full)

    # Add search metadata
    stats["query"] = query
    stats["search_type"] = search_type

    return market_snapshot, stats
