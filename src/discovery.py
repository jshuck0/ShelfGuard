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
from typing import Dict, List, Tuple, Optional
import requests

# Keepa API Configuration
KEEPA_API_KEY = os.getenv("KEEPA_API_KEY")
KEEPA_BASE_URL = "https://api.keepa.com"


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
    if search_type == "brand":
        # Search by brand name
        params = {
            "key": KEEPA_API_KEY,
            "domain": domain,
            "brand": query,
            "sort": "current_SALES desc",  # Sort by current sales rank (best sellers first)
            "range": f"0-{limit}",
            "stats": 90,  # Request 90-day stats
            "buybox": 1,  # Include buy box data
            "history": 0,  # Don't fetch full history yet (save tokens)
        }
    else:  # category search
        params = {
            "key": KEEPA_API_KEY,
            "domain": domain,
            "category": query,
            "sort": "current_SALES desc",
            "range": f"0-{limit}",
            "stats": 90,
            "buybox": 1,
            "history": 0,
        }

    try:
        response = requests.get(
            f"{KEEPA_BASE_URL}/product",
            params=params,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        return data.get("products", [])

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Keepa API Error: {str(e)}")
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
