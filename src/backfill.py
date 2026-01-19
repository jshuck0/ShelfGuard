"""
ShelfGuard Historical Backfill Engine
=======================================
Phase 3: Day 1 Historical Data Population

This module ensures new projects never show empty charts by:
1. Fetching 3 months (90 days) of historical Price & BSR data from Keepa
2. Converting Keepa minutes to Unix timestamps
3. Batch-inserting into historical_metrics table
"""

import pandas as pd
import numpy as np
import streamlit as st
from supabase import Client
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
import os
from threading import Thread


# Keepa API Configuration
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


KEEPA_BASE_URL = "https://api.keepa.com"

# Keepa CSV index constants
IDX_AMAZON = 0
IDX_NEW_FBA = 10
IDX_SALES_RANK = 3
IDX_BUY_BOX = 18


def keepa_minutes_to_unix(keepa_minutes: int) -> int:
    """
    Convert Keepa Minutes to Unix Timestamp (milliseconds).

    Keepa Formula: Unix Time (ms) = (KeepaMinutes + 21,564,000) × 60,000

    Args:
        keepa_minutes: Keepa time value (integer)

    Returns:
        Unix timestamp in milliseconds
    """
    return (keepa_minutes + 21_564_000) * 60_000


def keepa_minutes_to_datetime(keepa_minutes: int) -> datetime:
    """
    Convert Keepa Minutes to Python datetime object.

    Args:
        keepa_minutes: Keepa time value

    Returns:
        datetime object (timezone-naive UTC)
    """
    unix_ms = keepa_minutes_to_unix(keepa_minutes)
    return datetime.utcfromtimestamp(unix_ms / 1000.0)


def fetch_90day_history(asins: List[str], domain: int = 1, days: int = 90) -> List[Dict]:
    """
    Fetch historical data for a list of ASINs from Keepa.

    Args:
        asins: List of ASIN strings (max ~100 per API call)
        domain: Amazon marketplace (1 = US)
        days: Number of days of history to fetch (default: 90 days = ~3 months)

    Returns:
        List of product dictionaries with full csv history

    Note: This is a HEAVY call - use sparingly and batch ASINs
    """
    api_key = get_keepa_api_key()
    if not api_key:
        raise ValueError("KEEPA_API_KEY not found")

    # Batch ASINs (Keepa limit: 100 ASINs per request)
    batch_size = 100
    all_products = []

    for i in range(0, len(asins), batch_size):
        batch = asins[i:i + batch_size]
        asin_str = ",".join(batch)

        params = {
            "key": api_key,
            "domain": domain,
            "asin": asin_str,
            "history": 1,  # Request full history
            "days": days,  # Number of days of history (default: 90 = 3 months)
            "stats": 0,    # Don't need stats (we have history)
        }

        try:
            response = requests.get(
                f"{KEEPA_BASE_URL}/product",
                params=params,
                timeout=120  # Historical calls take longer
            )
            response.raise_for_status()
            data = response.json()

            products = data.get("products", [])
            all_products.extend(products)

        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Keepa backfill batch failed: {str(e)}")
            continue

    return all_products


def parse_historical_timeseries(
    product: Dict,
    csv_index: int,
    metric_name: str
) -> pd.DataFrame:
    """
    Extract a single time series from Keepa CSV data.

    Args:
        product: Keepa product dictionary
        csv_index: Index in csv array (e.g., 3 for BSR, 18 for Buy Box)
        metric_name: Column name for output (e.g., "sales_rank", "price")

    Returns:
        DataFrame with columns: [datetime, asin, metric_name]
    """
    asin = product.get("asin", "UNK")
    csv_data = product.get("csv", {})

    # Extract raw data
    try:
        raw_list = csv_data[csv_index] if isinstance(csv_data, list) else csv_data.get(str(csv_index)) or csv_data.get(csv_index)
    except (IndexError, KeyError, TypeError):
        return pd.DataFrame()

    if not raw_list or len(raw_list) == 0:
        return pd.DataFrame()

    # Ensure even length (pairs of [time, value])
    if len(raw_list) % 2 != 0:
        raw_list = raw_list[:-1]

    # Reshape to [time, value] pairs
    arr = np.array(raw_list, dtype=float).reshape(-1, 2)
    times = arr[:, 0]
    values = arr[:, 1]

    # Convert Keepa minutes to datetime
    datetimes = [keepa_minutes_to_datetime(int(t)) for t in times]

    # Build DataFrame
    df = pd.DataFrame({
        "datetime": datetimes,
        "asin": asin,
        metric_name: values
    })

    # Handle Keepa's -1 (no data) values
    df[metric_name] = df[metric_name].replace([-1, -2], np.nan)

    # Price normalization (cents → dollars)
    if "price" in metric_name:
        df[metric_name] = df[metric_name] / 100.0

    return df.dropna(subset=[metric_name])


def build_historical_metrics(products: List[Dict]) -> pd.DataFrame:
    """
    Parse all products and build a unified historical_metrics table.

    Tracks:
    - sales_rank (BSR)
    - buy_box_price
    - amazon_price
    - new_fba_price

    Returns:
        DataFrame with columns: [datetime, asin, sales_rank, buy_box_price, amazon_price, new_fba_price]
    """
    all_frames = []

    for product in products:
        asin = product.get("asin")
        if not asin:
            continue

        # Extract each time series
        bsr_df = parse_historical_timeseries(product, IDX_SALES_RANK, "sales_rank")
        bb_df = parse_historical_timeseries(product, IDX_BUY_BOX, "buy_box_price")
        amz_df = parse_historical_timeseries(product, IDX_AMAZON, "amazon_price")
        fba_df = parse_historical_timeseries(product, IDX_NEW_FBA, "new_fba_price")

        # Merge all metrics on datetime + asin
        frames_to_merge = [bsr_df, bb_df, amz_df, fba_df]
        frames_to_merge = [f for f in frames_to_merge if not f.empty]

        if not frames_to_merge:
            continue

        merged = frames_to_merge[0]
        for frame in frames_to_merge[1:]:
            merged = merged.merge(frame, on=["datetime", "asin"], how="outer")

        all_frames.append(merged)

    if not all_frames:
        return pd.DataFrame()

    # Concatenate all ASINs
    df_full = pd.concat(all_frames, ignore_index=True)

    # Sort by datetime
    df_full = df_full.sort_values(["asin", "datetime"]).reset_index(drop=True)

    # === PRICE AND BSR INTERPOLATION (similar to keepa_client.py) ===
    # Constants for interpolation limits (in data points - covers ~4 weeks even with sparse data)
    MAX_PRICE_FFILL_LIMIT = 50  # Forward fill prices up to ~4 weeks
    MAX_RANK_GAP_LIMIT = 30     # Interpolate BSR gaps up to ~3 weeks

    # Calculate effective price (fallback chain: buy_box → amazon → new_fba)
    if "buy_box_price" in df_full.columns or "amazon_price" in df_full.columns or "new_fba_price" in df_full.columns:
        df_full["eff_p"] = (
            df_full.get("buy_box_price", pd.Series(dtype=float))
            .fillna(df_full.get("amazon_price", pd.Series(dtype=float)))
            .fillna(df_full.get("new_fba_price", pd.Series(dtype=float)))
        )
        
        # Forward fill price (limit covers ~4 weeks of data)
        df_full["filled_price"] = df_full.groupby("asin")["eff_p"].ffill(limit=MAX_PRICE_FFILL_LIMIT)
    
    # Interpolate BSR (sales rank)
    if "sales_rank" in df_full.columns:
        # Use interpolation to fill gaps (limit covers ~3 weeks of data)
        df_full["sales_rank_filled"] = df_full.groupby("asin")["sales_rank"].transform(
            lambda x: x.interpolate(method='linear', limit=MAX_RANK_GAP_LIMIT) if len(x) > 1 else x
        )
        # Fallback: if interpolation didn't fill all values, forward fill remaining
        df_full["sales_rank_filled"] = df_full.groupby("asin")["sales_rank_filled"].ffill()

    return df_full


def upsert_historical_metrics(
    df: pd.DataFrame,
    project_id: str,
    supabase: Client
) -> int:
    """
    Batch-insert historical metrics into Supabase.

    Args:
        df: DataFrame from build_historical_metrics()
        project_id: UUID of the project
        supabase: Supabase client

    Returns:
        Number of records inserted

    Database Schema Required:
        Table: historical_metrics
        Columns:
            - id (uuid, primary key)
            - project_id (uuid, foreign key)
            - asin (text)
            - datetime (timestamp)
            - sales_rank (int)
            - buy_box_price (float)
            - amazon_price (float)
            - new_fba_price (float)
        Indexes:
            - (project_id, asin, datetime) for fast lookups
    """
    if df.empty:
        return 0

    # Add project_id column
    df["project_id"] = project_id

    # Convert datetime to ISO string for Supabase
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert to records
    records = df.to_dict(orient="records")

    # Batch insert (chunks of 500 to avoid timeouts)
    chunk_size = 500
    total_inserted = 0

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]

        try:
            supabase.table("historical_metrics").upsert(
                chunk,
                on_conflict="project_id,asin,datetime"  # Avoid duplicates
            ).execute()

            total_inserted += len(chunk)

        except Exception as e:
            st.warning(f"⚠️ Batch insert failed at {i}: {str(e)}")
            continue

    return total_inserted


def execute_backfill(project_id: str, asins: List[str], run_async: bool = True, days: int = 90) -> None:
    """
    Main backfill orchestrator.

    Workflow:
    1. Fetch Keepa history for all ASINs (default: 3 months / 90 days)
    2. Parse into historical_metrics format
    3. Batch-insert into Supabase

    Args:
        project_id: UUID of the new project
        asins: List of ASINs to backfill
        run_async: If True, runs in background thread (non-blocking)
        days: Number of days of history to fetch (default: 90 days = 3 months)

    Performance: ~5-10 seconds for 100 ASINs (depends on Keepa API latency)
    """
    from supabase import create_client

    def _backfill_task():
        try:
            # Fetch historical data (3 months default for User Dashboard)
            products = fetch_90day_history(asins, days=days)

            if not products:
                st.warning("⚠️ No historical data fetched from Keepa")
                return

            # Parse metrics
            df_metrics = build_historical_metrics(products)

            # Insert into DB
            supabase = create_client(st.secrets["url"], st.secrets["key"])
            inserted_count = upsert_historical_metrics(df_metrics, project_id, supabase)

            st.success(f"✅ Backfilled {inserted_count:,} historical data points")

        except Exception as e:
            st.error(f"❌ Backfill failed: {str(e)}")

    if run_async:
        # Run in background thread to keep UI responsive
        thread = Thread(target=_backfill_task, daemon=True)
        thread.start()
    else:
        # Run synchronously (for testing/debugging)
        _backfill_task()
