"""
Daily Panel Derivation — Market Misattribution Shield
======================================================
Derives a daily ASIN-day panel from the existing `historical_metrics` Supabase table.
No new Keepa API calls required.

Why:
- Discount persistence requires days/7 granularity
- The weekly panel smooths out intra-week pricing moves
- `historical_metrics` stores raw hourly-ish Keepa points — we bucket to calendar days

Fidelity levels:
    "daily"   — historical_metrics table is populated → true day-level
    "weekly"  — historical_metrics empty → use weekly panel as proxy
    "none"    — no data at all

Usage:
    from data.daily_panel import get_daily_panel, get_discount_persistence
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def get_daily_panel(
    project_id: Optional[str] = None,
    asins: Optional[List[str]] = None,
    days: int = 56,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch or derive a daily ASIN-day panel.

    Args:
        project_id: Supabase project_id (used to query historical_metrics)
        asins: Optional list of ASINs to filter (None = all)
        days: How many days of history to return (default 56 = 8 weeks)

    Returns:
        (df_daily, fidelity) where fidelity is "daily" | "weekly" | "none"
        df_daily columns: asin, date, price, sales_rank, buy_box_price
    """
    # Try Option A: query existing historical_metrics table
    df_daily, fidelity = _try_historical_metrics(project_id, asins, days)

    if fidelity == "daily" and not df_daily.empty:
        return df_daily, fidelity

    # Option B: fall back — caller should use weekly panel proxy
    return pd.DataFrame(), "weekly"


def _try_historical_metrics(
    project_id: Optional[str],
    asins: Optional[List[str]],
    days: int,
) -> tuple[pd.DataFrame, str]:
    """
    Query the `historical_metrics` Supabase table and bucket to daily.

    Returns (df_daily, "daily") if data found, (empty, "weekly") otherwise.
    """
    try:
        from src.supabase_reader import get_supabase_client
        supabase = get_supabase_client()
        if supabase is None:
            return pd.DataFrame(), "weekly"
    except (ImportError, Exception):
        return pd.DataFrame(), "weekly"

    try:
        since = (datetime.now() - timedelta(days=days)).isoformat()

        query = (
            supabase.table("historical_metrics")
            .select("asin,datetime,sales_rank_filled,filled_price,buy_box_price")
            .gte("datetime", since)
        )
        if project_id:
            query = query.eq("project_id", project_id)
        if asins:
            query = query.in_("asin", asins[:100])  # Supabase IN limit

        response = query.execute()
        rows = response.data if response and hasattr(response, "data") else []

        if not rows:
            return pd.DataFrame(), "weekly"

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        if df.empty:
            return pd.DataFrame(), "weekly"

        # Bucket to calendar day
        df["date"] = df["datetime"].dt.date
        df["date"] = pd.to_datetime(df["date"])

        # Aggregate to daily: price = last of day, rank = mean of day
        agg = df.groupby(["asin", "date"]).agg(
            price=("filled_price", "last"),
            sales_rank=("sales_rank_filled", "mean"),
            buy_box_price=("buy_box_price", "last"),
        ).reset_index()

        if agg.empty:
            return pd.DataFrame(), "weekly"

        # Forward-fill price gaps within each ASIN (max 4 days)
        agg = agg.sort_values(["asin", "date"])
        agg["price"] = agg.groupby("asin")["price"].transform(
            lambda x: x.ffill(limit=4).bfill(limit=4)
        )

        return agg, "daily"

    except Exception:
        return pd.DataFrame(), "weekly"


# ─── DISCOUNT PERSISTENCE ────────────────────────────────────────────────────

def get_discount_persistence(
    df_daily: pd.DataFrame,
    base_prices: Optional[Dict[str, float]] = None,
    discount_threshold: float = 0.97,
    window_days: int = 7,
) -> Dict[str, float]:
    """
    Compute discount persistence (days discounted / 7) for each ASIN.

    Args:
        df_daily: Daily panel (asin, date, price columns required)
        base_prices: Dict of asin → base price. If None, uses 28-day median from df_daily.
        discount_threshold: Price below this fraction of base = discounted (default 0.97 = 3% off)
        window_days: Lookback window for persistence (default 7 = last week)

    Returns:
        Dict of asin → discount_persistence (0.0 to 1.0)
    """
    if df_daily.empty or "price" not in df_daily.columns:
        return {}

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
    recent = df_daily[df_daily["date"] >= cutoff] if "date" in df_daily.columns else df_daily

    # Compute base prices if not provided
    if base_prices is None:
        base_cutoff = pd.Timestamp.now() - pd.Timedelta(days=28)
        base_df = df_daily[df_daily["date"] < base_cutoff] if "date" in df_daily.columns else df_daily
        base_prices = base_df.groupby("asin")["price"].median().to_dict()

    result = {}
    for asin, group in recent.groupby("asin"):
        base_p = base_prices.get(asin)
        if not base_p or base_p == 0:
            # Fallback: use this ASIN's own median in the daily panel
            base_p = df_daily[df_daily["asin"] == asin]["price"].median()

        if not base_p or base_p == 0:
            result[asin] = 0.0
            continue

        threshold_price = base_p * discount_threshold
        days_total = len(group)
        days_discounted = (group["price"] < threshold_price).sum()
        result[asin] = round(days_discounted / window_days, 2) if window_days > 0 else 0.0

    return result


# ─── WEEKLY PROXY ─────────────────────────────────────────────────────────────

def get_weekly_discount_proxy(
    df_weekly: pd.DataFrame,
    discount_threshold: float = 0.97,
    lookback_weeks: int = 2,
) -> Dict[str, float]:
    """
    Fallback when daily data is unavailable.
    Estimates discount persistence from weekly panel as weeks_discounted / lookback_weeks.
    Labels fidelity as LOW.

    Returns: Dict of asin → proxy persistence (0.0 to 1.0)
    """
    if df_weekly.empty:
        return {}

    price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    if price_col not in df_weekly.columns:
        return {}

    latest_week = df_weekly["week_start"].max()
    recent_weeks = sorted(df_weekly["week_start"].unique())[-lookback_weeks:]
    recent = df_weekly[df_weekly["week_start"].isin(recent_weeks)]

    # Per-ASIN base = median price in weeks before the recent window
    older = df_weekly[~df_weekly["week_start"].isin(recent_weeks)]
    base_prices = older.groupby("asin")[price_col].median().to_dict()

    result = {}
    for asin, group in recent.groupby("asin"):
        base_p = base_prices.get(asin) or group[price_col].median()
        if not base_p or base_p == 0:
            result[asin] = 0.0
            continue
        discounted_weeks = (group[price_col] < base_p * discount_threshold).sum()
        result[asin] = round(discounted_weeks / lookback_weeks, 2)

    return result


def get_discount_data(
    df_weekly: pd.DataFrame,
    project_id: Optional[str] = None,
    asins: Optional[List[str]] = None,
    discount_threshold: float = 0.97,
) -> tuple[Dict[str, float], str]:
    """
    Convenience wrapper: try daily panel first, fall back to weekly proxy.

    Returns:
        (persistence_map, fidelity) where fidelity is "daily" | "weekly"
    """
    asin_list = asins or (list(df_weekly["asin"].unique()) if "asin" in df_weekly.columns else None)

    df_daily, fidelity = get_daily_panel(project_id=project_id, asins=asin_list, days=56)

    if fidelity == "daily" and not df_daily.empty:
        persistence = get_discount_persistence(df_daily, discount_threshold=discount_threshold)
        return persistence, "daily"

    # Weekly fallback
    persistence = get_weekly_discount_proxy(df_weekly, discount_threshold=discount_threshold)
    return persistence, "weekly"
