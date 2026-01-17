"""
ShelfGuard Outcome Triangulation Engine
=========================================
Phase 4: Resolution Cards & Actionable Alerts

This module generates proactive recommendations based on:
1. Historical velocity analysis (from historical_metrics)
2. Competitive dynamics (price/BSR changes)
3. Mission profile priorities

Alert Types:
- Volume Stealer: Competitor with rapid BSR improvement + price drop
- Efficiency Gap: Top ASINs with review count below category average
- Buy Box Loss: Declining Buy Box share
- New Entrant: New ASINs entering top 100
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from supabase import Client


def calculate_bsr_velocity(
    df_metrics: pd.DataFrame,
    lookback_days: int = 7
) -> pd.DataFrame:
    """
    Calculate 7-day BSR improvement percentage.

    Formula: BSR Velocity = (BSR_7d_ago - BSR_now) / BSR_7d_ago

    Positive velocity = Rank improved (lower number)
    Negative velocity = Rank worsened (higher number)

    Args:
        df_metrics: Historical metrics DataFrame (from historical_metrics table)
        lookback_days: Days to look back for comparison

    Returns:
        DataFrame with columns: [asin, bsr_now, bsr_7d_ago, bsr_velocity_pct]
    """
    if df_metrics.empty or "sales_rank" not in df_metrics.columns:
        return pd.DataFrame()

    # Convert datetime
    df_metrics["datetime"] = pd.to_datetime(df_metrics["datetime"])

    # Get current BSR (most recent)
    now = df_metrics["datetime"].max()
    df_now = df_metrics[df_metrics["datetime"] == now][["asin", "sales_rank"]].copy()
    df_now.rename(columns={"sales_rank": "bsr_now"}, inplace=True)

    # Get BSR from 7 days ago
    cutoff = now - timedelta(days=lookback_days)
    df_7d = df_metrics[df_metrics["datetime"] <= cutoff].copy()
    df_7d = df_7d.sort_values("datetime").groupby("asin").last().reset_index()
    df_7d = df_7d[["asin", "sales_rank"]].rename(columns={"sales_rank": "bsr_7d_ago"})

    # Merge
    df_velocity = df_now.merge(df_7d, on="asin", how="inner")

    # Calculate velocity
    df_velocity["bsr_velocity_pct"] = (
        (df_velocity["bsr_7d_ago"] - df_velocity["bsr_now"]) /
        df_velocity["bsr_7d_ago"]
    ) * 100

    return df_velocity


def calculate_price_delta(
    df_metrics: pd.DataFrame,
    lookback_days: int = 7
) -> pd.DataFrame:
    """
    Calculate 7-day price change percentage.

    Args:
        df_metrics: Historical metrics DataFrame
        lookback_days: Days to look back

    Returns:
        DataFrame with columns: [asin, price_now, price_7d_ago, price_delta_pct]
    """
    if df_metrics.empty:
        return pd.DataFrame()

    # Use buy_box_price as primary, fallback to amazon_price
    df_metrics["effective_price"] = df_metrics["buy_box_price"].fillna(
        df_metrics["amazon_price"]
    ).fillna(df_metrics["new_fba_price"])

    # Convert datetime
    df_metrics["datetime"] = pd.to_datetime(df_metrics["datetime"])

    # Current price
    now = df_metrics["datetime"].max()
    df_now = df_metrics[df_metrics["datetime"] == now][["asin", "effective_price"]].copy()
    df_now.rename(columns={"effective_price": "price_now"}, inplace=True)

    # Price 7 days ago
    cutoff = now - timedelta(days=lookback_days)
    df_7d = df_metrics[df_metrics["datetime"] <= cutoff].copy()
    df_7d = df_7d.sort_values("datetime").groupby("asin").last().reset_index()
    df_7d = df_7d[["asin", "effective_price"]].rename(columns={"effective_price": "price_7d_ago"})

    # Merge
    df_price_delta = df_now.merge(df_7d, on="asin", how="inner")

    # Calculate delta
    df_price_delta["price_delta_pct"] = (
        (df_price_delta["price_now"] - df_price_delta["price_7d_ago"]) /
        df_price_delta["price_7d_ago"]
    ) * 100

    return df_price_delta


def detect_volume_stealers(
    df_metrics: pd.DataFrame,
    bsr_threshold: float = 20.0,
    price_threshold: float = -10.0
) -> List[Dict]:
    """
    Identify "Volume Stealer" ASINs.

    Criteria:
    - BSR improved by >20% in 7 days (velocity > 20%)
    - Price dropped by >10% in same period (delta < -10%)

    Returns:
        List of alert dictionaries with ASIN, metrics, and severity
    """
    df_bsr = calculate_bsr_velocity(df_metrics)
    df_price = calculate_price_delta(df_metrics)

    if df_bsr.empty or df_price.empty:
        return []

    # Merge BSR and price data
    df_combined = df_bsr.merge(df_price, on="asin", how="inner")

    # Filter for volume stealers
    stealers = df_combined[
        (df_combined["bsr_velocity_pct"] > bsr_threshold) &
        (df_combined["price_delta_pct"] < price_threshold)
    ]

    # Build alerts
    alerts = []
    for _, row in stealers.iterrows():
        alerts.append({
            "type": "volume_stealer",
            "severity": "high",
            "asin": row["asin"],
            "title": "Volume Stealer Alert",
            "message": (
                f"Competitor ASIN {row['asin']} is stealing market share. "
                f"BSR improved {row['bsr_velocity_pct']:.1f}% while dropping price {abs(row['price_delta_pct']):.1f}%."
            ),
            "bsr_now": row["bsr_now"],
            "bsr_velocity": row["bsr_velocity_pct"],
            "price_delta": row["price_delta_pct"],
            "action": "🎯 INVESTIGATE: Review pricing strategy or increase ad spend"
        })

    return alerts


def detect_efficiency_gaps(
    df_current: pd.DataFrame,
    review_gap_threshold: float = 0.50
) -> List[Dict]:
    """
    Identify "Efficiency Gap" ASINs.

    Criteria:
    - ASIN is in Top 20 by revenue
    - Review count < 50% of category average

    Args:
        df_current: Current week snapshot (from run_weekly_analysis)
        review_gap_threshold: Minimum gap to trigger alert (default 50%)

    Returns:
        List of alert dictionaries
    """
    if df_current.empty or "review_count" not in df_current.columns:
        return []

    # Calculate category average reviews
    avg_reviews = df_current["review_count"].mean()

    # Filter Top 20 by revenue
    top_20 = df_current.nlargest(20, "weekly_sales_filled")

    # Find gaps
    gaps = top_20[top_20["review_count"] < (avg_reviews * review_gap_threshold)]

    alerts = []
    for _, row in gaps.iterrows():
        gap_pct = ((avg_reviews - row["review_count"]) / avg_reviews) * 100

        alerts.append({
            "type": "efficiency_gap",
            "severity": "medium",
            "asin": row["asin"],
            "title": "Review Gap Opportunity",
            "message": (
                f"ASIN {row['asin']} has {int(row['review_count'])} reviews, "
                f"{gap_pct:.0f}% below category average ({int(avg_reviews)})."
            ),
            "review_count": row["review_count"],
            "category_avg": avg_reviews,
            "gap_pct": gap_pct,
            "action": "📦 ACTION: Launch Vine campaign or email follow-up"
        })

    return alerts


def detect_buybox_loss(
    df_metrics: pd.DataFrame,
    buybox_threshold: float = 0.50
) -> List[Dict]:
    """
    Identify ASINs with declining Buy Box share.

    Criteria:
    - Buy Box share dropped below 50% in last 7 days

    Args:
        df_metrics: Historical metrics with Buy Box data
        buybox_threshold: Minimum share to trigger alert

    Returns:
        List of alert dictionaries
    """
    # TODO: Implement Buy Box share tracking in historical_metrics
    # Placeholder for now
    return []


def detect_new_entrants(
    df_metrics: pd.DataFrame,
    bsr_threshold: int = 50000
) -> List[Dict]:
    """
    Identify new ASINs entering the market with strong BSR.

    Criteria:
    - ASIN first appeared in last 7 days
    - Current BSR < 50,000 (top performers)

    Args:
        df_metrics: Historical metrics
        bsr_threshold: Max BSR to qualify as "strong entrant"

    Returns:
        List of alert dictionaries
    """
    if df_metrics.empty:
        return []

    # Convert datetime
    df_metrics["datetime"] = pd.to_datetime(df_metrics["datetime"])

    # Group by ASIN and find first appearance
    first_seen = df_metrics.groupby("asin")["datetime"].min().reset_index()
    first_seen.rename(columns={"datetime": "first_seen"}, inplace=True)

    # Filter for ASINs first seen in last 7 days
    cutoff = df_metrics["datetime"].max() - timedelta(days=7)
    new_asins = first_seen[first_seen["first_seen"] >= cutoff]

    # Get current BSR for these ASINs
    latest = df_metrics.sort_values("datetime").groupby("asin").last().reset_index()
    new_asins = new_asins.merge(latest[["asin", "sales_rank"]], on="asin", how="inner")

    # Filter for strong performers
    strong_entrants = new_asins[new_asins["sales_rank"] < bsr_threshold]

    alerts = []
    for _, row in strong_entrants.iterrows():
        alerts.append({
            "type": "new_entrant",
            "severity": "medium",
            "asin": row["asin"],
            "title": "New Competitor Alert",
            "message": (
                f"New ASIN {row['asin']} entered market with BSR {int(row['sales_rank']):,}. "
                f"First seen {row['first_seen'].strftime('%Y-%m-%d')}."
            ),
            "bsr": row["sales_rank"],
            "first_seen": row["first_seen"],
            "action": "🔍 MONITOR: Add to watch list and track pricing"
        })

    return alerts


def generate_resolution_cards(
    df_metrics: pd.DataFrame,
    df_current: pd.DataFrame,
    mission_type: str = "bodyguard"
) -> List[Dict]:
    """
    Master orchestrator for all alert types.

    Combines all detection logic and prioritizes based on mission profile.

    Args:
        df_metrics: Historical metrics from historical_metrics table
        df_current: Current week snapshot
        mission_type: Mission profile ("bodyguard", "scout", "surgeon")

    Returns:
        List of resolution cards, sorted by priority
    """
    from src.persistence import get_mission_profile_config

    # Get mission profile config
    config = get_mission_profile_config(mission_type)
    priorities = config["priorities"]

    # Run all detectors
    all_alerts = []

    all_alerts.extend(detect_volume_stealers(df_metrics))
    all_alerts.extend(detect_efficiency_gaps(df_current))
    all_alerts.extend(detect_new_entrants(df_metrics))
    # all_alerts.extend(detect_buybox_loss(df_metrics))  # TODO: Enable when BB data available

    # Score each alert based on mission profile
    for alert in all_alerts:
        alert_type = alert["type"]
        priority_weight = priorities.get(alert_type, 0.5)

        # Calculate final score (severity + priority)
        severity_score = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(alert["severity"], 0.5)
        alert["priority_score"] = severity_score * priority_weight

    # Sort by priority
    all_alerts = sorted(all_alerts, key=lambda x: x["priority_score"], reverse=True)

    return all_alerts


def render_resolution_card(alert: Dict) -> None:
    """
    Render a single resolution card using Streamlit.

    Uses st.container with custom styling for visual consistency.

    Args:
        alert: Alert dictionary from generate_resolution_cards()
    """
    import streamlit as st

    # Color scheme by severity
    color_map = {
        "high": "#dc3545",    # Red
        "medium": "#ffc107",  # Yellow
        "low": "#6c757d"      # Gray
    }

    border_color = color_map.get(alert["severity"], "#6c757d")

    with st.container():
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 1px solid #e0e0e0;
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            ">
                <div style="font-weight: 700; font-size: 1rem; color: #1a1a1a; margin-bottom: 8px;">
                    {alert['title']}
                </div>
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 12px;">
                    {alert['message']}
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; color: {border_color};">
                    {alert['action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
