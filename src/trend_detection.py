"""
ShelfGuard Trend Detection Engine
==================================
The Oracle's Intelligence Layer

This module analyzes time-series snapshots to detect:
- Price movements (undercuts, spikes)
- Rank velocity (acceleration, deceleration)
- Buy Box changes (loss, gain)
- Competitive pressure signals

Designed to run as a background job and generate alerts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendSignal(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # Immediate action required
    WARNING = "warning"    # Attention needed
    INFO = "info"          # Informational only
    POSITIVE = "positive"  # Good news


class AlertType(Enum):
    """Types of alerts the system can generate."""
    PRICE_UNDERCUT = "price_undercut"
    PRICE_SPIKE = "price_spike"
    RANK_DECAY = "rank_decay"
    RANK_SURGE = "rank_surge"
    BUYBOX_LOSS = "buybox_loss"
    BUYBOX_GAIN = "buybox_gain"
    COMPETITOR_SURGE = "competitor_surge"
    REVIEW_VELOCITY = "review_velocity"


@dataclass
class TrendAlert:
    """Represents a detected trend or anomaly."""
    asin: str
    alert_type: AlertType
    signal: TrendSignal
    title: str
    message: str
    metric_current: float
    metric_previous: float
    change_pct: float
    detected_at: datetime
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "asin": self.asin,
            "alert_type": self.alert_type.value,
            "severity": self.signal.value,
            "title": self.title,
            "message": self.message,
            "metric_current": self.metric_current,
            "metric_previous": self.metric_previous,
            "change_pct": self.change_pct,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata or {}
        }


# Threshold configuration
THRESHOLDS = {
    "price_undercut_pct": -5.0,     # Alert if price drops 5%+
    "price_spike_pct": 10.0,         # Alert if price spikes 10%+
    "rank_decay_pct": -20.0,         # Alert if rank worsens 20%+ (higher number = worse)
    "rank_surge_pct": 20.0,          # Alert if rank improves 20%+
    "buybox_loss_pct": -15.0,        # Alert if BB share drops 15+ points
    "buybox_gain_pct": 10.0,         # Alert if BB share gains 10+ points
    "competitor_surge_count": 3,     # Alert if 3+ new competitors
    "review_velocity_pct": 50.0,     # Alert if review velocity changes 50%+
}


def detect_price_trends(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> List[TrendAlert]:
    """
    Detect price-related trends.
    
    Args:
        current_df: Today's snapshot data
        previous_df: Yesterday's (or previous) snapshot data
        
    Returns:
        List of price-related alerts
    """
    alerts = []
    
    if current_df.empty or previous_df.empty:
        return alerts
    
    # Merge on ASIN to compare
    merged = current_df.merge(
        previous_df[["asin", "buy_box_price", "filled_price"]],
        on="asin",
        suffixes=("_current", "_prev"),
        how="inner"
    )
    
    price_col_curr = "buy_box_price_current" if "buy_box_price_current" in merged.columns else "filled_price_current"
    price_col_prev = "buy_box_price_prev" if "buy_box_price_prev" in merged.columns else "filled_price_prev"
    
    for _, row in merged.iterrows():
        curr_price = row.get(price_col_curr)
        prev_price = row.get(price_col_prev)
        
        if not curr_price or not prev_price or prev_price == 0:
            continue
        
        change_pct = (curr_price - prev_price) / prev_price * 100
        
        # Price undercut detection
        if change_pct <= THRESHOLDS["price_undercut_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.PRICE_UNDERCUT,
                signal=TrendSignal.WARNING,
                title=f"Price Drop: {row.get('title', row['asin'])[:40]}",
                message=f"Price dropped {abs(change_pct):.1f}% from ${prev_price:.2f} to ${curr_price:.2f}",
                metric_current=curr_price,
                metric_previous=prev_price,
                change_pct=change_pct,
                detected_at=datetime.now()
            ))
        
        # Price spike detection
        elif change_pct >= THRESHOLDS["price_spike_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.PRICE_SPIKE,
                signal=TrendSignal.INFO,
                title=f"Price Spike: {row.get('title', row['asin'])[:40]}",
                message=f"Price increased {change_pct:.1f}% from ${prev_price:.2f} to ${curr_price:.2f}",
                metric_current=curr_price,
                metric_previous=prev_price,
                change_pct=change_pct,
                detected_at=datetime.now()
            ))
    
    return alerts


def detect_rank_trends(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> List[TrendAlert]:
    """
    Detect sales rank trends (velocity changes).
    
    Note: Lower rank = better. A "decay" means rank number increased (worse).
    """
    alerts = []
    
    if current_df.empty or previous_df.empty:
        return alerts
    
    rank_col = "sales_rank" if "sales_rank" in current_df.columns else "sales_rank_filled"
    
    merged = current_df.merge(
        previous_df[["asin", rank_col]],
        on="asin",
        suffixes=("_current", "_prev"),
        how="inner"
    )
    
    rank_curr_col = f"{rank_col}_current"
    rank_prev_col = f"{rank_col}_prev"
    
    for _, row in merged.iterrows():
        curr_rank = row.get(rank_curr_col)
        prev_rank = row.get(rank_prev_col)
        
        if not curr_rank or not prev_rank or prev_rank == 0:
            continue
        
        # Positive change_pct = improvement (rank went down/better)
        # Negative change_pct = decay (rank went up/worse)
        change_pct = (prev_rank - curr_rank) / prev_rank * 100
        
        # Rank decay (worsening)
        if change_pct <= THRESHOLDS["rank_decay_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.RANK_DECAY,
                signal=TrendSignal.CRITICAL if change_pct <= -40 else TrendSignal.WARNING,
                title=f"Rank Decay: {row.get('title', row['asin'])[:40]}",
                message=f"Sales rank worsened {abs(change_pct):.1f}% (#{int(prev_rank):,} → #{int(curr_rank):,})",
                metric_current=curr_rank,
                metric_previous=prev_rank,
                change_pct=change_pct,
                detected_at=datetime.now()
            ))
        
        # Rank surge (improving)
        elif change_pct >= THRESHOLDS["rank_surge_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.RANK_SURGE,
                signal=TrendSignal.POSITIVE,
                title=f"Rank Surge: {row.get('title', row['asin'])[:40]}",
                message=f"Sales rank improved {change_pct:.1f}% (#{int(prev_rank):,} → #{int(curr_rank):,})",
                metric_current=curr_rank,
                metric_previous=prev_rank,
                change_pct=change_pct,
                detected_at=datetime.now()
            ))
    
    return alerts


def detect_buybox_trends(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> List[TrendAlert]:
    """
    Detect Buy Box ownership changes.
    """
    alerts = []
    
    if current_df.empty or previous_df.empty:
        return alerts
    
    if "amazon_bb_share" not in current_df.columns:
        return alerts
    
    merged = current_df.merge(
        previous_df[["asin", "amazon_bb_share"]],
        on="asin",
        suffixes=("_current", "_prev"),
        how="inner"
    )
    
    for _, row in merged.iterrows():
        curr_bb = row.get("amazon_bb_share_current")
        prev_bb = row.get("amazon_bb_share_prev")
        
        if curr_bb is None or prev_bb is None:
            continue
        
        # Convert to percentage points (0-100 scale)
        curr_bb_pct = curr_bb * 100 if curr_bb <= 1 else curr_bb
        prev_bb_pct = prev_bb * 100 if prev_bb <= 1 else prev_bb
        
        change_pts = curr_bb_pct - prev_bb_pct
        
        # Buy Box loss
        if change_pts <= THRESHOLDS["buybox_loss_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.BUYBOX_LOSS,
                signal=TrendSignal.CRITICAL,
                title=f"Buy Box Loss: {row.get('title', row['asin'])[:40]}",
                message=f"Buy Box share dropped {abs(change_pts):.1f} points ({prev_bb_pct:.0f}% → {curr_bb_pct:.0f}%)",
                metric_current=curr_bb_pct,
                metric_previous=prev_bb_pct,
                change_pct=change_pts,
                detected_at=datetime.now()
            ))
        
        # Buy Box gain
        elif change_pts >= THRESHOLDS["buybox_gain_pct"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.BUYBOX_GAIN,
                signal=TrendSignal.POSITIVE,
                title=f"Buy Box Gain: {row.get('title', row['asin'])[:40]}",
                message=f"Buy Box share increased {change_pts:.1f} points ({prev_bb_pct:.0f}% → {curr_bb_pct:.0f}%)",
                metric_current=curr_bb_pct,
                metric_previous=prev_bb_pct,
                change_pct=change_pts,
                detected_at=datetime.now()
            ))
    
    return alerts


def detect_competitor_trends(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> List[TrendAlert]:
    """
    Detect competitive pressure changes (new offer count).
    """
    alerts = []
    
    if current_df.empty or previous_df.empty:
        return alerts
    
    if "new_offer_count" not in current_df.columns:
        return alerts
    
    merged = current_df.merge(
        previous_df[["asin", "new_offer_count"]],
        on="asin",
        suffixes=("_current", "_prev"),
        how="inner"
    )
    
    for _, row in merged.iterrows():
        curr_offers = row.get("new_offer_count_current") or 0
        prev_offers = row.get("new_offer_count_prev") or 0
        
        new_competitors = curr_offers - prev_offers
        
        if new_competitors >= THRESHOLDS["competitor_surge_count"]:
            alerts.append(TrendAlert(
                asin=row["asin"],
                alert_type=AlertType.COMPETITOR_SURGE,
                signal=TrendSignal.WARNING,
                title=f"Competitor Surge: {row.get('title', row['asin'])[:40]}",
                message=f"+{new_competitors} new sellers entered ({prev_offers} → {curr_offers} total)",
                metric_current=curr_offers,
                metric_previous=prev_offers,
                change_pct=new_competitors,
                detected_at=datetime.now()
            ))
    
    return alerts


def run_trend_detection(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame
) -> List[TrendAlert]:
    """
    Run all trend detection algorithms.
    
    Args:
        current_df: Today's snapshot data
        previous_df: Previous day's snapshot data
        
    Returns:
        Combined list of all detected alerts, sorted by severity
    """
    all_alerts = []
    
    # Run each detector
    all_alerts.extend(detect_price_trends(current_df, previous_df))
    all_alerts.extend(detect_rank_trends(current_df, previous_df))
    all_alerts.extend(detect_buybox_trends(current_df, previous_df))
    all_alerts.extend(detect_competitor_trends(current_df, previous_df))
    
    # Sort by severity (critical first)
    severity_order = {
        TrendSignal.CRITICAL: 0,
        TrendSignal.WARNING: 1,
        TrendSignal.INFO: 2,
        TrendSignal.POSITIVE: 3
    }
    
    all_alerts.sort(key=lambda x: severity_order.get(x.signal, 99))
    
    logger.info(f"🔍 Trend detection complete: {len(all_alerts)} alerts generated")
    return all_alerts


def load_and_compare_snapshots(supabase_client, days_back: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load today's and yesterday's snapshots from Supabase.
    
    Args:
        supabase_client: Supabase client
        days_back: Number of days to look back for comparison
        
    Returns:
        Tuple of (current_df, previous_df)
    """
    today = date.today()
    prev_date = today - timedelta(days=days_back)
    
    try:
        # Load today's snapshots
        current_result = supabase_client.table("product_snapshots").select("*").eq(
            "snapshot_date", today.isoformat()
        ).execute()
        
        # Load previous day's snapshots
        prev_result = supabase_client.table("product_snapshots").select("*").eq(
            "snapshot_date", prev_date.isoformat()
        ).execute()
        
        current_df = pd.DataFrame(current_result.data) if current_result.data else pd.DataFrame()
        previous_df = pd.DataFrame(prev_result.data) if prev_result.data else pd.DataFrame()
        
        return current_df, previous_df
        
    except Exception as e:
        logger.error(f"Failed to load snapshots: {e}")
        return pd.DataFrame(), pd.DataFrame()


def save_alerts_to_supabase(
    supabase_client,
    alerts: List[TrendAlert],
    project_id: Optional[str] = None
) -> int:
    """
    Save detected alerts to resolution_cards table.
    
    Args:
        supabase_client: Supabase client
        alerts: List of TrendAlert objects
        project_id: Optional project ID to associate alerts with
        
    Returns:
        Number of alerts saved
    """
    if not alerts:
        return 0
    
    records = []
    for alert in alerts:
        record = {
            "asin": alert.asin,
            "alert_type": alert.alert_type.value,
            "severity": alert.signal.value,
            "title": alert.title,
            "message": alert.message,
            "action": None,  # Can be enriched by AI later
            "priority_score": abs(alert.change_pct),
            "metadata": alert.to_dict()
        }
        if project_id:
            record["project_id"] = project_id
        records.append(record)
    
    try:
        supabase_client.table("resolution_cards").insert(records).execute()
        logger.info(f"✅ Saved {len(records)} alerts to resolution_cards")
        return len(records)
    except Exception as e:
        logger.error(f"Failed to save alerts: {e}")
        return 0


def run_daily_trend_detection():
    """
    Main entry point for daily trend detection cron job.
    
    This function:
    1. Connects to Supabase
    2. Loads today's and yesterday's snapshots
    3. Runs trend detection
    4. Saves alerts to resolution_cards
    """
    import os
    from supabase import create_client
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        return False
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        logger.info("=" * 60)
        logger.info("🔍 STARTING DAILY TREND DETECTION")
        logger.info(f"📅 Date: {date.today().isoformat()}")
        logger.info("=" * 60)
        
        # Load snapshots
        current_df, previous_df = load_and_compare_snapshots(supabase)
        
        if current_df.empty:
            logger.warning("⚠️ No current snapshots found. Run harvest first.")
            return False
        
        if previous_df.empty:
            logger.warning("⚠️ No previous snapshots for comparison. Need 2+ days of data.")
            return True  # Not an error, just need more data
        
        # Run detection
        alerts = run_trend_detection(current_df, previous_df)
        
        # Save alerts
        if alerts:
            saved = save_alerts_to_supabase(supabase, alerts)
            logger.info(f"💾 Saved {saved} alerts to database")
        else:
            logger.info("✅ No significant trends detected")
        
        logger.info("=" * 60)
        logger.info("🏁 TREND DETECTION COMPLETE")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Trend detection failed: {e}")
        return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    success = run_daily_trend_detection()
    exit(0 if success else 1)
