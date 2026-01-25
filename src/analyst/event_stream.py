"""
Sparse Event Stream Transformer

Converts raw Keepa weekly data (18,000 rows) into a compressed event stream
(~2,000 events) that fits in an LLM context window.

Key behaviors:
1. First row for each ASIN emits BASELINE event (full state snapshot)
2. Subsequent rows only emit if something meaningful changed
3. Events are tagged as "portfolio" (yours) vs "competitor"
4. Chain linking: events reference prior related events
5. Semantic enrichment: extract tags from product titles
6. Pre-computed derivatives: do math in Python, not in LLM
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.analyst.models import EnrichedEvent, EventOwner


# =============================================================================
# SIGNIFICANCE THRESHOLDS
# =============================================================================

THRESHOLDS = {
    "price_change_pct": 3.0,        # Price moved 3%+
    "rank_change_pct": 10.0,        # Rank moved 10%+
    "review_delta": 5,              # 5+ new reviews
    "bb_share_change": 0.10,        # Buy Box share shifted 10%+
    "revenue_change_pct": 15.0,     # Revenue moved 15%+
}


# =============================================================================
# SEMANTIC TAG EXTRACTION
# =============================================================================

SEMANTIC_PATTERNS = {
    # Scent types
    "scent": [
        (r"\b(fiji|tropical|ocean|fresh)\b", "tropical"),
        (r"\b(swagger|wolf|timber)\b", "bold"),
        (r"\b(lavender|chamomile)\b", "calming"),
        (r"\b(cedar|sandalwood|wood)\b", "woody"),
        (r"\b(mint|eucalyptus)\b", "fresh"),
        (r"\bunscented\b", "unscented"),
    ],
    # Product type
    "type": [
        (r"\baluminum[- ]?free\b", "aluminum-free"),
        (r"\bantiperspirant\b", "antiperspirant"),
        (r"\bdeodorant\b", "deodorant"),
        (r"\bbody wash\b", "body-wash"),
        (r"\bshampoo\b", "shampoo"),
    ],
    # Pack size
    "pack_size": [
        (r"\b(\d+)[- ]?pack\b", lambda m: f"{m.group(1)}-pack"),
        (r"\btwin[- ]?pack\b", "2-pack"),
        (r"\bvariety\b", "variety"),
    ],
    # Form factor
    "form": [
        (r"\bstick\b", "stick"),
        (r"\bspray\b", "spray"),
        (r"\bcream\b", "cream"),
        (r"\bgel\b", "gel"),
        (r"\broll[- ]?on\b", "roll-on"),
    ],
}


def extract_semantic_tags(title: str) -> List[str]:
    """Extract semantic tags from product title."""
    if not title:
        return []
    
    title_lower = title.lower()
    tags = []
    
    for category, patterns in SEMANTIC_PATTERNS.items():
        for pattern, tag in patterns:
            match = re.search(pattern, title_lower, re.IGNORECASE)
            if match:
                if callable(tag):
                    tags.append(tag(match))
                else:
                    tags.append(tag)
                break  # Only one tag per category
    
    return tags


# =============================================================================
# DERIVATIVE CALCULATIONS
# =============================================================================

def compute_price_tier(price: float, category_median: float) -> str:
    """Classify price tier relative to category."""
    if not price or not category_median or category_median == 0:
        return "unknown"
    
    ratio = price / category_median
    if ratio > 1.2:
        return f"+{(ratio - 1) * 100:.0f}% (Premium)"
    elif ratio < 0.8:
        return f"{(ratio - 1) * 100:.0f}% (Budget)"
    else:
        return "Mid-tier"


def compute_rank_velocity(current_rank: float, previous_rank: float) -> str:
    """Describe rank velocity in human terms."""
    if not current_rank or not previous_rank or previous_rank == 0:
        return "unknown"
    
    # Lower rank is better, so negative change_pct means improvement
    change_pct = ((current_rank - previous_rank) / previous_rank) * 100
    
    if change_pct < -20:
        return f"{change_pct:.0f}% (Surging)"
    elif change_pct < -5:
        return f"{change_pct:.0f}% (Improving)"
    elif change_pct > 20:
        return f"+{change_pct:.0f}% (Crashing)"
    elif change_pct > 5:
        return f"+{change_pct:.0f}% (Declining)"
    else:
        return "Stable"


def compute_competitive_context(
    asin: str,
    all_events: List[EnrichedEvent],
    lookback_days: int = 7
) -> str:
    """Summarize what competitors are doing."""
    # Count competitor OOS events in recent days
    competitor_oos = 0
    competitor_price_drops = 0
    
    for event in all_events:
        if event.owner == EventOwner.COMPETITOR:
            if event.event_type == "INVENTORY_ZERO":
                competitor_oos += 1
            elif event.event_type == "PRICE_DROP":
                competitor_price_drops += 1
    
    parts = []
    if competitor_oos > 0:
        parts.append(f"{competitor_oos} competitors OOS")
    if competitor_price_drops > 0:
        parts.append(f"{competitor_price_drops} price drops")
    
    return " | ".join(parts) if parts else ""


# =============================================================================
# CHANGE DETECTION
# =============================================================================

def detect_changes(
    current: pd.Series,
    previous: pd.Series
) -> List[Dict[str, Any]]:
    """
    Detect significant changes between two rows.
    Returns list of change dicts, one per significant metric.
    """
    changes = []
    
    # Price change
    curr_price = _safe_float(current.get('buy_box_price') or current.get('price'))
    prev_price = _safe_float(previous.get('buy_box_price') or previous.get('price'))
    if curr_price and prev_price and prev_price > 0:
        pct = ((curr_price - prev_price) / prev_price) * 100
        if abs(pct) >= THRESHOLDS["price_change_pct"]:
            event_type = "PRICE_DROP" if pct < 0 else "PRICE_SPIKE"
            changes.append({
                "event_type": event_type,
                "metric_name": "price",
                "old_value": prev_price,
                "new_value": curr_price,
                "change_pct": pct,
            })
    
    # Rank change
    curr_rank = _safe_float(current.get('sales_rank') or current.get('rank'))
    prev_rank = _safe_float(previous.get('sales_rank') or previous.get('rank'))
    if curr_rank and prev_rank and prev_rank > 0:
        # For rank, improvement = going lower
        pct = ((curr_rank - prev_rank) / prev_rank) * 100
        if abs(pct) >= THRESHOLDS["rank_change_pct"]:
            event_type = "RANK_SPIKE" if pct < 0 else "RANK_DROP"
            changes.append({
                "event_type": event_type,
                "metric_name": "rank",
                "old_value": prev_rank,
                "new_value": curr_rank,
                "change_pct": pct,
            })
    
    # Review count change
    curr_reviews = _safe_float(current.get('review_count') or current.get('reviews'))
    prev_reviews = _safe_float(previous.get('review_count') or previous.get('reviews'))
    if curr_reviews and prev_reviews:
        delta = curr_reviews - prev_reviews
        if abs(delta) >= THRESHOLDS["review_delta"]:
            changes.append({
                "event_type": "REVIEW_SURGE" if delta > 0 else "REVIEW_STALL",
                "metric_name": "reviews",
                "old_value": prev_reviews,
                "new_value": curr_reviews,
                "change_pct": (delta / prev_reviews * 100) if prev_reviews > 0 else 0,
            })
    
    # Buy Box share change
    curr_bb = _safe_float(current.get('amazon_bb_share') or current.get('bb_share'))
    prev_bb = _safe_float(previous.get('amazon_bb_share') or previous.get('bb_share'))
    if curr_bb is not None and prev_bb is not None:
        delta = curr_bb - prev_bb
        if abs(delta) >= THRESHOLDS["bb_share_change"]:
            changes.append({
                "event_type": "BUYBOX_GAINED" if delta > 0 else "BUYBOX_LOST",
                "metric_name": "buybox_share",
                "old_value": prev_bb,
                "new_value": curr_bb,
                "change_pct": delta * 100,  # Already a ratio
            })
    
    # Inventory zero (OOS)
    curr_stock = _safe_float(current.get('stock') or current.get('inventory'))
    prev_stock = _safe_float(previous.get('stock') or previous.get('inventory'))
    if curr_stock is not None and prev_stock is not None:
        if curr_stock == 0 and prev_stock > 0:
            changes.append({
                "event_type": "INVENTORY_ZERO",
                "metric_name": "stock",
                "old_value": prev_stock,
                "new_value": 0,
                "change_pct": -100,
            })
        elif curr_stock > 0 and prev_stock == 0:
            changes.append({
                "event_type": "RESTOCKED",
                "metric_name": "stock",
                "old_value": 0,
                "new_value": curr_stock,
                "change_pct": 100,
            })
    
    return changes


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, returning None for invalid values."""
    if val is None:
        return None
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


# =============================================================================
# CHAIN LINKING
# =============================================================================

CAUSAL_PAIRS = {
    # (prior_event, current_event) pairs that suggest causality
    ("PRICE_DROP", "RANK_SPIKE"): "Price cut drove rank improvement",
    ("PRICE_DROP", "RANK_DROP"): "Price cut failed to help rank",
    ("COMPETITOR_PRICE_DROP", "RANK_DROP"): "Competitor undercut you",
    ("INVENTORY_ZERO", "RANK_DROP"): "OOS hurt your rank",
    ("RESTOCKED", "RANK_SPIKE"): "Restock recovered rank",
    ("REVIEW_SURGE", "RANK_SPIKE"): "Reviews drove rank",
}


def find_related_prior_event(
    events: List[EnrichedEvent],
    current_asin: str,
    current_event_type: str,
    current_date: str,
    lookback_hours: int = 72
) -> Optional[str]:
    """Link events in causal chains."""
    for prior in reversed(events[-50:]):  # Only check recent events
        if prior.asin != current_asin:
            continue
        
        pair_key = (prior.event_type, current_event_type)
        if pair_key in CAUSAL_PAIRS:
            return f"Followed {prior.event_type} | {CAUSAL_PAIRS[pair_key]}"
    
    return None


# =============================================================================
# MAIN TRANSFORMER
# =============================================================================

def transform_to_event_stream(
    df_weekly: pd.DataFrame,
    portfolio_asins: List[str],
    category_median_price: float = 15.0,
) -> List[EnrichedEvent]:
    """
    Transform raw weekly data into sparse event stream.
    
    Args:
        df_weekly: Weekly data from build_keepa_weekly_table()
        portfolio_asins: List of ASINs that belong to you (vs competitors)
        category_median_price: For computing price tier derivatives
        
    Returns:
        List of EnrichedEvent objects, sorted by date
    """
    if df_weekly.empty:
        return []
    
    events: List[EnrichedEvent] = []
    
    # Ensure we have required columns
    if 'asin' not in df_weekly.columns:
        return []
    
    # Get date column
    date_col = 'week_start' if 'week_start' in df_weekly.columns else 'date'
    if date_col not in df_weekly.columns:
        return []
    
    # Sort by ASIN and date
    df_sorted = df_weekly.sort_values([date_col, 'asin'])
    
    # Track previous row per ASIN
    prev_rows: Dict[str, pd.Series] = {}
    
    for idx, row in df_sorted.iterrows():
        asin = row['asin']
        date_str = str(row[date_col])[:10]  # YYYY-MM-DD
        
        is_portfolio = asin in portfolio_asins
        owner = EventOwner.PORTFOLIO if is_portfolio else EventOwner.COMPETITOR
        
        # Get metadata
        brand = str(row.get('brand', ''))[:50] if row.get('brand') else ""
        title = str(row.get('title', ''))[:100] if row.get('title') else ""
        tags = extract_semantic_tags(title)
        
        # Extract new Keepa metrics (available in df_weekly)
        monthly_sold = int(row.get('monthly_sold', 0) or 0) if row.get('monthly_sold') else None
        velocity_30d = _safe_float(row.get('velocity_30d'))
        oos_pct_30 = _safe_float(row.get('oos_pct_30'))
        seller_count = int(row.get('seller_count', 0) or 0) if row.get('seller_count') else None
        
        if asin not in prev_rows:
            # BASELINE event - first time seeing this ASIN
            events.append(EnrichedEvent(
                date=date_str,
                asin=asin,
                event_type="BASELINE",
                owner=owner,
                brand=brand,
                title=title,
                tags=tags,
                monthly_sold=monthly_sold,
                velocity_30d=velocity_30d,
                oos_pct_30=oos_pct_30,
                seller_count=seller_count,
                state_snapshot={
                    "price": _safe_float(row.get('buy_box_price') or row.get('price')),
                    "rank": _safe_float(row.get('sales_rank') or row.get('rank')),
                    "reviews": _safe_float(row.get('review_count')),
                    "rating": _safe_float(row.get('rating')),
                    "bb_share": _safe_float(row.get('amazon_bb_share')),
                    "monthly_sold": monthly_sold,
                    "velocity_30d": velocity_30d,
                    "oos_pct_30": oos_pct_30,
                    "seller_count": seller_count,
                },
            ))
        else:
            # Check for significant changes
            prev = prev_rows[asin]
            changes = detect_changes(row, prev)
            
            for change in changes:
                # Compute derivatives
                curr_price = _safe_float(row.get('buy_box_price') or row.get('price'))
                prev_rank = _safe_float(prev.get('sales_rank') or prev.get('rank'))
                curr_rank = _safe_float(row.get('sales_rank') or row.get('rank'))
                
                price_tier = compute_price_tier(curr_price, category_median_price) if curr_price else ""
                rank_velocity = compute_rank_velocity(curr_rank, prev_rank) if curr_rank and prev_rank else ""
                
                # Find related prior event
                prior_link = find_related_prior_event(
                    events, asin, change["event_type"], date_str
                )
                
                events.append(EnrichedEvent(
                    date=date_str,
                    asin=asin,
                    event_type=change["event_type"],
                    owner=owner,
                    brand=brand,
                    title=title,
                    tags=tags,
                    metric_name=change["metric_name"],
                    old_value=change["old_value"],
                    new_value=change["new_value"],
                    change_pct=change["change_pct"],
                    price_gap_vs_category=price_tier,
                    rank_velocity_7d=rank_velocity,
                    monthly_sold=monthly_sold,
                    velocity_30d=velocity_30d,
                    oos_pct_30=oos_pct_30,
                    seller_count=seller_count,
                    prior_event=prior_link,
                ))
        
        # Store for next iteration
        prev_rows[asin] = row
    
    # Add competitive context to portfolio events
    for event in events:
        if event.owner == EventOwner.PORTFOLIO and event.event_type != "BASELINE":
            event.competitive_context = compute_competitive_context(event.asin, events)
    
    # Sort by date
    events.sort(key=lambda e: e.date)
    
    return events


def format_events_for_llm(events: List[EnrichedEvent], max_tokens: int = 70000) -> str:
    """
    Format event stream as a string for LLM consumption.
    
    Estimates ~40 tokens per event line.
    """
    lines = []
    
    # Header
    lines.append("=== MARKET EVENT STREAM ===")
    lines.append(f"Total Events: {len(events)}")
    lines.append("")
    
    # Count by type
    portfolio_events = [e for e in events if e.owner == EventOwner.PORTFOLIO]
    competitor_events = [e for e in events if e.owner == EventOwner.COMPETITOR]
    lines.append(f"Your Products: {len(portfolio_events)} events")
    lines.append(f"Competitors: {len(competitor_events)} events")
    lines.append("")
    lines.append("Legend: 📍YOUR = Your product | 🎯COMP = Competitor")
    lines.append("---")
    lines.append("")
    
    # Estimate tokens and truncate if needed
    max_events = max_tokens // 40
    events_to_show = events[:max_events]
    
    for event in events_to_show:
        lines.append(event.to_llm_string())
    
    if len(events) > max_events:
        lines.append(f"... ({len(events) - max_events} more events truncated)")
    
    return "\n".join(lines)


def get_event_summary(events: List[EnrichedEvent]) -> Dict[str, Any]:
    """Get summary statistics about the event stream."""
    if not events:
        return {"total": 0}
    
    portfolio = [e for e in events if e.owner == EventOwner.PORTFOLIO]
    competitor = [e for e in events if e.owner == EventOwner.COMPETITOR]
    
    # Count by type
    type_counts = {}
    for event in events:
        type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
    
    return {
        "total": len(events),
        "portfolio_events": len(portfolio),
        "competitor_events": len(competitor),
        "date_range": f"{events[0].date} to {events[-1].date}",
        "event_types": type_counts,
        "unique_asins": len(set(e.asin for e in events)),
    }
