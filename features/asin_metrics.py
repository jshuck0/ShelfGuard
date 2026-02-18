"""
ASIN Metrics — Market Misattribution Shield
============================================
Rule-based assignment of roles, tags, and Ad Waste Risk flags.
No LLM required. All logic is deterministic from the weekly panel.

Outputs per ASIN:
    role:           "Core" | "Challenger" | "Long-tail"
    tag:            One tag from the allowed set
    ad_waste_risk:  "High" | "Med" | "Low"
    discount_persistence: days_discounted / 7 (from daily panel) or weekly proxy
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


# Allowed tags (exactly as spec'd)
ALLOWED_TAGS = {
    "Undercut victim",
    "Discount-driven gain",
    "Stable leader",
    "Losing visibility",
    "Likely stock/BB issue",  # Only if proxy available
}


def _band_price_vs_tier(ratio: float) -> str:
    """Convert price_vs_tier ratio to a readable position band."""
    if ratio < -0.15:
        return "well below tier"
    elif ratio < -0.05:
        return "below tier"
    elif ratio <= 0.05:
        return "in line"
    elif ratio <= 0.15:
        return "above tier"
    else:
        return "well above tier"


@dataclass
class ASINMetrics:
    asin: str
    brand: str
    role: str                       # "Core" | "Challenger" | "Long-tail"
    tag: str                        # One of ALLOWED_TAGS
    ad_waste_risk: str              # "High" | "Med" | "Low"
    discount_persistence: float     # days/7 (0.0–1.0) or None if unknown
    price_vs_tier: float            # (own_price - tier_median) / tier_median — used for calc
    price_vs_tier_band: str         # "well below tier" … "well above tier" | "not comparable"
    tier_comparable: bool           # False if number_of_items is missing/extreme
    bsr_wow: float                  # BSR week-over-week % change
    has_momentum: bool              # True if rank improving
    fidelity: str                   # "daily" | "weekly" | "snapshot"


def compute_asin_metrics(
    df_weekly: pd.DataFrame,
    role_cfg: dict,
    risk_cfg: dict,
    your_brand: str = "",
    df_daily: Optional[pd.DataFrame] = None,
) -> Dict[str, ASINMetrics]:
    """
    Compute role, tag, and Ad Waste Risk for every ASIN in the arena.

    Args:
        df_weekly: Arena weekly panel
        role_cfg: ASIN_ROLE_THRESHOLDS from market_misattribution_module.py
        risk_cfg: AD_WASTE_RISK_THRESHOLDS from market_misattribution_module.py
        your_brand: Your brand name (for context, not filtering)
        df_daily: Optional daily panel for discount persistence

    Returns:
        Dict mapping asin → ASINMetrics
    """
    if df_weekly.empty:
        return {}

    price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    rank_col = "sales_rank_filled" if "sales_rank_filled" in df_weekly.columns else "sales_rank"

    # Latest week snapshot per ASIN
    latest_week = df_weekly["week_start"].max()
    prev_week_options = sorted(df_weekly["week_start"].unique())
    prev_week = prev_week_options[-2] if len(prev_week_options) >= 2 else None

    latest = df_weekly[df_weekly["week_start"] == latest_week].set_index("asin")
    prev = df_weekly[df_weekly["week_start"] == prev_week].set_index("asin") if prev_week else None

    # Arena-level aggregates (use last 4 weeks for stability)
    recent = df_weekly[df_weekly["week_start"] >= latest_week - pd.Timedelta(weeks=4)]

    # Comparability gating: exclude pack-size outliers from tier stats.
    # An ASIN is "comparable" if number_of_items is present, > 0, and
    # not more than 4× the arena median (catches 48-pack vs single-unit mismatch).
    items_col = "number_of_items"
    items_median: Optional[float] = None
    comparable_asins: set = set()
    if items_col in recent.columns:
        items_vals = recent[items_col].replace(0, np.nan).dropna()
        if not items_vals.empty:
            items_median = float(items_vals.median())
            comparable_mask = (
                recent[items_col].notna()
                & (recent[items_col] > 0)
                & (recent[items_col] <= items_median * 4)
            )
            comparable_asins = set(recent.loc[comparable_mask, "asin"].unique())
    if not comparable_asins:
        comparable_asins = set(recent["asin"].unique())  # all comparable if no items data

    # Tier median computed from comparable ASINs only (normalized price_per_unit)
    recent_comparable = recent[recent["asin"].isin(comparable_asins)]
    tier_median = recent_comparable[price_col].median() if price_col in recent_comparable.columns else None

    # Total arena revenue for share calculation
    rev_col = "weekly_revenue_adjusted" if "weekly_revenue_adjusted" in recent.columns else "weekly_revenue"
    total_arena_rev = recent.groupby("asin")[rev_col].mean().sum() if rev_col in recent.columns else 0

    # Per-ASIN base price (28-day median)
    base_cutoff = latest_week - pd.Timedelta(days=28)
    base_prices = (
        df_weekly[df_weekly["week_start"] < base_cutoff]
        .groupby("asin")[price_col]
        .median()
        if price_col in df_weekly.columns else pd.Series(dtype=float)
    )

    # Discount persistence (prefer daily panel, fall back to weekly proxy)
    discount_persistence_map = {}
    fidelity_map = {}

    if df_daily is not None and not df_daily.empty and price_col in df_daily.columns:
        # Daily panel available: compute days discounted in last 7 days
        cutoff_7d = pd.Timestamp.now() - pd.Timedelta(days=7)
        daily_recent = df_daily[df_daily["date"] >= cutoff_7d] if "date" in df_daily.columns else df_daily.tail(7 * len(df_daily["asin"].unique()))
        for asin, asin_df in daily_recent.groupby("asin"):
            base_p = base_prices.get(asin, tier_median)
            if base_p and base_p > 0:
                discount_days = (asin_df[price_col] < base_p * 0.97).sum()
                discount_persistence_map[asin] = discount_days / 7
                fidelity_map[asin] = "daily"
    else:
        # Weekly proxy: use price vs base in latest 2 weeks
        for asin in df_weekly["asin"].unique():
            asin_df = df_weekly[df_weekly["asin"] == asin].sort_values("week_start").tail(2)
            base_p = base_prices.get(asin, tier_median)
            if base_p and base_p > 0 and price_col in asin_df.columns:
                discounted_weeks = (asin_df[price_col] < base_p * 0.97).sum()
                discount_persistence_map[asin] = discounted_weeks / 2  # proxy for days/7
                fidelity_map[asin] = "weekly"

    results = {}

    for asin in latest.index:
        row = latest.loc[asin]
        brand = str(row.get("brand", "Unknown"))

        # ── Role Assignment ────────────────────────────────────────────────
        bsr = row.get(rank_col, np.nan)
        core_bsr_max = role_cfg.get("core_bsr_max", 5000)
        challenger_bsr_max = role_cfg.get("challenger_bsr_max", 25000)
        core_rev_min = role_cfg.get("core_revenue_share_min", 0.03)

        asin_rev = recent[recent["asin"] == asin][rev_col].mean() if rev_col in recent.columns else 0
        rev_share = asin_rev / total_arena_rev if total_arena_rev > 0 else 0

        if pd.notna(bsr) and bsr <= core_bsr_max:
            role = "Core"
        elif pd.notna(bsr) and bsr <= challenger_bsr_max:
            role = "Challenger"
        elif rev_share >= core_rev_min:
            role = "Core"  # Revenue-based fallback
        else:
            role = "Long-tail"

        # ── BSR WoW ───────────────────────────────────────────────────────
        bsr_wow = 0.0
        if prev is not None and asin in prev.index and rank_col in prev.columns:
            prev_bsr = prev.loc[asin, rank_col]
            if pd.notna(prev_bsr) and prev_bsr > 0 and pd.notna(bsr):
                bsr_wow = (bsr - prev_bsr) / prev_bsr  # positive = worsening

        has_momentum = bsr_wow < -0.03  # 3%+ improvement in rank

        # ── Price vs Tier ─────────────────────────────────────────────────
        own_price = row.get(price_col, np.nan)

        # Comparability: ASIN must have a valid, non-extreme number_of_items
        asin_items = row.get(items_col, np.nan) if items_col in row.index else np.nan
        tier_comparable = (
            pd.notna(asin_items)
            and asin_items > 0
            and (items_median is None or asin_items <= items_median * 4)
        ) if items_median is not None else True  # treat all as comparable when no items data

        price_vs_tier = (
            (own_price - tier_median) / tier_median
            if (tier_comparable and pd.notna(own_price) and tier_median and tier_median > 0)
            else 0.0
        )
        price_vs_tier_band = (
            _band_price_vs_tier(price_vs_tier) if tier_comparable and tier_median
            else "not comparable"
        )

        # ── Ad Waste Risk ─────────────────────────────────────────────────
        price_above_threshold = risk_cfg.get("price_above_tier_pct", 0.10)
        bsr_deterioration = risk_cfg.get("bsr_deterioration_pct", 0.20)
        competitive_threshold = risk_cfg.get("competitive_price_below_tier_pct", 0.05)
        improving_threshold = risk_cfg.get("rank_improving_pct", -0.05)

        is_expensive = price_vs_tier >= price_above_threshold
        is_deteriorating = bsr_wow >= bsr_deterioration
        is_competitive = price_vs_tier <= -competitive_threshold
        is_improving = bsr_wow <= improving_threshold

        if is_expensive and is_deteriorating:
            ad_waste_risk = "High"
        elif is_competitive and is_improving:
            ad_waste_risk = "Low"
        else:
            ad_waste_risk = "Med"

        # ── Tag Assignment ────────────────────────────────────────────────
        discount_pers = discount_persistence_map.get(asin, 0.0)
        has_bb_proxy = (
            "oos_pct_30" in row.index and
            "amazon_bb_share" in row.index and
            (row.get("oos_pct_30", 0) > 0.10 or row.get("amazon_bb_share", 1.0) < 0.40)
        )

        # Priority order for tag selection
        if is_expensive and is_deteriorating and discount_pers < 0.20:
            tag = "Undercut victim"
        elif discount_pers >= 0.50 and has_momentum:
            tag = "Discount-driven gain"
        elif bsr_wow >= 0.10:
            tag = "Losing visibility"
        elif has_bb_proxy:
            tag = "Likely stock/BB issue"
        elif role == "Core" and abs(bsr_wow) <= 0.03 and not is_expensive:
            tag = "Stable leader"
        elif is_competitive and has_momentum:
            tag = "Discount-driven gain"
        elif is_deteriorating:
            tag = "Losing visibility"
        else:
            tag = "Stable leader"

        results[asin] = ASINMetrics(
            asin=asin,
            brand=brand,
            role=role,
            tag=tag,
            ad_waste_risk=ad_waste_risk,
            discount_persistence=round(discount_pers, 2),
            price_vs_tier=round(price_vs_tier, 4),
            price_vs_tier_band=price_vs_tier_band,
            tier_comparable=tier_comparable,
            bsr_wow=round(bsr_wow, 4),
            has_momentum=has_momentum,
            fidelity=fidelity_map.get(asin, "snapshot"),
        )

    return results


def to_compact_table(
    asin_metrics: Dict[str, ASINMetrics],
    df_weekly: pd.DataFrame,
    max_asins: int = 30,
    band_fn=None,
) -> pd.DataFrame:
    """
    Build the compact per-ASIN table for the brief's Layer B.

    Columns: ASIN (short), Brand, Role, Price vs Tier, Discount Persistence,
             BSR WoW, Momentum, Tag

    Args:
        asin_metrics: Output of compute_asin_metrics()
        df_weekly: Arena weekly panel (for ASIN short name lookup)
        max_asins: Cap at this many rows (sorted by role then BSR)
        band_fn: Optional callable(value, band_type) → str for banding

    Returns:
        DataFrame ready to render in brief
    """
    if not asin_metrics:
        return pd.DataFrame()

    # Short name: first 3 words of title or ASIN
    title_map = {}
    if "title" in df_weekly.columns and "asin" in df_weekly.columns:
        latest = df_weekly["week_start"].max()
        for _, row in df_weekly[df_weekly["week_start"] == latest].iterrows():
            title = str(row.get("title", ""))
            words = title.split()[:3]
            title_map[row["asin"]] = " ".join(words) if words else row["asin"]

    role_order = {"Core": 0, "Challenger": 1, "Long-tail": 2}
    rows = []
    for asin, m in asin_metrics.items():
        short_name = title_map.get(asin, asin[-8:])

        # Band display — use pre-computed band (normalized, pack-size aware)
        if band_fn:
            price_band = band_fn(m.price_vs_tier, "price_vs_tier")
            bsr_band = band_fn(m.bsr_wow, "rank_change")
        else:
            price_band = m.price_vs_tier_band  # "well below tier" … "not comparable"
            bsr_band = f"{m.bsr_wow*100:+.1f}%"

        rows.append({
            "ASIN": short_name,
            "Brand": m.brand,
            "Role": m.role,
            "Price vs Tier": price_band,
            "Disc Persist (d/7)": f"{m.discount_persistence:.1f}",
            "BSR WoW": bsr_band,
            "Momentum": "Y" if m.has_momentum else "N",
            "Tag": m.tag,
            "_role_sort": role_order.get(m.role, 3),
            "_bsr_sort": m.bsr_wow,
            "_asin": asin,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["_role_sort", "_bsr_sort"]).head(max_asins)
    return df.drop(columns=["_role_sort", "_bsr_sort", "_asin"], errors="ignore")


def receipts_list(
    asin_metrics: Dict[str, ASINMetrics],
    your_brand: str,
    max_items: int = 8,
) -> List[str]:
    """
    Build Layer A: the 5–8 line ASIN receipts list for the brief.
    Format: "Brand (ASIN…): 2 signals + confidence"
    """
    lines = []
    sorted_asins = sorted(
        asin_metrics.values(),
        key=lambda m: (0 if m.brand.lower() == your_brand.lower() else 1, m.bsr_wow)
    )

    for m in sorted_asins[:max_items]:
        conf = "High" if m.ad_waste_risk == "Low" and m.has_momentum else (
            "Low" if m.ad_waste_risk == "High" else "Med"
        )
        price_str = f"price {m.price_vs_tier_band}"
        bsr_str = f"BSR {m.bsr_wow*100:+.1f}% WoW"
        lines.append(
            f"{m.brand} ({m.asin[-6:]}): {price_str}, {bsr_str} — {m.tag} [{conf} conf]"
        )

    return lines
