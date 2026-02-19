"""
Regime Detectors — Market Misattribution Shield
================================================
Five deterministic, cross-ASIN regime detectors.

Each detector:
- Reads from the arena weekly DataFrame (df_weekly: ASIN × week rows)
- Takes optional df_daily for higher-fidelity discount persistence
- Returns a RegimeSignal with verdict, confidence, evidence list, and receipts
- Uses thresholds from config/market_misattribution_module.py (never hardcoded here)

Detector functions:
    detect_tier_compression(df_weekly, cfg) → RegimeSignal
    detect_promo_war(df_weekly, cfg, df_daily=None) → RegimeSignal
    detect_competitor_compounding(df_weekly, cfg, your_brand) → RegimeSignal
    detect_demand_tailwind(df_weekly, cfg) → RegimeSignal
    detect_new_entrant(df_weekly, cfg) → RegimeSignal

    detect_all_regimes(df_weekly, your_brand, cfg=None, df_daily=None) → dict[str, RegimeSignal]
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class Receipt:
    """A single auditable evidence item for a regime claim."""
    label: str          # Short label shown in brief (e.g. "CeraVe: -12% price, BSR +8%")
    metric: str         # The metric observed (e.g. "price_per_unit")
    value: float        # Observed value
    baseline: float     # What normal looks like
    delta_pct: float    # (value - baseline) / baseline
    asin: Optional[str] = None
    brand: Optional[str] = None
    week: Optional[str] = None


@dataclass
class RegimeSignal:
    """
    Output of a single regime detector.

    confidence: "High" | "Med" | "Low"
    active: True if the regime is currently firing
    verdict: Concise verdict string for the brief (e.g. "Tier compression active for 3 weeks")
    driver_type: "Market-driven" | "Brand-driven" | "Unknown"
    receipts: Exactly 2 receipts shown in the brief
    evidence: Full list (for diagnostics)
    """
    regime: str                        # "tier_compression", "promo_war", etc.
    active: bool
    confidence: str                    # "High" | "Med" | "Low"
    verdict: str
    driver_type: str                   # "Market-driven" | "Brand-driven" | "Unknown"
    receipts: List[Receipt] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def _empty_signal(regime: str, reason: str = "Insufficient data") -> RegimeSignal:
    return RegimeSignal(
        regime=regime,
        active=False,
        confidence="Low",
        verdict=reason,
        driver_type="Unknown",
    )


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _weeks_in_range(df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Return only the last n_weeks of data."""
    if "week_start" not in df.columns or df.empty:
        return df
    cutoff = df["week_start"].max() - pd.Timedelta(weeks=n_weeks)
    return df[df["week_start"] >= cutoff]


def _base_window(df: pd.DataFrame, base_days: int) -> pd.DataFrame:
    """Return rows in the base window (used as the price baseline)."""
    if "week_start" not in df.columns or df.empty:
        return df
    cutoff = df["week_start"].max() - pd.Timedelta(days=base_days)
    return df[df["week_start"] < cutoff]


def _tier_median_price(df: pd.DataFrame) -> Optional[float]:
    """Compute the arena's median price_per_unit."""
    col = "price_per_unit" if "price_per_unit" in df.columns else "filled_price"
    vals = df[col].dropna()
    return float(vals.median()) if not vals.empty else None


def _safe_pct_change(new: float, old: float) -> Optional[float]:
    if old and old != 0:
        return (new - old) / abs(old)
    return None


# ─── A: TIER COMPRESSION ─────────────────────────────────────────────────────

def detect_tier_compression(
    df_weekly: pd.DataFrame,
    cfg: dict,
) -> RegimeSignal:
    """
    Detect sustained category-wide price floor compression.

    Fires when:
    - Tier median price_per_unit is below the 28-day base by cfg threshold
    - For at least cfg["persistence_weeks"] consecutive weeks
    - Across cfg["arena_coverage_min"] fraction of the arena
    """
    regime = "tier_compression"

    if df_weekly.empty or "week_start" not in df_weekly.columns:
        return _empty_signal(regime)

    base_days = cfg.get("base_window_days", 28)
    compress_pct = cfg.get("compression_threshold_pct", 0.05)
    persist_weeks = cfg.get("persistence_weeks", 2)
    coverage_min = cfg.get("arena_coverage_min", 0.30)
    coverage_high = cfg.get("arena_coverage_high", 0.50)

    price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    if price_col not in df_weekly.columns:
        return _empty_signal(regime, "No price column available")

    cols = ["asin", "brand", "week_start", price_col]
    if "number_of_items" in df_weekly.columns:
        cols.append("number_of_items")
    df = df_weekly[cols].dropna(subset=[price_col]).copy()
    if df.empty:
        return _empty_signal(regime)

    # Comparability gating: exclude pack-size outliers from tier stats.
    # Uses same 4× median rule as asin_metrics.py for consistency.
    if "number_of_items" in df.columns:
        items_med = df["number_of_items"].replace(0, np.nan).median()
        if pd.notna(items_med) and items_med > 0:
            comparable_mask = (
                df["number_of_items"].notna()
                & (df["number_of_items"] > 0)
                & (df["number_of_items"] <= items_med * 4)
            )
            df = df[comparable_mask].copy()
    if df.empty:
        return _empty_signal(regime, "No comparable ASINs after pack-size gating")

    all_weeks = sorted(df["week_start"].unique())
    if len(all_weeks) < persist_weeks + 1:
        return _empty_signal(regime, "Not enough weeks of data")

    # Baseline: median price across the base window (older weeks)
    base_cutoff = df["week_start"].max() - pd.Timedelta(days=base_days)
    base_df = df[df["week_start"] < base_cutoff]
    if base_df.empty:
        return _empty_signal(regime, "Base window has no data")

    base_median = base_df[price_col].median()
    if not base_median or base_median == 0:
        return _empty_signal(regime, "Base median price is zero")

    # Recent weeks: compute per-week tier medians
    recent_df = df[df["week_start"] >= base_cutoff]
    weekly_medians = recent_df.groupby("week_start")[price_col].median()
    total_asins = df["asin"].nunique()

    # Count how many weeks are compressed
    compressed_weeks = []
    for wk, med in weekly_medians.items():
        if base_median > 0 and (base_median - med) / base_median >= compress_pct:
            # Also check coverage: how many ASINs are below base median?
            week_df = df[df["week_start"] == wk]
            discounted_count = (week_df[price_col] < base_median * (1 - compress_pct)).sum()
            coverage = discounted_count / max(total_asins, 1)
            if coverage >= coverage_min:
                compressed_weeks.append((wk, med, coverage))

    active = len(compressed_weeks) >= persist_weeks

    # Build confidence
    if not active:
        confidence = "Low"
    else:
        max_coverage = max(c for _, _, c in compressed_weeks)
        if max_coverage >= coverage_high and len(compressed_weeks) >= 3:
            confidence = "High"
        elif max_coverage >= coverage_min:
            confidence = "Med"
        else:
            confidence = "Low"

    # Receipts (top 2 most discounted brands)
    receipts = []
    if active:
        recent_cutoff = df["week_start"].max() - pd.Timedelta(weeks=2)
        recent_brands = df[df["week_start"] >= recent_cutoff].groupby("brand")[price_col].mean()
        below_base = recent_brands[recent_brands < base_median * (1 - compress_pct)]
        for brand, avg_price in below_base.nsmallest(2).items():
            delta = _safe_pct_change(avg_price, base_median) or 0
            receipts.append(Receipt(
                label=f"{brand}: avg {price_col} {delta*100:+.1f}% vs 28d base",
                metric=price_col,
                value=round(avg_price, 2),
                baseline=round(base_median, 2),
                delta_pct=round(delta, 4),
                brand=brand,
            ))

    # Verdict
    if active:
        n = len(compressed_weeks)
        latest_med = weekly_medians.iloc[-1] if not weekly_medians.empty else base_median
        delta = _safe_pct_change(latest_med, base_median) or 0
        verdict = (
            f"Tier compression active for {n} week(s): "
            f"arena median price {delta*100:+.1f}% vs 28d base"
        )
    else:
        verdict = "No sustained tier compression detected"

    return RegimeSignal(
        regime=regime,
        active=active,
        confidence=confidence,
        verdict=verdict,
        driver_type="Market-driven" if active else "Unknown",
        receipts=receipts[:2],
        evidence=[f"Week {str(wk)[:10]}: median={med:.2f}, coverage={cov:.0%}"
                  for wk, med, cov in compressed_weeks],
        metadata={
            "base_median": round(base_median, 2),
            "compressed_weeks": len(compressed_weeks),
            "base_days_used": base_days,
        },
    )


# ─── B: PROMO WAR PROXY ──────────────────────────────────────────────────────

def detect_promo_war(
    df_weekly: pd.DataFrame,
    cfg: dict,
    df_daily: Optional[pd.DataFrame] = None,
) -> RegimeSignal:
    """
    Detect synchronized discounting across multiple competitors.

    Fires when:
    - N+ distinct brands are simultaneously discounting vs their own baseline
    - Rank gains are concentrated among the discounted set
    """
    regime = "promo_war"

    if df_weekly.empty:
        return _empty_signal(regime)

    discount_pct = cfg.get("discount_threshold_pct", 0.07)
    min_brands = cfg.get("min_brands_discounting", 3)
    rank_gain_min = cfg.get("rank_gain_from_discounters_min", 0.50)
    sync_weeks = cfg.get("sync_window_weeks", 2)
    high_brand_count = cfg.get("high_confidence_brand_count", 5)

    price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    if price_col not in df_weekly.columns or "sales_rank_filled" not in df_weekly.columns:
        return _empty_signal(regime, "Missing price or rank columns")

    df = df_weekly[["asin", "brand", "week_start", price_col, "sales_rank_filled"]].dropna().copy()
    if df.empty or df["week_start"].nunique() < 2:
        return _empty_signal(regime)

    # Per-ASIN 28-day base price
    all_weeks = sorted(df["week_start"].unique())
    cutoff = df["week_start"].max() - pd.Timedelta(weeks=sync_weeks)
    base_cutoff = df["week_start"].max() - pd.Timedelta(days=28)

    base_prices = (
        df[df["week_start"] < base_cutoff]
        .groupby("asin")[price_col]
        .median()
        .rename("base_price")
    )
    df = df.merge(base_prices, on="asin", how="left")
    df["is_discounted"] = df[price_col] < df["base_price"] * (1 - discount_pct)

    recent = df[df["week_start"] >= cutoff]
    if recent.empty:
        return _empty_signal(regime)

    # Brands discounting in recent window
    brand_discount = recent.groupby("brand")["is_discounted"].any()
    discounting_brands = brand_discount[brand_discount].index.tolist()
    n_discounting = len(discounting_brands)

    # Rank gain check: are rank gainers mostly discounters?
    prev_week = sorted(df["week_start"].unique())[-2] if len(all_weeks) >= 2 else None
    rank_gain_from_discounters = 0.0

    if prev_week is not None:
        curr_rank = df[df["week_start"] == df["week_start"].max()].set_index("asin")["sales_rank_filled"]
        prev_rank = df[df["week_start"] == prev_week].set_index("asin")["sales_rank_filled"]
        rank_change = (curr_rank - prev_rank).dropna()  # negative = improved
        rank_gainers = rank_change[rank_change < 0].index.tolist()

        if rank_gainers:
            discounting_asins = recent[recent["is_discounted"]]["asin"].unique()
            gainers_discounting = sum(1 for a in rank_gainers if a in discounting_asins)
            rank_gain_from_discounters = gainers_discounting / len(rank_gainers)

    active = (n_discounting >= min_brands) and (rank_gain_from_discounters >= rank_gain_min)

    # Confidence
    if not active:
        confidence = "Low"
    elif n_discounting >= high_brand_count and rank_gain_from_discounters >= 0.60:
        confidence = "High"
    elif n_discounting >= min_brands:
        confidence = "Med"
    else:
        confidence = "Low"

    # Receipts: 2 most discounted brands with rank gain info
    receipts = []
    if active and discounting_brands:
        for brand in discounting_brands[:2]:
            brand_df = recent[recent["brand"] == brand]
            avg_price = brand_df[price_col].mean()
            base_p = (df[df["brand"] == brand]["base_price"].mean())
            delta = _safe_pct_change(avg_price, base_p) or 0
            receipts.append(Receipt(
                label=f"{brand}: price {delta*100:+.1f}% vs own 28d base",
                metric=price_col,
                value=round(avg_price, 2),
                baseline=round(base_p, 2) if base_p else 0,
                delta_pct=round(delta, 4),
                brand=brand,
            ))

    verdict = (
        f"Promo war proxy: {n_discounting} brands discounting simultaneously; "
        f"{rank_gain_from_discounters:.0%} of rank gainers are discounters"
        if active else
        f"No promo war: {n_discounting}/{min_brands} brands needed discounting"
    )

    return RegimeSignal(
        regime=regime,
        active=active,
        confidence=confidence,
        verdict=verdict,
        driver_type="Market-driven" if active else "Unknown",
        receipts=receipts[:2],
        evidence=[f"{b}: discounting" for b in discounting_brands],
        metadata={
            "discounting_brands": discounting_brands,
            "n_discounting": n_discounting,
            "rank_gain_from_discounters": round(rank_gain_from_discounters, 3),
        },
    )


# ─── C: COMPETITOR COMPOUNDING ───────────────────────────────────────────────

def detect_competitor_compounding(
    df_weekly: pd.DataFrame,
    cfg: dict,
    your_brand: str,
) -> RegimeSignal:
    """
    Detect a specific competitor gaining rank persistently while your brand declines.

    Fires when:
    - A competitor ASIN improves BSR by cfg threshold for N+ consecutive weeks
    - Your brand's BSR is flat or worsening over the same period
    """
    regime = "competitor_compounding"

    if df_weekly.empty or not your_brand:
        return _empty_signal(regime)

    rank_gain_pct = cfg.get("competitor_rank_gain_pct", -0.15)
    persist_weeks = cfg.get("persistence_weeks", 2)
    your_decline_threshold = cfg.get("your_rank_decline_threshold", 0.0)
    max_price_premium = cfg.get("max_price_premium_pct", 0.15)
    high_conf_weeks = cfg.get("high_confidence_weeks", 3)

    if "sales_rank_filled" not in df_weekly.columns:
        return _empty_signal(regime, "No rank data")

    _price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    _sel = ["asin", "brand", "week_start", "sales_rank_filled"]
    if _price_col in df_weekly.columns:
        _sel.append(_price_col)
    df = df_weekly[[c for c in _sel if c in df_weekly.columns]].dropna(subset=["sales_rank_filled"]).copy()
    price_col = _price_col if _price_col in df.columns else None

    _brand_lower = df["brand"].fillna("").str.lower()
    your_df = df[_brand_lower == your_brand.lower()]
    comp_df = df[_brand_lower != your_brand.lower()]

    if your_df.empty or comp_df.empty:
        return _empty_signal(regime, "Missing your brand or competitor data")

    all_weeks = sorted(df["week_start"].unique())
    if len(all_weeks) < persist_weeks + 1:
        return _empty_signal(regime, "Not enough weeks for persistence check")

    # Tier median price for the max premium filter
    tier_median_price = df[price_col].median() if price_col in df.columns else None

    # For each competitor ASIN, check if it gains rank persistently
    compounders = []

    for asin, asin_df in comp_df.groupby("asin"):
        asin_df = asin_df.sort_values("week_start")
        if len(asin_df) < persist_weeks + 1:
            continue

        ranks = asin_df.set_index("week_start")["sales_rank_filled"]
        # Check consecutive week-over-week improvements
        improvements = 0
        for i in range(1, len(ranks)):
            prev = ranks.iloc[i - 1]
            curr = ranks.iloc[i]
            if prev > 0 and (curr - prev) / prev <= rank_gain_pct:  # negative = better rank
                improvements += 1
            else:
                improvements = 0  # reset streak

        if improvements < persist_weeks:
            continue

        # Skip if competitor is priced too far above tier (organic demand, not competitive threat)
        if tier_median_price and max_price_premium is not None:
            avg_price = asin_df[price_col].mean() if price_col in asin_df.columns else None
            if avg_price and avg_price > tier_median_price * (1 + max_price_premium):
                continue

        brand = asin_df["brand"].iloc[0]
        latest_rank = ranks.iloc[-1]
        earliest_rank = ranks.iloc[0]
        total_gain = _safe_pct_change(latest_rank, earliest_rank) or 0

        compounders.append({
            "asin": asin,
            "brand": brand,
            "streak_weeks": improvements,
            "rank_gain_pct": total_gain,
            "latest_rank": latest_rank,
        })

    # Check if your brand is declining
    your_ranks = your_df.groupby("week_start")["sales_rank_filled"].mean().sort_index()
    your_declining = False
    if len(your_ranks) >= 2:
        your_gain = _safe_pct_change(your_ranks.iloc[-1], your_ranks.iloc[0]) or 0
        your_declining = your_gain >= your_decline_threshold  # positive = worsening BSR

    active = len(compounders) > 0 and your_declining

    # Confidence
    if not active:
        confidence = "Low"
    else:
        max_streak = max(c["streak_weeks"] for c in compounders)
        if max_streak >= high_conf_weeks:
            confidence = "High"
        else:
            confidence = "Med"

    # Sort compounders by strength
    compounders.sort(key=lambda x: x["rank_gain_pct"])  # most improved first

    # Receipts
    receipts = []
    for c in compounders[:2]:
        receipts.append(Receipt(
            label=f"{c['brand']}: BSR {c['rank_gain_pct']*100:+.1f}% over {c['streak_weeks']}wk streak",
            metric="sales_rank_filled",
            value=c["latest_rank"],
            baseline=0,
            delta_pct=c["rank_gain_pct"],
            asin=c["asin"],
            brand=c["brand"],
        ))

    verdict = (
        f"{len(compounders)} competitor(s) compounding: "
        f"{compounders[0]['brand']} gained {compounders[0]['rank_gain_pct']*100:+.1f}% BSR "
        f"over {compounders[0]['streak_weeks']} weeks"
        if active and compounders else
        "No sustained competitor compounding detected"
    )

    return RegimeSignal(
        regime=regime,
        active=active,
        confidence=confidence,
        verdict=verdict,
        driver_type="Market-driven" if active else "Unknown",
        receipts=receipts[:2],
        evidence=[f"{c['brand']} ({c['asin']}): {c['streak_weeks']}wk gain" for c in compounders],
        metadata={
            "compounders": compounders,
            "your_brand_declining": your_declining,
        },
    )


# ─── D: DEMAND TAILWIND / HEADWIND ───────────────────────────────────────────

def detect_demand_tailwind(
    df_weekly: pd.DataFrame,
    cfg: dict,
) -> RegimeSignal:
    """
    Detect broad-based rank movement across the arena not driven by individual pricing.

    Fires when:
    - A large fraction of the arena moves in the same rank direction
    - No single brand accounts for too much of the movement (diffuse)
    """
    regime = "demand_tailwind"

    if df_weekly.empty:
        return _empty_signal(regime)

    arena_fraction_min = cfg.get("arena_fraction_min", 0.50)
    median_rank_change_pct = cfg.get("median_rank_change_pct", 0.10)
    max_single_brand_share = cfg.get("max_single_brand_share", 0.40)
    lookback_weeks = cfg.get("lookback_weeks", 4)
    high_conf_fraction = cfg.get("high_confidence_fraction", 0.60)

    if "sales_rank_filled" not in df_weekly.columns:
        return _empty_signal(regime, "No rank data")

    df = df_weekly[["asin", "brand", "week_start", "sales_rank_filled"]].dropna().copy()
    weeks = sorted(df["week_start"].unique())

    if len(weeks) < 2:
        return _empty_signal(regime, "Need at least 2 weeks")

    # Compare last week vs lookback_weeks ago
    latest = weeks[-1]
    reference = weeks[max(-lookback_weeks - 1, -len(weeks))]

    latest_df = df[df["week_start"] == latest].set_index("asin")
    ref_df = df[df["week_start"] == reference].set_index("asin")

    common_asins = latest_df.index.intersection(ref_df.index)
    if len(common_asins) < 5:
        return _empty_signal(regime, "Too few ASINs in common between periods")

    rank_changes = (
        (latest_df.loc[common_asins, "sales_rank_filled"]
         - ref_df.loc[common_asins, "sales_rank_filled"])
        / ref_df.loc[common_asins, "sales_rank_filled"]
    )
    median_change = rank_changes.median()

    # Direction: negative = improvement (lower BSR), positive = decline
    direction = "tailwind" if median_change < 0 else "headwind"

    # Fraction moving in the same direction as median
    if direction == "tailwind":
        movers = rank_changes[rank_changes < -median_rank_change_pct / 2]
    else:
        movers = rank_changes[rank_changes > median_rank_change_pct / 2]

    fraction_moving = len(movers) / len(rank_changes)

    # Concentration check: what fraction of movers are from one brand?
    mover_brands = latest_df.loc[movers.index, "brand"]
    top_brand_share = mover_brands.value_counts(normalize=True).iloc[0] if not mover_brands.empty else 1.0
    concentrated = top_brand_share >= max_single_brand_share

    active = (
        fraction_moving >= arena_fraction_min
        and abs(median_change) >= median_rank_change_pct
        and not concentrated
    )

    # ── Corroboration: monthly_sold_delta (Phase A) ─────────────────────────
    # When Amazon's own monthlySold data is available, it provides independent
    # confirmation (or contradiction) of the BSR-based demand direction.
    n_asins = df_weekly["asin"].nunique() if "asin" in df_weekly.columns else 0
    monthly_sold_corroboration: Optional[str] = None
    _ms_delta_available = False
    if "monthly_sold_delta" in df_weekly.columns:
        _latest_full = df_weekly[df_weekly["week_start"] == latest] if "week_start" in df_weekly.columns else df_weekly
        _ms_vals = _latest_full["monthly_sold_delta"].dropna()
        if len(_ms_vals) >= max(5, n_asins * 0.20 if n_asins > 0 else 5):
            _ms_delta_available = True
            _ms_median = _ms_vals.median()
            _ms_positive_frac = (_ms_vals > 0).mean()
            _ms_negative_frac = (_ms_vals < 0).mean()
            if direction == "tailwind" and _ms_positive_frac >= 0.50:
                monthly_sold_corroboration = "confirms"
            elif direction == "headwind" and _ms_negative_frac >= 0.50:
                monthly_sold_corroboration = "confirms"
            elif direction == "tailwind" and _ms_negative_frac >= 0.50:
                monthly_sold_corroboration = "contradicts"
            elif direction == "headwind" and _ms_positive_frac >= 0.50:
                monthly_sold_corroboration = "contradicts"
            else:
                monthly_sold_corroboration = "neutral"

    # ── Corroboration: sales_rank_drops (Phase A) ────────────────────────────
    # sales_rank_drops_30 = count of BSR improvement events in 30 days.
    # A high arena-wide count supports tailwind; low count weakens it.
    _srd_corroboration: Optional[str] = None
    if "sales_rank_drops_30" in df_weekly.columns:
        _latest_full_srd = df_weekly[df_weekly["week_start"] == latest] if "week_start" in df_weekly.columns else df_weekly
        _srd_vals = _latest_full_srd["sales_rank_drops_30"].dropna()
        if len(_srd_vals) >= 5:
            _srd_median = _srd_vals.median()
            if direction == "tailwind" and _srd_median >= 3:
                _srd_corroboration = "confirms"
            elif direction == "tailwind" and _srd_median <= 1:
                _srd_corroboration = "contradicts"
            else:
                _srd_corroboration = "neutral"

    # Confidence (with Phase A corroboration)
    if not active:
        confidence = "Low"
    elif fraction_moving >= high_conf_fraction and not concentrated:
        confidence = "High"
        # Phase A can upgrade Med→High or downgrade High→Med
        if monthly_sold_corroboration == "contradicts":
            confidence = "Med"  # BSR says tailwind but monthlySold disagrees
    else:
        confidence = "Med"
        # Phase A corroboration can upgrade Med→High
        if monthly_sold_corroboration == "confirms" and _srd_corroboration == "confirms":
            confidence = "High"

    # Receipts
    receipts = []
    if active:
        # Top 2 brands driving the movement
        brand_change = (
            df.groupby(["brand", "week_start"])["sales_rank_filled"]
            .mean()
            .unstack("week_start")
        )
        if latest in brand_change.columns and reference in brand_change.columns:
            brand_delta = (brand_change[latest] - brand_change[reference]) / brand_change[reference]
            top2 = brand_delta.dropna().nsmallest(2) if direction == "tailwind" else brand_delta.dropna().nlargest(2)
            for brand, delta in top2.items():
                receipts.append(Receipt(
                    label=f"{brand}: BSR {delta*100:+.1f}% ({direction})",
                    metric="sales_rank_filled",
                    value=0,
                    baseline=0,
                    delta_pct=round(delta, 4),
                    brand=brand,
                ))

    # Build verdict with corroboration note
    _corr_note = ""
    if monthly_sold_corroboration == "confirms":
        _corr_note = " — corroborated by Amazon monthlySold"
    elif monthly_sold_corroboration == "contradicts":
        _corr_note = " — note: Amazon monthlySold data diverges"

    verdict = (
        f"Demand {direction}: {fraction_moving:.0%} of arena moved "
        f"{'up' if direction == 'tailwind' else 'down'} "
        f"(median BSR {median_change*100:+.1f}% vs {lookback_weeks}wk ago){_corr_note}"
        if active else
        f"No broad demand shift: {fraction_moving:.0%} of arena moved, "
        f"{'concentrated in one brand' if concentrated else 'below threshold'}"
    )

    # Evidence
    evidence = [
        f"Arena fraction moving: {fraction_moving:.0%}",
        f"Median BSR change: {median_change*100:+.1f}%",
        f"Top brand concentration: {top_brand_share:.0%}",
    ]
    if monthly_sold_corroboration:
        evidence.append(f"monthly_sold_delta: {monthly_sold_corroboration}")
    if _srd_corroboration:
        evidence.append(f"sales_rank_drops_30: {_srd_corroboration}")

    return RegimeSignal(
        regime=regime,
        active=active,
        confidence=confidence,
        verdict=verdict,
        driver_type="Market-driven" if active else "Unknown",
        receipts=receipts[:2],
        evidence=evidence,
        metadata={
            "direction": direction,
            "fraction_moving": round(fraction_moving, 3),
            "median_rank_change": round(median_change, 4),
            "concentrated": concentrated,
            "monthly_sold_corroboration": monthly_sold_corroboration,
            "sales_rank_drops_corroboration": _srd_corroboration,
        },
    )


# ─── E: NEW ENTRANT ──────────────────────────────────────────────────────────

def detect_new_entrant(
    df_weekly: pd.DataFrame,
    cfg: dict,
) -> RegimeSignal:
    """
    Detect a new ASIN appearing with fast rank acceleration (aggressive launch).

    Fires when:
    - An ASIN's first appearance is within cfg["new_asin_days"] days
    - It reaches a threatening BSR (cfg["threat_bsr_threshold"])
    - Its rank is improving at cfg["rank_velocity_pct_per_week"] or faster
    """
    regime = "new_entrant"

    if df_weekly.empty:
        return _empty_signal(regime)

    new_days = cfg.get("new_asin_days", 56)
    threat_bsr = cfg.get("threat_bsr_threshold", 20000)
    rank_velocity = cfg.get("rank_velocity_pct_per_week", -0.10)
    review_velocity = cfg.get("review_velocity_weekly", 20)

    if "sales_rank_filled" not in df_weekly.columns:
        return _empty_signal(regime, "No rank data")

    _cols = ["asin", "brand", "week_start", "sales_rank_filled"]
    if "review_count" in df_weekly.columns:
        _cols.append("review_count")
    df = df_weekly[_cols].copy()
    df = df.dropna(subset=["sales_rank_filled"])

    cutoff = df["week_start"].max() - pd.Timedelta(days=new_days)

    # ASINs whose first week of data is within the new window
    first_seen = df.groupby("asin")["week_start"].min()
    new_asins = first_seen[first_seen >= cutoff].index.tolist()

    if not new_asins:
        return _empty_signal(regime, "No new ASINs detected in arena")

    threats = []
    for asin in new_asins:
        asin_df = df[df["asin"] == asin].sort_values("week_start")
        if asin_df.empty:
            continue

        latest_bsr = asin_df["sales_rank_filled"].iloc[-1]
        if latest_bsr > threat_bsr:
            continue  # Not threatening enough

        # Rank velocity: average week-over-week % improvement
        if len(asin_df) >= 2:
            ranks = asin_df["sales_rank_filled"].values
            wk_changes = [(ranks[i] - ranks[i - 1]) / ranks[i - 1] for i in range(1, len(ranks)) if ranks[i - 1] > 0]
            avg_velocity = np.mean(wk_changes) if wk_changes else 0
        else:
            avg_velocity = 0

        if avg_velocity > rank_velocity:  # Not improving fast enough
            continue

        # Review velocity (optional)
        rev_vel = None
        if "review_count" in asin_df.columns and len(asin_df) >= 2:
            rev_counts = asin_df["review_count"].dropna()
            if len(rev_counts) >= 2:
                rev_vel = (rev_counts.iloc[-1] - rev_counts.iloc[0]) / max(len(rev_counts) - 1, 1)

        brand = asin_df["brand"].iloc[0]
        threats.append({
            "asin": asin,
            "brand": brand,
            "latest_bsr": latest_bsr,
            "rank_velocity": avg_velocity,
            "review_velocity": rev_vel,
            "weeks_tracked": len(asin_df),
        })

    active = len(threats) > 0

    # Confidence
    if not active:
        confidence = "Low"
    else:
        has_rev_signal = any(
            t["review_velocity"] is not None and t["review_velocity"] >= review_velocity
            for t in threats
        )
        confidence = "High" if has_rev_signal else "Med"

    # Sort by BSR (lowest = most dangerous)
    threats.sort(key=lambda x: x["latest_bsr"])

    # Receipts
    receipts = []
    for t in threats[:2]:
        receipts.append(Receipt(
            label=(
                f"{t['brand']} ({t['asin']}): BSR {t['latest_bsr']:,.0f}, "
                f"rank {t['rank_velocity']*100:+.1f}%/wk"
            ),
            metric="sales_rank_filled",
            value=t["latest_bsr"],
            baseline=threat_bsr,
            delta_pct=t["rank_velocity"],
            asin=t["asin"],
            brand=t["brand"],
        ))

    verdict = (
        f"{len(threats)} new entrant(s) detected: "
        f"{threats[0]['brand']} at BSR {threats[0]['latest_bsr']:,.0f}, "
        f"gaining {threats[0]['rank_velocity']*100:+.1f}%/wk"
        if active else
        "No threatening new entrants detected"
    )

    return RegimeSignal(
        regime=regime,
        active=active,
        confidence=confidence,
        verdict=verdict,
        driver_type="Market-driven" if active else "Unknown",
        receipts=receipts[:2],
        evidence=[f"{t['brand']} ({t['asin']}): BSR {t['latest_bsr']:,.0f}" for t in threats],
        metadata={"threats": threats},
    )


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────

def detect_all_regimes(
    df_weekly: pd.DataFrame,
    your_brand: str,
    cfg: dict = None,
    df_daily: Optional[pd.DataFrame] = None,
) -> Dict[str, RegimeSignal]:
    """
    Run all 5 regime detectors and return results keyed by regime name.

    Args:
        df_weekly: Arena weekly panel (all ASINs × weeks)
        your_brand: Brand name string (to split portfolio vs competitors)
        cfg: Full config dict (from market_misattribution_module.py REGIME_THRESHOLDS).
             If None, falls back to module defaults.
        df_daily: Optional daily panel for higher-fidelity discount persistence.

    Returns:
        Dict mapping regime name → RegimeSignal
    """
    if cfg is None:
        try:
            from config.market_misattribution_module import REGIME_THRESHOLDS
            cfg = REGIME_THRESHOLDS
        except ImportError:
            cfg = {}

    results = {
        "tier_compression": detect_tier_compression(
            df_weekly, cfg.get("tier_compression", {})
        ),
        "promo_war": detect_promo_war(
            df_weekly, cfg.get("promo_war", {}), df_daily=df_daily
        ),
        "competitor_compounding": detect_competitor_compounding(
            df_weekly, cfg.get("competitor_compounding", {}), your_brand
        ),
        "demand_tailwind": detect_demand_tailwind(
            df_weekly, cfg.get("demand_tailwind", {})
        ),
        "new_entrant": detect_new_entrant(
            df_weekly, cfg.get("new_entrant", {})
        ),
    }

    return results


def active_regimes(signals: Dict[str, RegimeSignal]) -> List[RegimeSignal]:
    """Return only the firing regimes, sorted by confidence (High first)."""
    order = {"High": 0, "Med": 1, "Low": 2}
    active = [s for s in signals.values() if s.active]
    return sorted(active, key=lambda s: order.get(s.confidence, 3))


def build_baseline_signal(
    your_bsr_wow: Optional[float],
    arena_bsr_wow: Optional[float],
    band_fn=None,
    data_confidence: str = "Med",
) -> RegimeSignal:
    """
    Explicit Baseline RegimeSignal for weeks when no other regime fires.

    Provides Driver 1 receipts (brand vs arena divergence card) and routes
    to the "Baseline (No dominant market regime)" verdict path.
    driver_type="Unknown" so _build_misattribution_verdict() uses the
    tracking/divergence branch rather than the Market-driven branch.

    data_confidence: pass conf_score.label from build_brief() so the Baseline
    signal's confidence matches the data quality — not a hardcoded "Low".
    """
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"

    receipts: List[Receipt] = []

    if arena_bsr_wow is not None:
        receipts.append(Receipt(
            label=f"Arena median BSR {band_fn(arena_bsr_wow, 'rank_change')} WoW — no structural shift",
            metric="arena_bsr_wow",
            value=arena_bsr_wow,
            baseline=0.0,
            delta_pct=arena_bsr_wow,
        ))

    if your_bsr_wow is not None:
        delta = (your_bsr_wow - arena_bsr_wow) if arena_bsr_wow is not None else your_bsr_wow
        if arena_bsr_wow is not None:
            r2_label = (
                f"Brand BSR {band_fn(your_bsr_wow, 'rank_change')} WoW vs arena "
                f"{band_fn(arena_bsr_wow, 'rank_change')} — delta {band_fn(delta, 'rank_change')}"
            )
        else:
            r2_label = f"Brand BSR {band_fn(your_bsr_wow, 'rank_change')} WoW — no arena context"
        receipts.append(Receipt(
            label=r2_label,
            metric="brand_vs_arena_delta",
            value=delta,
            baseline=0.0,
            delta_pct=delta,
        ))

    while len(receipts) < 2:
        receipts.append(Receipt(
            label="No regime-level signals above confidence threshold",
            metric="",
            value=0.0,
            baseline=0.0,
            delta_pct=0.0,
        ))

    return RegimeSignal(
        regime="baseline",
        active=True,
        confidence=data_confidence,   # inherits data confidence — not a forced "Low"
        verdict="Baseline (No dominant market regime)",
        driver_type="Unknown",
        receipts=receipts[:2],
    )
