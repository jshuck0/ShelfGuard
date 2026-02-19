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
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# Posture display layer — translates internal ads_stance to marketer-friendly labels.
# Internal ads_stance values are unchanged; these dicts are display-only.
_POSTURE_DISPLAY = {
    "Scale":          "Scale",
    "Defend":         "Defend",
    "Hold":           "Hold budget",
    "Pause+Diagnose": "Pause incremental",
}

_POSTURE_NEXT_ACTION = {
    "Scale":          "Increase budget cap on winner; expand high-intent keywords",
    "Defend":         "Protect branded + hero terms; cap conquesting",
    "Hold":           "Maintain branded coverage; reallocate away from weak SKUs",
    "Pause+Diagnose": "Reduce bids; stop conquesting; fix retail readiness",
}

_VALIDATE_BY_STANCE = {
    "Scale":          "Gate: In ads console, confirm efficiency stable before adding budget.",
    "Defend":         "Gate: In ads console, confirm BB + impression share holding.",
    "Hold":           "Gate: In ads console, confirm no Buy Box loss.",
    "Pause+Diagnose": "Gate: Check OOS and listing health before resuming spend.",
}


# Allowed tags (exactly as spec'd)
ALLOWED_TAGS = {
    "Undercut victim",
    "Discount-driven gain",
    "Gaining in promo environment",
    "Stable leader",
    "Losing visibility",
    "Likely stock/BB issue",  # Only if proxy available
}


def _safe_int(v) -> Optional[int]:
    """Convert a value to int, returning None for missing/NaN values."""
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _discount_label(persistence: float) -> str:
    """Convert discount_persistence (0.0–1.0) to a promo activity level.

    Low (0–1 days discounted last 7), Medium (2–4 days), High (5–7 days).
    """
    days = round(persistence * 7)
    if days <= 1:
        return "Low"
    elif days <= 4:
        return "Medium"
    else:
        return "High"


def _derive_signal(m: "ASINMetrics") -> str:
    """Derive a one-word Signal label from the ASIN metrics row.

    Promo-heavy: Sustained discounting + gaining
    Undercut risk: Below benchmark + gaining
    Premium vulnerable: Above benchmark + losing
    Stable: everything else
    """
    disc_label = _discount_label(m.discount_persistence)
    is_above = m.price_vs_tier > 0.05
    is_below = m.price_vs_tier < -0.05
    gaining = m.has_momentum
    losing = m.bsr_wow > 0.05  # BSR worsening > 5%

    if disc_label == "High" and gaining:
        return "Promo-heavy"
    if is_below and gaining:
        return "Undercut risk"
    if is_above and losing:
        return "Premium vulnerable"
    return "Stable"


def _derive_group_signal(g: "ProductGroupMetrics") -> str:
    """Derive a one-word Signal label for a group row."""
    disc_label = _discount_label(g.pct_discounted)
    is_above = g.median_price_vs_tier > 0.05
    is_below = g.median_price_vs_tier < -0.05
    gaining = g.momentum_label in ("gaining", "mixed")
    losing = g.momentum_label == "losing"

    if disc_label == "High" and gaining:
        return "Promo-heavy"
    if is_below and gaining:
        return "Undercut risk"
    if is_above and losing:
        return "Premium vulnerable"
    return "Stable"


def _band_price_vs_tier(ratio: float) -> str:
    """Convert price_vs_tier ratio to a readable position band."""
    if ratio < -0.15:
        return "well below category median"
    elif ratio < -0.05:
        return "below category median"
    elif ratio <= 0.05:
        return "in line"
    elif ratio <= 0.15:
        return "above category median"
    else:
        return "well above category median"


@dataclass
class ASINMetrics:
    asin: str
    brand: str
    role: str                       # "Core" | "Challenger" | "Long-tail"
    tag: str                        # One of ALLOWED_TAGS
    ad_waste_risk: str              # "High" | "Med" | "Low"
    discount_persistence: float     # days/7 (0.0–1.0) or None if unknown
    price_vs_tier: float            # (own_price - tier_median) / tier_median — used for calc
    price_vs_tier_band: str         # "well below category median" … "well above category median" | "not comparable"
    tier_comparable: bool           # False if number_of_items is missing/extreme
    bsr_wow: float                  # BSR week-over-week % change
    has_momentum: bool              # True if rank improving
    fidelity: str                   # "daily" | "weekly" | "snapshot"
    ads_stance: str = "Hold"        # "Scale" | "Defend" | "Hold" | "Pause+Diagnose"
    product_type: str = "other"     # e.g. "serum", "moisturizer" — from classify_title(title)
    family_id: str = ""             # parent_asin if known, else own asin — used for group dedup
    concerns: List[str] = field(default_factory=list)  # 0–2 concern/active labels from tag_concerns()
    return_rate: Optional[int] = None          # 1=low, 2=high, None=unknown (Keepa returnRate)
    sales_rank_drops_30: Optional[int] = None  # Count of BSR improvements in last 30d
    sales_rank_drops_90: Optional[int] = None  # Count of BSR improvements in last 90d
    monthly_sold_delta: Optional[int] = None   # monthlySold change since previous Keepa reading
    top_comp_bb_share_30: Optional[float] = None  # Highest non-Amazon BB share (0–1), None = no data
    ad_waste_reason: Optional[str] = None     # Explains why ad_waste_risk was set (for brief explainability)
    has_buybox_stats: bool = False             # Data presence flag — suppress BB language when False
    has_monthly_sold_history: bool = False     # Data presence flag — suppress demand delta when False


@dataclass
class ProductGroupMetrics:
    """Aggregated metrics for a product_type × brand group."""
    product_type: str
    brand: str
    asin_count: int
    rev_share_pct: float            # fraction of total arena revenue (0.0–1.0)
    median_price_vs_tier: float     # median price_vs_tier across group ASINs
    pct_discounted: float           # fraction with discount_persistence >= 4/7
    momentum_label: str             # "gaining" | "mixed" | "losing" | "flat"
    dominant_ads_stance: str        # most common ads_stance across group
    top_asins: List[str]            # up to 5 ASINs sorted by |bsr_wow| for receipt expansion
    pct_losing: float = 0.0         # fraction of group ASINs with bsr_wow > 0.05


@dataclass
class ConcernGroupMetrics:
    """Aggregated metrics for a concern × brand group (multi-label explode)."""
    concern: str
    brand: str
    asin_count: int
    rev_share_pct: float
    pct_discounted: float
    momentum_label: str             # "gaining" | "mixed" | "losing" | "flat"
    dominant_ads_stance: str
    top_asins: List[str]


def compute_asin_metrics(
    df_weekly: pd.DataFrame,
    role_cfg: dict,
    risk_cfg: dict,
    your_brand: str = "",
    df_daily: Optional[pd.DataFrame] = None,
    competitor_pressure: bool = False,
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

    # Import taxonomy classifier (lazy — avoids hard dep at module level)
    try:
        from config.market_misattribution_module import classify_title as _classify_title
    except ImportError:
        _classify_title = lambda t: "other"

    for asin in latest.index:
        row = latest.loc[asin]
        brand = str(row.get("brand", "Unknown"))

        # ── Product taxonomy ───────────────────────────────────────────────
        _title = str(row.get("title", "")) if "title" in row.index else ""
        _itk = str(row.get("item_type_keyword", "") or "") if "item_type_keyword" in row.index else ""
        product_type = _classify_title(_title, item_type_keyword=_itk)
        from config.market_misattribution_module import tag_concerns as _tag_concerns
        _ingredients = str(row.get("active_ingredients_raw", "") or "")
        concerns = _tag_concerns(_title, ingredients=_ingredients)

        # family_id: parent_asin when available (for variant grouping), else own asin
        _parent_rows = (
            df_weekly[df_weekly["asin"] == asin]["parent_asin"]
            if "parent_asin" in df_weekly.columns
            else pd.Series(dtype=str)
        )
        family_id = (
            str(_parent_rows.iloc[0])
            if not _parent_rows.empty and pd.notna(_parent_rows.iloc[0])
            else asin
        )

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
        tier_comparable = bool(
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
        _return_rate = _safe_int(row.get("return_rate"))

        if _return_rate == 2:
            ad_waste_risk = "High"
            ad_waste_reason = "High return rate (Keepa returnRate=2)"
        elif is_expensive and is_deteriorating:
            ad_waste_risk = "High"
            ad_waste_reason = "Price above category median + BSR deteriorating"
        elif is_competitive and is_improving:
            ad_waste_risk = "Low"
            ad_waste_reason = "Competitive pricing + rank improving"
        else:
            ad_waste_risk = "Med"
            ad_waste_reason = None

        # ── Tag Assignment ────────────────────────────────────────────────
        discount_pers = discount_persistence_map.get(asin, 0.0)
        has_bb_proxy = (
            "oos_pct_30" in row.index and
            "amazon_bb_share" in row.index and
            (row.get("oos_pct_30", 0) > 0.10 or row.get("amazon_bb_share", 1.0) < 0.40)
        )

        # Priority order for tag selection
        # "Discount-driven gain" only when the SKU *itself* is discounted
        # (discount_pers ≥ threshold).  If the SKU isn't discounted but is
        # competitive & gaining, use "Gaining in promo environment".
        _sku_is_discounted = discount_pers >= 0.30  # SKU actively on promo
        if is_expensive and is_deteriorating and discount_pers < 0.20:
            tag = "Undercut victim"
        elif _sku_is_discounted and has_momentum:
            tag = "Discount-driven gain"
        elif bsr_wow >= 0.10:
            tag = "Losing visibility"
        elif has_bb_proxy:
            tag = "Likely stock/BB issue"
        elif role == "Core" and abs(bsr_wow) <= 0.03 and not is_expensive:
            tag = "Stable leader"
        elif is_competitive and has_momentum:
            tag = "Gaining in promo environment"
        elif is_deteriorating:
            tag = "Losing visibility"
        else:
            tag = "Stable leader"

        # ── Ads Stance ────────────────────────────────────────────────────────
        # Keepa-safe: uses only price competitiveness + BSR momentum + promo pressure
        # Phrased as conditional — does NOT assert brand is running ads
        if not tier_comparable:
            ads_stance = "Hold"
        elif not is_competitive and is_deteriorating and competitor_pressure:
            ads_stance = "Pause+Diagnose"
        elif is_competitive and bsr_wow < -0.10:   # significant rank gain
            ads_stance = "Scale"
        elif is_competitive:                        # competitive, flat or modest gain
            ads_stance = "Defend"
        else:
            ads_stance = "Hold"

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
            ads_stance=ads_stance,
            product_type=product_type,
            family_id=family_id,
            concerns=concerns,
            return_rate=_return_rate,
            sales_rank_drops_30=_safe_int(row.get("sales_rank_drops_30")),
            sales_rank_drops_90=_safe_int(row.get("sales_rank_drops_90")),
            monthly_sold_delta=_safe_int(row.get("monthly_sold_delta")),
            top_comp_bb_share_30=row.get("top_comp_bb_share_30"),
            ad_waste_reason=ad_waste_reason,
            has_buybox_stats=bool(row.get("has_buybox_stats", False)),
            has_monthly_sold_history=bool(row.get("has_monthly_sold_history", False)),
        )

    return results


def compute_group_metrics(
    asin_metrics: Dict[str, "ASINMetrics"],
    df_weekly: pd.DataFrame,
    your_brand: str,
) -> List["ProductGroupMetrics"]:
    """
    Aggregate ASINMetrics into product_type × brand groups.

    Returns list sorted: your brand first, then by rev_share_pct descending.
    """
    from collections import defaultdict

    # Total arena revenue for share calculation (latest week)
    rev_col = "weekly_revenue_adjusted" if "weekly_revenue_adjusted" in df_weekly.columns else "weekly_revenue"
    latest_week = df_weekly["week_start"].max() if "week_start" in df_weekly.columns else None
    latest_snap = df_weekly[df_weekly["week_start"] == latest_week] if latest_week is not None else df_weekly
    total_rev = (
        latest_snap.groupby("asin")[rev_col].mean().sum()
        if rev_col in latest_snap.columns else 0
    )

    # Group by (product_type, brand)
    groups: dict = defaultdict(list)
    for m in asin_metrics.values():
        groups[(m.product_type, m.brand)].append(m)

    result = []
    for (ptype, brand), members in groups.items():
        asin_set = {m.asin for m in members}

        # Revenue share
        group_snap = latest_snap[latest_snap["asin"].isin(asin_set)] if rev_col in latest_snap.columns else pd.DataFrame()
        group_rev = group_snap.groupby("asin")[rev_col].mean().sum() if not group_snap.empty else 0
        rev_share = group_rev / total_rev if total_rev > 0 else 0

        med_price = float(np.median([m.price_vs_tier for m in members]))
        pct_disc = sum(1 for m in members if m.discount_persistence >= 4 / 7) / len(members)

        gaining = sum(1 for m in members if m.has_momentum)
        losing = sum(1 for m in members if m.bsr_wow > 0.05)
        n = len(members)
        pct_losing_val = losing / n
        if gaining > losing and gaining > n // 3:
            momentum = "gaining"
        elif losing > gaining and losing > n // 3:
            momentum = "losing"
        elif gaining == 0 and losing == 0:
            momentum = "flat"
        else:
            momentum = "mixed"

        stances = [m.ads_stance for m in members]
        dominant_stance = max(set(stances), key=stances.count)
        top_5 = sorted(members, key=lambda m: -abs(m.bsr_wow))[:5]

        result.append(ProductGroupMetrics(
            product_type=ptype,
            brand=brand,
            asin_count=len(members),
            rev_share_pct=round(rev_share, 4),
            median_price_vs_tier=round(med_price, 4),
            pct_discounted=round(pct_disc, 2),
            momentum_label=momentum,
            dominant_ads_stance=dominant_stance,
            top_asins=[m.asin for m in top_5],
            pct_losing=round(pct_losing_val, 2),
        ))

    # Sort: your brand first, then by rev_share_pct desc
    result.sort(key=lambda g: (0 if g.brand.lower() == your_brand.lower() else 1, -g.rev_share_pct))
    return result


def _pressure_score(pct_losing: float, pct_discounted: float, median_price_vs_tier: float) -> float:
    """Composite pressure score for a segment. Higher = more under pressure."""
    above_tier_norm = min(max(median_price_vs_tier, 0) / 0.15, 1.0)
    return 0.4 * pct_losing + 0.4 * pct_discounted + 0.2 * above_tier_norm


def compute_concern_metrics(
    asin_metrics: Dict[str, "ASINMetrics"],
    df_weekly: "pd.DataFrame",
    your_brand: str,
) -> List["ConcernGroupMetrics"]:
    """
    Aggregate ASINMetrics into concern × brand groups (multi-label explode).
    ASINs with 0 concerns are excluded. ASINs with 2 concerns appear in 2 groups.
    Returns list sorted: your brand first, then by rev_share_pct desc.
    """
    from collections import defaultdict

    rev_col = "weekly_revenue_adjusted" if "weekly_revenue_adjusted" in df_weekly.columns else "weekly_revenue"
    latest_week = df_weekly["week_start"].max() if "week_start" in df_weekly.columns else None
    latest_snap = df_weekly[df_weekly["week_start"] == latest_week] if latest_week is not None else df_weekly
    total_rev = (
        latest_snap.groupby("asin")[rev_col].mean().sum()
        if rev_col in latest_snap.columns else 0
    )

    groups: dict = defaultdict(list)
    for m in asin_metrics.values():
        for concern in m.concerns:
            groups[(concern, m.brand)].append(m)

    result = []
    for (concern, brand), members in groups.items():
        asin_set = {m.asin for m in members}
        group_snap = (latest_snap[latest_snap["asin"].isin(asin_set)]
                      if rev_col in latest_snap.columns else pd.DataFrame())
        group_rev = group_snap.groupby("asin")[rev_col].mean().sum() if not group_snap.empty else 0
        rev_share = group_rev / total_rev if total_rev > 0 else 0

        pct_disc = sum(1 for m in members if m.discount_persistence >= 4 / 7) / len(members)
        gaining = sum(1 for m in members if m.has_momentum)
        losing = sum(1 for m in members if m.bsr_wow > 0.05)
        n = len(members)
        if gaining > losing and gaining > n // 3:
            momentum = "gaining"
        elif losing > gaining and losing > n // 3:
            momentum = "losing"
        elif gaining == 0 and losing == 0:
            momentum = "flat"
        else:
            momentum = "mixed"

        stances = [m.ads_stance for m in members]
        dominant_stance = max(set(stances), key=stances.count)
        top_5 = sorted(members, key=lambda m: -abs(m.bsr_wow))[:5]

        result.append(ConcernGroupMetrics(
            concern=concern,
            brand=brand,
            asin_count=len(members),
            rev_share_pct=round(rev_share, 4),
            pct_discounted=round(pct_disc, 2),
            momentum_label=momentum,
            dominant_ads_stance=dominant_stance,
            top_asins=[m.asin for m in top_5],
        ))

    result.sort(key=lambda g: (0 if g.brand.lower() == your_brand.lower() else 1, -g.rev_share_pct))
    return result


def to_group_table(
    group_metrics: List["ProductGroupMetrics"],
    band_fn=None,
    max_groups: int = 15,
) -> pd.DataFrame:
    """
    Build a product_type × brand summary DataFrame for Layer B group view.

    One row per group. Used in skincare module mode instead of flat per-ASIN table.
    """
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"
    rows = []
    for g in group_metrics[:max_groups]:
        rows.append({
            "Product Type": g.product_type.replace("_", " ").title(),
            "Brand": g.brand,
            "SKUs": g.asin_count,
            "Est. Share (proxy)": f"{g.rev_share_pct:.0%}",
            "Price vs Category": band_fn(g.median_price_vs_tier, "price_vs_tier"),
            "Discounting": _discount_label(g.pct_discounted),
            "Momentum": g.momentum_label.title(),
            "Signal": _derive_group_signal(g),
            "If on ads": g.dominant_ads_stance,
            "_share_sort": g.rev_share_pct,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("_share_sort", ascending=False).drop(columns=["_share_sort"])


def to_compact_table(
    asin_metrics: Dict[str, ASINMetrics],
    df_weekly: pd.DataFrame,
    max_asins: int = 30,
    band_fn=None,
) -> pd.DataFrame:
    """
    Build the compact per-ASIN table for the brief's Layer B.

    Columns: ASIN (short), Brand, Role, Price vs Category, Discounting,
             Visibility WoW, Signal, Tag

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

    # Collapse variants: for each parent_asin, keep only the top-role / most-improving child
    _keeper_asins: set = set(asin_metrics.keys())
    if "parent_asin" in df_weekly.columns and "asin" in df_weekly.columns:
        _parent_map: dict = {}
        for _a in asin_metrics:
            _parent_rows = df_weekly[df_weekly["asin"] == _a]["parent_asin"]
            _parent = (
                str(_parent_rows.iloc[0])
                if not _parent_rows.empty and pd.notna(_parent_rows.iloc[0])
                else _a
            )
            _parent_map.setdefault(_parent, []).append(_a)
        _role_pri = {"Core": 0, "Challenger": 1, "Long-tail": 2}
        _keeper_asins = set()
        for _children in _parent_map.values():
            _best = min(
                _children,
                key=lambda a: (
                    _role_pri.get(asin_metrics[a].role, 3),
                    asin_metrics[a].bsr_wow,
                ),
            )
            _keeper_asins.add(_best)
        # Safety: if dedup produced empty set (shouldn't happen), show all
        if not _keeper_asins:
            _keeper_asins = set(asin_metrics.keys())

    role_order = {"Core": 0, "Challenger": 1, "Long-tail": 2}
    rows = []
    for asin, m in asin_metrics.items():
        if asin not in _keeper_asins:
            continue
        _title_part = title_map.get(asin, asin[-8:])
        short_name = f"{_title_part} ({asin[-6:]})"

        # Band display — use pre-computed band (normalized, pack-size aware)
        if band_fn:
            price_band = band_fn(m.price_vs_tier, "price_vs_tier")
            bsr_band = band_fn(m.bsr_wow, "rank_change")
        else:
            price_band = m.price_vs_tier_band  # "well below category median" … "not comparable"
            bsr_band = f"{m.bsr_wow*100:+.1f}%"

        rows.append({
            "ASIN": short_name,
            "Brand": m.brand,
            "Role": m.role,
            "Price vs Category": price_band,
            "Discounting": _discount_label(m.discount_persistence),
            "Visibility WoW": bsr_band,
            "Signal": _derive_signal(m),
            "Tag": m.tag,
            "Posture": _POSTURE_DISPLAY.get(m.ads_stance, m.ads_stance),
            "Next action": _POSTURE_NEXT_ACTION.get(m.ads_stance, ""),
            "_role_sort": role_order.get(m.role, 3),
            "_bsr_sort": m.bsr_wow,
            "_asin": asin,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["_role_sort", "_bsr_sort"]).head(max_asins)
    return df.drop(columns=["_role_sort", "_bsr_sort", "_asin"], errors="ignore")


def phase_a_receipt_extras(m: "ASINMetrics") -> str:
    """Build compact suffix with Phase A signals for per-ASIN receipt lines.

    Missing-data rules:
      - If has_monthly_sold_history is False, suppress demand delta.
      - If has_buybox_stats is False, suppress Buy Box competition line
        and append "Buy Box signal unavailable."
    """
    parts = []

    # Demand delta — only when monthlySold history is actually present
    _has_msh = getattr(m, "has_monthly_sold_history", False)
    _delta = getattr(m, "monthly_sold_delta", None)
    if _has_msh and _delta is not None:
        if _delta > 0:
            parts.append(f"demand +{_delta}")
        elif _delta < 0:
            parts.append(f"demand {_delta}")

    # Visibility drops (terminology-locked: "visibility" not "BSR")
    _drops = getattr(m, "sales_rank_drops_30", None)
    if _drops is not None and _drops >= 5:
        parts.append(f"{_drops} visibility drops/30d")

    # Buy Box competition — only when buybox stats are available
    _has_bb = getattr(m, "has_buybox_stats", False)
    _bb = getattr(m, "top_comp_bb_share_30", None)
    if _has_bb and _bb is not None and _bb > 0:
        parts.append(f"BB competition elevated (top non-Amazon seller won {_bb*100:.0f}% in last 30d)")
    elif not _has_bb:
        parts.append("Buy Box signal unavailable")

    # Surface ad_waste_reason when risk is not Low (explainability)
    _ad_risk = getattr(m, "ad_waste_risk", "")
    _reason = getattr(m, "ad_waste_reason", None)
    if _reason and _ad_risk and _ad_risk != "Low":
        parts.append(f"ad risk: {_reason}")

    if not parts:
        return ""
    return " | " + ", ".join(parts)


def receipts_list(
    asin_metrics: Dict[str, ASINMetrics],
    your_brand: str,
    max_items: int = 8,
    band_fn=None,
    runs_ads: Optional[bool] = None,
) -> List[str]:
    """
    Build Layer A: curated ads-relevant ASIN receipts.

    Curation order (ads-priority):
      1. Your brand Scale SKUs (up to 2) — spend opportunities
      2. Your brand Pause+Diagnose SKUs (up to 2) — action items
      3. Top discounting competitor gaining rank (up to 1) — pressure driver
      4. Your brand Defend hero SKU (up to 1) — protect coverage
    Falls back to pure |bsr_wow| sort if no brand ASINs in dict.

    Each line includes a Validate hint when runs_ads is not False.
    """
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"
    lines = []

    yours = [m for m in asin_metrics.values() if m.brand.lower() == your_brand.lower()]
    comps = [m for m in asin_metrics.values() if m.brand.lower() != your_brand.lower()]

    scale_skus = sorted(
        [m for m in yours if m.ads_stance == "Scale"],
        key=lambda m: -abs(m.bsr_wow),
    )[:2]
    pause_skus = sorted(
        [m for m in yours if m.ads_stance == "Pause+Diagnose"],
        key=lambda m: -abs(m.bsr_wow),
    )[:2]
    hero_skus = sorted(
        [m for m in yours if m.ads_stance == "Defend"],
        key=lambda m: -abs(m.bsr_wow),
    )[:1]
    comp_pressure = sorted(
        [m for m in comps if m.discount_persistence >= 4 / 7 and m.bsr_wow < -0.03],
        key=lambda m: -abs(m.bsr_wow),
    )[:1]

    curated = (scale_skus + pause_skus + comp_pressure + hero_skus)[:max_items]
    if not curated:
        curated = sorted(asin_metrics.values(), key=lambda m: -abs(m.bsr_wow))[:max_items]

    for m in curated:
        conf = "High" if m.ad_waste_risk == "Low" and m.has_momentum else (
            "Low" if m.ad_waste_risk == "High" else "Med"
        )
        _bsr_dir = "BSR improving" if m.bsr_wow < -0.01 else ("BSR worsening" if m.bsr_wow > 0.01 else "BSR flat")
        _promo_lvl = _discount_label(m.discount_persistence).lower()
        price_str = f"priced {m.price_vs_tier_band}"
        ads_hint = f" | If on ads: {_POSTURE_DISPLAY.get(m.ads_stance, m.ads_stance)}"
        validate_hint = ""
        if runs_ads is not False:
            validate_hint = f" | Validate: {_VALIDATE_BY_STANCE.get(m.ads_stance, '')}"
        _extra = phase_a_receipt_extras(m)
        lines.append(
            f"{m.brand} ({m.asin[-6:]}): {_bsr_dir} WoW; promo activity {_promo_lvl}; {price_str} — "
            f"{m.tag} [{conf} conf]{_extra}{ads_hint}{validate_hint}"
        )

    return lines
