"""
Weekly Brief Generator — Market Misattribution Shield
======================================================
Renders the 1-page forwardable brief as a markdown string.

No LLM required for structure. All sections are deterministic.
The brief can be rendered to:
    - Streamlit (via render_brief_tab())
    - Raw markdown string (via generate_brief_markdown())
    - Future: email/Slack via downstream caller

Brief sections (per spec):
    1. Headline
    2. What changed (max 3 bullets)
    3. Why (top 2 drivers, each with claim + 2 receipts + confidence)
    4. Misattribution Verdict
    5. Implications (max 3 bullets)
    6. Requests (2 validation asks + 1 coordination ask)
    7. What to watch (2 triggers)
    8. Scoreboard (last week's calls ✅/❌)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from features.regimes import RegimeSignal, detect_all_regimes, active_regimes, build_baseline_signal
from features.asin_metrics import (
    ASINMetrics, compute_asin_metrics, to_compact_table, receipts_list,
    ProductGroupMetrics, compute_group_metrics, to_group_table,
    ConcernGroupMetrics, compute_concern_metrics,
    phase_a_receipt_extras as _phase_a_receipt_extras,
)
from scoring.confidence import ConfidenceScore, score_confidence, score_driver_confidence


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class BriefDriver:
    """One driver in Section 3 (Why)."""
    claim: str
    receipts: List[str]          # Exactly 2 auditable receipt strings
    confidence: str              # "High" | "Med" | "Low"
    regime: Optional[str] = None # Which regime signal backs this


@dataclass
class SecondarySignal:
    """A secondary market signal with 2 auditable receipts."""
    claim: str
    receipts: List[str]          # Exactly 2


@dataclass
class WeeklyBrief:
    """Complete brief for one arena × one week."""
    brand: str
    arena_name: str
    week_label: str              # e.g. "Week of Feb 17, 2026"
    generated_at: datetime

    # Section 1
    headline: str

    # Section 2
    what_changed: List[str]      # Max 3 bullets

    # Section 3
    drivers: List[BriefDriver]   # Top 2 only

    # Section 4
    misattribution_verdict: str       # "Market-driven" | "Brand-driven" | "Unknown"
    misattribution_confidence: str    # "High" | "Med" | "Low"
    misattribution_receipts: List[str]  # Exactly 2

    # Section 5
    implications: List[str]      # Max 3 bullets (marketer language)
    plan_stance: str             # "Hold" | "Reallocate" | "Pause+Diagnose"
    measurement_focus: str       # What to validate internally

    # Section 6
    validation_asks: List[str]   # Exactly 2 yes/no asks
    coordination_ask: str        # 1 ask for who needs to do what

    # Section 7
    watch_triggers: List[str]    # Exactly 2

    # Section 8 (populated from eval/scoreboard.py)
    scoreboard_lines: List[str] = field(default_factory=list)

    # Metadata
    confidence_score: Optional[ConfidenceScore] = None
    runs_ads: Optional[bool] = None   # None = omit budget language
    active_regime_names: List[str] = field(default_factory=list)
    secondary_signals: List[SecondarySignal] = field(default_factory=list)
    actions_block: List[str] = field(default_factory=list)   # Section 5.5 — ad action checklist
    pressure_buckets: List[BriefDriver] = field(default_factory=list)  # §2.5 — top 2 segment drivers
    opportunity_bucket: Optional[BriefDriver] = None                    # §2.6 — top 1 concern signal
    concern_summary: List = field(default_factory=list)                 # List[ConcernGroupMetrics]
    data_quality: str = "Partial"
    data_fidelity: str = "weekly"     # "daily" | "weekly proxy"
    module_id: str = "generic"        # "skincare" | "generic" — controls taxonomy rendering
    group_summary: List = field(default_factory=list)   # List[ProductGroupMetrics]


# ─── BRIEF ASSEMBLY ──────────────────────────────────────────────────────────

def build_brief(
    df_weekly: pd.DataFrame,
    your_brand: str,
    arena_name: str = "",
    runs_ads: Optional[bool] = None,
    regime_cfg: dict = None,
    role_cfg: dict = None,
    risk_cfg: dict = None,
    confidence_cfg: dict = None,
    df_daily: Optional[pd.DataFrame] = None,
    scoreboard_lines: List[str] = None,
    category_path: str = "",      # For infer_module() — pass from session state or golden_run
    module_id: str = "",          # Override infer_module() if already known
) -> WeeklyBrief:
    """
    Build a WeeklyBrief from the arena weekly panel.

    Args:
        df_weekly: Arena weekly panel (all ASINs × weeks)
        your_brand: Brand name string
        arena_name: Human-readable arena label
        runs_ads: True/False/None — controls budget action language
        regime_cfg: REGIME_THRESHOLDS dict
        role_cfg: ASIN_ROLE_THRESHOLDS dict
        risk_cfg: AD_WASTE_RISK_THRESHOLDS dict
        confidence_cfg: CONFIDENCE_RUBRIC dict
        df_daily: Optional daily panel
        scoreboard_lines: Previous week's ✅/❌ lines from eval/scoreboard.py

    Returns:
        WeeklyBrief instance
    """
    # Load config defaults
    if regime_cfg is None or role_cfg is None or risk_cfg is None or confidence_cfg is None:
        try:
            from config.market_misattribution_module import (
                REGIME_THRESHOLDS, ASIN_ROLE_THRESHOLDS, AD_WASTE_RISK_THRESHOLDS,
                CONFIDENCE_RUBRIC, band_value,
            )
            regime_cfg = regime_cfg or REGIME_THRESHOLDS
            role_cfg = role_cfg or ASIN_ROLE_THRESHOLDS
            risk_cfg = risk_cfg or AD_WASTE_RISK_THRESHOLDS
            confidence_cfg = confidence_cfg or CONFIDENCE_RUBRIC
        except ImportError:
            regime_cfg = regime_cfg or {}
            role_cfg = role_cfg or {}
            risk_cfg = risk_cfg or {}
            confidence_cfg = confidence_cfg or {}
            band_value = lambda v, t: f"{v*100:+.1f}%"
    else:
        try:
            from config.market_misattribution_module import band_value
        except ImportError:
            band_value = lambda v, t: f"{v*100:+.1f}%"

    # ── Module / taxonomy ─────────────────────────────────────────────────────
    try:
        from config.market_misattribution_module import infer_module as _infer_module
        _module_id = module_id or (_infer_module(category_path) if category_path else "generic")
    except ImportError:
        _module_id = module_id or "generic"

    week_label = _format_week_label(df_weekly)
    price_col = "price_per_unit" if "price_per_unit" in df_weekly.columns else "filled_price"
    rank_col = "sales_rank_filled" if "sales_rank_filled" in df_weekly.columns else "sales_rank"

    # ── Detect regimes ────────────────────────────────────────────────────────
    regime_signals = detect_all_regimes(df_weekly, your_brand, regime_cfg, df_daily)
    firing = active_regimes(regime_signals)

    # ── ASIN metrics — pass competitor_pressure from pre-baseline firing ──────
    competitor_pressure = any(s.regime in ("promo_war", "tier_compression") for s in firing)
    asin_metrics = compute_asin_metrics(
        df_weekly, role_cfg, risk_cfg, your_brand, df_daily,
        competitor_pressure=competitor_pressure,
    )

    # ── Group metrics (product_type × brand aggregation) ─────────────────────
    group_metrics = compute_group_metrics(asin_metrics, df_weekly, your_brand)

    # ── Concern metrics (concern × brand aggregation, multi-label) ────────────
    concern_metrics = compute_concern_metrics(asin_metrics, df_weekly, your_brand)

    # ── Confidence score ─────────────────────────────────────────────────────
    conf_score = score_confidence(df_weekly, regime_signals, confidence_cfg)

    # ── Portfolio-level aggregates ────────────────────────────────────────────
    _brand_lower = df_weekly["brand"].fillna("").str.lower() if "brand" in df_weekly.columns else pd.Series("", index=df_weekly.index)
    your_asins_df = df_weekly[_brand_lower == your_brand.lower()]
    comp_df = df_weekly[_brand_lower != your_brand.lower()]

    latest_week = df_weekly["week_start"].max() if "week_start" in df_weekly.columns else None
    if latest_week is None:
        raise ValueError("df_weekly missing week_start column — cannot build brief")
    prev_week_opts = sorted(df_weekly["week_start"].unique())
    prev_week = prev_week_opts[-2] if len(prev_week_opts) >= 2 else None

    your_latest = your_asins_df[your_asins_df["week_start"] == latest_week]
    your_prev = your_asins_df[your_asins_df["week_start"] == prev_week] if prev_week else pd.DataFrame()

    # Demand: use median arena BSR change as proxy
    arena_bsr_wow = _compute_arena_bsr_wow(df_weekly, latest_week, prev_week, rank_col)
    your_bsr_wow = _compute_bsr_wow(your_asins_df, latest_week, prev_week, rank_col)

    # Inject Baseline if no regime fired — must come AFTER arena/brand BSR computed
    if not firing:
        firing = [build_baseline_signal(
            your_bsr_wow, arena_bsr_wow,
            band_fn=band_value,
            data_confidence=conf_score.label,  # Baseline inherits data confidence
        )]

    active_names = [s.regime for s in firing]

    # Price/promo regime — use comparable ASINs only for tier median
    # (exclude extreme pack-size outliers that would skew the arena baseline)
    _latest_snap = df_weekly[df_weekly["week_start"] == latest_week]
    if "number_of_items" in _latest_snap.columns:
        _items_med = _latest_snap["number_of_items"].replace(0, np.nan).median()
        if _items_med and _items_med > 0:
            _cmp = _latest_snap[
                _latest_snap["number_of_items"].notna()
                & (_latest_snap["number_of_items"] > 0)
                & (_latest_snap["number_of_items"] <= _items_med * 4)
            ]
            tier_median = _cmp[price_col].median() if not _cmp.empty and price_col in _cmp.columns else _latest_snap[price_col].median()
        else:
            tier_median = _latest_snap[price_col].median()
    else:
        tier_median = _latest_snap[price_col].median()
    # Use Core-role ASINs only for brand price position (avoids Long-tail outliers skewing tier call)
    _core_asins = {a for a, m in asin_metrics.items()
                   if m.role == "Core" and m.brand.lower() == your_brand.lower()}
    _price_df = (your_latest[your_latest["asin"].isin(_core_asins)]
                 if _core_asins and "asin" in your_latest.columns else your_latest)
    your_price = _price_df[price_col].mean() if not _price_df.empty and price_col in _price_df.columns else None
    price_vs_tier = (your_price - tier_median) / tier_median if your_price and tier_median and tier_median > 0 else None

    # Biggest mover (ASIN with largest BSR change WoW)
    biggest_mover = _find_biggest_mover(asin_metrics)

    # ── Secondary signals (max 2 — surfaced only when not already a driver) ──
    secondary_signals = _build_secondary_signals(
        price_vs_tier, asin_metrics, your_brand, firing, band_value
    )

    # ── Section 1: Headline ───────────────────────────────────────────────────
    headline = _build_headline(
        your_brand, your_bsr_wow, arena_bsr_wow, firing, conf_score,
        module_id=_module_id, group_metrics=group_metrics,
    )

    # ── Section 2: What Changed ───────────────────────────────────────────────
    what_changed = _build_what_changed(
        your_bsr_wow, arena_bsr_wow, price_vs_tier, biggest_mover, firing, band_value,
        module_id=_module_id, asin_metrics=asin_metrics,
        your_brand=your_brand, group_metrics=group_metrics,
    )

    # ── Section 3: Drivers ────────────────────────────────────────────────────
    drivers = _build_drivers(firing, regime_signals, conf_score, your_bsr_wow, arena_bsr_wow, band_fn=band_value)

    # ── Section 4: Misattribution Verdict ─────────────────────────────────────
    verdict, verdict_conf, verdict_receipts = _build_misattribution_verdict(
        firing, regime_signals, conf_score, your_bsr_wow, arena_bsr_wow, band_fn=band_value
    )

    # ── Section 5: Implications ───────────────────────────────────────────────
    implications, plan_stance, measurement_focus = _build_implications(
        firing, asin_metrics, your_brand, runs_ads, risk_cfg, conf_score,
        your_bsr_wow=your_bsr_wow, arena_bsr_wow=arena_bsr_wow,
    )

    # ── Pressure buckets + opportunity bucket (§2.5 / §2.6) ─────────────────
    pressure_buckets = _build_pressure_buckets(group_metrics, asin_metrics, your_brand, band_value)
    opportunity_bucket = _build_opportunity_bucket(concern_metrics, asin_metrics, your_brand, band_value)
    _pressure_ptypes = [d.regime for d in pressure_buckets] if pressure_buckets else None
    _opportunity_concern = opportunity_bucket.regime if opportunity_bucket else None

    # ── Section 5.5: Actions block ────────────────────────────────────────────
    actions_block = _build_actions_block(
        firing, asin_metrics, your_brand, plan_stance, runs_ads,
        band_fn=band_value, group_metrics=group_metrics,
        pressure_ptypes=_pressure_ptypes, opportunity_ptype=_opportunity_concern,
    )

    # ── Section 6: Requests ───────────────────────────────────────────────────
    validation_asks, coordination_ask = _build_requests(
        firing, verdict, your_brand, asin_metrics,
        module_id=_module_id, group_metrics=group_metrics,
    )

    # ── Section 7: Watch Triggers ─────────────────────────────────────────────
    watch_triggers = _build_watch_triggers(
        firing, regime_signals, your_bsr_wow, band_fn=band_value,
        asin_metrics=asin_metrics, your_brand=your_brand,
        module_id=_module_id, group_metrics=group_metrics,
    )

    return WeeklyBrief(
        brand=your_brand,
        arena_name=arena_name or f"{your_brand} Arena",
        week_label=week_label,
        generated_at=datetime.now(),
        headline=headline,
        what_changed=what_changed,
        drivers=drivers,
        misattribution_verdict=verdict,
        misattribution_confidence=verdict_conf,
        misattribution_receipts=verdict_receipts,
        implications=implications,
        plan_stance=plan_stance,
        measurement_focus=measurement_focus,
        validation_asks=validation_asks,
        coordination_ask=coordination_ask,
        watch_triggers=watch_triggers,
        scoreboard_lines=scoreboard_lines or [],
        confidence_score=conf_score,
        runs_ads=runs_ads,
        active_regime_names=active_names,
        secondary_signals=secondary_signals,
        actions_block=actions_block,
        pressure_buckets=pressure_buckets,
        opportunity_bucket=opportunity_bucket,
        concern_summary=concern_metrics,
        data_quality=conf_score.data_quality,
        data_fidelity="daily" if (df_daily is not None and not df_daily.empty) else "weekly proxy",
        module_id=_module_id,
        group_summary=group_metrics,
    )


# ─── SECTION BUILDERS ────────────────────────────────────────────────────────

def _format_week_label(df: pd.DataFrame) -> str:
    if "week_start" in df.columns and not df.empty:
        latest = df["week_start"].max()
        return f"Week of {latest.strftime('%b %d, %Y')}"
    return f"Week of {datetime.now().strftime('%b %d, %Y')}"


def _compute_bsr_wow(df: pd.DataFrame, latest: pd.Timestamp, prev, rank_col: str) -> Optional[float]:
    if prev is None or df.empty or rank_col not in df.columns:
        return None
    l = df[df["week_start"] == latest][rank_col].mean()
    p = df[df["week_start"] == prev][rank_col].mean()
    if pd.notna(l) and pd.notna(p) and p > 0:
        return (l - p) / p
    return None


def _compute_arena_bsr_wow(df: pd.DataFrame, latest, prev, rank_col: str) -> Optional[float]:
    if prev is None or df.empty or rank_col not in df.columns:
        return None
    l = df[df["week_start"] == latest][rank_col].median()
    p = df[df["week_start"] == prev][rank_col].median()
    if pd.notna(l) and pd.notna(p) and p > 0:
        return (l - p) / p
    return None


def _find_biggest_mover(asin_metrics: Dict[str, ASINMetrics]) -> Optional[ASINMetrics]:
    if not asin_metrics:
        return None
    return max(asin_metrics.values(), key=lambda m: abs(m.bsr_wow), default=None)


def _has_buckets(group_metrics) -> bool:
    """True when group_metrics has at least one non-'other' product_type."""
    return bool(group_metrics and any(g.product_type != "other" for g in group_metrics))


def _ptype_arena_stats(group_metrics: list) -> dict:
    """
    Aggregate per-(product_type, brand) groups into per-product_type arena stats.
    Returns: {ptype: {asin_count, pct_discounted, pct_gaining, pct_losing}}
    """
    from collections import defaultdict
    stats: dict = defaultdict(lambda: {"n": 0, "disc": 0.0, "gaining": 0, "losing": 0})
    for g in (group_metrics or []):
        if g.product_type == "other":
            continue
        pt = g.product_type
        stats[pt]["n"] += g.asin_count
        stats[pt]["disc"] += g.pct_discounted * g.asin_count
        if g.momentum_label == "gaining":
            stats[pt]["gaining"] += g.asin_count
        elif g.momentum_label == "losing":
            stats[pt]["losing"] += g.asin_count
    result = {}
    for pt, s in stats.items():
        n = s["n"] or 1
        result[pt] = {
            "asin_count": s["n"],
            "pct_discounted": s["disc"] / n,
            "pct_gaining": s["gaining"] / n,
            "pct_losing": s["losing"] / n,
        }
    return result


def _ptype_pressure_arena_stats(group_metrics: list) -> dict:
    """
    Aggregate all brands' group_metrics into per-product_type arena pressure stats.
    Uses pct_losing from ProductGroupMetrics (computed in Round 8).
    Returns: {ptype: {pct_losing, pct_discounted, median_price_vs_tier}}
    """
    from collections import defaultdict
    stats: dict = defaultdict(lambda: {"n": 0, "losing": 0.0, "disc": 0.0, "price_sum": 0.0})
    for g in (group_metrics or []):
        if g.product_type == "other":
            continue
        pt = g.product_type
        stats[pt]["n"] += g.asin_count
        stats[pt]["losing"] += getattr(g, "pct_losing", 0.0) * g.asin_count
        stats[pt]["disc"] += g.pct_discounted * g.asin_count
        stats[pt]["price_sum"] += g.median_price_vs_tier * g.asin_count
    result = {}
    for pt, s in stats.items():
        n = s["n"] or 1
        result[pt] = {
            "pct_losing": s["losing"] / n,
            "pct_discounted": s["disc"] / n,
            "median_price_vs_tier": s["price_sum"] / n,
        }
    return result


def _build_headline(brand, your_bsr, arena_bsr, firing, conf,
                    module_id="generic", group_metrics=None) -> str:
    if firing:
        top = firing[0]
        regime_label = top.regime.replace("_", " ").title()
        # Baseline regime: richer, marketer-facing language
        if top.regime == "baseline":
            return (
                f"{brand}: Baseline / No dominant market regime [{conf.label} confidence] — "
                "Market not forcing a directional move; act only on clear SKU-level pressure."
            )
        # Append top brand product_type bucket context when buckets available
        bucket_note = ""
        if _has_buckets(group_metrics):
            brand_groups = [
                g for g in group_metrics
                if g.brand.lower() == brand.lower() and g.product_type != "other"
            ]
            if brand_groups:
                top_g = brand_groups[0]  # sorted by rev_share_pct desc already
                bucket_note = f" — led by {top_g.product_type.replace('_', ' ').title()} bucket"
        return f"{brand}: {regime_label} detected this week{bucket_note} [{conf.label} confidence]"
    if your_bsr is not None:
        direction = "gaining" if your_bsr < -0.03 else ("losing" if your_bsr > 0.03 else "holding")
        arena_note = "in a broadly stable arena" if (arena_bsr is None or abs(arena_bsr) < 0.03) else "alongside broad arena movement"
        return f"{brand} {direction} visibility {arena_note} [{conf.label} confidence]"
    return f"{brand} arena — weekly brief [{conf.label} confidence]"


def _build_what_changed(your_bsr, arena_bsr, price_vs_tier, biggest_mover, firing, band_fn,
                        module_id="generic", asin_metrics=None,
                        your_brand="", group_metrics=None) -> List[str]:
    # Bucket-first path: when meaningful product_type groups are available
    if _has_buckets(group_metrics):
        bullets = []

        # Bullet 1: Your brand's top product_type momentum
        brand_groups = [
            g for g in group_metrics
            if g.brand.lower() == your_brand.lower() and g.product_type != "other"
        ]
        if brand_groups:
            top_g = brand_groups[0]  # sorted by rev_share_pct desc already
            ptype_name = top_g.product_type.replace("_", " ").title()
            bullets.append(f"**Your {ptype_name}:** {top_g.momentum_label.title()} BSR trend this week")

        # Bullet 2: Most discounted competitor bucket (if pct_disc >= 30%)
        ptype_stats = _ptype_arena_stats(group_metrics)
        comp_ptypes = {
            g.product_type for g in group_metrics
            if g.brand.lower() != your_brand.lower() and g.product_type != "other"
        }
        if comp_ptypes:
            pressured_pt = max(comp_ptypes,
                               key=lambda pt: ptype_stats.get(pt, {}).get("pct_discounted", 0))
            pt_stat = ptype_stats.get(pressured_pt, {})
            pct_disc = pt_stat.get("pct_discounted", 0)
            if pct_disc >= 0.30:
                ptype_name = pressured_pt.replace("_", " ").title()
                pct_gaining = pt_stat.get("pct_gaining", 0)
                gain_str = "competitors gaining" if pct_gaining > 0.30 else "flat competition"
                bullets.append(
                    f"**{ptype_name} arena:** {round(pct_disc * 100)}% discounted, {gain_str}"
                )

        # Bullet 3: Biggest mover with bucket tag (universal — no module_id gate)
        if biggest_mover and abs(biggest_mover.bsr_wow) > 0.05:
            bsr_move = band_fn(biggest_mover.bsr_wow, "rank_change")
            ptype_note = ""
            if asin_metrics:
                m = asin_metrics.get(biggest_mover.asin, biggest_mover)
                ptype = getattr(m, "product_type", "other")
                if ptype != "other":
                    ptype_note = f" [{ptype.replace('_', ' ').title()}]"
            bullets.append(
                f"**Biggest mover:** {biggest_mover.brand} ({biggest_mover.asin[-6:]})"
                f"{ptype_note} — BSR {bsr_move} this week"
            )

        return bullets[:3]

    # Fallback: standard arena/promo/mover bullets (no product_type groups)
    bullets = []

    # Demand bullet
    if arena_bsr is not None:
        direction = band_fn(arena_bsr, "rank_change")
        bullets.append(f"**Demand:** Arena-wide BSR {direction} WoW")
    elif your_bsr is not None:
        direction = band_fn(your_bsr, "rank_change")
        bullets.append(f"**Demand:** Your portfolio BSR {direction} WoW")

    # Price/promo regime bullet — only when a regime actually fired (no orphan fallback)
    promo_regime = next((s for s in firing if s.regime in ("promo_war", "tier_compression")), None)
    if promo_regime:
        bullets.append(f"**Price/Promo regime:** {promo_regime.verdict}")

    # Biggest mover bullet
    if biggest_mover and abs(biggest_mover.bsr_wow) > 0.05:
        bsr_move = band_fn(biggest_mover.bsr_wow, "rank_change")
        bullets.append(
            f"**Biggest mover:** {biggest_mover.brand} ({biggest_mover.asin[-6:]}) "
            f"— BSR {bsr_move} this week"
        )

    return bullets[:3]


def _build_pressure_buckets(
    group_metrics: list,
    asin_metrics: dict,
    your_brand: str,
    band_fn=None,
) -> List[BriefDriver]:
    """
    Returns up to 2 BriefDriver objects for the top product_type segments by pressure score.
    Minimum pressure score threshold: 0.15. Returns [] when insufficient signal.
    """
    from features.asin_metrics import _pressure_score
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"

    arena_stats = _ptype_pressure_arena_stats(group_metrics)
    if not arena_stats:
        return []

    scored = []
    all_ptypes = {g.product_type for g in (group_metrics or []) if g.product_type != "other"}
    for pt in all_ptypes:
        s = arena_stats.get(pt, {})
        score = _pressure_score(
            s.get("pct_losing", 0), s.get("pct_discounted", 0), s.get("median_price_vs_tier", 0)
        )
        if score >= 0.15:
            scored.append((pt, score, s))

    scored.sort(key=lambda x: -x[1])

    drivers = []
    for pt, score, s in scored[:2]:
        ptype_name = pt.replace("_", " ").title()
        pct_losing_pct = round(s.get("pct_losing", 0) * 100)
        pct_disc_pct = round(s.get("pct_discounted", 0) * 100)
        price_pos = band_fn(s.get("median_price_vs_tier", 0), "price_vs_tier")

        claim = (
            f"**{ptype_name}** under pressure: {pct_losing_pct}% of ASINs losing visibility, "
            f"{pct_disc_pct}% discounted, arena price {price_pos}"
        )

        brand_groups_pt = [g for g in (group_metrics or [])
                           if g.product_type == pt and g.brand.lower() == your_brand.lower()]
        if brand_groups_pt:
            bg = brand_groups_pt[0]
            r1 = (
                f"Your brand in {ptype_name}: {bg.momentum_label} trend, "
                f"{round(bg.pct_discounted * 7)}/7 days discounted, "
                f"price {band_fn(bg.median_price_vs_tier, 'price_vs_tier')}"
            )
        else:
            r1 = f"Your brand has no ASINs in {ptype_name} — no brand-side context available"

        comp_groups_pt = sorted(
            [g for g in (group_metrics or [])
             if g.product_type == pt and g.brand.lower() != your_brand.lower()
             and g.momentum_label in ("gaining", "mixed")],
            key=lambda g: -g.pct_discounted,
        )
        if comp_groups_pt:
            cg = comp_groups_pt[0]
            r2 = (
                f"{cg.brand} ({ptype_name}): {cg.momentum_label}, "
                f"{round(cg.pct_discounted * 7)}/7 days discounted, "
                f"price {band_fn(cg.median_price_vs_tier, 'price_vs_tier')}"
            )
        else:
            r2 = f"{ptype_name}: No single competitor dominating — distributed pressure"

        drivers.append(BriefDriver(claim=claim, receipts=[r1, r2], confidence="Med", regime=pt))

    return drivers


def _build_opportunity_bucket(
    concern_metrics: list,
    asin_metrics: dict,
    your_brand: str,
    band_fn=None,
) -> Optional[BriefDriver]:
    """
    Returns a BriefDriver for the top concern-level opportunity signal, or None.
    Signal A: your brand gaining while competitors discounting ≥30%.
    Signal B: your brand losing while arena concern is flat/gaining.
    Suppressed when your brand has < 2 ASINs in the concern.
    """
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"

    if not concern_metrics:
        return None

    best = None
    best_score = 0.0

    concerns_present = {g.concern for g in concern_metrics}
    for concern in concerns_present:
        concern_groups = [g for g in concern_metrics if g.concern == concern]
        brand_g = next(
            (g for g in concern_groups if g.brand.lower() == your_brand.lower()), None
        )
        if not brand_g or brand_g.asin_count < 2:
            continue

        comp_groups = [g for g in concern_groups if g.brand.lower() != your_brand.lower()]
        if not comp_groups:
            continue

        total_comp_asins = max(sum(g.asin_count for g in comp_groups), 1)
        comp_disc = sum(g.pct_discounted * g.asin_count for g in comp_groups) / total_comp_asins
        comp_momenta = [g.momentum_label for g in comp_groups]
        arena_momentum = max(set(comp_momenta), key=comp_momenta.count)

        if brand_g.momentum_label in ("gaining", "mixed") and comp_disc >= 0.30:
            opp_score = 0.6 * comp_disc + 0.4 * (1.0 if brand_g.momentum_label == "gaining" else 0.5)
        elif brand_g.momentum_label == "losing" and arena_momentum in ("flat", "gaining"):
            opp_score = 0.3
        else:
            continue

        if opp_score > best_score:
            best_score = opp_score
            best = (concern, brand_g, comp_groups, comp_disc, arena_momentum)

    if not best:
        return None

    concern, brand_g, comp_groups, comp_disc, arena_momentum = best
    concern_name = concern.replace("_", " ").title()

    if brand_g.momentum_label in ("gaining", "mixed"):
        claim = (
            f"**{concern_name} opportunity:** Your brand is {brand_g.momentum_label} "
            f"while {round(comp_disc * 100)}% of competitors are discounting — "
            "hold position and defend branded placements."
        )
    else:
        claim = (
            f"**{concern_name} watch:** Your brand is losing ground while the arena is {arena_momentum} — "
            "investigate SKU-level issues before attributing to market."
        )

    r1 = (
        f"Your brand {concern_name}: {brand_g.momentum_label} trend, "
        f"{round(brand_g.pct_discounted * 7)}/7 days discounted, "
        f"{brand_g.asin_count} ASINs in concern"
    )
    top_comp = sorted(comp_groups, key=lambda g: -g.pct_discounted)[0] if comp_groups else None
    r2 = (
        f"{top_comp.brand} ({concern_name}): {top_comp.momentum_label}, "
        f"{round(top_comp.pct_discounted * 7)}/7 days discounted"
        if top_comp else "Competitor concern data sparse"
    )

    return BriefDriver(claim=claim, receipts=[r1, r2], confidence="Med", regime=concern)


def _build_drivers(firing, all_signals, conf_score, your_bsr, arena_bsr, band_fn=None) -> List[BriefDriver]:
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"
    drivers = []

    for signal in firing[:2]:
        receipts_strs = [
            f"{r.label}" for r in signal.receipts[:2]
        ]
        while len(receipts_strs) < 2:
            receipts_strs.append("(additional data needed)")

        driver_conf = score_driver_confidence(signal.regime, signal.receipts, signal, conf_score)
        drivers.append(BriefDriver(
            claim=signal.verdict,
            receipts=receipts_strs,
            confidence=driver_conf,
            regime=signal.regime,
        ))

    # If no regimes fired, add a generic demand driver
    if not drivers:
        if arena_bsr is not None:
            drivers.append(BriefDriver(
                claim=f"Arena-wide rank movement ({band_fn(arena_bsr, 'rank_change')} median BSR WoW) — no specific regime detected",
                receipts=["Arena median BSR shift vs prior week", "No single brand driving movement"],
                confidence=conf_score.label,
            ))

    return drivers[:2]


def _build_misattribution_verdict(firing, all_signals, conf_score, your_bsr, arena_bsr, band_fn=None):
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"
    market_regimes = [s for s in firing if s.driver_type == "Market-driven"]
    brand_signals = []  # Would require Seller Central data — not available

    receipts = []

    if len(market_regimes) >= 2:
        verdict = "Market-driven"
        verdict_conf = "High" if all(s.confidence in ("High", "Med") for s in market_regimes) else "Med"
        for r in market_regimes[:2]:
            if r.receipts:
                receipts.append(r.receipts[0].label)
    elif len(market_regimes) == 1:
        verdict = "Market-driven (partial)"
        verdict_conf = "Med"
        if market_regimes[0].receipts:
            receipts.append(market_regimes[0].receipts[0].label)
        if arena_bsr is not None:
            receipts.append(f"Arena BSR {band_fn(arena_bsr, 'rank_change')} WoW (market context)")
    else:
        # Distinguish "we understand it's steady-state" from "we can't tell"
        _tracking = (
            your_bsr is not None
            and arena_bsr is not None
            and (your_bsr - arena_bsr) <= 0.07
        )
        if _tracking:
            verdict = "Baseline (No dominant market regime)"
            verdict_conf = conf_score.label if conf_score else "Med"  # data confidence, not forced "Low"
            receipts = [
                "Arena-wide BSR stable WoW — no active regime detected",
                "Brand tracking arena: performance consistent with steady-state",
            ]
        else:
            verdict = "Unknown"
            verdict_conf = "Low"
            receipts = [
                "No market-level regime detected with sufficient confidence",
                "Cannot rule out brand-level factors (listing, stock, content) without internal data",
            ]

    while len(receipts) < 2:
        receipts.append("(additional signal needed)")

    return verdict, verdict_conf, receipts[:2]


def _build_implications(
    firing, asin_metrics, your_brand, runs_ads, risk_cfg, conf_score,
    your_bsr_wow: Optional[float] = None,
    arena_bsr_wow: Optional[float] = None,
) -> tuple:
    bullets = []
    plan_stance = "Hold"

    # Exec narrative
    market_driven = any(s.driver_type == "Market-driven" for s in firing)
    if market_driven:
        bullets.append(
            "**Exec narrative:** Performance movement is consistent with market-level dynamics. "
            "Attribution to any single internal lever (ads, content, pricing) is uncertain."
        )
        plan_stance = "Hold"
    else:
        # Gate Pause+Diagnose on divergence from arena, not just "no regime"
        # If brand is tracking the arena → Hold (market conditions are likely driver)
        # If brand is underperforming the arena → Pause+Diagnose (internal factors likely)
        _is_divergent = (
            your_bsr_wow is not None
            and arena_bsr_wow is not None
            and (your_bsr_wow - arena_bsr_wow) > 0.07   # brand ≥7pp worse than arena
        ) or (
            your_bsr_wow is not None
            and arena_bsr_wow is None
            and your_bsr_wow > 0.10                      # brand clearly deteriorating, no context
        )
        if _is_divergent:
            bullets.append(
                "**Exec narrative:** Brand is underperforming the arena — internal factors "
                "(listing health, stock, pricing) merit investigation."
            )
            plan_stance = "Pause+Diagnose"
        else:
            bullets.append(
                "**Exec narrative:** No active market regime detected — arena stable, "
                "brand in steady state. Escalate only if brand diverges from arena movement."
            )
            plan_stance = "Hold"

    # Budget action
    high_risk_asins = [
        m for m in asin_metrics.values()
        if m.ad_waste_risk == "High" and m.brand.lower() == your_brand.lower()
    ]
    is_baseline_week = any(s.regime == "baseline" for s in firing)
    if is_baseline_week and runs_ads is not False:
        # Baseline week: conservative posture, always show if ads in scope (True or None)
        bullets.append(
            "**If you're supporting these ASINs with ads:** Don't scale incremental spend "
            "unless the SKU is price-competitive and momentum is positive. "
            "Reallocate away from High Ad Waste Risk SKUs."
        )
        if high_risk_asins:
            plan_stance = "Reallocate"
    elif runs_ads is not None:
        if runs_ads and high_risk_asins:
            # Build ASIN list with reasons for explainability
            _risk_parts = []
            for a in high_risk_asins[:3]:
                _reason = getattr(a, "ad_waste_reason", None)
                _label = f"{a.asin[-6:]}"
                if _reason:
                    _label += f" — {_reason}"
                _risk_parts.append(_label)
            asin_detail = "; ".join(_risk_parts)
            bullets.append(
                f"**If you are supporting these ASINs with ads**, avoid scaling incremental spend on "
                f"High Ad Waste Risk items ({asin_detail}) this week; reallocate toward price-competitive "
                f"core SKUs. Keep spend only if CVR and in-stock are stable (operator check)."
            )
            plan_stance = "Reallocate"
        elif not runs_ads:
            bullets.append(
                "**Ad spend:** Not applicable (ads not in scope for this arena)."
            )

    # Measurement focus
    promo = next((s for s in firing if s.regime == "promo_war"), None)
    comp = next((s for s in firing if s.regime == "competitor_compounding"), None)
    if promo:
        measurement_focus = "Validate whether your CVR has held despite price pressure in the arena"
    elif comp:
        measurement_focus = "Track the compounding competitor's review velocity and stock status over next 2 weeks"
    else:
        measurement_focus = "Check in-stock rate and Buy Box ownership for your core SKUs"

    bullets.append(f"**Measurement focus:** {measurement_focus}")

    return bullets[:3], plan_stance, measurement_focus


def _build_requests(firing, verdict, your_brand, asin_metrics,
                    module_id="generic", group_metrics=None) -> tuple:
    asks = []
    coordination = ""

    # Resolve top brand product_type for bucket-aware ask language
    _sku_label = "Core SKUs"
    if _has_buckets(group_metrics):
        _brand_groups = [
            g for g in group_metrics
            if g.brand.lower() == your_brand.lower() and g.product_type != "other"
        ]
        if _brand_groups:
            _top_ptype = _brand_groups[0].product_type.replace("_", " ").title()
            _sku_label = f"Core {_top_ptype} SKUs"

    # Validation ask 1: confirm market driver
    if "Market-driven" in verdict:
        asks.append(
            "Did your team observe any pricing, listing, or inventory changes last week "
            "that could explain the rank shift independent of market conditions? (Y/N)"
        )
    else:
        asks.append(
            f"Any Buy Box loss or suppression on {_sku_label} last week? (Y/N)"
        )

    # Validation ask 2: tied to specific regime
    comp_regime = next((s for s in firing if s.regime == "competitor_compounding"), None)
    promo_regime = next((s for s in firing if s.regime == "promo_war"), None)

    if comp_regime and comp_regime.metadata.get("compounders"):
        comp_brand = comp_regime.metadata["compounders"][0].get("brand", "the competitor")
        asks.append(
            f"Has {comp_brand} launched any new SKUs, run Vine, or changed their listing "
            f"in the past 30 days that you're aware of? (Y/N)"
        )
    elif promo_regime:
        asks.append(
            "Are any of your core SKUs running a coupon or deal this week that could "
            "explain rank gains concentrated among discounters? (Y/N)"
        )
    else:
        asks.append(f"Any OOS events on {_sku_label} last week? (Y/N)")

    # Coordination ask
    high_risk = [m for m in asin_metrics.values()
                 if m.ad_waste_risk == "High" and m.brand.lower() == your_brand.lower()]
    if high_risk:
        _hr_reasons = [getattr(m, "ad_waste_reason", None) for m in high_risk if getattr(m, "ad_waste_reason", None)]
        _reason_note = f" Reasons: {'; '.join(set(_hr_reasons))}." if _hr_reasons else ""
        coordination = (
            f"Ops: Confirm in-stock status for {len(high_risk)} High Ad Waste Risk ASIN(s) "
            f"by EOD — needed before any spend decisions.{_reason_note}"
        )
    elif any(s.regime == "tier_compression" for s in firing):
        coordination = (
            "Pricing team: Review whether current list prices remain defensible against "
            "sustained tier compression — flag any ASINs needing adjustment."
        )
    else:
        coordination = (
            "Agency: Annotate reporting with 'Baseline market week' — avoid attributing "
            "small performance deltas to creative or ads changes this period."
        )

    return asks[:2], coordination


def _build_actions_block(
    firing, asin_metrics, your_brand, plan_stance, runs_ads,
    band_fn=None, group_metrics=None,
    pressure_ptypes: Optional[List[str]] = None,
    opportunity_ptype: Optional[str] = None,
) -> List[str]:
    """
    Produce the 'If you run Sponsored Ads' action checklist.
    Always returns 3+ bullets (posture + optional cross-segment + reallocation + competitor pressure).
    When pressure_ptypes + opportunity_ptype both provided, prepends cross-segment reallocation.
    """
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"

    from features.asin_metrics import _POSTURE_DISPLAY  # noqa: F401 (imported for completeness)

    lines = []
    is_baseline = any(s.regime == "baseline" for s in firing)

    # 1. Budget posture
    if plan_stance == "Hold" or is_baseline:
        lines.append(
            "**Budget posture:** Hold total budget this week "
            "(Baseline market — no structural shift detected)."
        )
    elif plan_stance == "Reallocate":
        lines.append(
            "**Budget posture:** Reallocate — shift budget toward Scale/Defend SKUs; "
            "reduce on Pause incremental SKUs."
        )
    else:
        lines.append(
            "**Budget posture:** Pause incremental spend; protect core branded coverage only."
        )

    # 1.5. Cross-segment reallocation (concern-level signal)
    if pressure_ptypes and opportunity_ptype and opportunity_ptype not in (pressure_ptypes or []):
        from_names = ", ".join(p.replace("_", " ").title() for p in pressure_ptypes[:2])
        to_name = opportunity_ptype.replace("_", " ").title()
        lines.append(
            f"**Cross-segment reallocation:** Reduce spend in pressure segments ({from_names}) "
            f"\u2192 defend/scale in {to_name}."
        )

    # 2. Reallocation guidance — bucket-first when groups available, ASIN-level fallback
    yours = [m for m in asin_metrics.values() if m.brand.lower() == your_brand.lower()]
    pause_asins = [m for m in yours if m.ads_stance == "Pause+Diagnose"]
    scale_asins = [m for m in yours if m.ads_stance == "Scale"]
    defend_asins = [m for m in yours if m.ads_stance == "Defend"]

    brand_groups = [
        g for g in (group_metrics or [])
        if g.brand.lower() == your_brand.lower() and g.product_type != "other"
    ]
    if brand_groups:
        scale_buckets = [g for g in brand_groups if g.dominant_ads_stance == "Scale"]
        pause_buckets = [g for g in brand_groups if g.dominant_ads_stance == "Pause+Diagnose"]
        parts = []
        if scale_buckets:
            names = ", ".join(g.product_type.replace("_", " ").title() for g in scale_buckets[:2])
            parts.append(f"Scale \u2192 {names}")
        if pause_buckets:
            names = ", ".join(g.product_type.replace("_", " ").title() for g in pause_buckets[:2])
            parts.append(f"Pause incremental \u2192 {names}")
        if not parts:
            hold_names = ", ".join(g.product_type.replace("_", " ").title() for g in brand_groups[:3])
            parts.append(f"Hold budget \u2192 {hold_names}")
        lines.append(f"**Reallocation by bucket:** {' | '.join(parts)}")
    elif pause_asins and (scale_asins or defend_asins):
        pause_labels = ", ".join(m.asin[-6:] for m in pause_asins[:3])
        winner_labels = ", ".join(m.asin[-6:] for m in (scale_asins + defend_asins)[:3])
        lines.append(
            f"**Reallocation:** Move spend from Pause-incremental SKUs ({pause_labels}) "
            f"\u2192 Scale/Defend SKUs ({winner_labels})."
        )
    elif pause_asins:
        pause_labels = ", ".join(m.asin[-6:] for m in pause_asins[:3])
        lines.append(
            f"**Reallocation:** Reduce spend on Pause-incremental SKUs ({pause_labels}); "
            "hold branded coverage elsewhere."
        )
    else:
        lines.append(
            "**Reallocation:** No immediate reallocation needed — maintain current allocation."
        )

    # 3. Competitor pressure callout
    comps = [
        m for m in asin_metrics.values()
        if m.brand.lower() != your_brand.lower()
        and m.discount_persistence >= 4 / 7
        and m.bsr_wow < -0.03
    ]
    comps_sorted = sorted(comps, key=lambda m: -abs(m.bsr_wow))
    if comps_sorted:
        top_comp = comps_sorted[0]
        comp_bsr = band_fn(top_comp.bsr_wow, "rank_change")
        if scale_asins or defend_asins:
            lines.append(
                f"**Competitor pressure:** {top_comp.brand} is discounting and gaining "
                f"({comp_bsr} BSR WoW) — defend branded + hero placements; "
                "conquest selectively only on price-competitive SKUs."
            )
        else:
            lines.append(
                f"**Competitor pressure:** {top_comp.brand} is discounting and gaining "
                f"({comp_bsr} BSR WoW) — monitor; no conquest until your SKUs are "
                "price-competitive."
            )
    else:
        lines.append(
            "**Competitor pressure:** No discounting + gaining competitors detected this week."
        )

    return lines


def _build_watch_triggers(
    firing, all_signals, your_bsr, band_fn=None,
    asin_metrics=None, your_brand="",
    module_id="generic", group_metrics=None,
) -> List[str]:
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"
    triggers = []

    # Trigger 1: based on most active regime
    if firing:
        top = firing[0]
        if top.regime == "tier_compression":
            triggers.append(
                "If tier median price falls another 5%+ next week, "
                "treat as structural shift — reassess positioning."
            )
        elif top.regime == "promo_war":
            triggers.append(
                "If 2+ additional brands join the discount wave next week, "
                "escalate to pricing team for response decision."
            )
        elif top.regime == "competitor_compounding":
            triggers.append(
                "If the compounding competitor reaches top-3 BSR in the category, "
                "treat as a share-loss event requiring urgent response."
            )
        elif top.regime == "new_entrant":
            triggers.append(
                "If the new entrant accumulates 50+ reviews next week, "
                "flag for competitive intel — aggressive launch likely."
            )
        elif top.regime == "baseline":
            triggers.append(
                "If brand vs arena BSR delta exceeds ≥7pp next week, "
                "investigate internal factors (listing, stock, pricing) before attributing to market."
            )
        else:
            triggers.append(
                "If this regime persists for a third consecutive week, "
                "escalate from monitoring to active response."
            )

    # Trigger 2: for Baseline → SKU-pressure; otherwise → portfolio-level BSR/BB watch
    if firing and firing[0].regime == "baseline" and asin_metrics and your_brand:
        pause_skus = [
            m for m in asin_metrics.values()
            if m.brand.lower() == your_brand.lower() and m.ads_stance == "Pause+Diagnose"
        ]
        if pause_skus:
            skus = ", ".join(m.asin[-6:] for m in pause_skus[:3])
            # Add product_type context to SKU pressure trigger when buckets available
            bucket_note = ""
            if asin_metrics and _has_buckets(group_metrics):
                ptypes = list({
                    getattr(asin_metrics.get(m.asin, m), "product_type", "other")
                    for m in pause_skus[:3]
                } - {"other"})
                if ptypes:
                    bucket_note = f" [{', '.join(p.replace('_', ' ').title() for p in ptypes[:2])}]"
            triggers.append(
                f"{len(pause_skus)} Core SKU(s) above tier + losing visibility{bucket_note} ({skus}) — "
                "confirm in-stock and listing health before next week."
            )
        else:
            triggers.append(
                "If any Core SKU flips to 'above tier + losing visibility' next week, "
                "treat as SKU-level issue: check Buy Box, OOS, and listing status."
            )
    elif your_bsr is not None and your_bsr > 0.10:
        triggers.append(
            f"If your brand BSR continues declining next week (currently {band_fn(your_bsr, 'rank_change')} WoW), "
            "cross-check with Seller Central for listing suppression or stock issues."
        )
    else:
        triggers.append(
            "If Buy Box ownership for any Core SKU drops below 70% next week, "
            "treat as a stock or competitive pricing issue requiring immediate investigation."
        )

    return triggers[:2]


def _build_secondary_signals(price_vs_tier, asin_metrics, your_brand, firing, band_fn) -> List[SecondarySignal]:
    """Secondary market signals — capped at 2 — shown only when not already a driver regime."""
    signals: List[SecondarySignal] = []
    regime_names = {s.regime for s in firing}

    # Signal 1: Price pressure — only if not already a driver regime
    if (price_vs_tier is not None and price_vs_tier > 0.05
            and "tier_compression" not in regime_names
            and "promo_war" not in regime_names):
        tier_label = band_fn(price_vs_tier, "price_vs_tier")
        # Build 2 auditable receipts
        core_brand = [m for m in asin_metrics.values()
                      if m.role == "Core" and m.brand.lower() == your_brand.lower()]
        n_above = sum(1 for m in core_brand if m.price_vs_tier > 0.05) if core_brand else 0
        r1 = (f"{n_above}/{len(core_brand)} brand Core SKUs priced above tier median"
              if core_brand else "Brand Core SKUs priced above arena tier median")
        r2 = f"Brand vs arena price gap: {tier_label} — exceeds 5% monitoring threshold"
        signals.append(SecondarySignal(
            claim=f"Price pressure: brand Core SKUs priced {tier_label} vs arena median (Med confidence)",
            receipts=[r1, r2],
        ))

    # Signal 2: Discount concentration — arena-level, only if promo_war not already active
    if "promo_war" not in regime_names:
        core_asins = [m for m in asin_metrics.values() if m.role == "Core"]
        if core_asins:
            discounted = sum(1 for m in core_asins if m.discount_persistence >= 4 / 7)
            pct = discounted / len(core_asins)
            if pct >= 0.40:
                signals.append(SecondarySignal(
                    claim=(f"Discount concentration: {round(pct * 100)}% of arena Core SKUs "
                           f"discounted ≥4/7 days (Med confidence)"),
                    receipts=[
                        f"{discounted}/{len(core_asins)} arena Core SKUs discounted ≥4 of 7 days last week",
                        "Promo war regime not yet active — monitoring for escalation next week",
                    ],
                ))

    # Signal 3: Buy Box competitive pressure (Phase A) — surface top_comp_bb_share_30
    # Only meaningful when at least some ASINs have buybox stats data
    if len(signals) < 2:  # Only add if we haven't hit the cap
        brand_metrics = [m for m in asin_metrics.values()
                         if m.brand.lower() == your_brand.lower()
                         and m.top_comp_bb_share_30 is not None]
        if brand_metrics:
            avg_bb_share = sum(m.top_comp_bb_share_30 for m in brand_metrics) / len(brand_metrics)
            threatened = [m for m in brand_metrics if m.top_comp_bb_share_30 >= 0.15]
            if len(threatened) >= 2 or (len(threatened) >= 1 and avg_bb_share >= 0.10):
                signals.append(SecondarySignal(
                    claim=(
                        f"Buy Box competition: {len(threatened)} brand SKU(s) facing "
                        f"≥15% competitor Buy Box share (avg {avg_bb_share:.0%} across "
                        f"{len(brand_metrics)} tracked SKUs)"
                    ),
                    receipts=[
                        f"{len(threatened)}/{len(brand_metrics)} brand SKUs with top competitor BB share ≥15%",
                        f"Average top competitor BB share: {avg_bb_share:.1%} — monitor for Buy Box loss",
                    ],
                ))

    return signals[:2]


def grouped_receipts_list(
    asin_metrics: dict,
    your_brand: str,
    pressure_ptypes: Optional[List[str]] = None,
    opportunity_concern: Optional[str] = None,
    band_fn=None,
    runs_ads: Optional[bool] = None,
    max_per_group: int = 3,
) -> List[str]:
    """
    Layer A: ASIN receipts grouped under bucket headings.
    Falls back to flat receipts_list() when no pressure_ptypes or opportunity_concern.
    """
    from features.asin_metrics import _POSTURE_DISPLAY, _VALIDATE_BY_STANCE
    if band_fn is None:
        band_fn = lambda v, t: f"{v*100:+.1f}%"

    if not pressure_ptypes and not opportunity_concern:
        return receipts_list(asin_metrics, your_brand, max_items=8,
                             band_fn=band_fn, runs_ads=runs_ads)

    def _fmt(m) -> str:
        bsr_str = band_fn(m.bsr_wow, "rank_change")
        ads_hint = f" | If on ads: {_POSTURE_DISPLAY.get(m.ads_stance, m.ads_stance)}"
        val_hint = (
            f" | Validate: {_VALIDATE_BY_STANCE.get(m.ads_stance, '')}"
            if runs_ads is not False else ""
        )
        _extra = _phase_a_receipt_extras(m)
        return (
            f"{m.brand} ({m.asin[-6:]}): price {m.price_vs_tier_band}, "
            f"BSR {bsr_str} WoW \u2014 {m.tag}{_extra}{ads_hint}{val_hint}"
        )

    lines = []

    for pt in (pressure_ptypes or []):
        pt_name = pt.replace("_", " ").title()
        brand_pressure = sorted(
            [m for m in asin_metrics.values()
             if m.brand.lower() == your_brand.lower()
             and m.product_type == pt
             and m.ads_stance in ("Pause+Diagnose", "Hold")],
            key=lambda m: -m.bsr_wow,
        )[:max_per_group]
        if brand_pressure:
            lines.append(f"**Your brand \u2014 {pt_name} (pressure):**")
            lines.extend(f"  - {_fmt(m)}" for m in brand_pressure)

        comp_pressure = sorted(
            [m for m in asin_metrics.values()
             if m.brand.lower() != your_brand.lower()
             and m.product_type == pt
             and m.discount_persistence >= 4 / 7
             and m.bsr_wow < -0.03],
            key=lambda m: m.bsr_wow,
        )[:max_per_group]
        if comp_pressure:
            lines.append(f"**Competitors \u2014 {pt_name} (pressure):**")
            lines.extend(f"  - {_fmt(m)}" for m in comp_pressure)

    if opportunity_concern:
        opp_name = opportunity_concern.replace("_", " ").title()
        opp_asins = sorted(
            [m for m in asin_metrics.values()
             if m.brand.lower() == your_brand.lower()
             and opportunity_concern in m.concerns],
            key=lambda m: m.bsr_wow,
        )[:max_per_group]
        if opp_asins:
            lines.append(f"**Your brand \u2014 {opp_name} (opportunity):**")
            lines.extend(f"  - {_fmt(m)}" for m in opp_asins)

    # Safety fallback
    if not lines:
        return receipts_list(asin_metrics, your_brand, max_items=8,
                             band_fn=band_fn, runs_ads=runs_ads)
    return lines


# ─── MARKDOWN RENDERER ───────────────────────────────────────────────────────

def generate_brief_markdown(
    brief: WeeklyBrief,
    df_weekly: pd.DataFrame,
    asin_metrics: Dict[str, ASINMetrics],
    your_brand: str,
    include_per_asin: bool = True,
) -> str:
    """
    Render a WeeklyBrief to a markdown string.

    Args:
        brief: Built WeeklyBrief instance
        df_weekly: Arena weekly panel (for per-ASIN table)
        asin_metrics: ASIN metrics dict
        your_brand: Brand name
        include_per_asin: Whether to include Layer A + Layer B per-ASIN sections

    Returns:
        Markdown string ready for Slack/email/download
    """
    try:
        from config.market_misattribution_module import band_value
    except ImportError:
        band_value = lambda v, t: f"{v*100:+.1f}%"

    lines = []
    conf_label = brief.confidence_score.label if brief.confidence_score else "Med"
    data_q = brief.data_quality

    # ── Header ────────────────────────────────────────────────────────────────
    _brand_ct = sum(1 for m in asin_metrics.values() if m.brand.lower() == your_brand.lower())
    _comp_ct = len(asin_metrics) - _brand_ct
    lines += [
        f"# {brief.arena_name} — Weekly Brief",
        f"**{brief.week_label}** | Generated {brief.generated_at.strftime('%Y-%m-%d %H:%M')} | "
        f"Data confidence: **{conf_label}** | Data quality: {data_q} | Fidelity: {brief.data_fidelity}",
        f"Arena: **{_brand_ct}** brand + **{_comp_ct}** competitor ASINs "
        "| est. coverage within scanned universe",
        "",
        "---",
        "",
    ]

    # ── Section 1: Headline ───────────────────────────────────────────────────
    lines += [
        "## 1. Headline",
        f"> {brief.headline}",
        "",
    ]

    # ── Section 2: What Changed ───────────────────────────────────────────────
    lines += ["## 2. What Changed", ""]
    for b in brief.what_changed:
        lines.append(f"- {b}")
    lines.append("")

    # ── Section 2.5: Pressure Buckets ────────────────────────────────────────
    if brief.pressure_buckets:
        lines += ["", "## 2.5 Market Pressure by Bucket", ""]
        for driver in brief.pressure_buckets:
            lines.append(f"- {driver.claim}")
            for r in driver.receipts[:2]:
                lines.append(f"  - {r}")
        lines.append("")

    # ── Section 2.6: Opportunity Bucket ──────────────────────────────────────
    if brief.opportunity_bucket:
        ob = brief.opportunity_bucket
        lines += ["", "## 2.6 Opportunity Signal", ""]
        lines.append(f"- {ob.claim}")
        for r in ob.receipts[:2]:
            lines.append(f"  - {r}")
        lines.append("")

    # ── Section 3: Why ────────────────────────────────────────────────────────
    lines += ["## 3. Why", ""]
    for i, driver in enumerate(brief.drivers, 1):
        lines += [
            f"**Driver {i}:** {driver.claim}",
            f"  - Confidence: **{driver.confidence}**",
        ]
        for j, receipt in enumerate(driver.receipts[:2], 1):
            lines.append(f"  - Receipt {j}: {receipt}")
        lines.append("")

    # ── Section 4: Misattribution Verdict ─────────────────────────────────────
    lines += [
        "## 4. Misattribution Verdict",
        "",
        f"**Verdict:** {brief.misattribution_verdict} "
        f"[{brief.misattribution_confidence} driver confidence]",
        "",
    ]
    for r in brief.misattribution_receipts:
        lines.append(f"- {r}")
    lines.append("")

    # ── Section 4.5: Secondary Signals ───────────────────────────────────────
    if brief.secondary_signals:
        lines += ["", "## Secondary Signals (monitor — not dominant)", ""]
        for sig in brief.secondary_signals:
            lines.append(f"**{sig.claim}**")
            for r in sig.receipts[:2]:
                lines.append(f"  - {r}")
            lines.append("")

    # ── Section 5: Implications ───────────────────────────────────────────────
    lines += ["## 5. Implications", ""]
    for b in brief.implications:
        lines.append(f"- {b}")
    lines += [
        "",
        f"**Plan stance:** {brief.plan_stance}",
        "",
    ]

    # ── Section 5.5: Actions (If you run Sponsored Ads) ──────────────────────
    if brief.actions_block:
        lines += ["", "## Actions (if you run Sponsored Ads)", ""]
        for item in brief.actions_block:
            lines.append(f"- {item}")
        lines.append("")

    # ── Section 6: Requests ───────────────────────────────────────────────────
    lines += ["## 6. Requests", "", "**Validation asks:**", ""]
    for i, ask in enumerate(brief.validation_asks, 1):
        lines.append(f"{i}. {ask}")
    lines += [
        "",
        f"**Coordination ask:** {brief.coordination_ask}",
        "",
    ]

    # ── Section 7: What to Watch ──────────────────────────────────────────────
    lines += ["## 7. What to Watch", ""]
    for trigger in brief.watch_triggers:
        lines.append(f"- {trigger}")
    lines.append("")

    # ── Section 8: Scoreboard ─────────────────────────────────────────────────
    lines += ["## 8. Scoreboard (Last Week's Calls)", ""]
    if brief.scoreboard_lines:
        for line in brief.scoreboard_lines:
            lines.append(f"- {line}")
    else:
        lines.append("- *(No prior week calls to score — first run)*")
    lines.append("")

    # ── Per-ASIN Sections ─────────────────────────────────────────────────────
    if include_per_asin and asin_metrics:
        lines += ["---", "", "## Per-ASIN Detail", ""]

        # Layer A: Ads-relevant receipts (grouped by pressure/opportunity bucket when available)
        lines += ["### Layer A: Ads-relevant receipts this week", ""]
        _pressure_ptypes = [d.regime for d in brief.pressure_buckets] if brief.pressure_buckets else None
        _opp_concern = brief.opportunity_bucket.regime if brief.opportunity_bucket else None
        receipt_lines = grouped_receipts_list(
            asin_metrics, your_brand,
            pressure_ptypes=_pressure_ptypes,
            opportunity_concern=_opp_concern,
            band_fn=band_value, runs_ads=brief.runs_ads,
        )
        for line in receipt_lines:
            lines.append(f"- {line}")
        lines.append("")

        def _render_md_table(tdf):
            if tdf.empty:
                return
            cols = list(tdf.columns)
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("| " + " | ".join("---" for _ in cols) + " |")
            for row in tdf.itertuples(index=False, name=None):
                lines.append("| " + " | ".join(str(v) if v is not None else "" for v in row) + " |")

        # Layer B: group view when buckets available, flat brand/competitor view otherwise
        if _has_buckets(brief.group_summary):
            lines += ["### Layer B: Product Type Groups", ""]
            group_df = to_group_table(brief.group_summary, band_fn=band_value)
            _render_md_table(group_df)
            lines.append("")

            # Per-group ASIN receipts (up to 5 ASINs per group)
            lines += ["#### ASIN Detail by Group", ""]
            for g in brief.group_summary[:12]:
                if not g.top_asins:
                    continue
                lines.append(
                    f"**{g.brand} — {g.product_type.replace('_', ' ').title()}** "
                    f"({g.asin_count} SKUs, {g.rev_share_pct:.0%} rev share)"
                )
                for asin in g.top_asins[:5]:
                    m = asin_metrics.get(asin)
                    if m:
                        price_str = band_value(m.price_vs_tier, "price_vs_tier")
                        bsr_str = band_value(m.bsr_wow, "rank_change")
                        _extra = _phase_a_receipt_extras(m)
                        lines.append(
                            f"  - {m.brand} ({asin[-6:]}): {price_str}, "
                            f"BSR {bsr_str} — {m.tag}{_extra} | If on ads: {m.ads_stance}"
                        )
                lines.append("")
        else:
            # Flat view — unchanged for generic mode
            brand_metrics = {a: m for a, m in asin_metrics.items()
                             if m.brand.lower() == your_brand.lower()}
            comp_metrics = {a: m for a, m in asin_metrics.items()
                            if m.brand.lower() != your_brand.lower()}

            lines += ["### Layer B: Your Brand SKUs", ""]
            brand_table = to_compact_table(brand_metrics, df_weekly, max_asins=20, band_fn=band_value)
            _render_md_table(brand_table)
            lines.append("")

            # Competitor block: top 5 by |bsr_wow|
            comp_top5 = dict(sorted(comp_metrics.items(), key=lambda kv: -abs(kv[1].bsr_wow))[:5])
            if comp_top5:
                lines += ["### Layer B: Competitor Pressure (top movers)", ""]
                comp_table = to_compact_table(comp_top5, df_weekly, max_asins=5, band_fn=band_value)
                _render_md_table(comp_table)
                lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "_Brief generated by Market Misattribution Shield. "
        "All estimates are market-observable proxies only. "
        "No ad spend, CVR, or Seller Central data was used._",
        "",
    ]

    return "\n".join(lines)


# ─── STREAMLIT RENDERER ──────────────────────────────────────────────────────

def render_brief_tab(
    df_weekly: pd.DataFrame,
    your_brand: str,
    arena_name: str = "",
    runs_ads: Optional[bool] = None,
    df_daily: Optional[pd.DataFrame] = None,
    scoreboard_lines: List[str] = None,
    project_id: Optional[str] = None,
    category_path: str = "",    # Category breadcrumb — for module inference
    module_id: str = "",        # Override infer_module() if already known
):
    """
    Streamlit entry point. Renders the brief in a tab.
    Import and call from shelfguard_app.py or a dedicated tab.

    Args:
        project_id: Supabase project_id, used to fetch daily panel for discount persistence.
    """
    import streamlit as st

    if df_weekly.empty:
        st.info("Run Market Discovery to generate a brief.")
        return

    # Read category_path from session state if not supplied
    _cat_path = category_path or st.session_state.get("last_phase2_params", {}).get("category_path", "")
    _mod_id = module_id or st.session_state.get("category_module_id", "")

    # Auto-fetch daily panel if project_id provided and df_daily not supplied
    if df_daily is None and project_id:
        try:
            from data.daily_panel import get_daily_panel
            asins = list(df_weekly["asin"].unique()) if "asin" in df_weekly.columns else None
            df_daily, _fidelity = get_daily_panel(project_id=project_id, asins=asins)
        except Exception:
            df_daily = None

    with st.spinner("Building brief…"):
        try:
            brief = build_brief(
                df_weekly=df_weekly,
                your_brand=your_brand,
                arena_name=arena_name,
                runs_ads=runs_ads,
                df_daily=df_daily,
                scoreboard_lines=scoreboard_lines,
                category_path=_cat_path,
                module_id=_mod_id,
            )

            # ── Re-score scoreboard with REAL regime signals ──────────────
            # The caller may have passed placeholder scoreboard_lines (empty
            # signals / "Unknown" misattribution) because regime signals are
            # only available after build_brief.  Now that we have them, re-
            # score properly and patch the brief before markdown generation.
            try:
                from eval.scoreboard import get_scoreboard_lines as _get_sb
                _regime_signals_for_scoring = detect_all_regimes(
                    df_weekly, your_brand
                )
                _real_sb = _get_sb(
                    your_brand,
                    _regime_signals_for_scoring,
                    brief.misattribution_verdict,
                )
                brief.scoreboard_lines = _real_sb
            except Exception:
                pass  # Non-fatal — keep whatever was passed in

            # Use proper thresholds for ASIN metrics (same configs that build_brief uses)
            from config.market_misattribution_module import ASIN_ROLE_THRESHOLDS, AD_WASTE_RISK_THRESHOLDS
            asin_metrics_map = compute_asin_metrics(
                df_weekly, ASIN_ROLE_THRESHOLDS, AD_WASTE_RISK_THRESHOLDS,
                your_brand=your_brand, df_daily=df_daily,
            )

            md = generate_brief_markdown(brief, df_weekly, asin_metrics_map, your_brand)
        except Exception as e:
            st.error(f"❌ Brief generation failed: {e}")
            st.caption("Check that the market data loaded correctly and the brand name matches listings.")
            return

    # Persist for scoreboard
    import streamlit as st
    st.session_state["last_brief"] = brief
    st.session_state["last_brief_markdown"] = md

    # Render
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {brief.arena_name} — {brief.week_label}")
    with col2:
        st.download_button(
            "⬇ Download brief (.md)",
            data=md,
            file_name=f"brief_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )

    st.markdown("---")
    st.markdown(md)

    # ── Bucket drilldown (interactive) ───────────────────────────────────────
    if brief.group_summary and _has_buckets(brief.group_summary):
        st.markdown("---")
        st.markdown("### Bucket Drilldown")

        _ms = st.session_state.get("last_market_stats", {})
        if _ms:
            st.caption(
                f"Arena: **{_ms.get('brand_selected_count', '?')}** brand + "
                f"**{_ms.get('competitor_selected_count', '?')}** competitors | "
                "est. coverage within scanned universe"
            )

        try:
            from config.market_misattribution_module import band_value as _bv
            from features.asin_metrics import _POSTURE_DISPLAY as _PD
        except ImportError:
            _bv = lambda v, t: f"{v*100:+.1f}%"
            _PD = {}

        from collections import defaultdict as _dd
        by_ptype: dict = _dd(list)
        for g in brief.group_summary:
            if g.product_type != "other":
                by_ptype[g.product_type].append(g)

        for ptype, groups in list(by_ptype.items())[:10]:
            ptype_name = ptype.replace("_", " ").title()
            total_skus = sum(g.asin_count for g in groups)
            stances = [g.dominant_ads_stance for g in groups]
            dom_stance = max(set(stances), key=stances.count)
            posture_label = _PD.get(dom_stance, dom_stance)
            momenta = [g.momentum_label for g in groups]
            arena_momentum = max(set(momenta), key=momenta.count)

            with st.expander(
                f"**{ptype_name}** — {total_skus} SKUs | {arena_momentum} trend | {posture_label}"
            ):
                for g in sorted(groups, key=lambda g: (
                    0 if g.brand.lower() == your_brand.lower() else 1, -g.rev_share_pct
                ))[:6]:
                    g_posture = _PD.get(g.dominant_ads_stance, g.dominant_ads_stance)
                    st.markdown(
                        f"**{g.brand}** — {g.asin_count} SKU{'s' if g.asin_count != 1 else ''}, "
                        f"{g.rev_share_pct:.0%} rev share, {g_posture} posture, "
                        f"{g.momentum_label} trend"
                    )
                    for asin in g.top_asins[:3]:
                        m = asin_metrics_map.get(asin)
                        if m:
                            bsr_str = _bv(m.bsr_wow, "rank_change")
                            st.caption(
                                f"  \u2022 {asin[-6:]}: price {m.price_vs_tier_band}, "
                                f"BSR {bsr_str} WoW \u2014 {m.tag}"
                            )

    # ── Concern / active drilldown ────────────────────────────────────────────
    if brief.concern_summary:
        from collections import defaultdict as _dd2
        by_concern: dict = _dd2(list)
        for g in brief.concern_summary:
            by_concern[g.concern].append(g)
        if by_concern:
            st.markdown("---")
            st.markdown("#### Active/Concern Breakdown")
            for concern, groups in list(by_concern.items())[:8]:
                concern_name = concern.replace("_", " ").title()
                total = sum(g.asin_count for g in groups)
                momenta = [g.momentum_label for g in groups]
                arena_mom = max(set(momenta), key=momenta.count)
                with st.expander(f"**{concern_name}** — {total} SKUs | {arena_mom} trend"):
                    for g in sorted(groups, key=lambda g: (
                        0 if g.brand.lower() == your_brand.lower() else 1, -g.rev_share_pct
                    ))[:5]:
                        st.markdown(
                            f"**{g.brand}** — {g.asin_count} SKU{'s' if g.asin_count != 1 else ''}, "
                            f"{g.momentum_label} trend, {round(g.pct_discounted * 7)}/7 discounted"
                        )

    # Diagnostics expander
    with st.expander("🔧 Confidence Diagnostics"):
        if brief.confidence_score:
            st.write(f"**Score:** {brief.confidence_score.label} ({brief.confidence_score.score:+d})")
            st.write(f"**Data quality:** {brief.confidence_score.data_quality}")
            st.write(f"**Arena coverage:** {brief.confidence_score.arena_coverage:.0%}")
            for reason in brief.confidence_score.reasons:
                st.caption(reason)


def generate_golden_brief(runs_ads: Optional[bool] = None) -> Optional[str]:
    """
    Generate the brief for the golden brand from config/golden_run.py.
    Can be called from CLI: python -c "from report.weekly_brief import generate_golden_brief; print(generate_golden_brief())"

    Returns the markdown string or None if data not available.
    """
    try:
        from config.golden_run import (
            GOLDEN_BRAND, GOLDEN_PROJECT_NAME, GOLDEN_RUNS_ADS
        )
        import streamlit as st
        df_weekly = st.session_state.get("active_project_data", None)
        if df_weekly is None or df_weekly.empty:
            print("No data in session state. Run market discovery first.")
            return None

        effective_runs_ads = runs_ads if runs_ads is not None else GOLDEN_RUNS_ADS
        brief = build_brief(df_weekly, GOLDEN_BRAND, GOLDEN_PROJECT_NAME, effective_runs_ads)
        asin_metrics = compute_asin_metrics(df_weekly, {}, {}, GOLDEN_BRAND)
        return generate_brief_markdown(brief, df_weekly, asin_metrics, GOLDEN_BRAND)

    except Exception as e:
        print(f"Error generating golden brief: {e}")
        return None


# ─── MVP: NON-STREAMLIT ORCHESTRATION ────────────────────────────────────────

def generate_weekly_brief_markdown(
    df_weekly: pd.DataFrame,
    brand: str,
    arena_name: str = "",
    runs_ads: Optional[bool] = None,
    project_id: Optional[str] = None,
) -> tuple:
    """
    One-call MVP function. Orchestrates daily panel, regime detection, ASIN metrics,
    confidence scoring, brief assembly, and markdown rendering.

    Returns (markdown_string, WeeklyBrief) — no Streamlit required.
    Usable from CLI, tests, or non-UI contexts.

    Fidelity note: if daily panel unavailable, brief.data_fidelity == "weekly proxy"
    and the header is labelled accordingly.

    Example (CLI):
        python -c "
        import pandas as pd
        from report.weekly_brief import generate_weekly_brief_markdown
        df = pd.read_csv('data/sample.csv')
        md, brief = generate_weekly_brief_markdown(df, 'MyBrand')
        print(md)
        "
    """
    df_daily = None
    if project_id:
        try:
            from data.daily_panel import get_daily_panel
            asins = list(df_weekly["asin"].unique()) if "asin" in df_weekly.columns else None
            df_daily, _ = get_daily_panel(project_id=project_id, asins=asins)
        except Exception:
            pass

    brief = build_brief(df_weekly, brand, arena_name, runs_ads, df_daily=df_daily)

    asin_metrics = compute_asin_metrics(df_weekly, {}, {}, brand, df_daily)
    try:
        from config.market_misattribution_module import ASIN_ROLE_THRESHOLDS, AD_WASTE_RISK_THRESHOLDS
        asin_metrics = compute_asin_metrics(
            df_weekly, ASIN_ROLE_THRESHOLDS, AD_WASTE_RISK_THRESHOLDS, brand, df_daily
        )
    except ImportError:
        pass

    md = generate_brief_markdown(brief, df_weekly, asin_metrics, brand)
    return md, brief
