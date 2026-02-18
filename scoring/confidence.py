"""
Confidence Rubric — Market Misattribution Shield
=================================================
Produces HIGH/MED/LOW confidence per driver based on objective, parameterized criteria.

Design:
- Start at MED (neutral)
- Upgrade or downgrade based on each criterion
- Every criterion is parameterized from config/market_misattribution_module.py
- The rubric is fully deterministic — same data always produces same score
- Returns a ConfidenceScore with label AND reason string (for transparency in brief)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ConfidenceScore:
    label: str              # "High" | "Med" | "Low"
    score: int              # -3 to +3 internal tally
    reasons: List[str]      # Upgrade/downgrade reasons
    data_quality: str       # "Good" | "Partial" | "Sparse"
    arena_coverage: float   # Fraction of ASINs with complete data


def score_confidence(
    df_weekly: pd.DataFrame,
    regime_signals: Dict,
    cfg: dict,
    monthly_sold_check: bool = True,
) -> ConfidenceScore:
    """
    Compute a single brief-level confidence score.

    Args:
        df_weekly: Arena weekly panel
        regime_signals: Dict of regime name → RegimeSignal
        cfg: CONFIDENCE_RUBRIC dict from market_misattribution_module.py
        monthly_sold_check: Whether to apply the monthly_sold coverage downgrade

    Returns:
        ConfidenceScore
    """
    tally = 0
    reasons = []

    completeness_min = cfg.get("data_completeness_min", 0.70)
    min_peer_size = cfg.get("min_peer_set_size", 10)
    monthly_sold_min = cfg.get("monthly_sold_coverage_min", 0.40)
    pack_factor = cfg.get("pack_size_comparability_factor", 2.0)
    min_data_weeks = cfg.get("min_data_weeks", 4)

    # ── Criterion 1: Peer set size ────────────────────────────────────────────
    n_asins = df_weekly["asin"].nunique() if "asin" in df_weekly.columns else 0
    if n_asins < min_peer_size:
        tally -= 2
        reasons.append(f"Small peer set ({n_asins} ASINs < {min_peer_size} minimum) → LOW")
    elif n_asins >= min_peer_size * 2:
        tally += 1
        reasons.append(f"Large peer set ({n_asins} ASINs) → upgrade")

    # ── Criterion 2: Data completeness (last 14 days) ─────────────────────────
    if "week_start" in df_weekly.columns:
        cutoff_14d = df_weekly["week_start"].max() - pd.Timedelta(days=14)
        recent = df_weekly[df_weekly["week_start"] >= cutoff_14d]
        total_asin_weeks = n_asins * 2  # 2 weeks × n ASINs
        present = recent["asin"].nunique() if not recent.empty else 0
        completeness = present / n_asins if n_asins > 0 else 0
    else:
        completeness = 0.0

    arena_coverage = completeness
    if completeness < completeness_min:
        tally -= 1
        reasons.append(f"Incomplete data: {completeness:.0%} ASINs in last 14d (need {completeness_min:.0%}) → downgrade")
    elif completeness >= 0.90:
        tally += 1
        reasons.append(f"Strong data coverage: {completeness:.0%} → upgrade")

    # ── Criterion 3: Weeks of history ─────────────────────────────────────────
    if "week_start" in df_weekly.columns:
        n_weeks = df_weekly["week_start"].nunique()
    else:
        n_weeks = 0

    if n_weeks < min_data_weeks:
        tally -= 1
        reasons.append(f"Short history: {n_weeks} weeks (need {min_data_weeks}) → downgrade")
    elif n_weeks >= 10:
        tally += 1
        reasons.append(f"Long history: {n_weeks} weeks → upgrade")

    # ── Criterion 4: monthly_sold coverage ────────────────────────────────────
    if monthly_sold_check and "monthly_sold" in df_weekly.columns:
        ms_present = (df_weekly["monthly_sold"] > 0).sum()
        ms_total = len(df_weekly)
        ms_coverage = ms_present / ms_total if ms_total > 0 else 0
        if ms_coverage < monthly_sold_min:
            tally -= 1
            reasons.append(
                f"Low monthly_sold coverage ({ms_coverage:.0%} < {monthly_sold_min:.0%}): "
                f"demand claims based on BSR formula → downgrade"
            )
        else:
            reasons.append(f"Good monthly_sold coverage ({ms_coverage:.0%})")

    # ── Criterion 5: Cross-signal agreement (from regime signals) ─────────────
    cross_signal_required = cfg.get("cross_signal_agreement_required", True)
    if cross_signal_required and regime_signals:
        high_conf = sum(1 for s in regime_signals.values() if s.confidence == "High")
        if high_conf >= 2:
            tally += 1
            reasons.append(f"{high_conf} regimes agree at High confidence → upgrade")
        elif high_conf == 0:
            # Only penalize if some regime reached Med confidence (was borderline — genuine ambiguity).
            # A clean quiet week (ALL regimes at Low) means detectors ran and nothing was even close.
            # That is a confident "no regime" finding, not a data gap — don't penalize it.
            med_conf = sum(1 for s in regime_signals.values() if s.confidence == "Med")
            if med_conf > 0:
                tally -= 1
                reasons.append(f"No regimes at High confidence ({med_conf} borderline at Med) → downgrade")
            else:
                reasons.append("All regimes at Low confidence (clean Baseline) → no cross-signal penalty")

    # ── Criterion 6: Pack size comparability ──────────────────────────────────
    if "number_of_items" in df_weekly.columns:
        items = df_weekly["number_of_items"].dropna()
        if not items.empty:
            median_pack = items.median()
            within_range = ((items >= median_pack / pack_factor) & (items <= median_pack * pack_factor)).mean()
            if within_range < 0.60:
                tally -= 1
                reasons.append(
                    f"Mixed pack sizes ({within_range:.0%} within {pack_factor}x of median): "
                    f"price comparisons may be misleading → downgrade"
                )
            else:
                reasons.append(f"Pack sizes comparable ({within_range:.0%} within {pack_factor}x)")

    # ── Tally → Label ─────────────────────────────────────────────────────────
    if tally >= 2:
        label = "High"
    elif tally <= -2:
        label = "Low"
    else:
        label = "Med"

    # Data quality summary
    if completeness >= 0.85 and n_weeks >= 8:
        data_quality = "Good"
    elif completeness >= 0.60 and n_weeks >= 4:
        data_quality = "Partial"
    else:
        data_quality = "Sparse"

    return ConfidenceScore(
        label=label,
        score=tally,
        reasons=reasons,
        data_quality=data_quality,
        arena_coverage=round(arena_coverage, 3),
    )


def score_driver_confidence(
    driver_name: str,
    receipts: list,
    regime_signal,
    base_score: "ConfidenceScore",
) -> str:
    """
    Score confidence for a specific driver within the brief.

    Starts from the brief-level base score and adjusts based on:
    - Number of receipts (2 = good, 1 = weak)
    - Regime signal confidence
    - Whether regime is active

    Returns "High" | "Med" | "Low"
    """
    order = {"High": 2, "Med": 1, "Low": 0}
    base = order.get(base_score.label, 1)

    # Regime signal adjustment
    if regime_signal is not None:
        regime_order = order.get(regime_signal.confidence, 1)
        combined = (base + regime_order) // 2
    else:
        combined = max(base - 1, 0)

    # Receipt quality
    n_receipts = len([r for r in receipts if r is not None]) if receipts else 0
    if n_receipts < 2:
        combined = max(combined - 1, 0)

    labels = {2: "High", 1: "Med", 0: "Low"}
    return labels.get(combined, "Low")
