"""
Scoreboard — Market Misattribution Shield
==========================================
Persists this week's regime/driver predictions to Supabase.
On the next run, loads last week's predictions and marks them ✅ or ❌.

Table: `brief_predictions`
Schema:
    id              uuid (auto)
    brief_date      date (ISO, the Monday of the brief week)
    brand           text
    arena_name      text
    regime_call     text   (e.g. "promo_war")
    top_threat      text   (e.g. "CeraVe compounding")
    primary_driver  text   (e.g. "Market-driven")
    plan_stance     text   (e.g. "Hold")
    created_at      timestamptz

On next run, the new regime signals are compared to last week's stored call.
Outcome is ✅ if the regime was still active, ❌ if it resolved or reversed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict

import pandas as pd


@dataclass
class BriefPrediction:
    brief_date: str           # ISO date of the Monday of the brief week
    brand: str
    arena_name: str
    regime_call: str          # Primary firing regime ("none" if no regimes)
    top_threat: str           # Human-readable top threat
    primary_driver: str       # "Market-driven" | "Brand-driven" | "Unknown"
    plan_stance: str          # "Hold" | "Reallocate" | "Pause+Diagnose"
    created_at: Optional[str] = None


@dataclass
class ScoredCall:
    prediction: BriefPrediction
    outcome: str              # "✅" | "❌" | "⏳"
    outcome_reason: str


def _get_monday(dt: datetime = None) -> str:
    """Return the ISO date string for the Monday of the given week."""
    if dt is None:
        dt = datetime.now()
    days_since_monday = dt.weekday()
    monday = dt - timedelta(days=days_since_monday)
    return monday.strftime("%Y-%m-%d")


def save_prediction(
    brand: str,
    arena_name: str,
    regime_call: str,
    top_threat: str,
    primary_driver: str,
    plan_stance: str,
) -> bool:
    """
    Persist this week's brief predictions to Supabase.

    Returns True if saved successfully, False otherwise.
    """
    pred = BriefPrediction(
        brief_date=_get_monday(),
        brand=brand,
        arena_name=arena_name,
        regime_call=regime_call,
        top_threat=top_threat,
        primary_driver=primary_driver,
        plan_stance=plan_stance,
        created_at=datetime.now().isoformat(),
    )

    try:
        from src.supabase_reader import get_supabase_client
        supabase = get_supabase_client()
        if supabase is None:
            return False

        supabase.table("brief_predictions").upsert(
            {
                "brief_date": pred.brief_date,
                "brand": pred.brand,
                "arena_name": pred.arena_name,
                "regime_call": pred.regime_call,
                "top_threat": pred.top_threat,
                "primary_driver": pred.primary_driver,
                "plan_stance": pred.plan_stance,
                "created_at": pred.created_at,
            },
            on_conflict="brief_date,brand",
        ).execute()
        return True

    except Exception:
        return False


def load_last_prediction(brand: str) -> Optional[BriefPrediction]:
    """
    Load the most recent prediction for this brand (last week's call).

    Returns None if no prior predictions found.
    """
    try:
        from src.supabase_reader import get_supabase_client
        supabase = get_supabase_client()
        if supabase is None:
            return None

        response = (
            supabase.table("brief_predictions")
            .select("*")
            .eq("brand", brand)
            .order("brief_date", desc=True)
            .limit(2)  # Get last 2 to skip current week
            .execute()
        )
        rows = response.data if response and hasattr(response, "data") else []

        # Skip this week's row if present
        current_monday = _get_monday()
        prior = [r for r in rows if r.get("brief_date") != current_monday]

        if not prior:
            return None

        r = prior[0]
        return BriefPrediction(
            brief_date=r["brief_date"],
            brand=r["brand"],
            arena_name=r.get("arena_name", ""),
            regime_call=r["regime_call"],
            top_threat=r.get("top_threat", ""),
            primary_driver=r.get("primary_driver", "Unknown"),
            plan_stance=r.get("plan_stance", "Hold"),
            created_at=r.get("created_at"),
        )

    except Exception:
        return None


def score_last_prediction(
    last_pred: BriefPrediction,
    current_regime_signals: Dict,
    current_misattribution: str,
) -> ScoredCall:
    """
    Score last week's prediction against current signals.

    Rules:
    - regime_call ✅ if the same regime is still active this week
    - primary_driver ✅ if misattribution verdict matches
    - ❌ otherwise
    - ⏳ if data is insufficient to score
    """
    if last_pred is None:
        return None

    # Score regime call
    if last_pred.regime_call == "none":
        # Last week predicted no regime — check if that holds
        any_active = any(s.active for s in current_regime_signals.values() if hasattr(s, "active"))
        regime_outcome = "✅" if not any_active else "❌"
        regime_reason = "No environment active (correct)" if not any_active else "Environment appeared (incorrect)"
    else:
        matching_signal = current_regime_signals.get(last_pred.regime_call)
        if matching_signal is None:
            regime_outcome = "⏳"
            regime_reason = "Environment not evaluated this run"
        elif matching_signal.active:
            regime_outcome = "✅"
            regime_reason = f"{last_pred.regime_call} still active"
        else:
            regime_outcome = "❌"
            regime_reason = f"{last_pred.regime_call} resolved"

    # Score misattribution verdict
    if current_misattribution == last_pred.primary_driver:
        driver_outcome = "✅"
        driver_reason = f"Misattribution still {last_pred.primary_driver}"
    else:
        driver_outcome = "❌"
        driver_reason = f"Misattribution shifted from {last_pred.primary_driver} → {current_misattribution}"

    # Overall outcome
    outcomes = [regime_outcome, driver_outcome]
    if all(o == "✅" for o in outcomes):
        overall = "✅"
    elif "⏳" in outcomes:
        overall = "⏳"
    else:
        overall = "❌"

    reason = f"Environment: {regime_outcome} ({regime_reason}). Driver: {driver_outcome} ({driver_reason})."

    return ScoredCall(
        prediction=last_pred,
        outcome=overall,
        outcome_reason=reason,
    )


def get_scoreboard_lines(
    brand: str,
    current_regime_signals: Dict,
    current_misattribution: str,
) -> List[str]:
    """
    Load last week's prediction, score it, and return display lines for the brief.

    Returns:
        List of strings ready for Section 8 of the brief.
    """
    last_pred = load_last_prediction(brand)
    if last_pred is None:
        return ["*(No prior week calls to score — first run)*"]

    scored = score_last_prediction(last_pred, current_regime_signals, current_misattribution)
    if scored is None:
        return ["*(Could not score prior week)*"]

    return [
        f"**Week of {last_pred.brief_date}**",
        f"- Environment call: `{last_pred.regime_call}` → {scored.outcome}",
        f"- Driver call: `{last_pred.primary_driver}` → {'✅' if last_pred.primary_driver == current_misattribution else '❌'}",
        f"- Reason: {scored.outcome_reason}",
    ]


def save_brief_predictions_from_brief(brief) -> bool:
    """
    Convenience: extract prediction fields from a WeeklyBrief and save.

    Args:
        brief: WeeklyBrief instance from report/weekly_brief.py

    Returns:
        True if saved successfully
    """
    regime_call = brief.active_regime_names[0] if brief.active_regime_names else "none"
    top_threat = brief.drivers[0].claim[:100] if brief.drivers else "none"

    return save_prediction(
        brand=brief.brand,
        arena_name=brief.arena_name,
        regime_call=regime_call,
        top_threat=top_threat,
        primary_driver=brief.misattribution_verdict,
        plan_stance=brief.plan_stance,
    )
