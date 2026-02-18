"""
MVP App — Market Misattribution Brief
======================================
Standalone Streamlit entrypoint. Two modes:

  Golden run (default when configured):
    Pre-populates session state from config/golden_run.py.
    Shows the golden brand + [Load Market] button → renders brief.

  Manual mode:
    Bare keyword search → seed selector → [Map Market] → renders brief.

Launch:
    streamlit run apps/mvp_app.py

This file imports ONLY the MVP pipeline. No workflow_dashboard, analyst,
ai_engine, unified_dashboard, revenue_attribution, or predictive_forecasting.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from config.golden_run import (
    GOLDEN_RUN_ENABLED,
    GOLDEN_SEED_ASIN,
    GOLDEN_BRAND,
    GOLDEN_PROJECT_NAME,
    GOLDEN_CATEGORY_ID,
    GOLDEN_DOMAIN,
    GOLDEN_RUNS_ADS,
    GOLDEN_MARKET_SIZE,
    get_golden_session_state,
    is_golden_run_active,
)
from report.weekly_brief import render_brief_tab
from eval.scoreboard import get_scoreboard_lines, save_brief_predictions_from_brief


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ShelfGuard — MVP Brief",
    page_icon="🛡️",
    layout="wide",
)


# ─── GOLDEN RUN INIT ─────────────────────────────────────────────────────────

if is_golden_run_active() and "active_project_data" not in st.session_state:
    for k, v in get_golden_session_state().items():
        st.session_state.setdefault(k, v)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ ShelfGuard MVP")
    st.markdown("---")

    # Golden run toggle (only show if golden run is configured)
    if GOLDEN_RUN_ENABLED and not GOLDEN_SEED_ASIN.startswith("B07XYZ"):
        use_golden = st.toggle(
            "Use Golden Run",
            value=True,
            help=f"Pre-configured: {GOLDEN_BRAND} / {GOLDEN_PROJECT_NAME}",
        )
        st.session_state["_mvp_use_golden"] = use_golden
    else:
        st.session_state["_mvp_use_golden"] = False

    st.markdown("---")

    # Ads toggle
    runs_ads_option = st.selectbox(
        "Runs Sponsored Ads?",
        options=["Unknown", "Yes", "No"],
        index=0,
        help="Controls budget-action language in the brief.",
    )
    runs_ads: Optional[bool] = None
    if runs_ads_option == "Yes":
        runs_ads = True
    elif runs_ads_option == "No":
        runs_ads = False
    st.session_state["_mvp_runs_ads"] = runs_ads

    st.markdown("---")

    # Regenerate button (clears brief cache so a fresh build runs)
    if st.button("🔄 Regenerate Brief", use_container_width=True):
        for key in ["last_brief", "last_brief_markdown"]:
            st.session_state.pop(key, None)
        st.rerun()

    # Clear all data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        for key in [
            "active_project_data", "active_project_seed_brand",
            "active_project_name", "active_project_all_asins",
            "active_project_id", "_mvp_seed_candidates",
            "last_brief", "last_brief_markdown",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────

st.markdown("# 🛡️ Market Misattribution Shield — Weekly Brief")
st.markdown("---")

# ── Section 1: Arena ─────────────────────────────────────────────────────────

st.markdown("## 1. Arena")

use_golden = st.session_state.get("_mvp_use_golden", False)
df_weekly = st.session_state.get("active_project_data", pd.DataFrame())

if use_golden:
    # Golden run mode
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Brand:** {GOLDEN_BRAND}")
    with col2:
        st.markdown(f"**Arena:** {GOLDEN_PROJECT_NAME}")
    with col3:
        st.markdown(f"**Seed ASIN:** `{GOLDEN_SEED_ASIN}`")

    if df_weekly.empty:
        st.info(
            f"Market not yet loaded. Click **Load Market** to pull the top "
            f"{GOLDEN_MARKET_SIZE} ASINs from the {GOLDEN_PROJECT_NAME} category."
        )
        if st.button("⬇ Load Market", type="primary"):
            with st.spinner("Mapping market — this pulls ~90 days of Keepa history. Takes ~60s…"):
                try:
                    from src.two_phase_discovery import phase2_category_market_mapping
                    df_new, _ = phase2_category_market_mapping(
                        category_id=GOLDEN_CATEGORY_ID,
                        seed_product_title=GOLDEN_PROJECT_NAME,
                        seed_asin=GOLDEN_SEED_ASIN,
                        target_brand=GOLDEN_BRAND,
                        domain="US" if GOLDEN_DOMAIN == 1 else str(GOLDEN_DOMAIN),
                        max_products=GOLDEN_MARKET_SIZE,
                    )
                    st.session_state["active_project_data"] = df_new
                    st.session_state["active_project_seed_brand"] = GOLDEN_BRAND
                    st.session_state["active_project_name"] = GOLDEN_PROJECT_NAME
                    st.session_state["active_project_all_asins"] = list(df_new["asin"].unique())
                    st.success(f"Loaded {len(df_new['asin'].unique())} ASINs.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Market load failed: {e}")
    else:
        asin_count = df_weekly["asin"].nunique() if "asin" in df_weekly.columns else 0
        week_count = df_weekly["week_start"].nunique() if "week_start" in df_weekly.columns else 0
        st.success(f"Market loaded: {asin_count} ASINs × {week_count} weeks")

else:
    # Manual mode: bare seed search → map market
    st.markdown("### Search for a Market")

    from apps.search_to_state_ui import render_seed_search_and_map_mvp
    render_seed_search_and_map_mvp()


# ── Section 2: Brief ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 2. Brief")

df_weekly = st.session_state.get("active_project_data", pd.DataFrame())

if df_weekly.empty:
    st.info("Load a market above to generate the brief.")
else:
    your_brand = st.session_state.get("active_project_seed_brand", "")
    arena_name = st.session_state.get("active_project_name", "")
    project_id = st.session_state.get("active_project_id", None)
    _runs_ads = st.session_state.get("_mvp_runs_ads", None)

    if not your_brand:
        st.warning("Brand name is missing. Set `active_project_seed_brand` in session state.")
    else:
        # Fetch scoreboard lines (safe — returns placeholder if no prior predictions)
        try:
            scoreboard_lines = get_scoreboard_lines(your_brand, {}, "Unknown")
        except Exception:
            scoreboard_lines = ["*(Scoreboard unavailable)*"]

        # Render brief
        render_brief_tab(
            df_weekly=df_weekly,
            your_brand=your_brand,
            arena_name=arena_name,
            runs_ads=_runs_ads,
            scoreboard_lines=scoreboard_lines,
            project_id=project_id,
        )

        # Save predictions for next-week scoring
        brief = st.session_state.get("last_brief")
        if brief is not None:
            try:
                save_brief_predictions_from_brief(brief)
            except Exception:
                pass  # Non-fatal — scoreboard is best-effort
