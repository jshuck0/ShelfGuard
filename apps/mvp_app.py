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

import sys
from pathlib import Path

# Add project root and apps directory to sys.path (required on Streamlit Cloud)
_PROJECT_ROOT = Path(__file__).parent.parent
_APPS_DIR = Path(__file__).parent
for _p in [str(_PROJECT_ROOT), str(_APPS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from eval.scoreboard import save_brief_predictions_from_brief

# Supabase read path — enables return visits without re-fetching from Keepa
try:
    from src.supabase_reader import load_weekly_timeseries, _normalize_snapshot_to_dashboard
    _SUPABASE_READ_ENABLED = True
except ImportError:
    _SUPABASE_READ_ENABLED = False


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ShelfGuard — MVP Brief",
    page_icon="🛡️",
    layout="wide",
)

# ─── MVP SESSION FLAGS ────────────────────────────────────────────────────────
# Downstream components (search_to_state_ui, etc.) can check these to suppress
# verbose debug output and advanced UI that doesn't belong in the MVP flow.

st.session_state["mvp_mode"] = True
st.session_state.setdefault("show_advanced_debug", False)


# ─── GOLDEN RUN INIT ─────────────────────────────────────────────────────────

if is_golden_run_active() and "active_project_data" not in st.session_state:
    for k, v in get_golden_session_state().items():
        st.session_state.setdefault(k, v)


# ─── RETURN-VISIT LOADER ────────────────────────────────────────────────────
# If session_state has no data but Supabase has cached data from a previous
# session, load it automatically.  This makes the cache actually useful.

def _try_load_from_supabase() -> bool:
    """Attempt to hydrate session_state from Supabase cache. Returns True on success."""
    if not _SUPABASE_READ_ENABLED:
        return False
    try:
        from src.supabase_reader import get_supabase_client
        sb = get_supabase_client()
        # Find the most recent snapshot batch (grouped by fetched_at date)
        probe = (
            sb.table("product_snapshots")
            .select("asin, brand, category_name, fetched_at")
            .order("fetched_at", desc=True)
            .limit(1)
            .execute()
        )
        if not probe.data:
            return False

        latest_row = probe.data[0]
        category_name = latest_row.get("category_name", "Unknown Market")

        # Get all ASINs from the same category
        cat_result = (
            sb.table("product_snapshots")
            .select("asin")
            .eq("category_name", category_name)
            .execute()
        )
        if not cat_result.data:
            return False

        asins = list({r["asin"] for r in cat_result.data if r.get("asin")})
        if len(asins) < 5:
            return False  # Too few ASINs — probably stale/incomplete data

        df_weekly = load_weekly_timeseries(tuple(sorted(asins)), days=90)
        if df_weekly.empty or "asin" not in df_weekly.columns:
            return False

        # Normalize column names so downstream brief code works
        df_weekly = _normalize_snapshot_to_dashboard(df_weekly)

        # Detect brand (most-common brand in category, or from first row)
        brand_counts = df_weekly["brand"].value_counts() if "brand" in df_weekly.columns else pd.Series(dtype=int)
        detected_brand = brand_counts.index[0] if not brand_counts.empty else "Unknown"

        st.session_state["active_project_data"] = df_weekly
        st.session_state["active_project_seed_brand"] = detected_brand
        st.session_state["active_project_name"] = category_name
        st.session_state["active_project_all_asins"] = asins
        st.session_state["_mvp_loaded_from_cache"] = True
        return True
    except Exception:
        return False


if "active_project_data" not in st.session_state:
    _try_load_from_supabase()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ ShelfGuard")
    st.markdown("_Market context & competitive signals_")
    st.markdown("---")

    # Golden run toggle (only show if golden run is configured)
    if GOLDEN_RUN_ENABLED and not GOLDEN_SEED_ASIN.startswith("B07XYZ"):
        use_golden = st.toggle(
            "Use saved brand",
            value=True,
            help=f"Load pre-configured brand: {GOLDEN_BRAND} ({GOLDEN_PROJECT_NAME})",
        )
        st.session_state["_mvp_use_golden"] = use_golden
    else:
        st.session_state["_mvp_use_golden"] = False

    st.markdown("---")

    # Arena preset selector
    _preset = st.selectbox(
        "Competitive set size",
        ["Quick scan (200 SKUs)", "Standard (300 SKUs)", "Deep dive (500 SKUs)"],
        index=1,
        key="_mvp_arena_preset",
        help="How many SKUs to include in your competitive market. "
             "Standard covers most sub-categories well. "
             "Deep dive gives fuller coverage but takes longer and uses more API tokens.",
    )
    _preset_map = {
        "Quick scan (200 SKUs)": (200, 100, 100),
        "Standard (300 SKUs)": (300, 150, 150),
        "Deep dive (500 SKUs)": (500, 250, 250),
    }
    _arena_size, _min_comps, _brand_cap_mv = _preset_map[_preset]
    st.session_state["_mvp_arena_size"] = _arena_size
    st.session_state["_mvp_min_comps"] = _min_comps
    st.session_state["_mvp_brand_cap"] = _brand_cap_mv

    st.markdown("---")

    # Ads toggle
    runs_ads_option = st.selectbox(
        "Does this brand run Sponsored Ads?",
        options=["Unknown", "Yes", "No"],
        index=0,
        help="If Yes, the brief includes ad-specific actions and posture recommendations. "
             "If Unknown, ad language is included but hedged.",
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

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=80)
with col_title:
    st.markdown("# ShelfGuard Market Brief")
    st.caption("A weekly Amazon category brief that turns marketplace signals into a clear stance and SKU-level actions.")
st.markdown(
    "**Seller Central shows your store. ShelfGuard shows your market.** "
    "Using Amazon marketplace signals like pricing, promos, visibility, and assortment, it delivers a weekly category readout "
    "and SKU priorities so teams can coordinate decisions without flying blind on competitive dynamics."
)
st.markdown("---")

# ── Section 1: Market ─────────────────────────────────────────────────────────

st.markdown("## Market Setup")

use_golden = st.session_state.get("_mvp_use_golden", False)
df_weekly = st.session_state.get("active_project_data", pd.DataFrame())

if use_golden:
    # Golden run mode
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Brand:** {GOLDEN_BRAND}")
    with col2:
        st.markdown(f"**Market:** {GOLDEN_PROJECT_NAME}")
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
                    from apps.search_to_state_ui import ensure_weekly_panel
                    _sz = st.session_state.get("_mvp_arena_size", 300)
                    _mc = st.session_state.get("_mvp_min_comps", 150)
                    _bc = st.session_state.get("_mvp_brand_cap", 150)
                    df_snapshot, market_stats = phase2_category_market_mapping(
                        category_id=GOLDEN_CATEGORY_ID,
                        seed_product_title=GOLDEN_PROJECT_NAME,
                        seed_asin=GOLDEN_SEED_ASIN,
                        target_brand=GOLDEN_BRAND,
                        domain="US" if GOLDEN_DOMAIN == 1 else str(GOLDEN_DOMAIN),
                        max_products=2000,
                        leaf_category_id=GOLDEN_CATEGORY_ID,
                        mvp_mode=True,
                        arena_size=_sz,
                        min_competitors=_mc,
                        brand_cap=_bc,
                    )
                    asins = list(df_snapshot["asin"].unique()) if "asin" in df_snapshot.columns else []
                    df_new = ensure_weekly_panel(df_snapshot, market_stats, asins, mvp_mode=True)
                    st.session_state["active_project_data"] = df_new
                    st.session_state["active_project_seed_brand"] = GOLDEN_BRAND
                    st.session_state["active_project_name"] = GOLDEN_PROJECT_NAME
                    st.session_state["active_project_all_asins"] = asins
                    st.session_state["last_market_stats"] = market_stats
                    _ms = market_stats

                    # Cache to Supabase for instant return visits
                    try:
                        from src.supabase_reader import cache_market_snapshot, cache_weekly_timeseries
                        _cat_ctx = {
                            "category_id": GOLDEN_CATEGORY_ID,
                            "category_name": GOLDEN_PROJECT_NAME,
                            "category_tree": [],
                            "category_root": GOLDEN_PROJECT_NAME,
                        }
                        cache_market_snapshot(df_snapshot, df_new, _cat_ctx)
                        cache_weekly_timeseries(df_new, _cat_ctx)
                    except Exception:
                        pass  # Caching is best-effort

                    st.success(
                        f"Market loaded — {_ms.get('brand_selected_count', '?')} brand + "
                        f"{_ms.get('competitor_selected_count', '?')} competitors selected"
                    )
                    with st.expander("Market contract", expanded=True):
                        st.markdown(f"""\
| | |
|---|---|
| **Brand ASINs** | {_ms.get('brand_selected_count', '?')} |
| **Competitor ASINs** | {_ms.get('competitor_selected_count', '?')} |
| **Weeks of history** | {df_new['week_start'].nunique() if 'week_start' in df_new.columns else '?'} |
| **Excluded off-leaf** | {_ms.get('excluded_off_leaf_count', 0)} |
| **Selection basis** | Best sellers first (BSR ascending) |
| **Coverage** | {_ms.get('coverage_note', 'estimated within scanned universe')} |
""")
                    st.rerun()
                except SystemExit:
                    pass  # st.stop() — let spinner clear cleanly
                except Exception as e:
                    st.error(f"Market load failed: {e}")
    else:
        asin_count = df_weekly["asin"].nunique() if "asin" in df_weekly.columns else 0
        week_count = df_weekly["week_start"].nunique() if "week_start" in df_weekly.columns else 0
        _ms = st.session_state.get("last_market_stats", {})
        _from_cache = st.session_state.get("_mvp_loaded_from_cache", False)
        _source_label = " (from cache)" if _from_cache else ""
        if _ms:
            st.success(
                f"Market loaded{_source_label} — {_ms.get('brand_selected_count', asin_count)} brand + "
                f"{_ms.get('competitor_selected_count', '?')} competitors | {week_count} weeks"
            )
            with st.expander("Market contract", expanded=False):
                st.markdown(f"""\
| | |
|---|---|
| **Brand ASINs** | {_ms.get('brand_selected_count', '?')} |
| **Competitor ASINs** | {_ms.get('competitor_selected_count', '?')} |
| **Weeks of history** | {week_count} |
| **Excluded off-leaf** | {_ms.get('excluded_off_leaf_count', 0)} |
| **Selection basis** | Best sellers first (BSR ascending) |
| **Coverage** | {_ms.get('coverage_note', 'estimated within scanned universe')} |
""")
        else:
            st.success(f"Market loaded{_source_label}: {asin_count} ASINs × {week_count} weeks")

else:
    # Manual mode: bare seed search → map market
    st.markdown("### Search for a Market")

    from apps.search_to_state_ui import render_seed_search_and_map_mvp
    render_seed_search_and_map_mvp()


# ── Section 2: Brief ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Brief")

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
        # Scoreboard lines — pass placeholder; render_brief_tab re-scores
        # with real regime signals after build_brief completes.
        scoreboard_lines = ["*(Scoring in progress…)*"]

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
