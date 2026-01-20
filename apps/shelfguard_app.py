import sys
from pathlib import Path

# Add project root and apps directory to path for module resolution
PROJECT_ROOT = Path(__file__).parent.parent
APPS_DIR = Path(__file__).parent

for path in [str(PROJECT_ROOT), str(APPS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import hashlib
import re
import os
from dotenv import load_dotenv

# Optional OpenAI import (for chat feature)
try:
    from openai import OpenAI
    OPENAI_SYNC_AVAILABLE = True
except ImportError:
    OPENAI_SYNC_AVAILABLE = False
    OpenAI = None

# Load environment variables
load_dotenv()

# App imports - local modules (same directory)
# NOTE: Legacy engine.py removed - using unified AI engine only
from finance import f_money, f_pct
from search_to_state_ui import (
    render_discovery_ui, 
    render_project_dashboard, 
    render_project_selector, 
    render_user_dashboard
)

# Supabase Data Reader - The Oracle's Cache Layer
try:
    from src.supabase_reader import (
        load_project_data,
        load_snapshot_trends,
        check_data_freshness,
        get_market_snapshot_from_cache,
        # FIX 1.1: Historical metrics from DB for velocity extraction
        load_historical_metrics_from_db,
        load_historical_metrics_by_asins,
        # FIX 1.2: Network intelligence for competitive context
        get_market_snapshot_with_network_intelligence
    )
    SUPABASE_CACHE_ENABLED = True
except ImportError:
    SUPABASE_CACHE_ENABLED = False
    load_project_data = None
    load_snapshot_trends = None
    check_data_freshness = None
    get_market_snapshot_from_cache = None
    load_historical_metrics_from_db = None
    load_historical_metrics_by_asins = None
    get_market_snapshot_with_network_intelligence = None

# AI Engine - Strategic Triangulation (Unified AI Engine)
try:
    from utils.ai_engine import (
        StrategicTriangulator, 
        triangulate_portfolio,
        generate_portfolio_brief_sync,  # Unified portfolio brief
        calculate_expansion_alpha,       # Growth intelligence (offensive layer)
        is_growth_eligible               # Velocity validation gate
    )
    TRIANGULATION_ENABLED = True
except ImportError:
    TRIANGULATION_ENABLED = False
    generate_portfolio_brief_sync = None
    calculate_expansion_alpha = None
    is_growth_eligible = None

# Initialize OpenAI client (for chat feature - optional)
openai_client = None
if OPENAI_SYNC_AVAILABLE:
    try:
        openai_client = OpenAI(api_key=st.secrets.get("openai", {}).get("OPENAI_API_KEY"))
    except Exception:
        openai_client = None

# Pre-compile regex for performance
_METRICS_PATTERN = re.compile(r'\$[\d,]+|\d+\.\d+%|\d+ products')


def get_product_strategy(row: dict, revenue: float = 0, use_triangulation: bool = True, strategic_bias: str = "Balanced Defense",
                         enable_triggers: bool = False, enable_network: bool = False,
                         competitors_df: pd.DataFrame = None) -> dict:
    """
    UNIFIED AI ENGINE - Strategic Classification + Predictive Intelligence

    Single entry point for all AI analysis. Returns unified outputs that include:
    - Strategic state (FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL)
    - Predictive 30-day risk forecast
    - Actionable alerts (Inventory, Pricing, Rank)
    - Model certainty based on data quality
    - Trigger events (optional - requires historical data)
    - Network intelligence (optional - requires Supabase)
    - Competitive intelligence (uses competitors_df for market context)

    Args:
        row: Product row dictionary with metrics
        revenue: Product revenue for predictive calculations
        use_triangulation: Whether to use unified AI engine (default True)
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        enable_triggers: Enable trigger event detection (requires historical_df in row)
        enable_network: Enable network intelligence (requires category_id in row)
        competitors_df: DataFrame of competitor products in same market (for competitive analysis)

    Returns:
        dict with unified strategic + predictive outputs
    """
    # Convert row to Series for legacy fallback
    row_series = pd.Series(row) if isinstance(row, dict) else row

    # Add historical data from session state if available and triggers enabled
    if enable_triggers and 'historical_df' not in row:
        # Try to get historical data from session state
        if 'df_weekly' in st.session_state and not st.session_state['df_weekly'].empty:
            asin = row.get('asin', '')
            if asin:
                df_weekly = st.session_state['df_weekly']
                historical_df = df_weekly[df_weekly['asin'] == asin].copy()
                if not historical_df.empty:
                    row['historical_df'] = historical_df
    
    # Add competitor data for competitive intelligence
    if competitors_df is not None and not competitors_df.empty:
        asin = row.get('asin', '')
        # Filter out the current product from competitors
        row['competitors_df'] = competitors_df[competitors_df['asin'] != asin].copy()
        
        # Enrich row with competitive context metrics
        competitors = row['competitors_df']
        if not competitors.empty:
            # Calculate competitive benchmarks
            if 'buy_box_price' in competitors.columns or 'filled_price' in competitors.columns:
                price_col = 'buy_box_price' if 'buy_box_price' in competitors.columns else 'filled_price'
                median_price = competitors[price_col].median()
                current_price = row.get('buy_box_price', row.get('filled_price', row.get('price', 0)))
                if median_price and current_price:
                    row['price_gap_vs_median'] = (current_price - median_price) / median_price if median_price > 0 else 0
                    row['median_competitor_price'] = median_price
            
            if 'review_count' in competitors.columns:
                median_reviews = competitors['review_count'].median()
                row['median_competitor_reviews'] = median_reviews
                current_reviews = row.get('review_count', 0)
                if median_reviews and current_reviews:
                    row['review_advantage_pct'] = (current_reviews - median_reviews) / median_reviews if median_reviews > 0 else 0
            
            if 'sales_rank_filled' in competitors.columns or 'sales_rank' in competitors.columns:
                rank_col = 'sales_rank_filled' if 'sales_rank_filled' in competitors.columns else 'sales_rank'
                row['competitor_count'] = len(competitors)
                row['best_competitor_rank'] = competitors[rank_col].min()
                row['worst_competitor_rank'] = competitors[rank_col].max()
            
            # Out of stock percentage (if available)
            if 'outOfStockPercentage90' in competitors.columns:
                avg_oos = competitors['outOfStockPercentage90'].mean()
                if avg_oos > 1:
                    avg_oos = avg_oos / 100  # Normalize to 0-1
                row['competitor_oos_pct'] = avg_oos
    elif 'competitors_df' not in row:
        # Try to get competitor data from session state (market_snapshot)
        market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
        if not market_snapshot.empty:
            asin = row.get('asin', '')
            row['competitors_df'] = market_snapshot[market_snapshot['asin'] != asin].copy()

    # Use unified AI engine
    if use_triangulation and TRIANGULATION_ENABLED:
        try:
            # Single call to unified engine (strategic + predictive + triggers + network)
            triangulator = StrategicTriangulator(
                use_llm=True,
                strategic_bias=strategic_bias,
                enable_triggers=enable_triggers,
                enable_network=enable_network
            )
            brief = triangulator.analyze(row, strategic_bias=strategic_bias, revenue=revenue)
            
            # Map strategic state to legacy categories
            state_to_category = {
                "FORTRESS": "🏰 Fortress",
                "HARVEST": "🌾 Harvest",
                "TRENCH_WAR": "⚔️ Trench War",
                "DISTRESS": "🚨 Distress",
                "TERMINAL": "💀 Terminal",
            }
            
            state_to_zone = {
                "FORTRESS": "🏰 FORTRESS (Cash Flow)",
                "HARVEST": "🌾 HARVEST (Profit)",
                "TRENCH_WAR": "⚔️ TRENCH (Defense)",
                "DISTRESS": "🚨 DISTRESS (At Risk)",
                "TERMINAL": "💀 EXIT (Terminal)",
            }
            
            state_to_ad = {
                "FORTRESS": "⚖️ OPTIMIZE ROAS",
                "HARVEST": "📉 REDUCE SPEND",
                "TRENCH_WAR": "🎯 DEFENSIVE KEYWORDS",
                "DISTRESS": "⏸️ PAUSE & INVESTIGATE",
                "TERMINAL": "🛑 FULL STOP",
            }
            
            state_to_ecom = {
                "FORTRESS": "📈 TEST PRICE INCREASE",
                "HARVEST": "📈 RAISE PRICE",
                "TRENCH_WAR": "🎫 MATCH COMPETITOR",
                "DISTRESS": "🔍 FIX ROOT CAUSE",
                "TERMINAL": "💀 LIQUIDATE",
            }
            
            state = brief.strategic_state
            
            return {
                # === STRATEGIC CLASSIFICATION ===
                "strategic_state": state,
                "strategic_emoji": brief.state_emoji,
                "strategic_color": brief.state_color,
                "confidence_score": brief.confidence,  # Now = model_certainty
                "primary_outcome": brief.primary_outcome,
                "recommended_plan": brief.recommended_action,
                "reasoning": brief.reasoning,
                "signals_detected": brief.signals_detected,
                "source": brief.source,
                
                # === PREDICTIVE INTELLIGENCE (Defensive) ===
                "thirty_day_risk": brief.thirty_day_risk,
                "daily_burn_rate": brief.daily_burn_rate,
                "predictive_state": brief.predictive_state,
                "predictive_emoji": brief.predictive_emoji,
                "cost_of_inaction": brief.cost_of_inaction,
                "ai_recommendation": brief.ai_recommendation,
                "alert_type": brief.alert_type,
                "alert_urgency": brief.alert_urgency,
                "predicted_event_date": brief.predicted_event_date,
                "action_deadline": brief.action_deadline,
                "model_certainty": brief.confidence,
                "data_quality": brief.data_quality,
                
                # === GROWTH INTELLIGENCE (Offensive) ===
                "thirty_day_growth": brief.thirty_day_growth,
                "price_lift_opportunity": brief.price_lift_opportunity,
                "conquest_opportunity": brief.conquest_opportunity,
                "expansion_recommendation": brief.expansion_recommendation,
                "growth_validated": brief.growth_validated,
                "opportunity_type": brief.opportunity_type,
                
                # === PREDICTIVE-DERIVED OUTPUTS (replaces legacy capital_zone) ===
                "ad_action": state_to_ad.get(state, "⚖️ OPTIMIZE ROAS"),
                "ecom_action": state_to_ecom.get(state, "✅ MAINTAIN"),
                "problem_category": f"{brief.state_emoji} {state_to_category.get(state, state)}",
                "problem_reason": brief.reasoning[:80] + "..." if len(brief.reasoning) > 80 else brief.reasoning,
                
                # === PREDICTIVE RISK (North Star metric) ===
                "opportunity_value": brief.thirty_day_risk,  # Always use predictive risk
                
                # === PREDICTIVE STATE (replaces capital_zone) ===
                "predictive_zone": f"{brief.predictive_emoji} {brief.predictive_state}",
                "is_healthy": brief.predictive_state in ["HOLD", "EXPLOIT"]
            }
        except Exception as e:
            # Unified fallback using deterministic AI engine
            pass
    
    # Fallback: Use deterministic AI engine (no legacy engine.py)
    if TRIANGULATION_ENABLED:
        try:
            from utils.ai_engine import _determine_state_fallback
            brief = _determine_state_fallback(row, reason="Primary analysis failed", strategic_bias=strategic_bias)
            state = brief.strategic_state
            return {
                "ad_action": "⚖️ OPTIMIZE ROAS",
                "ecom_action": "✅ MAINTAIN",
                "problem_category": f"{brief.state_emoji} {state}",
                "problem_reason": brief.reasoning[:80] + "..." if len(brief.reasoning) > 80 else brief.reasoning,
                "opportunity_value": brief.thirty_day_risk if hasattr(brief, 'thirty_day_risk') else 0,
                "confidence_score": brief.confidence,
                "strategic_state": state,
                "recommended_plan": brief.recommended_action,
                "predictive_state": brief.predictive_state if hasattr(brief, 'predictive_state') else "HOLD",
                "predictive_emoji": brief.predictive_emoji if hasattr(brief, 'predictive_emoji') else "✅",
                "predictive_zone": f"{brief.predictive_emoji if hasattr(brief, 'predictive_emoji') else '✅'} {brief.predictive_state if hasattr(brief, 'predictive_state') else 'HOLD'}",
                "is_healthy": brief.predictive_state in ["HOLD", "EXPLOIT"] if hasattr(brief, 'predictive_state') else True,
                # Growth Intelligence
                "thirty_day_risk": brief.thirty_day_risk if hasattr(brief, 'thirty_day_risk') else 0,
                "thirty_day_growth": brief.thirty_day_growth if hasattr(brief, 'thirty_day_growth') else 0,
                "price_lift_opportunity": brief.price_lift_opportunity if hasattr(brief, 'price_lift_opportunity') else 0,
                "conquest_opportunity": brief.conquest_opportunity if hasattr(brief, 'conquest_opportunity') else 0,
                "expansion_recommendation": brief.expansion_recommendation if hasattr(brief, 'expansion_recommendation') else "",
                "growth_validated": brief.growth_validated if hasattr(brief, 'growth_validated') else True,
                "opportunity_type": brief.opportunity_type if hasattr(brief, 'opportunity_type') else "",
                "source": "fallback"
            }
        except Exception:
            pass
    
    # Final fallback if all analysis fails (minimal response)
    return {
        "ad_action": "⚖️ OPTIMIZE ROAS",
        "ecom_action": "✅ MAINTAIN",
        "problem_category": "📊 Awaiting Analysis",
        "problem_reason": "Analysis pending - insufficient data",
        "opportunity_value": 0,  # No predictive risk without analysis
        "confidence_score": 0.3,
        "strategic_state": "HARVEST",  # Neutral state
        "recommended_plan": "Awaiting analysis",
        "predictive_state": "HOLD",
        "predictive_emoji": "✅",
        "predictive_zone": "✅ HOLD",
        "is_healthy": True,
        # Growth defaults
        "thirty_day_risk": 0,
        "thirty_day_growth": 0,
        "price_lift_opportunity": 0,
        "conquest_opportunity": 0,
        "expansion_recommendation": "",
        "growth_validated": True,
        "opportunity_type": "",
        "source": "error"
    }

def _hash_portfolio_data(portfolio_summary: str) -> str:
    """
    Create a hash of key portfolio metrics to use as cache key.

    Performance: Pre-compiled regex pattern, optimized string operations.
    """
    # Extract key numbers from summary for hashing (using pre-compiled pattern)
    key_metrics = _METRICS_PATTERN.findall(portfolio_summary)
    metrics_str = '|'.join(key_metrics)
    return hashlib.md5(metrics_str.encode(), usedforsecurity=False).hexdigest()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_ai_brief(portfolio_summary: str, data_hash: str) -> str:
    """
    Generate an LLM-powered strategic brief for the portfolio.
    
    NOW USES THE SAME AI ENGINE as product-level classification
    for consistency across all AI outputs.

    Cached by data_hash (portfolio metrics), not by date, to avoid
    unnecessary API calls when only the date range changes.

    Performance: Cached results reduce API calls and latency.
    """
    # Use the unified AI engine from utils/ai_engine.py
    # This ensures the same client, model, and configuration as product classification
    if generate_portfolio_brief_sync is None:
        return None
    
    return generate_portfolio_brief_sync(portfolio_summary)


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def _cached_portfolio_intelligence(data_hash: str, total_revenue: float, strategic_bias: str, _df: pd.DataFrame) -> dict:
    """
    Cached wrapper for portfolio intelligence calculation.
    
    Caches the vectorized intelligence calculation to avoid redundant computation
    when sidebar toggles change but data hasn't.
    
    Args:
        data_hash: Hash of portfolio data for cache key
        total_revenue: Total monthly revenue
        strategic_bias: User's strategic focus
        _df: Portfolio DataFrame (underscore prefix = not hashed)
        
    Returns:
        Dict with portfolio intelligence metrics and enriched DataFrame
    """
    from utils.ai_engine import calculate_portfolio_predictive_risk
    return calculate_portfolio_predictive_risk(_df, total_revenue, strategic_bias)

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ShelfGuard OS", layout="wide", page_icon="🛡️")

# --- CLEAN LIGHT UI ---
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Metric tiles */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #00704A;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    [data-testid="stMetricValue"] div { color: #1a1a1a !important; font-weight: 700; }
    
    .custom-metric-container {
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #00704A;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .custom-metric-label { font-size: 0.875rem; color: #666; margin-bottom: 4px; }
    .custom-metric-value { font-size: 1.75rem; font-weight: 700; color: #1a1a1a; margin-bottom: auto; }

    .benchmark-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
    .benchmark-pos, .benchmark-elite { background: #d4edda; color: #155724; }
    .benchmark-neg, .benchmark-atrisk { background: #f8d7da; color: #721c24; }
    .benchmark-neu, .benchmark-standard { background: #fff3cd; color: #856404; }
    .benchmark-row { margin-top: auto; padding-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; }

    .pos { color: #28a745; }
    .neg { color: #dc3545; }
    .neu { color: #ffc107; }

    .metric-subtext { font-size: 0.8rem; font-weight: 600; margin-top: 6px; }
    .metric-subtext.pos { color: #28a745; }
    .metric-subtext.neg { color: #dc3545; }
    .metric-subtext.neu { color: #f57c00; }
    
    .alpha-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #00704A;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        min-height: 200px;
        display: flex;
        flex-direction: column;
    }
    .alpha-label { font-size: 0.875rem; color: #666; margin-bottom: 4px; }
    .alpha-score { font-size: 1.75rem; font-weight: 700; color: #1a1a1a; }
    .alpha-score-suffix { font-size: 1rem; color: #999; }
    .alpha-validation { font-size: 0.9rem; font-weight: 600; margin-top: 6px; }
    .alpha-validation.validated { color: #28a745; }
    .alpha-validation.rejected { color: #dc3545; }
    .alpha-validation.neutral { color: #666; }
    .alpha-divider { border-top: 1px solid #eee; margin: 10px 0; }
    .alpha-saved { font-size: 0.85rem; color: #666; }
    .alpha-saved-value { font-weight: 700; color: #00704A; }
    .alpha-status-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; margin-left: 6px; }
    .alpha-status-badge.human { background: #d4edda; color: #155724; }
    .alpha-status-badge.system { background: #cce5ff; color: #004085; }
    
    .product-card { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 12px; }
    .product-card:hover { border-color: #00704A; }
    .product-img { width: 100%; height: 120px; object-fit: contain; border-radius: 4px; }
    
    .ai-response-box { background: #f8f9fa; border-left: 3px solid #00704A; padding: 10px; border-radius: 4px; margin: 6px 0; font-size: 0.9rem; }
    .user-query-box { background: #e3f2fd; border-left: 3px solid #1976d2; padding: 10px; border-radius: 4px; margin: 6px 0; font-size: 0.9rem; }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# === STRATEGIC GOVERNOR (Global Context Selector) ===
st.sidebar.title("⚙️ Strategic Settings")
st.sidebar.markdown("---")

strategic_bias = st.sidebar.radio(
    "**🎯 Current Strategic Focus**",
    options=['💰 Profit Maximization', '⚖️ Balanced Defense', '🚀 Aggressive Growth'],
    index=1,  # Default to Balanced Defense
    help="""
    **Profit Mode**: Prioritize margins and efficiency. Penalize low-margin products.
    
    **Balanced Mode**: Standard defense scoring. Evaluate all factors equally.
    
    **Growth Mode**: Prioritize velocity and market share. Forgive margin compression if rank is improving.
    """,
    key='strategic_bias'
)

# Clean strategic bias string (remove emoji for internal use)
strategic_bias_clean = strategic_bias.split(' ', 1)[1] if ' ' in strategic_bias else strategic_bias

st.sidebar.markdown("---")
st.sidebar.caption(f"🎚️ AI Engine: **{strategic_bias_clean}**")

# === AI ENGINE DEBUG STATUS ===
with st.sidebar.expander("🔍 AI Engine Debug", expanded=False):
    # Check if OpenAI client can be initialized
    try:
        from utils.ai_engine import _get_openai_client, _get_model_name
        test_client = _get_openai_client()
        test_model = _get_model_name()
        
        if test_client is not None:
            st.success("✅ OpenAI Connected")
            st.caption(f"Model: `{test_model}`")
            
            # Show API key status (masked)
            try:
                api_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY", "")
                if api_key:
                    masked_key = api_key[:8] + "..." + api_key[-4:]
                    st.caption(f"Key: `{masked_key}`")
            except:
                pass
        else:
            st.error("❌ OpenAI Not Connected")
            st.caption("Check secrets.toml or .env file")
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:50]}")
    
    # Show LLM call statistics (if available in session state)
    if 'llm_stats' in st.session_state:
        stats = st.session_state.llm_stats
        st.markdown("---")
        st.caption("**Session Statistics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 LLM Calls", stats.get('llm_calls', 0))
        with col2:
            st.metric("📊 Fallback", stats.get('fallback_calls', 0))
        
        success_rate = 0
        total = stats.get('llm_calls', 0) + stats.get('fallback_calls', 0)
        if total > 0:
            success_rate = (stats.get('llm_calls', 0) / total) * 100
        st.progress(success_rate / 100)
        st.caption(f"Success Rate: {success_rate:.1f}%")

# === TOP LEVEL NAVIGATION ===
main_tab1, main_tab2, main_tab3 = st.tabs(["🛡️ Command Center", "🔍 Market Discovery", "📂 My Projects"])

with main_tab2:
    # Market Discovery - Always available, no data needed
    render_discovery_ui()

with main_tab3:
    # My Projects - Always available
    project_id = render_project_selector()
    if project_id:
        render_project_dashboard(project_id)
    else:
        st.info("💡 Create your first project using the Market Discovery tab!")

with main_tab1:
    # === URL PERSISTENCE (Cache-First Logic) ===
    # Sync active project with URL query params for page reload persistence
    query_params = st.query_params
    
    # If URL has project params but session doesn't, restore from URL
    # The Oracle will load data from Supabase cache automatically
    if 'project_asin' in query_params and not st.session_state.get('active_project_asin'):
        st.session_state['active_project_asin'] = query_params.get('project_asin')
        st.session_state['active_project_name'] = query_params.get('project_name', 'Restored Project')
        
        # Try to restore project ASINs from Supabase tracked_asins table
        if SUPABASE_CACHE_ENABLED:
            try:
                from src.persistence import load_project_asins
                # Note: This requires the project_id, which we don't have from URL
                # For now, just mark that we need to load from cache
                st.session_state['needs_cache_load'] = True
            except Exception:
                pass
    
    # === STATE MACHINE ROUTER ===
    # Check if user has activated a project
    active_project_asin = st.session_state.get('active_project_asin', None)
    
    # Sync to URL if we have an active project
    if active_project_asin and query_params.get('project_asin') != active_project_asin:
        st.query_params['project_asin'] = active_project_asin
        st.query_params['project_name'] = st.session_state.get('active_project_name', 'Project')

    if not active_project_asin:
        # === SYSTEM OFFLINE / COLD START STATE ===
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 60px 40px; background: white;
                        border: 2px dashed #e0e0e0; border-radius: 12px;">
                <div style="font-size: 72px; margin-bottom: 20px;">⚪</div>
                <div style="font-size: 28px; font-weight: 700; color: #666; margin-bottom: 12px;">
                    SYSTEM IDLE
                </div>
                <div style="font-size: 16px; color: #999; margin-bottom: 30px;">
                    No active defense perimeter detected
                </div>
                <div style="font-size: 14px; color: #666; margin-bottom: 20px;">
                    Initialize your first market position to activate the Command Center
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Redirect button to Market Discovery
            if st.button("🔍 Initialize Defense in Market Discovery", use_container_width=True, type="primary"):
                st.session_state['redirect_to_discovery'] = True
                st.info("💡 Navigate to the **Market Discovery** tab above to search for your product and activate the Command Center")

        st.stop()  # Don't render the rest of the dashboard

    # === ACTIVE PROJECT MODE ===
    # If we have an active project, continue with dashboard rendering
    try:
        # 2. DATA INGESTION - CACHE-FIRST ARCHITECTURE (The Oracle)
        # Priority: Supabase Cache > Session State > Redirect to Discovery
        
        project_name = st.session_state.get('active_project_name', 'Unknown Project')
        seed_asin = st.session_state.get('active_project_asin', None)
        project_asins = st.session_state.get('active_project_all_asins', [])
        seed_brand = st.session_state.get('active_project_seed_brand', '')
        
        # Initialize data containers
        df_weekly = pd.DataFrame()
        market_snapshot = pd.DataFrame()
        data_source = "none"
        
        # === SUPABASE CACHE (PRIMARY) - FIX 1.2: Enhanced with Network Intelligence ===
        # Try loading from harvested snapshots first (instant 0.1s load)
        # Now includes category benchmarks for competitive context
        category_id = st.session_state.get('active_project_category_id')
        network_stats = {}

        if SUPABASE_CACHE_ENABLED and project_asins:
            try:
                # Use enhanced function with network intelligence if available
                if get_market_snapshot_with_network_intelligence:
                    market_snapshot, cache_stats = get_market_snapshot_with_network_intelligence(
                        project_asins, seed_brand, category_id
                    )
                    if cache_stats.get('has_network_context'):
                        network_stats = cache_stats.get('network_intelligence', {})
                else:
                    market_snapshot, cache_stats = get_market_snapshot_from_cache(project_asins, seed_brand)

                if not market_snapshot.empty:
                    data_source = "supabase"
                    df_weekly = market_snapshot.copy()  # Use snapshot as weekly data

                    # Check freshness and show indicator
                    freshness = check_data_freshness(project_asins)
                    if freshness.get("is_stale"):
                        st.caption(f"⏰ Data last updated {freshness.get('freshness_hours', 'N/A')}h ago")

            except Exception as e:
                st.caption(f"⚠️ Cache unavailable, using session data")
        
        # === SESSION STATE FALLBACK ===
        # For newly created projects not yet harvested
        if market_snapshot.empty:
            df_weekly = st.session_state.get('active_project_data', pd.DataFrame())
            market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
            if not market_snapshot.empty:
                data_source = "session"
                # CRITICAL: Normalize session state data to ensure consistent column names and dtypes
                # Discovery phase uses: price, bsr, revenue_proxy, monthly_units
                # Dashboard expects: weekly_sales_filled (numeric), revenue_proxy (numeric)
                if 'revenue_proxy' in market_snapshot.columns:
                    market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['revenue_proxy'], errors='coerce').fillna(0)
                    market_snapshot['weekly_sales_filled'] = market_snapshot['revenue_proxy'].copy()
                if 'bsr' in market_snapshot.columns:
                    market_snapshot['sales_rank_filled'] = pd.to_numeric(market_snapshot['bsr'], errors='coerce').fillna(0)
                if 'price' in market_snapshot.columns:
                    market_snapshot['filled_price'] = pd.to_numeric(market_snapshot['price'], errors='coerce').fillna(0)
                    market_snapshot['buy_box_price'] = market_snapshot['filled_price'].copy()
                if 'monthly_units' in market_snapshot.columns:
                    market_snapshot['estimated_units'] = pd.to_numeric(market_snapshot['monthly_units'], errors='coerce').fillna(0)
                # Ensure AI-critical metrics are numeric (prevents "zero buy box" false positives)
                if 'amazon_bb_share' in market_snapshot.columns:
                    market_snapshot['amazon_bb_share'] = pd.to_numeric(market_snapshot['amazon_bb_share'], errors='coerce').fillna(0)
                if 'review_count' in market_snapshot.columns:
                    market_snapshot['review_count'] = pd.to_numeric(market_snapshot['review_count'], errors='coerce').fillna(0).astype(int)
                if 'rating' in market_snapshot.columns:
                    market_snapshot['rating'] = pd.to_numeric(market_snapshot['rating'], errors='coerce').fillna(0)
                if 'new_offer_count' in market_snapshot.columns:
                    market_snapshot['new_offer_count'] = pd.to_numeric(market_snapshot['new_offer_count'], errors='coerce').fillna(1).astype(int)
                
                # === DATA HEALER: Apply comprehensive gap-filling ===
                # Ensures AI receives complete data with no gaps
                try:
                    from utils.data_healer import clean_and_interpolate_metrics
                    market_snapshot = clean_and_interpolate_metrics(market_snapshot, group_by_column="asin", verbose=False)
                except ImportError:
                    pass  # Data healer not available

        if df_weekly.empty or market_snapshot.empty:
            st.warning("⚠️ No project data found. Please create a project in Market Discovery.")
            st.stop()
        
        # 3. VELOCITY EXTRACTION FROM DATABASE (FIX 1.1 - Critical Pipeline Fix)
        # Extract velocity trends from historical_metrics table instead of session state
        # This enables cross-session velocity tracking and uses the 90-day backfill data
        velocity_df = pd.DataFrame()
        historical_df_for_velocity = pd.DataFrame()

        # PRIORITY 1: Try loading from historical_metrics table in Supabase
        project_id = st.session_state.get('active_project_id')
        if SUPABASE_CACHE_ENABLED and load_historical_metrics_from_db and project_id:
            try:
                historical_df_for_velocity = load_historical_metrics_from_db(project_id)
                if not historical_df_for_velocity.empty:
                    st.session_state['velocity_source'] = 'database'
            except Exception as e:
                st.caption(f"⚠️ DB velocity load failed: {str(e)[:50]}")

        # PRIORITY 2: Try loading by ASINs if project_id failed
        if historical_df_for_velocity.empty and SUPABASE_CACHE_ENABLED and load_historical_metrics_by_asins and project_asins:
            try:
                asin_tuple = tuple(sorted([a.strip().upper() for a in project_asins]))
                historical_df_for_velocity = load_historical_metrics_by_asins(asin_tuple, days=90)
                if not historical_df_for_velocity.empty:
                    st.session_state['velocity_source'] = 'database_by_asin'
            except Exception as e:
                pass  # Silent fallback to session state

        # PRIORITY 3: Fallback to session state df_weekly (for newly created projects)
        if historical_df_for_velocity.empty and not df_weekly.empty:
            historical_df_for_velocity = df_weekly
            st.session_state['velocity_source'] = 'session_state'

        # Extract velocity trends from the historical data
        if not historical_df_for_velocity.empty and 'asin' in historical_df_for_velocity.columns:
            try:
                from utils.ai_engine import extract_portfolio_velocity
                velocity_df = extract_portfolio_velocity(historical_df_for_velocity)
                # Merge velocity data into market_snapshot
                if not velocity_df.empty:
                    market_snapshot = market_snapshot.merge(
                        velocity_df[['asin', 'velocity_trend_30d', 'velocity_trend_90d', 'data_quality', 'data_weeks']],
                        on='asin',
                        how='left'
                    )
                    # Fill NaN velocities with 0.0 (no change)
                    market_snapshot['velocity_trend_30d'] = market_snapshot['velocity_trend_30d'].fillna(0.0)
                    market_snapshot['velocity_trend_90d'] = market_snapshot['velocity_trend_90d'].fillna(0.0)
                    market_snapshot['data_quality'] = market_snapshot['data_quality'].fillna('VERY_LOW')
                    market_snapshot['data_weeks'] = market_snapshot['data_weeks'].fillna(0)
            except Exception as e:
                # Fallback: Set default velocity values
                market_snapshot['velocity_trend_30d'] = 0.0
                market_snapshot['velocity_trend_90d'] = 0.0
                market_snapshot['data_quality'] = 'VERY_LOW'
                market_snapshot['data_weeks'] = 0
        else:
            # No historical data available - set defaults
            market_snapshot['velocity_trend_30d'] = 0.0
            market_snapshot['velocity_trend_90d'] = 0.0
            market_snapshot['data_quality'] = 'VERY_LOW'
            market_snapshot['data_weeks'] = 0

        # === BRAND IDENTIFICATION ===
        # Step 1: Find the target brand from the seed ASIN
        if 'brand' not in market_snapshot.columns:
            # Fallback: Create brand column from title (first word)
            st.warning("⚠️ Brand column missing - creating from product titles...")
            market_snapshot['brand'] = market_snapshot['title'].apply(lambda x: x.split()[0] if pd.notna(x) and x else "Unknown")

        # Get the brand of the seed product
        seed_product = market_snapshot[market_snapshot['asin'] == seed_asin]
        if seed_product.empty:
            st.error(f"❌ Seed ASIN {seed_asin} not found in market data.")
            st.stop()

        target_brand = seed_product['brand'].iloc[0]
        
        # Data source indicator (The Oracle status)
        source_icons = {"supabase": "⚡", "session": "💾", "none": "❓"}
        source_labels = {"supabase": "Cached (Instant)", "session": "Session", "none": "No Data"}
        velocity_source = st.session_state.get('velocity_source', 'unknown')
        velocity_indicator = "📊" if velocity_source.startswith('database') else "💾"
        network_indicator = "🌐" if network_stats.get('median_price') else ""

        st.caption(
            f"🎯 Analyzing: **{target_brand}** vs. Market | "
            f"{source_icons.get(data_source, '❓')} {source_labels.get(data_source, 'Unknown')} | "
            f"{velocity_indicator} Velocity: {velocity_source.replace('_', ' ').title()} "
            f"{network_indicator}"
        )

        # Step 2: Create two dataframes
        # Use CONTAINS matching to catch brand variations:
        # - "Colgate" matches "Colgate", "Colgate-Palmolive", "Colgate Oral Care"
        # - Case-insensitive
        target_brand_lower = target_brand.lower().strip() if target_brand else ""
        market_snapshot['brand_lower'] = market_snapshot['brand'].str.lower().str.strip().fillna("")
        
        # Use contains matching instead of exact matching
        # This catches "Colgate-Palmolive" when searching for "Colgate"
        market_snapshot['is_your_brand'] = market_snapshot['brand_lower'].str.contains(
            target_brand_lower, case=False, na=False, regex=False
        ) if target_brand_lower else False
        
        # Also check if target brand is contained in the product title (backup)
        # This catches products where brand field is missing but title contains brand
        title_match = market_snapshot['title'].str.lower().str.contains(
            target_brand_lower, case=False, na=False, regex=False
        ) if target_brand_lower else False
        
        # Combine: is_your_brand if brand matches OR title contains brand name
        market_snapshot['is_your_brand'] = market_snapshot['is_your_brand'] | title_match
        
        portfolio_df = market_snapshot[market_snapshot['is_your_brand']].copy()
        market_df = market_snapshot  # View only - no modification needed
        
        # === CALCULATE COMPETITIVE INTELLIGENCE FOR GROWTH LAYER ===
        # Calculate price gaps (your price vs. competitor average price)
        # This enables the Growth Intelligence layer to detect opportunities
        price_col = 'current_price' if 'current_price' in market_snapshot.columns else 'avg_price' if 'avg_price' in market_snapshot.columns else None
        if price_col:
            # Calculate competitor average price (excluding your brand)
            competitor_prices = market_snapshot.loc[~market_snapshot['is_your_brand'], price_col]
            market_avg_price = competitor_prices.mean() if len(competitor_prices) > 0 else 0
            
            if market_avg_price > 0:
                # Price gap: (your_price - market_avg) / market_avg
                # Negative = you're cheaper (growth opportunity: price lift)
                # Positive = you're more expensive (risk: price erosion)
                market_snapshot['price_gap_vs_competitor'] = (
                    (market_snapshot[price_col] - market_avg_price) / market_avg_price
                ).fillna(0)
            else:
                market_snapshot['price_gap_vs_competitor'] = 0.0
        else:
            market_snapshot['price_gap_vs_competitor'] = 0.0
        
        # Use Keepa OOS data if available
        if 'outOfStockPercentage90' in market_snapshot.columns:
            # Calculate average competitor OOS rate
            competitor_oos = market_snapshot.loc[~market_snapshot['is_your_brand'], 'outOfStockPercentage90']
            avg_competitor_oos = competitor_oos.mean() if len(competitor_oos) > 0 else 0
            # Normalize to 0-1 range if needed
            if avg_competitor_oos > 1:
                avg_competitor_oos = avg_competitor_oos / 100
            market_snapshot['competitor_oos_pct'] = avg_competitor_oos
        else:
            market_snapshot['competitor_oos_pct'] = 0.0
        
        # Debug: Show brand distribution in market (always show for transparency)
        unique_brands = market_snapshot['brand'].value_counts().head(15)
        your_brand_count = len(portfolio_df)
        your_brand_revenue = portfolio_df['revenue_proxy'].sum() if 'revenue_proxy' in portfolio_df.columns else 0
        total_market_revenue_debug = market_snapshot['revenue_proxy'].sum() if 'revenue_proxy' in market_snapshot.columns else 0
        market_share_pct = (your_brand_revenue / total_market_revenue_debug * 100) if total_market_revenue_debug > 0 else 0
        
        # Count how many matched by brand vs title
        brand_match_count = market_snapshot['brand_lower'].str.contains(target_brand_lower, case=False, na=False, regex=False).sum() if target_brand_lower else 0
        title_match_count = market_snapshot['title'].str.lower().str.contains(target_brand_lower, case=False, na=False, regex=False).sum() if target_brand_lower else 0
        
        with st.expander(f"🔍 Brand Matching: {your_brand_count} products matched ({market_share_pct:.1f}% market share)", expanded=(your_brand_count <= 5)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Target Brand:** `{target_brand}` (searching for: `{target_brand_lower}`)")
                st.markdown(f"**Matched Products:** {your_brand_count} of {len(market_snapshot)}")
                st.markdown(f"**Matched by Brand Field:** {brand_match_count}")
                st.markdown(f"**Matched by Title:** {title_match_count}")
                st.markdown(f"**Your Revenue:** ${your_brand_revenue:,.0f}")
                st.markdown(f"**Market Revenue:** ${total_market_revenue_debug:,.0f}")
            with col2:
                st.markdown("**Top 15 Brands in Market:**")
                st.dataframe(unique_brands, use_container_width=True)
            
            # Show matched products for verification
            if your_brand_count > 0:
                st.markdown("**✅ Matched Products:**")
                matched_cols = ['asin', 'title', 'brand', 'revenue_proxy'] if 'revenue_proxy' in portfolio_df.columns else ['asin', 'title', 'brand']
                available_cols = [c for c in matched_cols if c in portfolio_df.columns]
                st.dataframe(portfolio_df[available_cols].head(20), use_container_width=True)
            
            # Show sample NON-matched products to help debug
            if your_brand_count < 10:
                st.markdown("**❌ Sample NON-matched products (check if these should have matched):**")
                non_matched = market_snapshot[~market_snapshot['is_your_brand']].head(10)
                non_matched_cols = ['asin', 'title', 'brand', 'revenue_proxy'] if 'revenue_proxy' in non_matched.columns else ['asin', 'title', 'brand']
                available_non_cols = [c for c in non_matched_cols if c in non_matched.columns]
                st.dataframe(non_matched[available_non_cols], use_container_width=True)

        # === ENHANCEMENT 2.1: NETWORK INTELLIGENCE DISPLAY ===
        # Show category benchmarks if available (from network intelligence tables)
        if network_stats and network_stats.get('median_price'):
            with st.expander("🌐 Category Intelligence (Network Data)", expanded=False):
                st.markdown("**Category Benchmarks from Network Intelligence**")
                st.caption("Data accumulated from searches across ShelfGuard users")

                col1, col2, col3 = st.columns(3)

                with col1:
                    median_price = network_stats.get('median_price')
                    if median_price:
                        st.metric("Median Price", f"${median_price:.2f}")

                        # Compare your portfolio average to median
                        if 'buy_box_price' in portfolio_df.columns or 'filled_price' in portfolio_df.columns or 'price' in portfolio_df.columns:
                            price_col = 'buy_box_price' if 'buy_box_price' in portfolio_df.columns else 'filled_price' if 'filled_price' in portfolio_df.columns else 'price'
                            your_avg_price = portfolio_df[price_col].mean() if price_col in portfolio_df.columns else 0
                            if your_avg_price > 0:
                                price_diff_pct = ((your_avg_price / median_price) - 1) * 100
                                price_indicator = "🟢" if -10 < price_diff_pct < 10 else "🟡" if -20 < price_diff_pct < 20 else "🔴"
                                st.caption(f"{price_indicator} Your avg: ${your_avg_price:.2f} ({price_diff_pct:+.1f}%)")

                with col2:
                    median_bsr = network_stats.get('median_bsr')
                    if median_bsr:
                        st.metric("Median BSR", f"{int(median_bsr):,}")

                        # Compare your portfolio average to median
                        if 'sales_rank_filled' in portfolio_df.columns or 'bsr' in portfolio_df.columns:
                            bsr_col = 'sales_rank_filled' if 'sales_rank_filled' in portfolio_df.columns else 'bsr'
                            your_avg_bsr = portfolio_df[bsr_col].median() if bsr_col in portfolio_df.columns else 0
                            if your_avg_bsr > 0:
                                bsr_diff_pct = ((your_avg_bsr / median_bsr) - 1) * 100
                                # Lower BSR is better, so invert the indicator logic
                                bsr_indicator = "🟢" if bsr_diff_pct < -10 else "🟡" if bsr_diff_pct < 20 else "🔴"
                                st.caption(f"{bsr_indicator} Your median: {int(your_avg_bsr):,} ({bsr_diff_pct:+.1f}%)")

                with col3:
                    median_reviews = network_stats.get('median_review_count')
                    if median_reviews:
                        st.metric("Median Reviews", f"{int(median_reviews):,}")

                        # Compare your portfolio average to median
                        if 'review_count' in portfolio_df.columns:
                            your_avg_reviews = portfolio_df['review_count'].median()
                            if your_avg_reviews > 0:
                                review_diff_pct = ((your_avg_reviews / median_reviews) - 1) * 100
                                review_indicator = "🟢" if review_diff_pct > 20 else "🟡" if review_diff_pct > -20 else "🔴"
                                st.caption(f"{review_indicator} Your median: {int(your_avg_reviews):,} ({review_diff_pct:+.1f}%)")

                # Additional network stats
                st.markdown("---")
                col4, col5 = st.columns(2)
                with col4:
                    total_tracked = network_stats.get('total_asins_tracked', 0)
                    data_quality = network_stats.get('data_quality', 'UNKNOWN')
                    st.caption(f"📊 Network Data: {total_tracked} ASINs tracked | Quality: {data_quality}")
                with col5:
                    snapshot_date = network_stats.get('snapshot_date', 'N/A')
                    st.caption(f"📅 Last updated: {snapshot_date}")

        # Step 3: Calculate key metrics
        # Portfolio (Your Brand) metrics
        portfolio_revenue = portfolio_df['revenue_proxy'].sum()
        portfolio_product_count = len(portfolio_df)

        # Market (Total Category) metrics
        total_market_revenue = market_df['revenue_proxy'].sum()
        total_market_products = len(market_df)

        # Competitor metrics
        competitor_revenue = total_market_revenue - portfolio_revenue
        competitor_product_count = total_market_products - portfolio_product_count

        # Market share
        your_market_share = (portfolio_revenue / total_market_revenue * 100) if total_market_revenue > 0 else 0

        # Transform portfolio_df into dashboard format
        # IMPORTANT: Dashboard should show ONLY the target brand's products
        portfolio_snapshot_df = portfolio_df.copy()

        # Add required columns for dashboard
        # NOTE: revenue_proxy is MONTHLY revenue (calculated from avg_weekly_revenue * 4.33)
        # We use 'weekly_sales_filled' as column name for backward compatibility with dashboard
        # but the value represents MONTHLY revenue (90-day average monthly estimate)
        # CRITICAL: Ensure numeric dtype to prevent 'nlargest' errors
        if 'revenue_proxy' in portfolio_snapshot_df.columns:
            portfolio_snapshot_df['revenue_proxy'] = pd.to_numeric(portfolio_snapshot_df['revenue_proxy'], errors='coerce').fillna(0)
        portfolio_snapshot_df['weekly_sales_filled'] = portfolio_snapshot_df.get('revenue_proxy', 0)
        portfolio_snapshot_df['monthly_revenue'] = portfolio_snapshot_df.get('revenue_proxy', 0)
        portfolio_snapshot_df['asin'] = portfolio_snapshot_df.get('asin', '')

        # All products in portfolio_df are "Your Brand - Healthy" (predictive state: HOLD)
        portfolio_snapshot_df['problem_category'] = '✅ Your Brand - Healthy'
        portfolio_snapshot_df['predictive_zone'] = '✅ HOLD'  # Predictive state (replaces capital_zone)
        portfolio_snapshot_df['is_healthy'] = True  # Healthy until analyzed
        
        # === CARRY COMPETITIVE INTELLIGENCE INTO PORTFOLIO ===
        # These columns power the Growth Intelligence layer
        if 'price_gap_vs_competitor' in market_snapshot.columns:
            # Merge competitive intel from market_snapshot (which was calculated after portfolio_df was created)
            portfolio_snapshot_df['price_gap_vs_competitor'] = portfolio_snapshot_df['asin'].map(
                market_snapshot.set_index('asin')['price_gap_vs_competitor'].to_dict()
            ).fillna(0.0)
        else:
            portfolio_snapshot_df['price_gap_vs_competitor'] = 0.0
            
        if 'competitor_oos_pct' in market_snapshot.columns:
            portfolio_snapshot_df['competitor_oos_pct'] = market_snapshot['competitor_oos_pct'].iloc[0] if len(market_snapshot) > 0 else 0.0
        else:
            portfolio_snapshot_df['competitor_oos_pct'] = 0.0

        # Create res object that dashboard expects (scoped to YOUR BRAND ONLY)
        res = {
            'data': portfolio_snapshot_df,  # ONLY your brand's products
            'total_rev': portfolio_revenue,  # YOUR revenue, not market revenue
            'yoy_delta': 0.0,
            'share_delta': 0.0,
            'predictive_zones': {  # Replaces capital_flow
                '✅ HOLD': portfolio_revenue,  # Healthy products
                '🛡️ DEFEND': 0,  # Products needing defense
                '⚡ EXPLOIT': 0,  # Growth opportunities
                '🔄 REPLENISH': 0  # Inventory alerts
            },
            'demand_forecast': {},
            'hierarchy': {}
        }

        # Create simplified finance metrics
        fin = {
            'efficiency_score': int(your_market_share),
            'portfolio_status': 'Active',
            'avg_velocity_decay': 1.0,
            'annualized_waste': 0,
            'growth_alloc': 0
        }

        # Initialize chat state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_open" not in st.session_state:
            st.session_state.chat_open = False

        # Now continue with the full dashboard rendering using res and fin
        total_rev_curr = res.get("total_rev", 0)
        yoy_delta = res.get("yoy_delta", 0)
        share_delta = your_market_share / 100  # Use market share as proxy

        # Build AI context - will be enhanced after Command Center metrics are calculated
        portfolio_context = f"""
        BRAND PERFORMANCE SNAPSHOT:
        - Brand: {target_brand}
        - Market Share: {your_market_share:.1f}%
        - Portfolio Revenue: ${portfolio_revenue:,.0f}/month
        - Total Market Size: ${total_market_revenue:,.0f}/month
        - Your Products: {portfolio_product_count} ASINs
        - Market Total: {total_market_products} ASINs
        - Competitor Revenue: ${competitor_revenue:,.0f}/month ({competitor_product_count} products)
        - Position: {"Market Leader" if your_market_share > 50 else "Challenger" if your_market_share > 20 else "Niche Player"}
        """

        # 4. SYSTEM STATUS BANNER + HEADER (will be rendered after ai_brief is generated)
        # Placeholder - status will be determined after checking ai_brief text
        total_rev_curr = res.get("total_rev", 0)
        target_revenue = total_rev_curr * 1.0  # Placeholder - would come from user's target
        revenue_gap = total_rev_curr - target_revenue

        # Header with AI chat
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.title("ShelfGuard OS")
    
        with header_col2:
            with st.popover("🤖 AI Assistant", use_container_width=True):
                # Display chat messages
                if st.session_state.chat_messages:
                    for msg in st.session_state.chat_messages:
                        if msg["role"] == "user":
                            st.markdown(f'<div class="user-query-box">💬 {msg["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="ai-response-box">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Chat input - simple text input with on_submit
                user_input = st.chat_input("Ask a question about your Command Center...", key="ai_chat_input")
                
                if user_input:
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    try:
                        if openai_client is None:
                            st.session_state.chat_messages.append({
                                "role": "assistant", 
                                "content": "⚠️ Chat feature requires OpenAI package. Please install: pip install openai"
                            })
                        else:
                            # Build conversation history for context
                            messages_for_api = [
                                {"role": "system", "content": f"You are ShelfGuard AI, a strategic advisor for the Command Center. Be concise and actionable. {portfolio_context}"}
                            ]
                            # Add conversation history (last 10 messages for context)
                            for msg in st.session_state.chat_messages[-10:]:
                                messages_for_api.append(msg)
                            
                            response = openai_client.chat.completions.create(
                                model=st.secrets.get("openai", {}).get("model", "gpt-4o-mini"),
                                messages=messages_for_api,
                                max_tokens=300
                            )
                            st.session_state.chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    except Exception as e:
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
                
                # Clear button
                if st.session_state.chat_messages:
                    if st.button("🗑️ Clear Chat", key="ai_clear", use_container_width=True):
                        st.session_state.chat_messages = []
                        st.rerun()
    
        # === COMPACT EXECUTIVE BRIEFING ===
        total_rev_curr = res.get("total_rev", 0)
        yoy_delta = res.get("yoy_delta", 0)
        share_delta = res.get("share_delta", 0)
        
        # NOTE: Predictive zones replace legacy capital_flow
        # Zone metrics will be calculated from early predictive risk (see below)
        # These defaults will be overridden by predictive intelligence
        status_emoji, status_text, status_color = "🟢", "HEALTHY", "#28a745"
        top_action = "OPTIMIZE efficiency"
    
        # === SIMPLIFIED METRICS FOR DISCOVERY DATA ===
        # Since we don't have historical data, use simplified calculations
        demand_forecast = {}
        portfolio_forecast = 0
        forecast_confidence = "N/A"
        max_years = 0

        # Skip all forecast calculations - not available for snapshot data
        # === CALCULATE FORECAST METRICS (for AI Brief) ===
        # demand_forecast = res.get("demand_forecast", {})
        if demand_forecast:
            # Calculate weighted average forecast change
            total_forecast_rev = sum(f.get('current_weekly_avg', 0) for f in demand_forecast.values())
            if total_forecast_rev > 0:
                portfolio_forecast = sum(
                    f.get('current_weekly_avg', 0) * f.get('forecast_change_pct', 0) 
                    for f in demand_forecast.values()
                ) / total_forecast_rev
            
                # Calculate confidence (weighted by revenue)
                high_conf_count = sum(1 for f in demand_forecast.values() if f.get('confidence', 'LOW') == 'HIGH')
                med_conf_count = sum(1 for f in demand_forecast.values() if f.get('confidence', 'LOW') == 'MEDIUM')
                total_count = len(demand_forecast)
            
                if total_count > 0:
                    if high_conf_count / total_count > 0.5:
                        forecast_confidence = "HIGH"
                    elif (high_conf_count + med_conf_count) / total_count > 0.5:
                        forecast_confidence = "MED"
                    else:
                        forecast_confidence = "LOW"
                else:
                    forecast_confidence = "LOW"
                
                # Get max years analyzed
                max_years = max((f.get('years_analyzed', 1) for f in demand_forecast.values()), default=1)
            else:
                portfolio_forecast = 0
                forecast_confidence = "LOW"
                max_years = 1
        else:
            portfolio_forecast = 0
            forecast_confidence = "LOW"
            max_years = 1
    
        # === DETERMINE DATE DISPLAY ===
        # For Discovery data, show current market snapshot date
        from datetime import datetime
        date_display = datetime.now().strftime('%b %d, %Y')
        date_label = "Market Snapshot"
    
        # === AI-GENERATED STRATEGIC BRIEF (LLM-Powered) ===
        data_df = res.get("data", pd.DataFrame())
        
        # === VECTORIZED INTELLIGENCE CALCULATION (Single Point - 100x faster) ===
        # Calculate ALL predictive metrics in ONE vectorized pass
        # CACHED: Avoids redundant computation on sidebar interactions
        
        # Create data hash for cache key (based on ASIN list + revenue totals)
        portfolio_data = res["data"]
        asin_list = portfolio_data['asin'].tolist() if 'asin' in portfolio_data.columns else []
        data_cache_hash = hashlib.md5(
            f"{sorted(asin_list)[:20]}|{total_rev_curr:.0f}|{len(portfolio_data)}".encode(),
            usedforsecurity=False
        ).hexdigest()
        
        # Use cached intelligence calculation
        early_predictive_risk = _cached_portfolio_intelligence(
            data_cache_hash,
            total_rev_curr,
            strategic_bias,
            portfolio_data
        )
        
        # CRITICAL: Extract the pre-computed enriched DataFrame
        # This contains all intelligence columns - NO NEED to recalculate per-row
        enriched_portfolio_df = early_predictive_risk.get("_enriched_df", res["data"])
        
        # Store in session for downstream use (eliminates redundant loops)
        st.session_state['_enriched_portfolio'] = enriched_portfolio_df
        
        # Extract aggregate metrics for UI components
        thirty_day_risk = early_predictive_risk.get("thirty_day_risk", 0)
        risk_pct = early_predictive_risk.get("risk_pct", 0)
        portfolio_status = early_predictive_risk.get("portfolio_status", "HEALTHY")
        status_emoji = early_predictive_risk.get("status_emoji", "✅")
        defend_count = early_predictive_risk.get("defend_count", 0)
        exploit_count = early_predictive_risk.get("exploit_count", 0)
        replenish_count = early_predictive_risk.get("replenish_count", 0)
        
        # Growth metrics (offensive layer)
        thirty_day_growth = early_predictive_risk.get("thirty_day_growth", 0)
        growth_pct = early_predictive_risk.get("growth_pct", 0)
        price_lift_count = early_predictive_risk.get("price_lift_count", 0)
        conquest_count = early_predictive_risk.get("conquest_count", 0)
        expand_count = early_predictive_risk.get("expand_count", 0)
        opportunity_alpha = early_predictive_risk.get("opportunity_alpha", 0)
        growth_opportunity_count = early_predictive_risk.get("growth_opportunity_count", 0)

        # Build market summary for LLM (Brand vs Market + Predictive Intelligence)
        if not data_df.empty:
            
            # Extract TOP RISK PRODUCTS with actual ASINs and individual risk amounts
            top_risk_lines = ""
            top_growth_lines = ""
            
            if 'thirty_day_risk' in enriched_portfolio_df.columns:
                # Get top 3 products by risk
                risk_products = enriched_portfolio_df.nlargest(3, 'thirty_day_risk')[['asin', 'title', 'thirty_day_risk', 'strategic_state']].to_dict('records')
                if risk_products:
                    top_risk_lines = "\n    TOP RISK PRODUCTS (Actual ASINs):\n"
                    for p in risk_products:
                        if p['thirty_day_risk'] > 100:  # Only show meaningful risk
                            title_short = str(p.get('title', ''))[:40]
                            state = str(p.get('strategic_state', 'UNKNOWN'))
                            top_risk_lines += f"    - {p['asin']}: ${p['thirty_day_risk']:.0f}/mo risk ({state}) - {title_short}\n"
                
                # Get top 3 products by growth
                if 'thirty_day_growth' in enriched_portfolio_df.columns:
                    growth_products = enriched_portfolio_df[enriched_portfolio_df['thirty_day_growth'] > 0].nlargest(3, 'thirty_day_growth')[['asin', 'title', 'thirty_day_growth', 'opportunity_type']].to_dict('records')
                    if growth_products:
                        top_growth_lines = "\n    TOP GROWTH PRODUCTS (Actual ASINs):\n"
                        for p in growth_products:
                            title_short = str(p.get('title', ''))[:40]
                            opp_type = str(p.get('opportunity_type', '')).replace('_', ' ').title()
                            top_growth_lines += f"    - {p['asin']}: ${p['thirty_day_growth']:.0f}/mo upside ({opp_type}) - {title_short}\n"
            
            # Build summary for LLM showing brand performance, risk, and growth opportunities
            portfolio_summary = f"""
    BRAND PERFORMANCE ANALYSIS:
    - Brand Name: {target_brand}
    - Market Share: {your_market_share:.1f}%

    YOUR PORTFOLIO ({target_brand}):
    - Products: {portfolio_product_count} ASINs
    - Revenue: {f_money(portfolio_revenue)}/month
    - Avg Revenue per Product: {f_money(portfolio_revenue / portfolio_product_count if portfolio_product_count > 0 else 0)}

    COMPETITIVE LANDSCAPE:
    - Competitor Products: {competitor_product_count} ASINs
    - Competitor Revenue: {f_money(competitor_revenue)}/month
    - Total Market Size: {f_money(total_market_revenue)}/month
    - Your Position: {"Market Leader" if your_market_share > 50 else "Strong Challenger" if your_market_share > 20 else "Niche Player"}

    30-DAY PREDICTIVE INTELLIGENCE:
    - Total Portfolio Risk: {f_money(thirty_day_risk)} across {defend_count} products ({risk_pct:.1f}% of revenue)
    - Total Growth Opportunity: {f_money(thirty_day_growth)} across {growth_opportunity_count} products ({growth_pct:.1f}% potential)
    - Opportunity Alpha (Risk + Growth): {f_money(opportunity_alpha)}
    - Price Lift Opportunities: {price_lift_count} products
    - Conquest Opportunities: {conquest_count} products (competitors vulnerable)
    - Strategic Focus: {strategic_bias}
    {top_risk_lines}{top_growth_lines}
    NOTE: Risk is distributed across {defend_count} products, not concentrated in 1-2 products.
    """
        
            # Hash portfolio data for smart caching (only regenerates if metrics actually change)
            portfolio_hash = _hash_portfolio_data(portfolio_summary)

            # Check if user wants to force refresh
            if "force_refresh_brief" not in st.session_state:
                st.session_state.force_refresh_brief = False

            # Try LLM-powered brief first (cached by data hash, not date)
            # If force refresh, use a unique hash to bypass cache
            cache_key = portfolio_hash + ("_refresh" if st.session_state.force_refresh_brief else "")
            llm_brief = generate_ai_brief(portfolio_summary, cache_key)

            # Reset force refresh flag after use
            if st.session_state.force_refresh_brief:
                st.session_state.force_refresh_brief = False

            if llm_brief:
                ai_brief = llm_brief
                brief_source = "🤖 STRATEGIC BRIEF"
            else:
                # Fallback to rule-based brief (Brand vs Market + Predictive Intelligence)
                brief_parts = []
                
                # Risk alerts
                if risk_pct > 15:
                    brief_parts.append(f"**{f_money(thirty_day_risk)} at risk** over next 30 days ({risk_pct:.0f}% of revenue) — {defend_count} products need defensive action.")
                
                # Growth opportunities
                if thirty_day_growth > 0:
                    if conquest_count > 0:
                        brief_parts.append(f"**{f_money(thirty_day_growth)} growth opportunity** identified — {conquest_count} competitors vulnerable to conquest.")
                    elif price_lift_count > 0:
                        brief_parts.append(f"**{f_money(thirty_day_growth)} upside** via price optimization across {price_lift_count} products.")
                
                # Market position
                if portfolio_product_count > 0 and your_market_share < 30:
                    brief_parts.append(f"**{your_market_share:.1f}% market share** — significant expansion potential.")
                
                if competitor_revenue > portfolio_revenue * 2:
                    brief_parts.append(f"Competitors control {f_money(competitor_revenue)}/month — monitor for conquest opportunities.")

                ai_brief = " ".join(brief_parts) if brief_parts else f"{target_brand} positioned in competitive market. Monitor and optimize."
                brief_source = "📊 ANALYSIS"
        else:
            ai_brief = "Analyzing brand data..."
            brief_source = "⏳ LOADING"

        # Market context (Brand positioning)
        if your_market_share > 50:
            share_context = f"{target_brand} is the market leader ({your_market_share:.1f}% share)."
        elif your_market_share > 20:
            share_context = f"{target_brand} is a challenger brand ({your_market_share:.1f}% share)."
        else:
            share_context = f"{target_brand} is a niche player ({your_market_share:.1f}% share) — significant growth opportunity."

        # Competitive context
        if competitor_product_count > 20:
            competitive_context = f"Highly fragmented market with {competitor_product_count} competitor products."
        elif competitor_product_count > 10:
            competitive_context = f"Competitive market with {competitor_product_count} rival products."
        else:
            competitive_context = f"Concentrated market with {competitor_product_count} key competitors."
    
        # Cascade predictive alerts to system banner (using early calculation from above)
        # FIX: Only count alerts for products with actual revenue or risk (not $0 products)
        if 'predictive_state' in enriched_portfolio_df.columns:
            rev_col = 'weekly_sales_filled' if 'weekly_sales_filled' in enriched_portfolio_df.columns else 'revenue_proxy'
            risk_col = 'thirty_day_risk' if 'thirty_day_risk' in enriched_portfolio_df.columns else None
            
            state_mask = enriched_portfolio_df['predictive_state'].isin(['DEFEND', 'REPLENISH'])
            if rev_col in enriched_portfolio_df.columns and risk_col:
                revenue_values = enriched_portfolio_df[rev_col].fillna(0)
                risk_values = enriched_portfolio_df[risk_col].fillna(0)
                has_stake = (revenue_values > 100) | (risk_values > 10)  # Revenue >$100 OR Risk >$10
                real_alert_count = (state_mask & has_stake).sum()
            else:
                real_alert_count = state_mask.sum()
            
            # Also recalculate defend/replenish counts with stake filter
            defend_mask = (enriched_portfolio_df['predictive_state'] == 'DEFEND') & has_stake if 'has_stake' in dir() else (enriched_portfolio_df['predictive_state'] == 'DEFEND')
            replenish_mask = (enriched_portfolio_df['predictive_state'] == 'REPLENISH') & has_stake if 'has_stake' in dir() else (enriched_portfolio_df['predictive_state'] == 'REPLENISH')
            defend_count = defend_mask.sum()
            replenish_count = replenish_mask.sum()
        else:
            real_alert_count = 0
        
        has_high_urgency_alerts = real_alert_count > 0
        is_predictive_critical = portfolio_status == "CRITICAL" and real_alert_count > 0
        is_predictive_elevated = portfolio_status == "ELEVATED" and real_alert_count > 0
        
        # === ACTION A: CHECK AI BRIEF FOR THREAT KEYWORDS AND OVERRIDE STATUS BANNER ===
        # Check if ai_brief contains threat, erosion, or critical keywords
        ai_brief_lower = ai_brief.lower() if ai_brief else ""
        has_threat_keywords = any(keyword in ai_brief_lower for keyword in ["threat", "erosion", "critical"])
        
        # Also check predictive alerts for threat detection
        has_threat_keywords = has_threat_keywords or is_predictive_critical
        
        # FIX 2: Override status header to match banner (red when threat detected)
        # This must happen AFTER status_emoji, status_text, and top_action are set above
        if has_threat_keywords or is_predictive_critical:
            # Override status header to match red banner (predictive threat)
            status_emoji = "🔴"
            status_text = "DEFENSE PROTOCOL"
            action_required_count = defend_count + replenish_count
            top_action = f"{action_required_count} ACTIONS REQUIRED"
        elif is_predictive_elevated or has_high_urgency_alerts:
            status_emoji = "🟡"
            status_text = "ATTENTION"
            action_required_count = defend_count + replenish_count
            top_action = f"{action_required_count} ALERTS"
        else:
            # Smart default based on portfolio state
            if thirty_day_risk > total_rev_curr * 0.10:  # >10% at risk
                top_action = "DEFEND revenue"
            elif thirty_day_growth > total_rev_curr * 0.05:  # >5% growth opportunity
                top_action = "CAPTURE growth"
            elif your_market_share > 50:  # Market leader
                top_action = "OPTIMIZE pricing"
            elif your_market_share < 20:  # Niche player
                top_action = "EXPAND share"
            else:  # Balanced
                top_action = "MAINTAIN position"
        
        # Override top status banner based on predictive intelligence
        # Calculate action counts for status banner
        action_required_count = defend_count + replenish_count
        total_opportunity_count = exploit_count + growth_opportunity_count
        
        if is_predictive_critical or has_threat_keywords:
            system_status = f"🔴 SYSTEM STATUS: {action_required_count} CRITICAL THREATS"
            status_bg = "#dc3545"
        elif is_predictive_elevated:
            system_status = f"🟡 SYSTEM STATUS: {action_required_count} ALERTS ACTIVE"
            status_bg = "#ffc107"
        elif total_opportunity_count > 0:
            system_status = f"🟢 SYSTEM STATUS: {total_opportunity_count} GROWTH OPPORTUNITIES"
            status_bg = "#28a745"
        else:
            # Default status based on revenue gap (original logic)
            if revenue_gap < 0:
                system_status = "🔴 SYSTEM STATUS: CRITICAL THREATS DETECTED"
                status_bg = "#dc3545"
            else:
                system_status = "🟢 SYSTEM STATUS: OPTIMIZED"
                status_bg = "#28a745"
        
        # Render System Status Banner (now checked against ai_brief)
        st.markdown(f"""
        <div style="background: {status_bg}; color: white; padding: 12px 20px; border-radius: 8px;
                    margin-bottom: 20px; text-align: center; font-weight: 700; font-size: 14px;">
            {system_status}
        </div>
        """, unsafe_allow_html=True)
        
        # === SHOW ALERT DETAILS (if alerts exist) ===
        if action_required_count > 0:
            with st.expander(f"📋 View {action_required_count} Alert Details", expanded=False):
                # Get products that triggered alerts - must have BOTH DEFEND/REPLENISH state AND actual revenue/risk
                if 'predictive_state' in enriched_portfolio_df.columns:
                    # Get revenue column
                    rev_col = 'weekly_sales_filled' if 'weekly_sales_filled' in enriched_portfolio_df.columns else 'revenue_proxy'
                    risk_col = 'thirty_day_risk' if 'thirty_day_risk' in enriched_portfolio_df.columns else None
                    
                    # Filter: DEFEND/REPLENISH AND (revenue > $100 OR risk > $10)
                    # Products with $0 revenue AND $0 risk are not real alerts
                    state_mask = enriched_portfolio_df['predictive_state'].isin(['DEFEND', 'REPLENISH'])
                    
                    if rev_col in enriched_portfolio_df.columns and risk_col:
                        revenue_values = enriched_portfolio_df[rev_col].fillna(0)
                        risk_values = enriched_portfolio_df[risk_col].fillna(0)
                        has_stake = (revenue_values > 100) | (risk_values > 10)
                        alert_products = enriched_portfolio_df[state_mask & has_stake].copy()
                    else:
                        alert_products = enriched_portfolio_df[state_mask].copy()
                else:
                    alert_products = pd.DataFrame()
                
                if not alert_products.empty:
                    # Sort by risk (highest first)
                    if 'thirty_day_risk' in alert_products.columns:
                        alert_products = alert_products.sort_values('thirty_day_risk', ascending=False)
                    
                    shown_count = 0
                    for _, row in alert_products.head(15).iterrows():  # Show top 15 alerts
                        asin = row.get('asin', 'Unknown')
                        title = row.get('title', asin)[:50] + "..." if len(str(row.get('title', asin))) > 50 else row.get('title', asin)
                        risk = row.get('thirty_day_risk', 0)
                        state = row.get('predictive_state', 'UNKNOWN')
                        rev = row.get('weekly_sales_filled', row.get('revenue_proxy', 0))
                        
                        # Skip if truly zero stake (extra safety check)
                        if rev <= 0 and risk <= 0:
                            continue
                        
                        shown_count += 1
                        state_emoji = "🛡️" if state == "DEFEND" else "📦" if state == "REPLENISH" else "⚠️"
                        
                        st.markdown(f"""
                        **{state_emoji} {state}** | `{asin}` | {title}  
                        💰 Revenue: ${rev:,.0f}/mo | ⚠️ Risk: ${risk:,.0f}
                        """)
                    
                    if shown_count == 0:
                        st.success("All flagged products have minimal revenue/risk - no action needed")
                else:
                    st.success("No products require immediate attention")
    
        # Render the AI brief (full width, no regenerate button)
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; 
                    margin-bottom: 20px; border-left: 5px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 4px;">{date_label} {date_display}</div>
                    <div style="font-size: 22px; font-weight: 700; color: #1a1a1a;">
                        {status_emoji} {status_text}: {top_action}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 28px; font-weight: 700; color: #00704A;">{f_money(total_rev_curr)}</div>
                    <div style="font-size: 11px; color: #666;">Monthly Revenue (90-day avg)</div>
                </div>
            </div>
            <div style="background: #f8f9fa; padding: 14px; border-radius: 6px; margin-top: 10px;">
                <div style="font-size: 11px; color: #00704A; font-weight: 600; margin-bottom: 6px;">{brief_source}</div>
                <div style="font-size: 14px; color: #333; line-height: 1.5;">
                    {ai_brief}
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 10px;">
                    {share_context} {competitive_context}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        # --- STRATEGIC TILES ---
        c1, c2, c3, c4 = st.columns(4)
    
        # TILE 1: Portfolio Revenue (Your Brand Only)
        with c1:
            # Market share metrics
            share_pct = your_market_share / 100
            share_class = "pos" if share_pct > 0.3 else "neg" if share_pct < 0.15 else "neu"
            share_icon = "↑" if share_pct > 0.3 else "↓" if share_pct < 0.15 else "→"

            # Display custom metric showing BRAND revenue (not total market)
            st.markdown(f"""
                <div class="custom-metric-container">
                    <div class="custom-metric-label">{target_brand} Portfolio Revenue</div>
                    <div class="custom-metric-value">{f_money(portfolio_revenue)}</div>
                    <div class="benchmark-row" style="flex-wrap: wrap; gap: 6px;">
                        <span class="benchmark-badge benchmark-{share_class}">{share_icon} {your_market_share:.1f}% Market Share</span>
                        <span class="benchmark-badge benchmark-neu">📊 {portfolio_product_count} Your ASINs</span>
                        <span class="benchmark-badge benchmark-neu">🎯 vs. {competitor_product_count} Competitors</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 2: Defense Score
        with c2:
            # ACTION B: Defense Score (Predictive-Based)
            # Uses early predictive risk calculation for consistent cascade
            # Score reflects % of portfolio NOT at risk (100 - risk_pct)
            is_critical_status = is_predictive_critical or has_threat_keywords
            is_elevated_status = is_predictive_elevated or has_high_urgency_alerts
            
            # Calculate based on predictive risk percentage
            defense_score_base = 98
            if is_critical_status:
                # Critical: High risk - score penalized by risk percentage
                defense_score = max(50, defense_score_base - risk_pct)
            elif is_elevated_status:
                # Elevated: Moderate risk - smaller penalty
                defense_score = max(70, defense_score_base - (risk_pct * 0.5))
            else:
                # Healthy: Calculate from healthy products
                # Healthy = total - (defend + replenish)
                defend_replenish_count = defend_count + replenish_count
                total_products = len(res["data"])
                healthy_pct = ((total_products - defend_replenish_count) / total_products * 100) if total_products > 0 else 100
                # Cap at 98 (never reach 100 - always room for improvement)
                defense_score = min(healthy_pct, 98)

            # Determine benchmark status
            if defense_score >= 85:
                benchmark_status = "Elite"
                benchmark_class = "benchmark-elite"
                benchmark_icon = "🏆"
            elif defense_score >= 60:
                benchmark_status = "Standard"
                benchmark_class = "benchmark-standard"
                benchmark_icon = "⚡"
            else:
                benchmark_status = "At Risk"
                benchmark_class = "benchmark-atrisk"
                benchmark_icon = "⚠️"

            # Display custom metric with benchmark
            st.markdown(f"""
                <div class="custom-metric-container">
                    <div class="custom-metric-label">Defense Score</div>
                    <div class="custom-metric-value">{defense_score:.0f} <span style="font-size: 1rem; color: #999;">/100</span></div>
                    <div class="benchmark-row">
                        <span class="benchmark-badge {benchmark_class}">{benchmark_icon} {benchmark_status}</span>
                        <span class="benchmark-target">Target: 80+</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 3: 30-Day Predictive Risk (Formerly Recoverable Alpha)
        with c3:
            # PREDICTIVE ALPHA: Use early calculation for consistency
            # (Enriched DataFrame already calculated - no redundant copy)
            
            # Reuse early calculation for consistency
            predictive_risk = early_predictive_risk
            thirty_day_risk = predictive_risk["thirty_day_risk"]
            risk_pct = predictive_risk["risk_pct"]
            portfolio_status = predictive_risk["portfolio_status"]
            risk_status_emoji = predictive_risk["status_emoji"]  # Renamed to avoid conflict
            defend_count = predictive_risk["defend_count"]
            exploit_count = predictive_risk["exploit_count"]
            replenish_count = predictive_risk["replenish_count"]
            
            # Extract GROWTH metrics (offensive layer)
            thirty_day_growth = predictive_risk.get("thirty_day_growth", 0)
            growth_pct = predictive_risk.get("growth_pct", 0)
            price_lift_count = predictive_risk.get("price_lift_count", 0)
            conquest_count = predictive_risk.get("conquest_count", 0)
            expand_count = predictive_risk.get("expand_count", 0)
            opportunity_alpha = predictive_risk.get("opportunity_alpha", thirty_day_risk + thirty_day_growth)
            growth_opportunity_count = predictive_risk.get("growth_opportunity_count", 0)
            
            # Also calculate static recoverable alpha for comparison/fallback
            # Use enriched_portfolio_df which has all columns
            if "opportunity_value" not in enriched_portfolio_df.columns:
                if "weekly_sales_filled" in enriched_portfolio_df.columns:
                    recoverable_alpha = (enriched_portfolio_df["weekly_sales_filled"] * 0.15).sum()
                else:
                    recoverable_alpha = thirty_day_risk  # Fallback to predictive risk
            else:
                recoverable_alpha = enriched_portfolio_df["opportunity_value"].fillna(0).sum()

            # Determine status color based on risk level
            if portfolio_status == "CRITICAL":
                status_color = "neg"
                risk_icon = "🚨"
            elif portfolio_status == "ELEVATED":
                status_color = "neu"
                risk_icon = "⚠️"
            elif portfolio_status == "MODERATE":
                status_color = "neu"
                risk_icon = "📊"
            else:
                status_color = "pos"
                risk_icon = "✅"

            # Build action count badge
            action_count = defend_count + replenish_count
            
            # Build action summary line with both risk and growth actions
            action_parts = []
            if action_count > 0:
                action_parts.append(f"🔴 <strong>{action_count}</strong> risks")
            if growth_opportunity_count > 0:
                action_parts.append(f"🟢 <strong>{growth_opportunity_count}</strong> growth")
            
            action_line = " • ".join(action_parts) if action_parts else "No actions needed"
            
            # Display Opportunity Alpha (Risk + Growth)
            st.markdown(f"""
<div class="custom-metric-container" title="Combined 30-day opportunity: Risk to avert + Growth to capture. Based on velocity trends and competitive signals.">
    <div class="custom-metric-label">Opportunity Alpha</div>
    <div class="custom-metric-value">
        <span style="color: #dc3545;">{f_money(thirty_day_risk)}</span>
        <span style="color: #666; font-size: 0.7em;"> + </span>
        <span style="color: #28a745;">{f_money(thirty_day_growth)}</span>
    </div>
    <div class="benchmark-row" style="flex-wrap: wrap; gap: 4px;">
        <span class="benchmark-badge benchmark-{status_color}">{risk_icon} {portfolio_status}</span>
        <span class="benchmark-target" style="font-size: 0.7rem;">{f_money(opportunity_alpha)} total</span>
    </div>
    <div style="font-size: 0.65rem; color: #666; margin-top: 6px;">{action_line}</div>
</div>
            """, unsafe_allow_html=True)
            
            # === OPPORTUNITY ALPHA BREAKDOWN ===
            if thirty_day_risk > 0 or thirty_day_growth > 0:
                # Count products with actual risk/growth values (not just predictive_state)
                actual_risk_count = 0
                actual_growth_count = 0
                if 'thirty_day_risk' in enriched_portfolio_df.columns:
                    actual_risk_count = (enriched_portfolio_df['thirty_day_risk'].fillna(0) > 0).sum()
                if 'thirty_day_growth' in enriched_portfolio_df.columns:
                    actual_growth_count = (enriched_portfolio_df['thirty_day_growth'].fillna(0) > 0).sum()
                
                with st.expander(f"📊 View Opportunity Breakdown ({actual_risk_count} risks, {actual_growth_count} growth)", expanded=False):
                    col_risk, col_growth = st.columns(2)
                    
                    with col_risk:
                        st.markdown("**🔴 Risk Products**")
                        # Filter by actual risk value, not just predictive_state
                        if 'thirty_day_risk' in enriched_portfolio_df.columns:
                            risk_products = enriched_portfolio_df[
                                enriched_portfolio_df['thirty_day_risk'].fillna(0) > 0
                            ].copy()
                            
                            if not risk_products.empty:
                                risk_products = risk_products.sort_values('thirty_day_risk', ascending=False)
                                for _, row in risk_products.head(10).iterrows():
                                    asin = row.get('asin', '')[:10]
                                    title = str(row.get('title', ''))[:25] + "..."
                                    risk = row.get('thirty_day_risk', 0)
                                    state = row.get('predictive_state', 'HOLD')
                                    st.markdown(f"• `{asin}` ${risk:,.0f} ({state})")
                            else:
                                st.info("No risk products identified")
                        else:
                            st.info("Risk data not available")
                    
                    with col_growth:
                        st.markdown("**🟢 Growth Products**")
                        # Filter by actual growth value, not just predictive_state
                        if 'thirty_day_growth' in enriched_portfolio_df.columns:
                            growth_products = enriched_portfolio_df[
                                enriched_portfolio_df['thirty_day_growth'].fillna(0) > 0
                            ].copy()
                            
                            if not growth_products.empty:
                                growth_products = growth_products.sort_values('thirty_day_growth', ascending=False)
                                for _, row in growth_products.head(10).iterrows():
                                    asin = row.get('asin', '')[:10]
                                    title = str(row.get('title', ''))[:25] + "..."
                                    growth = row.get('thirty_day_growth', 0)
                                    opp_type = row.get('opportunity_type', 'GROWTH')
                                    state = row.get('predictive_state', 'HOLD')
                                    st.markdown(f"• `{asin}` ${growth:,.0f} ({opp_type if opp_type != 'GROWTH' else state})")
                            else:
                                st.info("No growth opportunities identified")
                        else:
                            st.info("Growth data not available")
    
        # TILE 4: Risk Averted (Banked from Resolved Actions)
        with c4:
            # Initialize banked alpha tracking
            if 'banked_alpha' not in st.session_state:
                st.session_state['banked_alpha'] = 0
            if 'completed_tasks' not in st.session_state:
                st.session_state['completed_tasks'] = set()

            banked_alpha = st.session_state.get('banked_alpha', 0)
            task_count = len(st.session_state.get('completed_tasks', set()))

            # Alpha Status
            if banked_alpha > 0:
                alpha_status = "Risk Averted"
                alpha_class = "human"
                alpha_icon = "🛡️"
            else:
                alpha_status = "Awaiting Action"
                alpha_class = "system"
                alpha_icon = "⏳"

            # Render the Risk Averted Card
            st.markdown(f"""
                <div class="alpha-card">
                    <div class="alpha-label">Risk Averted</div>
                    <div>
                        <span class="alpha-score" style="color: #28a745;">{f_money(banked_alpha)}</span>
                    </div>
                    <div class="alpha-validation validated">
                        {alpha_icon} {task_count} Actions Resolved
                    </div>
                    <div class="alpha-divider"></div>
                    <div class="alpha-saved">
                        <span class="alpha-status-badge {alpha_class}">{alpha_status}</span>
                        <span style="color:#666; font-size: 0.75rem;">Projected loss prevented</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # === ENHANCE AI CONTEXT WITH COMMAND CENTER METRICS ===
        # Now that all Command Center metrics are calculated, enhance the portfolio context for AI assistant
        portfolio_context += f"""
        
        COMMAND CENTER METRICS:
        - System Status: {system_status}
        - Defense Score: {defense_score:.0f}/100 ({benchmark_status}) - {"Critical threats detected, score penalized by -15" if is_critical_status else "Healthy zones calculated"}
        - 30-Day Risk (Defensive): ${thirty_day_risk:,.0f} ({risk_pct:.1f}% at risk - {portfolio_status})
        - 30-Day Growth (Offensive): ${thirty_day_growth:,.0f} ({growth_pct:.1f}% potential)
        - Opportunity Alpha (Risk + Growth): ${opportunity_alpha:,.0f}
        - Banked Alpha: ${banked_alpha:,.0f} ({task_count} tasks completed - {alpha_status})
        - Defensive Actions Needed: {defend_count + replenish_count} products
        - Growth Opportunities: {growth_opportunity_count} products ({price_lift_count} price lifts, {conquest_count} conquests)
        - Status Banner: {system_status} - {"Threat detected in Strategic Brief" if has_threat_keywords else "System status based on performance"}
        - Strategic Brief: {ai_brief[:200] if ai_brief else "Generating..."} {"[THREAT DETECTED]" if has_threat_keywords else ""}
        
        ACTION QUEUE (Hit List):
        - Products prioritized by Opportunity Alpha (Risk + Growth combined)
        - 🔴 Risk items = Defensive actions needed (DEFEND, REPLENISH states)
        - 🟢 Growth items = Offensive opportunities (EXPLOIT, PRICE_LIFT, CONQUEST)
        - Model certainty based on data quality and trend consistency
        - Use "RESOLVE" button to mark actions complete
        - Total action items available: {portfolio_product_count} products
        """
    
        # --- AI ACTION QUEUE (Outside of columns - full width) ---
        tab1, tab2 = st.tabs(["🎯 AI Action Queue", "🖼️ Visual Audit"])

        with tab1:
            # AI ACTION QUEUE - Unified view with AI-prioritized actions
            # Use enriched DataFrame (already has all intelligence columns)
            display_df = enriched_portfolio_df if 'thirty_day_risk' in enriched_portfolio_df.columns else res["data"]
    
            # === TOP PRIORITIES (Brand Portfolio Analysis) ===
            # Note: display_df now contains ONLY the target brand's products
            st.markdown(f"#### 🎯 {target_brand} Portfolio Actions")

            # For single-brand portfolio, show top products by revenue
            # Find the revenue column (may have different names)
            rev_col = 'weekly_sales_filled' if 'weekly_sales_filled' in display_df.columns else 'revenue_proxy' if 'revenue_proxy' in display_df.columns else None
            if rev_col:
                top_products = display_df.nlargest(3, rev_col)
            else:
                top_products = display_df.head(3)

            if not top_products.empty:
                priority_cols = st.columns(len(top_products))

                for i, (idx, product) in enumerate(top_products.iterrows()):
                    rev = product.get('weekly_sales_filled', product.get('revenue_proxy', 0))
                    asin = product.get('asin', '')
                    title = product.get('title', asin)[:40] + "..." if len(product.get('title', asin)) > 40 else product.get('title', asin)

                    # === UNIFIED AI ENGINE (Strategic + Predictive in one call) ===
                    # Enable triggers, network intelligence, AND competitive intelligence
                    bias = st.session_state.get('strategic_bias', '⚖️ Balanced Defense')
                    bias_clean = bias.split(' ', 1)[1] if ' ' in bias else bias
                    
                    # Get market snapshot for competitive intelligence
                    market_df = st.session_state.get('active_project_market_snapshot', display_df)
                    
                    strategy = get_product_strategy(
                        product.to_dict(), 
                        revenue=rev, 
                        use_triangulation=TRIANGULATION_ENABLED, 
                        strategic_bias=bias_clean,
                        enable_triggers=True,   # Enable trigger event detection
                        enable_network=True,    # Enable network intelligence
                        competitors_df=market_df  # Pass market data for competitive intelligence
                    )
                    
                    # Strategic outputs
                    problem_category = strategy["problem_category"]
                    ad_action = strategy["ad_action"]
                    ecom_action = strategy["ecom_action"]
                    strategic_state = strategy.get("strategic_state", "")
                    confidence_score = strategy.get("confidence_score", 0)
                    recommended_plan = strategy.get("recommended_plan", "")
                    signals_detected = strategy.get("signals_detected", [])
                    
                    # Predictive outputs (now included in unified response)
                    thirty_day_risk = strategy.get("thirty_day_risk", strategy.get("opportunity_value", 0))
                    predictive_state = strategy.get("predictive_state", "HOLD")
                    predictive_emoji = strategy.get("predictive_emoji", "✅")
                    cost_of_inaction = strategy.get("cost_of_inaction", "")
                    model_certainty = strategy.get("model_certainty", confidence_score)
                    ai_recommendation = strategy.get("ai_recommendation", "")
                    alert_type = strategy.get("alert_type", "")
                    alert_urgency = strategy.get("alert_urgency", "")
                    data_quality = strategy.get("data_quality", "MEDIUM")
                    
                    # Growth Intelligence outputs (offensive layer)
                    thirty_day_growth = strategy.get("thirty_day_growth", 0)
                    price_lift_opportunity = strategy.get("price_lift_opportunity", 0)
                    conquest_opportunity = strategy.get("conquest_opportunity", 0)
                    expansion_recommendation = strategy.get("expansion_recommendation", "")
                    growth_validated = strategy.get("growth_validated", True)
                    opportunity_type = strategy.get("opportunity_type", "")
                    
                    # Use strategic color if available, otherwise use emoji-based logic
                    if "strategic_color" in strategy and strategy["strategic_color"]:
                        color = strategy["strategic_color"]
                    elif "Losing Money" in problem_category or "🔥" in problem_category:
                        color = "#dc3545"  # Red
                    elif "Losing Share" in problem_category or "📉" in problem_category:
                        color = "#ffc107"  # Yellow
                    elif "Scale Winner" in problem_category or "🚀" in problem_category:
                        color = "#28a745"  # Green
                    elif "Healthy" in problem_category or "✅" in problem_category:
                        color = "#00704A"  # Starbucks green
                    else:
                        color = "#6c757d"  # Gray

                    # Check if task is completed
                    task_id = f"priority_{i}_{asin}"
                    is_completed = task_id in st.session_state.get('completed_tasks', set())

                    with priority_cols[i]:
                        # Card styling changes based on completion
                        card_opacity = "0.5" if is_completed else "1.0"
                        card_bg = "#f0f0f0" if is_completed else "white"
                        
                        # Build model certainty badge (based on 90-day/36-month backfill depth)
                        confidence_badge = ""
                        if model_certainty > 0:
                            cert_pct = int(model_certainty * 100)
                            cert_color = "#28a745" if model_certainty >= 0.75 else "#ffc107" if model_certainty >= 0.5 else "#dc3545"
                            confidence_badge = f'<span style="font-size: 9px; color: {cert_color}; margin-left: 8px;">({cert_pct}% {data_quality})</span>'
                        
                        # === PREDICTIVE AI RECOMMENDATION (from unified engine) ===
                        import html
                        reasoning_preview = ""
                        source_badge = ""
                        
                        if ai_recommendation and alert_type:
                            # Use predictive alert as the main reasoning (Inventory/Pricing/Rank)
                            urgency_color = "#dc3545" if alert_urgency == "HIGH" else "#ffc107" if alert_urgency == "MEDIUM" else "#28a745"
                            escaped_recommendation = html.escape(ai_recommendation[:80])
                            if len(ai_recommendation) > 80:
                                escaped_recommendation += "..."
                            reasoning_preview = f'<div style="font-size: 10px; color: #333; margin-top: 6px; padding: 6px; background: linear-gradient(135deg, #fff5f5, #ffe6e6); border-radius: 4px; border-left: 3px solid {urgency_color};">{escaped_recommendation}</div>'
                            source_badge = f"🎯 {alert_type}"
                        else:
                            # Fallback to LLM reasoning
                            reasoning = strategy.get("reasoning", "")
                            source = strategy.get("source", "unknown")
                            source_badge = "🤖 AI" if source == "llm" else "📊 Rules" if source == "fallback" else ""
                            
                            if reasoning and len(reasoning) > 10:
                                clean_reasoning = reasoning.replace("[Fallback:", "").split("]")[0]
                                escaped_reasoning = html.escape(clean_reasoning[:60])
                                if len(clean_reasoning) > 60:
                                    escaped_reasoning += "..."
                                reasoning_preview = f'<div style="font-size: 10px; color: #555; margin-top: 6px; padding: 6px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 4px; font-style: italic; border-left: 2px solid {color};">"{escaped_reasoning}"</div>'
                        
                        # Build signals preview if available
                        signals_preview = ""
                        if signals_detected and len(signals_detected) > 0:
                            top_signals = signals_detected[:3]
                            escaped_signals = [html.escape(s[:20]) for s in top_signals]
                            signals_html = " • ".join([f"<span style='font-size: 9px;'>{sig}</span>" for sig in escaped_signals])
                            signals_preview = f'<div style="font-size: 9px; color: #888; margin-top: 4px;">{signals_html}</div>'

                        # Escape variables that will be inserted into HTML
                        escaped_title = html.escape(title[:30])
                        escaped_asin = html.escape(asin)
                        escaped_ad_action = html.escape(ad_action)
                        escaped_ecom_action = html.escape(ecom_action)
                        
                        # Build predictive state badge with type indicator
                        if predictive_state in ["DEFEND", "REPLENISH"]:
                            pred_state_badge = f'<span style="font-size: 9px; background: #f8d7da; color: #721c24; padding: 2px 6px; border-radius: 3px; margin-left: 4px;">🔴 {predictive_state}</span>'
                        elif predictive_state == "EXPLOIT" or (thirty_day_growth > 0 and growth_validated):
                            pred_state_badge = f'<span style="font-size: 9px; background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 3px; margin-left: 4px;">🟢 GROWTH</span>'
                        else:
                            pred_state_badge = ""
                        
                        # Escape cost of inaction for HTML
                        escaped_cost = html.escape(cost_of_inaction[:50]) + "..." if len(cost_of_inaction) > 50 else html.escape(cost_of_inaction)
                        
                        # Build growth section if growth opportunity exists
                        growth_section = ""
                        if thirty_day_growth > 0 and growth_validated:
                            escaped_growth_rec = html.escape(expansion_recommendation[:70]) + "..." if len(expansion_recommendation) > 70 else html.escape(expansion_recommendation)
                            growth_type_label = "Price Lift" if opportunity_type == "PRICE_LIFT" else "Conquest" if opportunity_type == "CONQUEST" else "Expansion" if opportunity_type == "EXPAND" else "Growth"
                            growth_section = f'''
<div style="font-size: 10px; color: #155724; margin-top: 6px; padding: 6px; background: #d4edda; border-radius: 4px; border-left: 2px solid #28a745;">
<strong>🚀 {growth_type_label}:</strong> {f_money(thirty_day_growth)} — {escaped_growth_rec}
</div>'''
                        
                        # Determine primary metric display (show Risk if defensive, Growth if offensive)
                        total_opportunity = thirty_day_risk + thirty_day_growth
                        
                        st.markdown(f"""<div style="background: {card_bg}; border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.08); opacity: {card_opacity};">
<div style="font-size: 11px; color: {color}; font-weight: 600; text-transform: uppercase; display: flex; justify-content: space-between; align-items: center;">
<span>#{i+1} PRIORITY{confidence_badge}</span>
<span style="font-size: 9px; color: #999;">{source_badge}{pred_state_badge}</span>
</div>
<div style="font-size: 13px; color: #1a1a1a; font-weight: 600; margin: 6px 0 2px 0;">{problem_category}</div>
<div style="display: flex; align-items: baseline; gap: 10px; margin: 4px 0;">
<span style="font-size: 22px; color: #dc3545; font-weight: 700;">{f_money(thirty_day_risk)}</span>
<span style="font-size: 14px; color: #666;">+</span>
<span style="font-size: 22px; color: #28a745; font-weight: 700;">{f_money(thirty_day_growth)}</span>
</div>
<div style="font-size: 11px; color: #666; margin-top: 2px;">30-Day Risk + Growth = {f_money(total_opportunity)}</div>
{reasoning_preview}
{signals_preview}
<div style="font-size: 10px; color: #c9302c; margin-top: 6px; padding: 6px; background: #fff5f5; border-radius: 4px; border-left: 2px solid #dc3545;">
<strong>⚡ Cost of Inaction:</strong> {escaped_cost}
</div>
{growth_section}
<div style="font-size: 12px; color: #1a1a1a; margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
<div style="margin-bottom: 4px;"><strong>Ad:</strong> {escaped_ad_action}</div>
<div><strong>Ecom:</strong> {escaped_ecom_action}</div>
</div>
<div style="font-size: 10px; color: #999; margin-top: 8px; font-family: monospace;">{escaped_asin}</div>
<div style="font-size: 10px; color: #666; margin-top: 2px;">{escaped_title}...</div>
</div>""", unsafe_allow_html=True)

                        # Execute button - ACTION 4: Rename to "RESOLVE"
                        if is_completed:
                            st.success("✅ Resolved", icon="✅")
                        else:
                            if st.button(f"RESOLVE", key=f"exec_{task_id}", use_container_width=True, type="primary"):
                                # Mark task as completed
                                if 'completed_tasks' not in st.session_state:
                                    st.session_state['completed_tasks'] = set()
                                st.session_state['completed_tasks'].add(task_id)
                                
                                # Increment Banked Alpha with total opportunity (Risk + Growth)
                                total_banked = thirty_day_risk + thirty_day_growth
                                st.session_state['banked_alpha'] = st.session_state.get('banked_alpha', 0) + total_banked
                                
                                # Show appropriate message based on what was captured
                                if thirty_day_growth > 0:
                                    st.success(f"✅ Resolved - {f_money(thirty_day_risk)} risk averted + {f_money(thirty_day_growth)} growth captured")
                                else:
                                    st.success(f"✅ Resolved - {f_money(thirty_day_risk)} risk averted")
                                st.rerun()
            else:
                st.info("No products found in portfolio.")
        
            st.markdown("---")
    
            # === FULL ACTION LIST ===
            st.markdown("#### 📋 Full Action Queue")
    
            # Filter controls
            if 'problem_category' in display_df.columns:
                problem_options = ["All Products"] + sorted(display_df['problem_category'].dropna().unique().tolist())
            else:
                problem_options = ["All Products"]
    
            selected_problem = st.selectbox("Filter by problem type", options=problem_options, key="problem_filter")
    
            # Apply filter
            if selected_problem and selected_problem != "All Products" and 'problem_category' in display_df.columns:
                display_df = display_df[display_df['problem_category'] == selected_problem]
    
            # Show count
            st.caption(f"Showing **{len(display_df)}** products" + (f" in **{selected_problem}**" if selected_problem != "All Products" else "") + " — sorted by revenue")
    
            try:
                # === HIGH PERFORMANCE: Use Pre-Computed Intelligence ===
                # Intelligence was calculated ONCE in vectorized pass above
                # NO per-row recalculation needed - just format for display
                
                # Get enriched DataFrame with all intelligence columns
                enriched_df = st.session_state.get('_enriched_portfolio', display_df)
                
                # Merge enriched columns back to display_df if needed
                if 'thirty_day_risk' not in display_df.columns and 'asin' in display_df.columns:
                    intel_cols = ['asin', 'thirty_day_risk', 'thirty_day_growth', 'opportunity_alpha', 
                                  'predictive_state', 'opportunity_type', 'growth_validated', 'model_certainty']
                    available_cols = [c for c in intel_cols if c in enriched_df.columns]
                    if available_cols and 'asin' in enriched_df.columns:
                        display_df = display_df.merge(
                            enriched_df[available_cols].drop_duplicates(subset=['asin']),
                            on='asin', how='left', suffixes=('', '_intel')
                        )
                
                # Build display table using vectorized operations
                strategy_data = []
                for idx, row in display_df.iterrows():
                    # Use PRE-COMPUTED values (no function calls!)
                    asin = row.get('asin', '')
                    title = row.get('title', asin)
                    # FIX #1: Show more of the product name (50 chars instead of 30)
                    short_title = title[:50] + "..." if len(str(title)) > 50 else title
                    rev = row.get('weekly_sales_filled', 0) or 0
                    
                    # === USE PRE-COMPUTED INTELLIGENCE ===
                    risk = row.get('thirty_day_risk', 0) or 0
                    growth = row.get('thirty_day_growth', 0) or 0
                    pred_state = row.get('predictive_state', 'HOLD')
                    strategic_state = row.get('strategic_state', 'HARVEST')  # NEW: Get AI strategic state
                    opp_type = row.get('opportunity_type', '')
                    certainty = row.get('model_certainty', 0.5) or 0.5
                    ai_reasoning = row.get('reasoning', '')  # Get AI reasoning for better actions
                    
                    # FIX #3: Determine action type based on ACTUAL risk/growth values, not just pred_state
                    # A product with $6K+ risk should NOT show "Hold" even if pred_state is HOLD
                    risk_pct = (risk / rev * 100) if rev > 0 else 0
                    
                    if risk > 1000 or risk_pct > 10:  # Significant risk threshold
                        action_type = "🔴 Risk"
                        action_emoji = "🚨" if risk_pct > 15 else "⚠️"
                    elif growth > 500:  # Growth opportunity
                        action_type = "🟢 Growth"
                        action_emoji = "💰" if opp_type == "CONQUEST" else "📈"
                    elif pred_state in ["DEFEND", "REPLENISH"]:
                        action_type = "🔴 Risk"
                        action_emoji = "⚠️"
                    elif pred_state == "EXPLOIT":
                        action_type = "🟢 Growth"
                        action_emoji = "📈"
                    else:
                        action_type = "⚪ Hold"
                        action_emoji = "✅"
                    
                    # FIX #3: Build action text based on actual risk/growth values
                    if opp_type == "CONQUEST":
                        action_display = f"{action_emoji} Competitor vulnerable - capture ${growth:.0f} in 30 days"
                    elif opp_type == "PRICE_LIFT":
                        action_display = f"{action_emoji} Price headroom - raise price to capture ${growth:.0f}"
                    elif opp_type == "PRICE_POWER":
                        action_display = f"{action_emoji} Test 4-5% price increase - rank #{int(row.get('sales_rank_filled', 0))} supports ${growth:.0f} upside"
                    elif opp_type == "REVIEW_MOAT":
                        action_display = f"{action_emoji} Premium pricing opportunity - strong reviews support ${growth:.0f} upside"
                    elif risk > 1000 and strategic_state == "HARVEST":
                        # Harvest products with risk = optimization opportunity, not actual threat
                        # Check if we have risk component data to explain source
                        price_risk = row.get('price_erosion_risk', 0) or 0
                        share_risk = row.get('share_erosion_risk', 0) or 0
                        stockout_risk = row.get('stockout_risk', 0) or 0
                        competitor_count = row.get('competitor_count', 0) or 0
                        
                        # Determine risk source
                        if stockout_risk > risk * 0.5:
                            action_display = f"{action_emoji} Inventory optimization - ${risk:.0f}/mo opportunity from stockout prevention"
                        elif price_risk > risk * 0.5:
                            action_display = f"{action_emoji} Pricing optimization - ${risk:.0f}/mo opportunity (rank #{int(row.get('sales_rank_filled', 0))} supports price test)"
                        elif share_risk > risk * 0.5:
                            action_display = f"{action_emoji} Market share protection - ${risk:.0f}/mo at risk from velocity decline"
                        elif competitor_count > 5:
                            action_display = f"{action_emoji} Competitive pressure - ${risk:.0f}/mo optimization opportunity ({competitor_count} competitors)"
                        else:
                            action_display = f"{action_emoji} Optimization opportunity - ${risk:.0f}/mo potential (stable position, test pricing/spend)"
                    elif risk > 1000 and strategic_state == "TRENCH_WAR":
                        # Trench War + risk = need to defend
                        competitor_count = row.get('competitor_count', 0) or 0
                        action_display = f"{action_emoji} Defend share - ${risk:.0f}/mo at risk from {competitor_count} competitors, match pricing"
                    elif risk > 1000 and strategic_state == "DISTRESS":
                        # Check risk source for distress
                        price_risk = row.get('price_erosion_risk', 0) or 0
                        share_risk = row.get('share_erosion_risk', 0) or 0
                        if share_risk > risk * 0.5:
                            action_display = f"{action_emoji} Urgent: ${risk:.0f}/mo at risk - velocity declining, investigate root cause"
                        elif price_risk > risk * 0.5:
                            action_display = f"{action_emoji} Urgent: ${risk:.0f}/mo at risk - pricing pressure, review competitor pricing"
                        else:
                            action_display = f"{action_emoji} Urgent: ${risk:.0f}/mo at risk - investigate root cause"
                    elif risk > 1000:
                        # High risk action - try to explain source
                        price_risk = row.get('price_erosion_risk', 0) or 0
                        share_risk = row.get('share_erosion_risk', 0) or 0
                        if price_risk > risk * 0.5:
                            action_display = f"{action_emoji} Pricing pressure - ${risk:.0f}/mo at risk, review competitor pricing"
                        elif share_risk > risk * 0.5:
                            action_display = f"{action_emoji} Market share decline - ${risk:.0f}/mo at risk, review velocity trends"
                        else:
                            action_display = f"{action_emoji} Action needed - ${risk:.0f}/mo at risk, review pricing/inventory"
                    elif risk > 100:
                        # Medium risk - still needs attention
                        action_display = f"⚠️ Monitor closely - ${risk:.0f}/mo at risk"
                    elif risk > 0:
                        # Low risk - flag for awareness
                        action_display = f"📊 Low risk flagged - ${risk:.0f}/mo exposure"
                    elif pred_state == "DEFEND":
                        action_display = f"{action_emoji} Defend position - velocity declining"
                    elif pred_state == "REPLENISH":
                        action_display = f"{action_emoji} Inventory alert - restock needed"
                    elif pred_state == "EXPLOIT":
                        action_display = f"{action_emoji} Exploit momentum - accelerate spend while rank improving"
                    elif growth > 500:
                        action_display = f"{action_emoji} Growth opportunity - ${growth:.0f} potential upside"
                    elif growth > 0:
                        action_display = f"🎯 Minor growth - ${growth:.0f} potential"
                    elif rev > 10000:  # High revenue, no risk, no growth
                        action_display = f"✅ Strong performer - maintain strategy"
                    else:
                        action_display = f"✅ Stable - no action needed"
                    
                    # FIX #2: Show strategic_state (from AI) which is more meaningful than pred_state
                    # Format: "TRENCH_WAR" → "Trench War"
                    state_display = strategic_state.replace("_", " ").title() if strategic_state else pred_state
                    
                    strategy_data.append({
                        "ASIN": asin,
                        "Product": short_title,
                        "State": state_display,  # FIX #2: Removed redundant Type column
                        "Action": action_display,
                        "Revenue": rev,
                        "Risk": risk,
                        "Growth": growth,
                        "Certainty": certainty * 100,
                    })
                
                final_df = pd.DataFrame(strategy_data)
                
                # Calculate combined opportunity (Risk + Growth) for sorting
                final_df["Opportunity"] = final_df["Risk"] + final_df["Growth"]
                
                # Sort by combined Opportunity highest to lowest
                final_df = final_df.drop_duplicates(subset=["ASIN"]).sort_values("Opportunity", ascending=False)

                # Configure column display - cleaner layout with actionable info
                column_config = {
                    "ASIN": st.column_config.TextColumn("ASIN", width="small"),
                    "Product": st.column_config.TextColumn("Product", width="large"),  # FIX #1: Wider product column
                    "State": st.column_config.TextColumn("AI State", width="small", help="AI Strategic Classification: Fortress, Harvest, Trench War, Distress, Terminal"),
                    "Action": st.column_config.TextColumn("Recommended Action", width="large"),  # FIX #2: Clearer header
                    "Revenue": st.column_config.NumberColumn("Mo. Rev", format="$%.0f", width="small", help="Monthly revenue (90-day avg)"),
                    "Risk": st.column_config.NumberColumn("Risk", format="$%.0f", width="small", help="Predicted 30-day revenue at risk if no action"),
                    "Growth": st.column_config.NumberColumn("Growth", format="$%.0f", width="small", help="Predicted 30-day revenue from growth opportunities"),
                    "Certainty": st.column_config.ProgressColumn(
                        "Certainty",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                        width="small",
                        help="Model certainty based on data quality and trend consistency"
                    ),
                    "Opportunity": None,  # Hide from display (used for sorting only)
                }

                # Display columns (removed redundant Type column)
                display_columns = ["ASIN", "Product", "State", "Action", "Revenue", "Risk", "Growth", "Certainty"]
                
                st.dataframe(
                    final_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
        
            except Exception as table_err:
                st.error("Unable to render Execution Queue.")
                st.exception(table_err)

        with tab2:
            st.markdown("### 🖼️ Visual Portfolio Audit")
            st.caption("Top 16 products by revenue")
    
            # Use enriched data for visual audit (no redundant copy)
            full_df = enriched_portfolio_df
            # Find the revenue column (may have different names)
            rev_col = 'weekly_sales_filled' if 'weekly_sales_filled' in full_df.columns else 'revenue_proxy' if 'revenue_proxy' in full_df.columns else None
            main_image_col = 'main_image'
            if rev_col is not None and main_image_col in full_df.columns:
                # Filter for products with images, then get top 16 by revenue
                filtered_df = full_df.loc[full_df[main_image_col] != ""]
                gallery_df = filtered_df.nlargest(16, str(rev_col))
            elif main_image_col in full_df.columns:
                # Filter for products with images, then get first 16
                filtered_df = full_df.loc[full_df[main_image_col] != ""]
                gallery_df = filtered_df.head(16)
            else:
                gallery_df = pd.DataFrame()
    
            if not gallery_df.empty:
                cols = st.columns(4)
                for i, (_, row) in enumerate(gallery_df.iterrows()):
                    with cols[i % 4]:
                        clean_title = (row['title'][:45] + '...') if len(row['title']) > 45 else row['title']
                        problem = row.get('problem_category', row.get('predictive_zone', 'N/A'))
                
                        # Color based on problem
                        if "Losing Money" in str(problem):
                            badge_color = "#dc3545"
                        elif "Losing Share" in str(problem) or "Price Problem" in str(problem):
                            badge_color = "#f57c00"
                        elif "Scale" in str(problem) or "Healthy" in str(problem):
                            badge_color = "#28a745"
                        else:
                            badge_color = "#666"
                
                        # Get 36M metrics
                        velocity = row.get('velocity_decay', 1.0)
                        velocity_color = "#28a745" if velocity < 0.9 else "#dc3545" if velocity > 1.2 else "#666"
                        velocity_label = "🚀 Accelerating" if velocity < 0.9 else "📉 Decaying" if velocity > 1.2 else "→ Stable"
                        forecast = row.get('forecast_signal', '→ STABLE')
                
                        st.markdown(f"""
                            <div class="product-card">
                                <img src="{row['main_image']}" class="product-img">
                                <div style="margin-top: 10px; height: 40px; overflow: hidden;">
                                    <span style="font-size: 0.8rem; color: #333; font-weight: 500;">{clean_title}</span>
                                </div>
                                <div style="font-size: 1.1rem; color: #00704A; font-weight: 700; margin-top: 6px;">
                                    {f_money(row.get('weekly_sales_filled', row.get('revenue_proxy', 0)))}
                                </div>
                                <div style="font-size: 0.7rem; color: {badge_color}; font-weight: 600; margin-top: 4px;">{problem}</div>
                                <div style="font-size: 0.65rem; color: {velocity_color}; margin-top: 4px;">
                                    <strong>36M:</strong> {velocity:.2f}x ({velocity_label})
                                </div>
                                <div style="font-size: 0.65rem; color: #666; margin-top: 2px;">{forecast}</div>
                                <div style="font-size: 0.6rem; color: #999; margin-top: 2px;">{row['asin']}</div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No product images available.")

    except Exception as e:
        st.error(f"🛡️ Command Center Offline: {e}")