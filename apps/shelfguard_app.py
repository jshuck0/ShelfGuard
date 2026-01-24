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
        get_market_snapshot_with_network_intelligence,
        get_supabase_client
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

# Intelligence Pipeline - Unified Orchestrator (ASIN Deep Dive)
try:
    from src.intelligence_pipeline import IntelligencePipeline
except ImportError:
    pass

# Revenue Attribution Engine - Causal Intelligence
try:
    from src.revenue_attribution import calculate_revenue_attribution, save_revenue_attribution
    from src.models.revenue_attribution import RevenueAttribution
    ATTRIBUTION_ENABLED = True
except ImportError:
    ATTRIBUTION_ENABLED = False
    calculate_revenue_attribution = None
    RevenueAttribution = None

# Unified Dashboard Logic (Unified 3.0)
try:
    from src.dashboard_logic import ensure_data_loaded
except ImportError:
    ensure_data_loaded = None

# Predictive Forecasting Engine - Phase 2.5
try:
    from src.predictive_forecasting import (
        generate_combined_intelligence,
        calculate_annual_projection,
        forecast_event_impacts,
        build_scenarios
    )
    from src.models.forecast_models import (
        AnticipatedEvent,
        RevenueForecast,
        Scenario,
        CombinedIntelligence,
        EventSeverity
    )
    FORECASTING_ENABLED = True
except ImportError:
    FORECASTING_ENABLED = False
    generate_combined_intelligence = None
    calculate_annual_projection = None
    forecast_event_impacts = None
    build_scenarios = None

# Unified Intelligence Pipeline - Consolidates ALL intelligence systems
try:
    from src.intelligence_pipeline import IntelligencePipeline, get_active_insights_from_db
    from src.network_intelligence import NetworkIntelligence
    INTELLIGENCE_PIPELINE_ENABLED = True
except ImportError:
    INTELLIGENCE_PIPELINE_ENABLED = False
    IntelligencePipeline = None
    NetworkIntelligence = None
    get_active_insights_from_db = None

# Strategic Triangulator - LLM-powered classification (already imported via TRIANGULATION_ENABLED)
# StrategicTriangulator provides: 5 strategic states, 30-day risk forecasts, cost of inaction


# Initialize OpenAI client (for chat feature - optional)
openai_client = None
if OPENAI_SYNC_AVAILABLE:
    try:
        openai_client = OpenAI(api_key=st.secrets.get("openai", {}).get("OPENAI_API_KEY"))
    except Exception:
        openai_client = None

# Pre-compile regex for performance
_METRICS_PATTERN = re.compile(r'\$[\d,]+|\d+\.\d+%|\d+ products')


# ========================================
# VISUALIZATION HELPERS
# ========================================

def generate_mini_sparkline(values: list, width: int = 60, height: int = 20, color: str = "#007bff") -> str:
    """
    Generate an inline SVG sparkline from a list of values.
    
    Returns base64-encoded SVG for embedding in HTML.
    
    Args:
        values: List of numeric values (e.g., prices or ranks over time)
        width: SVG width in pixels
        height: SVG height in pixels
        color: Line color (hex or CSS color name)
    
    Returns:
        HTML img tag with embedded SVG sparkline
    """
    if not values or len(values) < 2:
        return ""
    
    # Filter out None/NaN values
    clean_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if len(clean_values) < 2:
        return ""
    
    # Normalize values to fit in the SVG
    min_val = min(clean_values)
    max_val = max(clean_values)
    value_range = max_val - min_val if max_val != min_val else 1
    
    # Generate path points
    points = []
    for i, val in enumerate(clean_values):
        x = (i / (len(clean_values) - 1)) * width
        y = height - ((val - min_val) / value_range) * (height - 4) - 2  # 2px padding
        points.append(f"{x:.1f},{y:.1f}")
    
    path_d = "M" + " L".join(points)
    
    # Determine trend color (green if trending up, red if down)
    if clean_values[-1] < clean_values[0]:
        trend_color = "#28a745"  # Green - good for price (lower), or rank (lower is better)
    elif clean_values[-1] > clean_values[0]:
        trend_color = "#dc3545"  # Red - price up or rank worse
    else:
        trend_color = color
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <path d="{path_d}" fill="none" stroke="{trend_color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>'''
    
    import base64
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{encoded}" style="vertical-align: middle; margin-left: 4px;"/>'


def get_velocity_badge(velocity_30d: float, velocity_90d: float = None) -> str:
    """
    Generate HTML badge showing velocity trend with directional indicator.
    
    Args:
        velocity_30d: 30-day velocity change (positive = accelerating, negative = declining)
        velocity_90d: Optional 90-day velocity for context
    
    Returns:
        HTML span with velocity badge
    """
    if velocity_30d is None or pd.isna(velocity_30d):
        return ""
    
    # Determine trend direction and color
    if velocity_30d >= 0.15:  # 15%+ acceleration
        arrow = "↑↑"
        label = "HOT"
        bg_color = "#28a745"
        text_color = "#fff"
    elif velocity_30d >= 0.05:  # 5%+ growth
        arrow = "↑"
        label = "RISING"
        bg_color = "#d4edda"
        text_color = "#155724"
    elif velocity_30d <= -0.15:  # 15%+ decline
        arrow = "↓↓"
        label = "COLD"
        bg_color = "#dc3545"
        text_color = "#fff"
    elif velocity_30d <= -0.05:  # 5%+ decline
        arrow = "↓"
        label = "FALLING"
        bg_color = "#f8d7da"
        text_color = "#721c24"
    else:  # Stable (-5% to +5%)
        arrow = "→"
        label = "STABLE"
        bg_color = "#e9ecef"
        text_color = "#495057"
    
    pct = f"{velocity_30d*100:+.0f}%"
    
    return f'''<span style="
        font-size: 9px; 
        background: {bg_color}; 
        color: {text_color}; 
        padding: 2px 6px; 
        border-radius: 3px; 
        margin-left: 6px;
        font-weight: 600;
    ">{arrow} {label} ({pct})</span>'''


def format_trigger_timeline(triggers: list, max_events: int = 5) -> str:
    """
    Format trigger events as a compact timeline HTML.
    
    Args:
        triggers: List of TriggerEvent objects
        max_events: Maximum events to display
    
    Returns:
        HTML string with trigger timeline
    """
    if not triggers:
        return ""
    
    events_html = []
    for t in triggers[:max_events]:
        # Severity-based styling
        if t.severity >= 8:
            severity_color = "#dc3545"  # Critical
            severity_icon = "🔴"
        elif t.severity >= 6:
            severity_color = "#ffc107"  # Warning
            severity_icon = "🟡"
        else:
            severity_color = "#28a745"  # Opportunity
            severity_icon = "🟢"
        
        # Event nature styling
        nature_badge = ""
        if hasattr(t, 'nature'):
            if t.nature == "THREAT":
                nature_badge = '<span style="font-size: 8px; color: #dc3545;">THREAT</span>'
            elif t.nature == "OPPORTUNITY":
                nature_badge = '<span style="font-size: 8px; color: #28a745;">OPP</span>'
        
        delta_str = f"{t.delta_pct:+.1f}%" if hasattr(t, 'delta_pct') and t.delta_pct else ""
        
        events_html.append(f'''
            <div style="font-size: 10px; padding: 4px 0; border-bottom: 1px solid #eee;">
                <span style="color: {severity_color};">{severity_icon}</span>
                <strong>{t.event_type}</strong>: {t.metric_name} 
                <span style="color: #666;">{delta_str}</span>
                {nature_badge}
            </div>
        ''')
    
    if len(triggers) > max_events:
        events_html.append(f'<div style="font-size: 9px; color: #999; padding-top: 4px;">+{len(triggers) - max_events} more events</div>')
    
    return f'''
        <div style="background: #f8f9fa; border-radius: 6px; padding: 8px; margin-top: 8px;">
            <div style="font-size: 10px; font-weight: 600; color: #333; margin-bottom: 4px;">📅 Recent Market Events</div>
            {''.join(events_html)}
        </div>
    '''


# ========================================
# PHASE 2/2.5: CAUSAL INTELLIGENCE UI
# ========================================

def render_attribution_breakdown(attribution) -> str:
    """
    Render revenue attribution breakdown as visual HTML.
    
    Shows what's driving revenue changes:
    - Internal factors (price, inventory, content)
    - Competitive factors (market share, competitor actions)
    - Macro factors (category trends, seasonality)
    - Platform factors (algorithm, Amazon actions)
    
    Args:
        attribution: RevenueAttribution object from calculate_revenue_attribution()
    
    Returns:
        HTML string for rendering in Streamlit
    """
    if attribution is None:
        return ""
    
    # Calculate total change and percentages
    total_change = (attribution.internal_contribution + 
                    attribution.competitive_contribution + 
                    attribution.macro_contribution + 
                    attribution.platform_contribution)
    
    if abs(total_change) < 0.01:
        return ""
    
    def get_pct(value):
        if abs(total_change) < 0.01:
            return 0
        return (value / abs(total_change)) * 100
    
    def get_bar_color(value):
        if value > 0:
            return "#28a745"  # green
        elif value < 0:
            return "#dc3545"  # red
        return "#6c757d"  # gray
    
    def render_attribution_bar(label, value, emoji):
        pct = abs(get_pct(value))
        color = get_bar_color(value)
        sign = "+" if value > 0 else ""
        return f'''
            <div style="margin: 4px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 11px;">
                    <span>{emoji} {label}</span>
                    <span style="color: {color}; font-weight: 600;">{sign}${value:,.0f}</span>
                </div>
                <div style="background: #e9ecef; border-radius: 3px; height: 6px; margin-top: 2px;">
                    <div style="background: {color}; width: {min(pct, 100):.0f}%; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>
        '''
    
    bars_html = [
        render_attribution_bar("Internal", attribution.internal_contribution, "🏠"),
        render_attribution_bar("Competitive", attribution.competitive_contribution, "⚔️"),
        render_attribution_bar("Macro", attribution.macro_contribution, "📊"),
        render_attribution_bar("Platform", attribution.platform_contribution, "🔧"),
    ]
    
    # Determine overall direction
    direction = "📈" if total_change > 0 else "📉"
    direction_text = "Growth" if total_change > 0 else "Decline"
    direction_color = "#28a745" if total_change > 0 else "#dc3545"
    
    return f'''
        <div style="background: #f8f9fa; border-radius: 8px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 8px;">
                {direction} Revenue Attribution
                <span style="color: {direction_color}; float: right;">
                    {direction_text}: ${abs(total_change):,.0f}
                </span>
            </div>
            {''.join(bars_html)}
            <div style="font-size: 9px; color: #666; margin-top: 6px; text-align: right;">
                Confidence: {attribution.confidence:.0%}
            </div>
        </div>
    '''


def render_scenario_cards(scenarios: list, sustainable_run_rate: float = 0) -> str:
    """
    Render predictive scenarios as compact cards.
    
    Shows Base/Optimistic/Pessimistic 30-day revenue forecasts.
    
    Args:
        scenarios: List of Scenario objects from build_scenarios()
        sustainable_run_rate: Monthly revenue after temporary factors removed
    
    Returns:
        HTML string for rendering in Streamlit
    """
    if not scenarios:
        return ""
    
    cards_html = []
    for scenario in scenarios[:3]:  # Max 3 scenarios
        # Determine card styling
        if scenario.name == "Optimistic":
            bg_color = "#d4edda"
            border_color = "#28a745"
            emoji = "🚀"
        elif scenario.name == "Pessimistic":
            bg_color = "#f8d7da"
            border_color = "#dc3545"
            emoji = "⚠️"
        else:  # Base
            bg_color = "#fff3cd"
            border_color = "#ffc107"
            emoji = "📊"
        
        # Calculate change from sustainable run rate
        delta = scenario.projected_revenue - sustainable_run_rate if sustainable_run_rate else 0
        delta_sign = "+" if delta > 0 else ""
        delta_color = "#28a745" if delta > 0 else "#dc3545" if delta < 0 else "#666"
        
        cards_html.append(f'''
            <div style="flex: 1; background: {bg_color}; border: 1px solid {border_color}; 
                        border-radius: 6px; padding: 8px; min-width: 100px;">
                <div style="font-size: 10px; font-weight: 600; color: #333;">
                    {emoji} {scenario.name}
                </div>
                <div style="font-size: 14px; font-weight: 700; color: #000; margin: 4px 0;">
                    ${scenario.projected_revenue:,.0f}
                </div>
                <div style="font-size: 9px; color: {delta_color};">
                    {delta_sign}${delta:,.0f} vs baseline
                </div>
                <div style="font-size: 8px; color: #666; margin-top: 4px;">
                    {scenario.probability:.0%} probability
                </div>
            </div>
        ''')
    
    return f'''
        <div style="margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 600; color: #333; margin-bottom: 8px;">
                🔮 30-Day Revenue Scenarios
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                {''.join(cards_html)}
            </div>
            {f'<div style="font-size: 9px; color: #666; margin-top: 6px;">Sustainable Run Rate: ${sustainable_run_rate:,.0f}/mo</div>' if sustainable_run_rate else ''}
        </div>
    '''


def render_anticipated_events(anticipated_events: list, max_events: int = 3) -> str:
    """
    Render anticipated future events timeline.
    
    Shows predicted market events with timing and impact.
    
    Args:
        anticipated_events: List of AnticipatedEvent objects
        max_events: Maximum events to display
    
    Returns:
        HTML string for rendering in Streamlit
    """
    if not anticipated_events:
        return ""
    
    events_html = []
    for event in anticipated_events[:max_events]:
        # Severity styling
        if event.severity.value == "critical":
            severity_color = "#dc3545"
            severity_icon = "🔴"
        elif event.severity.value == "high":
            severity_color = "#fd7e14"
            severity_icon = "🟠"
        elif event.severity.value == "medium":
            severity_color = "#ffc107"
            severity_icon = "🟡"
        else:
            severity_color = "#28a745"
            severity_icon = "🟢"
        
        # Impact styling
        impact_color = "#dc3545" if event.projected_impact < 0 else "#28a745"
        impact_sign = "+" if event.projected_impact > 0 else ""
        
        events_html.append(f'''
            <div style="font-size: 10px; padding: 6px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between;">
                <div>
                    <span style="color: {severity_color};">{severity_icon}</span>
                    <strong>{event.event_type}</strong>
                    <span style="color: #666; font-size: 9px;">({event.days_until}d)</span>
                </div>
                <div style="color: {impact_color}; font-weight: 600;">
                    {impact_sign}${event.projected_impact:,.0f}
                </div>
            </div>
        ''')
    
    if len(anticipated_events) > max_events:
        events_html.append(f'<div style="font-size: 9px; color: #999; padding: 4px;">+{len(anticipated_events) - max_events} more anticipated</div>')
    
    return f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 12px; font-weight: 600; color: white; margin-bottom: 8px;">
                ⏳ Anticipated Events (Next 30 Days)
            </div>
            <div style="background: white; border-radius: 6px; overflow: hidden;">
                {''.join(events_html)}
            </div>
        </div>
    '''


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
                # SEMANTIC SPLIT: thirty_day_risk = actual threats only
                "thirty_day_risk": brief.thirty_day_risk,
                "optimization_value": brief.optimization_value,  # NEW: separate from risk
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
                
                # === OPPORTUNITY VALUE (for display) ===
                # Use actual risk if present, otherwise optimization value
                "opportunity_value": brief.thirty_day_risk if brief.thirty_day_risk > 0 else brief.optimization_value,
                
                # === PREDICTIVE STATE (replaces capital_zone) ===
                "predictive_zone": f"{brief.predictive_emoji} {brief.predictive_state}",
                "is_healthy": brief.predictive_state in ["HOLD", "EXPLOIT", "STABLE", "GROW"]
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
            risk = brief.thirty_day_risk if hasattr(brief, 'thirty_day_risk') else 0
            opt_val = brief.optimization_value if hasattr(brief, 'optimization_value') else 0
            return {
                "ad_action": "⚖️ OPTIMIZE ROAS",
                "ecom_action": "✅ MAINTAIN",
                "problem_category": f"{brief.state_emoji} {state}",
                "problem_reason": brief.reasoning[:80] + "..." if len(brief.reasoning) > 80 else brief.reasoning,
                "opportunity_value": risk if risk > 0 else opt_val,
                "confidence_score": brief.confidence,
                "strategic_state": state,
                "recommended_plan": brief.recommended_action,
                "predictive_state": brief.predictive_state if hasattr(brief, 'predictive_state') else "HOLD",
                "predictive_emoji": brief.predictive_emoji if hasattr(brief, 'predictive_emoji') else "✅",
                "predictive_zone": f"{brief.predictive_emoji if hasattr(brief, 'predictive_emoji') else '✅'} {brief.predictive_state if hasattr(brief, 'predictive_state') else 'HOLD'}",
                "is_healthy": brief.predictive_state in ["HOLD", "EXPLOIT", "STABLE", "GROW"] if hasattr(brief, 'predictive_state') else True,
                # SEMANTIC SPLIT: Risk vs Optimization
                "thirty_day_risk": risk,
                "optimization_value": opt_val,
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
        "opportunity_value": 0,
        "confidence_score": 0.3,
        "strategic_state": "HARVEST",  # Neutral state
        "recommended_plan": "Awaiting analysis",
        "predictive_state": "STABLE",
        "predictive_emoji": "✅",
        "predictive_zone": "✅ STABLE",
        "is_healthy": True,
        # SEMANTIC SPLIT: Risk vs Optimization
        "thirty_day_risk": 0,
        "optimization_value": 0,
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
def generate_ai_brief(portfolio_summary: str, data_hash: str, strategic_bias: str = "Balanced Defense") -> str:
    """
    Generate an LLM-powered strategic brief for the portfolio.
    
    NOW USES THE SAME AI ENGINE as product-level classification
    for consistency across all AI outputs.

    Cached by data_hash (portfolio metrics + strategic_bias), not by date, to avoid
    unnecessary API calls when only the date range changes.

    Args:
        portfolio_summary: Portfolio metrics summary string
        data_hash: Cache key (includes strategic_bias)
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)

    Performance: Cached results reduce API calls and latency.
    """
    # Use the unified AI engine from utils/ai_engine.py
    # This ensures the same client, model, and configuration as product classification
    if generate_portfolio_brief_sync is None:
        return None
    
    # Pass strategic_bias to the brief generator
    # Note: generate_portfolio_brief_sync will incorporate strategic_bias into the prompt
    return generate_portfolio_brief_sync(portfolio_summary, client=None, model=None, strategic_bias=strategic_bias)


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

# Store cleaned version in session state for use throughout the app (including discovery path)
# Note: We use '_value' suffix because 'strategic_bias' is a widget key managed by Streamlit
st.session_state['strategic_bias_value'] = strategic_bias
st.session_state['strategic_bias_clean_value'] = strategic_bias_clean

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
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["🛡️ Command Center", "🧩 Command Center 2.0", "🔍 Market Discovery", "📂 My Projects"])

with main_tab2:
    # === COMMAND CENTER 2.0: CAUSAL INTELLIGENCE PLATFORM ===
    # Transforms revenue analysis from "What happened?" to "Why it happened?"
    # with quantified attribution across 4 causal categories

    st.markdown("## 🧩 Command Center 2.0: Causal Intelligence")
    st.caption("**Revenue Attribution:** Understand where your revenue came from with 4-category causal breakdown")

    # Check if we have an active project
    active_project_asin = st.session_state.get('active_project_asin', None)

    # === UNIFIED COMMAND CENTER 3.0 (Analysis & Action) ===
    from src.unified_dashboard import render_unified_dashboard
    render_unified_dashboard()

    # LEGACY CODE DISABLED (Preserved for reference but unreachable)
    if False:
        pass
    elif False: # Replaces 'else:' to skip the legacy block below
        # === ACTIVE PROJECT MODE ===
        try:
            # Get project data from session state - USE SAME KEYS AS COMMAND CENTER 1
            project_name = st.session_state.get('active_project_name', 'Unknown Project')
            target_brand = st.session_state.get('active_brand', '')

            # Get data using the same keys as Command Center 1 (lines 3394-3396)
            df_weekly = st.session_state.get('df_weekly', pd.DataFrame())
            market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())

            # === DETECT TRIGGER EVENTS (Same approach as CC1, lines 3398-3430) ===
            trigger_events = []
            project_asins = []

            if not df_weekly.empty and 'asin' in df_weekly.columns:
                project_asins = list(df_weekly['asin'].unique())
                try:
                    from src.trigger_detection import detect_trigger_events

                    # Get top revenue products for trigger detection
                    if target_brand and 'brand' in df_weekly.columns:
                        brand_asins = df_weekly[df_weekly['brand'].str.lower() == target_brand.lower()]['asin'].unique()[:15]
                    else:
                        brand_asins = df_weekly['asin'].unique()[:15]

                    for asin in brand_asins:
                        asin_history = df_weekly[df_weekly['asin'] == asin]
                        if not asin_history.empty and len(asin_history) >= 3:
                            try:
                                triggers = detect_trigger_events(
                                    asin=asin,
                                    df_historical=asin_history,
                                    df_competitors=market_snapshot,
                                    lookback_days=30
                                )
                                for t in triggers:
                                    t.asin = asin
                                trigger_events.extend(triggers)
                            except Exception:
                                pass

                    # Sort by severity, take top events
                    trigger_events = sorted(trigger_events, key=lambda e: e.severity, reverse=True)[:20]
                except ImportError:
                    pass
                except Exception:
                    pass

            # === REVENUE CALCULATION ===
            # Calculate current and previous revenue from portfolio
            if not df_weekly.empty:
                # Ensure we have date column
                if 'date' not in df_weekly.columns and 'week' in df_weekly.columns:
                    df_weekly['date'] = pd.to_datetime(df_weekly['week'])

                # Sort by date
                df_weekly = df_weekly.sort_values('date')

                # Calculate revenue proxy (sales rank proxy or actual revenue if available)
                revenue_col = None
                for col in ['revenue', 'sales', 'revenue_proxy', 'revenue_proxy_adjusted']:
                    if col in df_weekly.columns:
                        revenue_col = col
                        break

                if revenue_col:
                    # Get last 30 days and previous 30 days
                    from datetime import datetime, timedelta
                    cutoff_recent = datetime.now() - timedelta(days=30)
                    cutoff_previous = datetime.now() - timedelta(days=60)

                    recent_df = df_weekly[df_weekly['date'] >= cutoff_recent]
                    previous_df = df_weekly[(df_weekly['date'] >= cutoff_previous) & (df_weekly['date'] < cutoff_recent)]

                    current_revenue = recent_df[revenue_col].sum() if not recent_df.empty else 0
                    previous_revenue = previous_df[revenue_col].sum() if not previous_df.empty else 0
                else:
                    # Fallback: use market snapshot revenue if available
                    if not market_snapshot.empty:
                        # Try multiple revenue columns
                        rev_col = None
                        for col in ['revenue_proxy_adjusted', 'revenue_proxy', 'monthly_revenue']:
                            if col in market_snapshot.columns:
                                rev_col = col
                                break
                        if rev_col:
                            # Filter to target brand if available
                            if target_brand and 'brand' in market_snapshot.columns:
                                brand_mask = market_snapshot['brand'].str.lower().str.contains(target_brand.lower(), case=False, na=False)
                                brand_df = market_snapshot[brand_mask]
                                current_revenue = brand_df[rev_col].sum() if not brand_df.empty else market_snapshot[rev_col].sum()
                            else:
                                current_revenue = market_snapshot[rev_col].sum()
                            previous_revenue = current_revenue * 0.85  # Estimate 15% growth
                        else:
                            current_revenue = 100000  # Placeholder
                            previous_revenue = 85000
                    else:
                        current_revenue = 100000  # Placeholder
                        previous_revenue = 85000
            else:
                # Use market snapshot if df_weekly is empty
                if not market_snapshot.empty:
                    rev_col = None
                    for col in ['revenue_proxy_adjusted', 'revenue_proxy', 'monthly_revenue']:
                        if col in market_snapshot.columns:
                            rev_col = col
                            break
                    if rev_col:
                        if target_brand and 'brand' in market_snapshot.columns:
                            brand_mask = market_snapshot['brand'].str.lower().str.contains(target_brand.lower(), case=False, na=False)
                            brand_df = market_snapshot[brand_mask]
                            current_revenue = brand_df[rev_col].sum() if not brand_df.empty else 0
                        else:
                            current_revenue = market_snapshot[rev_col].sum()
                        previous_revenue = current_revenue * 0.85
                    else:
                        current_revenue = 100000
                        previous_revenue = 85000
                else:
                    current_revenue = 100000  # Placeholder
                    previous_revenue = 85000

            # === ATTRIBUTION CALCULATION ===
            attribution = None

            if ATTRIBUTION_ENABLED and calculate_revenue_attribution:
                try:
                    # Prepare market snapshot dict for attribution
                    market_snapshot_dict = None
                    if not market_snapshot.empty:
                        # Create category benchmarks from market data
                        # Safely get median price from available columns
                        median_price = 0
                        for price_col in ['price_per_unit', 'buy_box_price', 'price']:
                            if price_col in market_snapshot.columns:
                                price_series = market_snapshot[price_col].dropna()
                                if len(price_series) > 0:
                                    median_price = float(price_series.median())
                                    break
                        
                        market_snapshot_dict = {
                            'category_benchmarks': {
                                'growth_rate_30d': 0,  # Will be calculated if historical data available
                                'median_price': median_price
                            }
                        }

                    # Calculate attribution
                    attribution = calculate_revenue_attribution(
                        previous_revenue=previous_revenue,
                        current_revenue=current_revenue,
                        df_weekly=df_weekly,
                        trigger_events=trigger_events,
                        market_snapshot=market_snapshot_dict,
                        lookback_days=30,
                        portfolio_asins=project_asins
                    )

                    # === PERSISTENCE INJECTION (Added 2026-01-23) ===
                    # Automatically save intelligence for longitudinal monitoring
                    active_project_id = st.session_state.get('active_project_id')
                    if attribution and active_project_id:
                        try:
                            from datetime import date, timedelta
                            today = date.today()
                            # Define analysis period (matches lookback_days=30)
                            end_date_iso = today.isoformat()
                            start_date_iso = (today - timedelta(days=30)).isoformat()
                            
                            supabase_client = get_supabase_client()
                            
                            saved = save_revenue_attribution(
                                attribution=attribution,
                                project_id=active_project_id,
                                start_date=start_date_iso,
                                end_date=end_date_iso,
                                supabase=supabase_client
                            )
                            if saved:
                                st.toast("✅ Intelligence saved to history", icon="💾")
                        except Exception as save_err:
                            # Non-blocking error
                            st.caption(f"Note: Could not save history ({str(save_err)})")
                except Exception as e:
                    st.error(f"⚠️ Attribution calculation failed: {str(e)}")
                    st.caption("Debug: Check that revenue_attribution module is properly installed")

            # === DISPLAY DASHBOARD ===
            if attribution:
                # Header metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    delta_sign = "+" if attribution.total_delta >= 0 else ""
                    st.metric(
                        "Total Revenue Change",
                        f"${abs(attribution.total_delta):,.0f}",
                        delta=f"{delta_sign}{attribution.delta_pct:.1f}%",
                        delta_color="normal" if attribution.total_delta >= 0 else "inverse"
                    )

                with col2:
                    earned_pct = attribution.get_earned_percentage()
                    st.metric(
                        "Earned Growth (Your Actions)",
                        f"${attribution.internal_contribution:,.0f}",
                        delta=f"{earned_pct:.0f}% of total"
                    )

                with col3:
                    opportunistic = attribution.get_opportunistic_growth()
                    opp_pct = attribution.get_opportunistic_percentage()
                    st.metric(
                        "Opportunistic Growth (Market)",
                        f"${opportunistic:,.0f}",
                        delta=f"{opp_pct:.0f}% of total"
                    )

                # Confidence badge
                st.markdown(f"""
                **Explained Variance:** {attribution.get_variance_badge()} {attribution.get_variance_label()} |
                **Unexplained:** ${abs(attribution.residual):,.0f}
                """)

                st.markdown("---")

                # === WATERFALL CHART ===
                st.markdown("### 📊 Revenue Attribution Waterfall")
                st.caption("Shows cumulative contribution of each causal category to revenue change")

                try:
                    import plotly.graph_objects as go

                    # Build waterfall data
                    labels = [
                        "Starting Revenue",
                        "Internal Actions",
                        "Competitive Factors",
                        "Market Trends",
                        "Platform Changes",
                        "Ending Revenue"
                    ]

                    values = [
                        previous_revenue,
                        attribution.internal_contribution,
                        attribution.competitive_contribution,
                        attribution.macro_contribution,
                        attribution.platform_contribution,
                        current_revenue
                    ]

                    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]

                    # Color coding
                    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#3498db"]

                    fig = go.Figure(go.Waterfall(
                        name="Revenue Attribution",
                        orientation="v",
                        measure=measures,
                        x=labels,
                        y=values,
                        text=[f"${v:,.0f}" for v in values],
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": "#e74c3c"}},
                        increasing={"marker": {"color": "#2ecc71"}},
                        totals={"marker": {"color": "#3498db"}}
                    ))

                    fig.update_layout(
                        title="Revenue Change Attribution (30-Day Period)",
                        showlegend=False,
                        height=400,
                        yaxis_title="Revenue ($)",
                        xaxis_title="Category"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Waterfall chart unavailable: {str(e)}")

                st.markdown("---")

                # === CAUSAL MATRIX TABLE ===
                st.markdown("### 🧩 Causal Matrix: Revenue Change Drivers")
                st.caption("Color-coded by category | 🟢 High Confidence | 🟡 Medium | 🔴 Low")

                # Build matrix data
                all_drivers = attribution.get_all_drivers()

                if all_drivers:
                    matrix_data = []
                    for driver in all_drivers:
                        # Category label with color
                        category_labels = {
                            "internal": "🔵 Internal",
                            "competitive": "🟠 Competitive",
                            "macro": "🟣 Macro",
                            "platform": "🟢 Platform"
                        }

                        impact_str = f"+${driver.impact:,.0f}" if driver.impact > 0 else f"-${abs(driver.impact):,.0f}"
                        control_str = "High ✓" if driver.controllable else "None ✗"

                        matrix_data.append({
                            "Event": driver.description,
                            "Type": category_labels.get(driver.category.value, driver.category.value.title()),
                            "Impact": impact_str,
                            "Control": control_str,
                            "Confidence": f"{driver.get_confidence_badge()} {driver.confidence:.0%}"
                        })

                    matrix_df = pd.DataFrame(matrix_data)

                    # Display table with styling
                    st.dataframe(
                        matrix_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No significant drivers detected in this period")

                st.markdown("---")

                # === ATTRIBUTION PIE CHART ===
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📈 Attribution Breakdown")

                    try:
                        import plotly.graph_objects as go

                        # Build pie chart
                        labels = ['Internal Actions', 'Competitive', 'Market Trends', 'Platform']
                        values = [
                            abs(attribution.internal_contribution),
                            abs(attribution.competitive_contribution),
                            abs(attribution.macro_contribution),
                            abs(attribution.platform_contribution)
                        ]

                        # Filter out zero values
                        filtered_labels = []
                        filtered_values = []
                        for label, value in zip(labels, values):
                            if value > 0:
                                filtered_labels.append(label)
                                filtered_values.append(value)

                        if filtered_values:
                            fig = go.Figure(data=[go.Pie(
                                labels=filtered_labels,
                                values=filtered_values,
                                hole=.3,
                                marker=dict(colors=['#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
                            )])

                            fig.update_layout(
                                showlegend=True,
                                height=300
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No attribution data to display")
                    except Exception as e:
                        st.warning(f"⚠️ Pie chart unavailable: {str(e)}")

                with col2:
                    st.markdown("#### 💡 Executive Summary")

                    # Generate executive summary
                    if attribution.total_delta >= 0:
                        direction = "grew"
                        direction_emoji = "📈"
                    else:
                        direction = "declined"
                        direction_emoji = "📉"

                    earned_pct = attribution.get_earned_percentage()
                    opp_pct = attribution.get_opportunistic_percentage()

                    summary_html = f"""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db;">
                        <p style="font-size: 16px; margin-bottom: 12px;">
                            {direction_emoji} <strong>Revenue {direction} ${abs(attribution.total_delta):,.0f} ({attribution.delta_pct:+.1f}%)</strong>
                        </p>
                        <p style="font-size: 14px; color: #666; margin-bottom: 8px;">
                            <strong>Earned:</strong> ${attribution.internal_contribution:,.0f} ({earned_pct:.0f}%)<br>
                            <em>Growth from your actions (controllable)</em>
                        </p>
                        <p style="font-size: 14px; color: #666; margin-bottom: 8px;">
                            <strong>Opportunistic:</strong> ${attribution.get_opportunistic_growth():,.0f} ({opp_pct:.0f}%)<br>
                            <em>Growth from market conditions (temporary)</em>
                        </p>
                        <p style="font-size: 13px; color: #999; margin-top: 12px;">
                            {attribution.get_variance_badge()} <strong>Confidence:</strong> {attribution.explained_variance:.0%} explained variance
                        </p>
                    </div>
                    """

                    st.markdown(summary_html, unsafe_allow_html=True)

                    # Strategic insight
                    if earned_pct < 50 and attribution.total_delta > 0:
                        st.warning("⚠️ **Caution:** Most growth is opportunistic (external factors). Focus on converting temporary gains into sustainable advantages.")
                    elif earned_pct >= 70:
                        st.success("✅ **Strong:** Growth is primarily from your actions. This is sustainable.")
            else:
                # Attribution not available
                st.info("""
                🧩 **Causal Intelligence Unavailable**

                Revenue attribution requires:
                - Historical revenue data (30+ days)
                - Trigger event detection
                - Market snapshot data

                Please ensure your project has sufficient data loaded.
                """)

                if not ATTRIBUTION_ENABLED:
                    st.error("⚠️ Attribution engine not loaded. Check installation.")

                # === PREDICTIVE INTELLIGENCE (FORECAST) ===
                st.markdown("---")
                st.markdown("### 🔮 Predictive Intelligence: Future Outlook")
                st.caption("Forecast based on current run rate, seasonality, and anticipated events")

                try:
                    # Generate forecast using the unified pipeline
                    from src.predictive_forecasting import generate_combined_intelligence
                    
                    # Generate combined intelligence
                    combined_intel = generate_combined_intelligence(
                        current_revenue=current_revenue,
                        previous_revenue=previous_revenue,
                        attribution=attribution,
                        trigger_events=trigger_events,
                        df_historical=df_weekly
                    )
                    
                    if combined_intel and combined_intel.forecast:
                        portfolio_forecast = combined_intel.forecast
                        
                        pred_cols = st.columns([2, 1])
                        
                        with pred_cols[0]:
                            # Prepare Forecast Chart Data
                            # 1. Historical Data (Monthly)
                            if not df_weekly.empty:
                                if 'date' not in df_weekly.columns:
                                    df_weekly['date'] = pd.to_datetime(df_weekly['week'])
                                    
                                df_weekly['month'] = df_weekly['date'].dt.to_period('M')
                                # Use revenue proxy or calc from sum
                                rev_col_chart = 'revenue_proxy' if 'revenue_proxy' in df_weekly.columns else 'sales' if 'sales' in df_weekly.columns else None
                                
                                if rev_col_chart:
                                    monthly_rev = df_weekly.groupby('month')[rev_col_chart].sum().reset_index()
                                    monthly_rev['month'] = monthly_rev['month'].dt.to_timestamp()
                                    
                                    # Filter to last 6 months
                                    cutoff_month = pd.Timestamp.now() - pd.Timedelta(days=180)
                                    monthly_rev = monthly_rev[monthly_rev['month'] >= cutoff_month]
                                    
                                    # 2. Future Data Point
                                    last_date = monthly_rev['month'].max() if not monthly_rev.empty else pd.Timestamp.now()
                                    next_date = last_date + pd.Timedelta(days=30)
                                    
                                    proj_rev = portfolio_forecast.projected_revenue
                                    lower_bound = portfolio_forecast.lower_bound
                                    upper_bound = portfolio_forecast.upper_bound
                                    
                                    # Create Chart
                                    import plotly.graph_objects as go
                                    fig_pred = go.Figure()
                                    
                                    # Historical Bars
                                    fig_pred.add_trace(go.Bar(
                                        x=monthly_rev['month'],
                                        y=monthly_rev[rev_col_chart],
                                        name='Historical Revenue',
                                        marker_color='#e0e0e0'
                                    ))
                                    
                                    # Forecast Line
                                    fig_pred.add_trace(go.Scatter(
                                        x=[last_date, next_date],
                                        y=[monthly_rev[rev_col_chart].iloc[-1] if not monthly_rev.empty else 0, proj_rev],
                                        name='Projected Trend',
                                        line=dict(color='#007bff', width=3, dash='dot'),
                                        mode='lines+markers'
                                    ))
                                    
                                    # Confidence Interval (Error Bars)
                                    fig_pred.add_trace(go.Scatter(
                                        x=[next_date, next_date],
                                        y=[upper_bound, lower_bound],
                                        mode='markers',
                                        marker=dict(color='#007bff', size=1),
                                        error_y=dict(
                                            type='data',
                                            symmetric=False,
                                            array=[upper_bound - proj_rev],
                                            arrayminus=[proj_rev - lower_bound],
                                            color='rgba(0,123,255,0.3)',
                                            thickness=10,
                                            width=10
                                        ),
                                        name='Confidence Interval (80%)'
                                    ))
                                    
                                    fig_pred.update_layout(
                                        height=300,
                                        margin=dict(l=40, r=40, t=30, b=30),
                                        showlegend=True,
                                        legend=dict(orientation="h", y=1.1),
                                        title="Monthly Revenue Trajectory"
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                else:
                                    st.info("Insufficient historical data for forecast chart")
                        
                        with pred_cols[1]:
                            st.markdown("#### Annual Projection")
                            st.metric(
                                "Est. Annual Revenue",
                                f"${portfolio_forecast.projected_annual_sales:,.0f}",
                                delta=None,
                                help="Based on sustainable run rate and seasonality"
                            )
                            
                            st.markdown("#### Forecast Analysis")
                            st.info(f"📈 **Trajectory:** On track for ${proj_rev:,.0f} next month.")
                            
                            if portfolio_forecast.event_adjustments != 0:
                                st.warning(f"⚠️ **Event Impact:** ${portfolio_forecast.event_adjustments:,.0f} adjustment included due to upcoming events.")
                    
                except Exception as e:
                    st.warning(f"Note: Predictive pipeline not fully active yet ({str(e)})")

        except Exception as e:
            st.error(f"⚠️ Error loading Command Center 2.0: {str(e)}")
            st.code(traceback.format_exc())

with main_tab3:
    # Market Discovery - Always available, no data needed
    render_discovery_ui()

with main_tab4:
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

        # Display seed ASIN debug trail if available (persists from Market Discovery)
        if "active_project_debug_trail" in st.session_state and st.session_state["active_project_debug_trail"]:
            with st.expander("🔍 Seed ASIN Debug Trail (from Market Discovery)", expanded=True):
                st.caption("This shows the tracking of your seed ASIN through the market mapping process:")
                for msg in st.session_state["active_project_debug_trail"]:
                    if "❌" in msg:
                        st.error(msg)
                    elif "✅" in msg:
                        st.success(msg)
                    else:
                        st.write(msg)

                # Also check if seed ASIN is actually in the project
                if seed_asin:
                    if seed_asin in project_asins:
                        st.success(f"✅ **FINAL VERIFICATION:** Seed ASIN {seed_asin} is present in project ASINs ({len(project_asins)} total)")
                    else:
                        st.error(f"❌ **FINAL VERIFICATION:** Seed ASIN {seed_asin} NOT FOUND in project ASINs!")
                        st.write(f"Project has {len(project_asins)} ASINs")
                        st.write(f"First 10 ASINs: {project_asins[:10]}")

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
                    # DON'T use market_snapshot as df_weekly - that's the aggregated view!
                    # Real weekly data should come from session state (has week_start column)

                    # Check freshness and show indicator
                    freshness = check_data_freshness(project_asins)
                    if freshness.get("is_stale"):
                        st.caption(f"⏰ Data last updated {freshness.get('freshness_hours', 'N/A')}h ago")

            except Exception as e:
                st.caption(f"⚠️ Cache unavailable, using session data")
        
        # === GET REAL WEEKLY DATA FROM SESSION STATE ===
        # The actual weekly time-series data (with week_start) is stored in active_project_data
        # This contains the Keepa historical data with multiple rows per ASIN (one per week)
        df_weekly = st.session_state.get('active_project_data', pd.DataFrame())
        
        # Fallback to market snapshot if no weekly data
        if df_weekly.empty:
            df_weekly = market_snapshot.copy() if not market_snapshot.empty else pd.DataFrame()
        
        # Get market snapshot from session if not from Supabase
        if market_snapshot.empty:
            market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
            if not market_snapshot.empty:
                data_source = "session"

        if df_weekly.empty or market_snapshot.empty:
            st.warning("⚠️ No project data found. Please create a project in Market Discovery.")
            st.stop()
        
        # === DATA HEALER: Apply comprehensive gap-filling (ALL code paths) ===
        # This ensures ALL critical metrics exist with valid values for dashboard compatibility
        # MUST run BEFORE brand filtering to ensure portfolio_df inherits healed data
        try:
            from utils.data_healer import heal_market_snapshot, clean_and_interpolate_metrics
            # Use comprehensive market snapshot healer (ensures all dashboard columns exist)
            market_snapshot = heal_market_snapshot(market_snapshot, verbose=False)
            # Then apply time-series healing for gap interpolation
            market_snapshot = clean_and_interpolate_metrics(market_snapshot, group_by_column="asin", verbose=False)
            # Also heal df_weekly if it's different from market_snapshot
            if not df_weekly.equals(market_snapshot):
                df_weekly = heal_market_snapshot(df_weekly, verbose=False)
            
            # === ENSURE week_start COLUMN EXISTS FOR CAUSALITY CHART ===
            # The real Keepa weekly data should already have week_start from build_keepa_weekly_table()
            # Only create it from timestamps if truly missing (edge case)
            if 'week_start' not in df_weekly.columns:
                # Try to derive from date columns
                for date_col in ['timestamp', 'last_update', 'created_at']:
                    if date_col in df_weekly.columns:
                        df_weekly['week_start'] = pd.to_datetime(df_weekly[date_col]).dt.to_period('W').dt.start_time
                        break
            
            # Debug: Log weekly data stats for causality chart
            if 'week_start' in df_weekly.columns:
                unique_weeks = df_weekly['week_start'].nunique()
                unique_asins = df_weekly['asin'].nunique() if 'asin' in df_weekly.columns else 0
                st.session_state['_weekly_data_stats'] = f"{len(df_weekly)} rows, {unique_weeks} weeks, {unique_asins} ASINs"
            
            # Store df_weekly in session state for causality chart and other components
            st.session_state['df_weekly'] = df_weekly
            
        except ImportError:
            # Fallback: Manual column creation if healer not available
            if 'revenue_proxy' not in market_snapshot.columns or market_snapshot['revenue_proxy'].isna().all():
                if 'avg_weekly_revenue' in market_snapshot.columns:
                    market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
                elif 'price' in market_snapshot.columns and 'monthly_units' in market_snapshot.columns:
                    market_snapshot['revenue_proxy'] = (
                        pd.to_numeric(market_snapshot['price'], errors='coerce').fillna(0) *
                        pd.to_numeric(market_snapshot['monthly_units'], errors='coerce').fillna(0)
                    )
                else:
                    market_snapshot['revenue_proxy'] = 0.0
            if 'weekly_sales_filled' not in market_snapshot.columns:
                market_snapshot['weekly_sales_filled'] = market_snapshot['revenue_proxy'].copy()
            # Ensure other critical columns exist
            for col, default in [('velocity_trend_30d', 0.0), ('velocity_trend_90d', 0.0), 
                                  ('data_quality', 'VERY_LOW'), ('data_weeks', 0),
                                  ('bsr', 1000000), ('review_count', 0), ('rating', 0.0),
                                  ('amazon_bb_share', 0.5), ('competitor_oos_pct', 0.0)]:
                if col not in market_snapshot.columns:
                    market_snapshot[col] = default
            
            # CRITICAL: Also store df_weekly in fallback path
            st.session_state['df_weekly'] = df_weekly
        
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
            st.write(f"**Debug Info:**")
            st.write(f"- Data source: {data_source}")
            st.write(f"- Total ASINs in market_snapshot: {len(market_snapshot)}")
            st.write(f"- Total ASINs in project_asins list: {len(project_asins)}")
            st.write(f"- Seed ASIN: {seed_asin}")
            st.write(f"- Seed ASIN in project_asins list: {seed_asin in project_asins}")
            st.write(f"- Seed ASIN in market_snapshot['asin']: {seed_asin in market_snapshot['asin'].values if 'asin' in market_snapshot.columns else 'No asin column'}")
            st.write(f"- First 10 ASINs in market_snapshot: {market_snapshot['asin'].tolist()[:10] if 'asin' in market_snapshot.columns and len(market_snapshot) > 0 else 'Empty'}")
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
        
        # CRITICAL FIX: Ensure revenue_proxy exists in market_snapshot BEFORE creating portfolio_df
        # This prevents KeyError when discovery data has poor quality (80% missing price/BSR)
        # Discovery may create: revenue_proxy, monthly_units, price, avg_weekly_revenue
        if 'revenue_proxy' not in market_snapshot.columns or market_snapshot['revenue_proxy'].isna().all():
            if 'avg_weekly_revenue' in market_snapshot.columns:
                market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
            elif 'price' in market_snapshot.columns and 'monthly_units' in market_snapshot.columns:
                market_snapshot['revenue_proxy'] = (
                    pd.to_numeric(market_snapshot['price'], errors='coerce').fillna(0) *
                    pd.to_numeric(market_snapshot['monthly_units'], errors='coerce').fillna(0)
                )
            else:
                market_snapshot['revenue_proxy'] = 0.0
        else:
            # Column exists - ensure it's numeric (handles edge case where it's string or mixed type)
            market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['revenue_proxy'], errors='coerce').fillna(0)
        
        # NOW create portfolio_df - revenue_proxy is guaranteed to exist
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
        
        # DATA QUALITY WARNING: Alert user if most products have $0 revenue (common in niche B2B categories)
        products_with_revenue = (market_snapshot['revenue_proxy'] > 0).sum() if 'revenue_proxy' in market_snapshot.columns else 0
        data_quality_pct = (products_with_revenue / len(market_snapshot) * 100) if len(market_snapshot) > 0 else 0
        if data_quality_pct < 20 and len(market_snapshot) > 10:
            st.warning(
                f"⚠️ **Low Data Quality**: Only {products_with_revenue}/{len(market_snapshot)} products ({data_quality_pct:.0f}%) "
                f"have tracked sales data. This often happens in niche or B2B categories where Keepa has limited tracking. "
                f"Revenue estimates may be unreliable."
            )
        
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
        benchmark_valid = network_stats.get('benchmark_valid', True) if network_stats else True
        data_quality = network_stats.get('data_quality', 'UNKNOWN') if network_stats else 'UNKNOWN'
        
        if network_stats and network_stats.get('median_price'):
            with st.expander("🌐 Category Intelligence (Network Data)", expanded=False):
                # Show category context so user knows what benchmarks are from
                cat_id = network_stats.get('category_id')
                cat_name = st.session_state.get('active_project_category_name', 'Unknown Category')
                st.markdown(f"**Category Benchmarks: {cat_name}**")
                st.caption(f"Category ID: {cat_id} | Data from ShelfGuard network searches")
                
                # Check if benchmark is stale (detected during fetch)
                if data_quality == 'STALE' or not benchmark_valid:
                    st.warning("""
⚠️ **Stale benchmark data detected** - the stored benchmarks don't match your category.

**Why this happens**: Previous searches may have stored benchmarks from a broader parent category.

**Fix**: Run a fresh search in the "Search" tab. The new search will update the benchmarks with accurate data from your specific category.
""")
                else:
                    # Additional validation for display
                    median_price = network_stats.get('median_price', 0)
                    your_avg_price = 0
                    if 'buy_box_price' in portfolio_df.columns or 'filled_price' in portfolio_df.columns or 'price' in portfolio_df.columns:
                        price_col = 'buy_box_price' if 'buy_box_price' in portfolio_df.columns else 'filled_price' if 'filled_price' in portfolio_df.columns else 'price'
                        your_avg_price = portfolio_df[price_col].mean() if price_col in portfolio_df.columns else 0
                    
                    if median_price > 0 and your_avg_price > 0:
                        price_ratio = your_avg_price / median_price
                        if price_ratio > 5 or price_ratio < 0.2:
                            # NOTE: Escape $ signs to prevent LaTeX interpretation in Streamlit
                            st.warning(f"⚠️ **Benchmark mismatch detected**: Your portfolio avg (\\${your_avg_price:.2f}) is {price_ratio:.1f}x the category median (\\${median_price:.2f}). Run a fresh search to update benchmarks.")

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

                # BUY BOX & COMPETITION BENCHMARKS (new row)
                col_bb, col_offers = st.columns(2)
                with col_bb:
                    avg_bb_share = network_stats.get('avg_bb_share')
                    if avg_bb_share:
                        st.metric("Avg Buy Box %", f"{avg_bb_share*100:.0f}%")
                        
                        # Compare your portfolio average to category
                        if 'amazon_bb_share' in portfolio_df.columns:
                            your_avg_bb = portfolio_df['amazon_bb_share'].mean()
                            if your_avg_bb and your_avg_bb > 0:
                                bb_diff_pct = (your_avg_bb - avg_bb_share) * 100
                                bb_indicator = "🟢" if bb_diff_pct > 10 else "🟡" if bb_diff_pct > -10 else "🔴"
                                st.caption(f"{bb_indicator} Your avg: {your_avg_bb*100:.0f}% ({bb_diff_pct:+.1f}pp)")
                
                with col_offers:
                    avg_offer_count = network_stats.get('avg_offer_count')
                    if avg_offer_count:
                        st.metric("Avg Sellers/Listing", f"{avg_offer_count:.1f}")
                        
                        # Compare your portfolio average to category
                        # Prefer seller_count (from sellerIds) over new_offer_count
                        if 'seller_count' in portfolio_df.columns and not portfolio_df['seller_count'].isna().all():
                            your_avg_offers = portfolio_df['seller_count'].mean()
                        elif 'new_offer_count' in portfolio_df.columns:
                            your_avg_offers = portfolio_df['new_offer_count'].mean()
                        else:
                            your_avg_offers = None
                            
                        if your_avg_offers and your_avg_offers > 0:
                            offer_diff_pct = ((your_avg_offers / avg_offer_count) - 1) * 100
                            # Lower seller count is better (less competition)
                            offer_indicator = "🟢" if offer_diff_pct < -20 else "🟡" if offer_diff_pct < 20 else "🔴"
                            st.caption(f"{offer_indicator} Your avg: {your_avg_offers:.1f} ({offer_diff_pct:+.1f}%)")

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
        # CRITICAL FIX: Ensure revenue_proxy exists in both DataFrames before accessing
        # This prevents KeyError when discovery data doesn't have revenue_proxy column
        # Discovery creates: revenue_proxy, monthly_units, price, avg_weekly_revenue
        if 'revenue_proxy' not in portfolio_df.columns or portfolio_df['revenue_proxy'].isna().all():
            # Create revenue_proxy from available columns (priority order)
            if 'avg_weekly_revenue' in portfolio_df.columns:
                # Historical data available - use weekly * 4.33
                portfolio_df['revenue_proxy'] = pd.to_numeric(portfolio_df['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
            elif 'price' in portfolio_df.columns and 'monthly_units' in portfolio_df.columns:
                # Use price * monthly_units
                portfolio_df['revenue_proxy'] = (
                    pd.to_numeric(portfolio_df['price'], errors='coerce').fillna(0) *
                    pd.to_numeric(portfolio_df['monthly_units'], errors='coerce').fillna(0)
                )
            else:
                portfolio_df['revenue_proxy'] = 0.0
        
        if 'revenue_proxy' not in market_df.columns or market_df['revenue_proxy'].isna().all():
            # Create revenue_proxy from available columns (priority order)
            if 'avg_weekly_revenue' in market_df.columns:
                # Historical data available - use weekly * 4.33
                market_df['revenue_proxy'] = pd.to_numeric(market_df['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
            elif 'price' in market_df.columns and 'monthly_units' in market_df.columns:
                # Use price * monthly_units
                market_df['revenue_proxy'] = (
                    pd.to_numeric(market_df['price'], errors='coerce').fillna(0) *
                    pd.to_numeric(market_df['monthly_units'], errors='coerce').fillna(0)
                )
            else:
                market_df['revenue_proxy'] = 0.0
        
        # === FIX: Use ADJUSTED revenue (variation-deduplicated) to match Market Discovery ===
        # revenue_proxy_adjusted accounts for sibling variations sharing the same parent ASIN
        # Without this, variations would be counted multiple times, inflating revenue estimates
        
        # Portfolio (Your Brand) metrics - use adjusted if available
        portfolio_rev_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in portfolio_df.columns else 'revenue_proxy'
        portfolio_revenue = portfolio_df[portfolio_rev_col].sum() if portfolio_rev_col in portfolio_df.columns else 0
        portfolio_product_count = len(portfolio_df)

        # Market (Total Category) metrics - use adjusted if available
        market_rev_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in market_df.columns else 'revenue_proxy'
        total_market_revenue = market_df[market_rev_col].sum() if market_rev_col in market_df.columns else 0
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
        # FIX: Always ensure revenue_proxy exists as a column (not scalar)
        if 'revenue_proxy' in portfolio_snapshot_df.columns:
            portfolio_snapshot_df['revenue_proxy'] = pd.to_numeric(portfolio_snapshot_df['revenue_proxy'], errors='coerce').fillna(0)
        else:
            portfolio_snapshot_df['revenue_proxy'] = 0.0  # Create as column of zeros
        portfolio_snapshot_df['weekly_sales_filled'] = portfolio_snapshot_df['revenue_proxy'].copy()
        portfolio_snapshot_df['monthly_revenue'] = portfolio_snapshot_df['revenue_proxy'].copy()
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
        
        # Create data hash for cache key (based on ASIN list + revenue totals + strategic bias)
        # Include strategic_bias in cache key so changes trigger recalculation
        portfolio_data = res["data"]
        asin_list = portfolio_data['asin'].tolist() if 'asin' in portfolio_data.columns else []
        
        # Retrieve strategic_bias from session state (set in sidebar)
        strategic_bias = st.session_state.get('strategic_bias_value', '⚖️ Balanced Defense')
        strategic_bias_clean = st.session_state.get('strategic_bias_clean_value', 'Balanced Defense')
        
        data_cache_hash = hashlib.md5(
            f"{sorted(asin_list)[:20]}|{total_rev_curr:.0f}|{len(portfolio_data)}|{strategic_bias_clean}".encode(),
            usedforsecurity=False
        ).hexdigest()
        
        # Use cached intelligence calculation
        early_predictive_risk = _cached_portfolio_intelligence(
            data_cache_hash,
            total_rev_curr,
            strategic_bias_clean,  # Use cleaned version for internal calculations
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
            
            # Extract new intelligence signals from enriched data
            amazon_1p_count = 0
            backorder_count = 0
            amazon_unstable_count = 0
            sns_count = 0
            subscribe_opps = 0
            amazon_conquest_opps = 0
            
            if not enriched_portfolio_df.empty:
                if 'amazon_1p_competitor' in enriched_portfolio_df.columns:
                    amazon_1p_count = enriched_portfolio_df['amazon_1p_competitor'].sum()
                if 'buybox_is_backorder' in enriched_portfolio_df.columns:
                    backorder_count = enriched_portfolio_df['buybox_is_backorder'].fillna(False).sum()
                if 'amazon_unstable' in enriched_portfolio_df.columns:
                    amazon_unstable_count = enriched_portfolio_df['amazon_unstable'].sum()
                if 'sns_eligible' in enriched_portfolio_df.columns:
                    sns_count = enriched_portfolio_df['sns_eligible'].sum()
                if 'opportunity_type' in enriched_portfolio_df.columns:
                    subscribe_opps = (enriched_portfolio_df['opportunity_type'] == 'SUBSCRIBE').sum()
                if 'predictive_state' in enriched_portfolio_df.columns:
                    # Check if amazon_unstable column exists, otherwise create Series of False
                    if 'amazon_unstable' in enriched_portfolio_df.columns:
                        amazon_conquest_opps = ((enriched_portfolio_df['predictive_state'] == 'EXPLOIT') & (enriched_portfolio_df['amazon_unstable'] == True)).sum()
                    else:
                        amazon_conquest_opps = 0  # No column means no opportunities
            
            # Build intelligence signals section
            intel_signals = ""
            if amazon_1p_count > 0:
                intel_signals += f"\n    - Amazon 1P Competition: {amazon_1p_count} products facing Amazon as competitor"
            if backorder_count > 0:
                intel_signals += f"\n    - SUPPLY CRISIS: {backorder_count} products BACKORDERED (urgent action needed)"
            if amazon_unstable_count > 0:
                intel_signals += f"\n    - Amazon OOS Opportunity: {amazon_unstable_count} products have Amazon supply unstable (conquest target)"
            if sns_count > 0:
                intel_signals += f"\n    - Subscribe & Save: {sns_count} products eligible for subscription push"
            
            # Retrieve strategic_bias for portfolio summary
            strategic_bias_for_summary = st.session_state.get('strategic_bias_clean_value', 'Balanced Defense')
            
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
    - Strategic Focus: {strategic_bias_for_summary}

    INTELLIGENCE SIGNALS:{intel_signals if intel_signals else " (No special signals detected)"}
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
            # Include strategic_bias in cache key so brief updates when strategy changes
            strategic_bias_clean = st.session_state.get('strategic_bias_clean_value', 'Balanced Defense')
            cache_key = portfolio_hash + f"|{strategic_bias_clean}" + ("_refresh" if st.session_state.force_refresh_brief else "")
            llm_brief = generate_ai_brief(portfolio_summary, cache_key, strategic_bias_clean)

            # Reset force refresh flag after use
            if st.session_state.force_refresh_brief:
                st.session_state.force_refresh_brief = False

            if llm_brief:
                ai_brief = llm_brief
                brief_source = "🤖 STRATEGIC BRIEF"
            else:
                # Fallback to rule-based brief (Brand vs Market + Predictive Intelligence)
                brief_parts = []
                
                # URGENT: Backorder/Supply crisis alerts (highest priority)
                if backorder_count > 0:
                    brief_parts.append(f"**URGENT: {backorder_count} products BACKORDERED** — supply chain action required immediately.")
                
                # Risk alerts
                if risk_pct > 15:
                    brief_parts.append(f"**{f_money(thirty_day_risk)} at risk** over next 30 days ({risk_pct:.0f}% of revenue) — {defend_count} products need defensive action.")
                
                # Growth opportunities (prioritize Amazon conquest)
                if thirty_day_growth > 0:
                    if amazon_unstable_count > 0:
                        brief_parts.append(f"**{f_money(thirty_day_growth)} conquest opportunity** — Amazon supply unstable on {amazon_unstable_count} products. Attack now!")
                    elif conquest_count > 0:
                        brief_parts.append(f"**{f_money(thirty_day_growth)} growth opportunity** identified — {conquest_count} competitors vulnerable to conquest.")
                    elif price_lift_count > 0:
                        brief_parts.append(f"**{f_money(thirty_day_growth)} upside** via price optimization across {price_lift_count} products.")
                
                # Amazon 1P competition warning
                if amazon_1p_count > 0:
                    brief_parts.append(f"**Amazon 1P competition** detected on {amazon_1p_count} products — differentiate on brand value.")
                
                # S&S opportunity
                if sns_count > 0 and subscribe_opps > 0:
                    brief_parts.append(f"**{sns_count} products S&S eligible** — push subscription messaging for recurring revenue.")
                
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
        
        # === UNIFIED RISK COUNT (FIX: One definition used everywhere) ===
        # "Meaningful risk" = products where risk > $100 AND > 2% of product revenue
        # This prevents counting ALL products as "at risk" due to baseline optimization
        if 'thirty_day_risk' in enriched_portfolio_df.columns:
            product_risk = enriched_portfolio_df['thirty_day_risk'].fillna(0)
            # Use revenue_proxy if available, else weekly_sales_filled, else default to 1000
            if 'revenue_proxy' in enriched_portfolio_df.columns:
                product_rev = enriched_portfolio_df['revenue_proxy'].fillna(1000)
            elif 'weekly_sales_filled' in enriched_portfolio_df.columns:
                product_rev = enriched_portfolio_df['weekly_sales_filled'].fillna(1000)
            else:
                product_rev = pd.Series([1000] * len(enriched_portfolio_df))
            # Meaningful = either high absolute risk OR high % of that product's revenue
            meaningful_risk_mask = (product_risk > 100) & (product_risk > product_rev * 0.02)
            meaningful_risk_count = meaningful_risk_mask.sum()
        else:
            meaningful_risk_count = defend_count + replenish_count
        
        # SEMANTIC FIX: Distinguish URGENT alerts from OPTIMIZATION reviews
        # - Urgent alerts (DEFEND/REPLENISH): You'll LOSE money if you don't act
        # - Optimization reviews (HOLD): You COULD gain by optimizing, but no urgent loss
        urgent_alert_count = defend_count + replenish_count
        optimization_review_count = meaningful_risk_count - urgent_alert_count
        
        # Growth count: products with >$100 growth opportunity
        if 'thirty_day_growth' in enriched_portfolio_df.columns:
            meaningful_growth_count = (enriched_portfolio_df['thirty_day_growth'].fillna(0) > 100).sum()
        else:
            meaningful_growth_count = exploit_count
        
        # === ACTION A: CHECK AI BRIEF FOR THREAT KEYWORDS AND OVERRIDE STATUS BANNER ===
        ai_brief_lower = ai_brief.lower() if ai_brief else ""
        has_threat_keywords = any(keyword in ai_brief_lower for keyword in ["threat", "erosion", "critical"])
        has_threat_keywords = has_threat_keywords or is_predictive_critical
        
        # === STATUS LOGIC (FIXED: Semantic clarity between alerts vs optimizations) ===
        # Urgent alerts = DEFEND + REPLENISH (actual risk)
        # Optimization reviews = HOLD products with value (not urgent)
        action_required_count = meaningful_risk_count  # For backwards compat
        
        if has_threat_keywords or is_predictive_critical:
            status_emoji = "🔴"
            status_text = "DEFENSE PROTOCOL"
            top_action = f"{urgent_alert_count} URGENT + {optimization_review_count} to optimize" if urgent_alert_count > 0 else f"{meaningful_risk_count} ACTIONS REQUIRED"
        elif urgent_alert_count > 0:
            # Has actual risk items (DEFEND/REPLENISH)
            status_emoji = "🟡"
            status_text = "ATTENTION"
            top_action = f"{urgent_alert_count} ALERTS" + (f" + {optimization_review_count} to optimize" if optimization_review_count > 0 else "")
        elif is_predictive_elevated or has_high_urgency_alerts:
            status_emoji = "🟡"
            status_text = "ATTENTION"
            top_action = f"{optimization_review_count} items to optimize" if optimization_review_count > 0 else f"{meaningful_risk_count} items need review"
        elif meaningful_growth_count > 5:
            # Significant growth opportunities
            status_emoji = "🟢"
            status_text = "HEALTHY"
            top_action = f"CAPTURE {meaningful_growth_count} opportunities"
        elif risk_pct < 10:
            # Low risk, healthy
            status_emoji = "🟢"
            status_text = "HEALTHY"
            if thirty_day_risk > total_rev_curr * 0.10:
                top_action = "DEFEND revenue"
            elif thirty_day_growth > total_rev_curr * 0.05:
                top_action = "CAPTURE growth"
            elif your_market_share > 50:
                top_action = "OPTIMIZE pricing"
            elif your_market_share < 20:
                top_action = "EXPAND share"
            else:
                top_action = "MAINTAIN position"
        else:
            status_emoji = "🟡"
            status_text = "ATTENTION"
            top_action = f"{meaningful_risk_count} items need review"
        
        # === SYSTEM STATUS BANNER (FIXED: Uses same counts) ===
        total_opportunity_count = meaningful_growth_count
        
        if is_predictive_critical or has_threat_keywords:
            system_status = f"🔴 SYSTEM STATUS: {meaningful_risk_count} CRITICAL THREATS"
            status_bg = "#dc3545"
        elif is_predictive_elevated or meaningful_risk_count > len(enriched_portfolio_df) * 0.2:
            # More than 20% of products at meaningful risk
            system_status = f"🟡 SYSTEM STATUS: {meaningful_risk_count} ALERTS ACTIVE"
            status_bg = "#ffc107"
        elif total_opportunity_count > 0:
            system_status = f"🟢 SYSTEM STATUS: {total_opportunity_count} GROWTH OPPORTUNITIES"
            status_bg = "#28a745"
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
        
        # === EXECUTIVE SUMMARY: TOP 3 ACTIONS ===
        # FIX: Always show up to 3 actions, dynamically pulling from risk + growth pools
        # Even when no "risk" exists, show growth opportunities
        top_actions = []
        if 'thirty_day_risk' in enriched_portfolio_df.columns and 'thirty_day_growth' in enriched_portfolio_df.columns:
            # Get top risk products - ANY product with meaningful risk (>$50, lowered threshold)
            # Includes HARVEST (optimization), DEFEND (defense), REPLENISH (inventory)
            risk_products = enriched_portfolio_df[
                (enriched_portfolio_df['thirty_day_risk'].fillna(0) > 50)  # Lowered from $100
            ].copy()
            
            # Get top growth products (>$50 opportunity, lowered threshold)
            growth_products = enriched_portfolio_df[
                (enriched_portfolio_df['thirty_day_growth'].fillna(0) > 50)  # Lowered from $100
            ].copy()
            
            # Sort by risk/growth value
            if not risk_products.empty:
                risk_products = risk_products.sort_values('thirty_day_risk', ascending=False)
            if not growth_products.empty:
                growth_products = growth_products.sort_values('thirty_day_growth', ascending=False)
            
            # Build up to 3 actions dynamically
            # Priority: 1 risk (if exists) → 2 growth → fill remaining with risk OR growth
            action_count = 0
            asins_shown = set()
            
            # Calculate how many of each type to show based on availability
            has_meaningful_risk = len(risk_products) > 0
            has_meaningful_growth = len(growth_products) > 0
            
            if has_meaningful_risk and has_meaningful_growth:
                max_risk_items = 1  # Show 1 risk, then 2 growth
            elif has_meaningful_risk:
                max_risk_items = 3  # Only risk available
            else:
                max_risk_items = 0  # Only growth available
            for _, row in risk_products.head(max_risk_items).iterrows():
                if action_count >= 3:
                    break
                asin = row.get('asin', '')
                title = str(row.get('title', ''))[:40] + "..." if len(str(row.get('title', ''))) > 40 else row.get('title', '')
                risk = row.get('thirty_day_risk', 0)
                pred_state = row.get('predictive_state', '')
                strategic_state = row.get('strategic_state', 'HARVEST')
                current_price = row.get('buy_box_price', row.get('price', 0)) or 0
                rank = int(row.get('sales_rank_filled', row.get('sales_rank', 0)) or 0)
                bb_share = row.get('amazon_bb_share', 0) or 0
                competitor_count = row.get('competitor_count', 0) or 0
                price_risk = row.get('price_erosion_risk', 0) or 0
                share_risk = row.get('share_erosion_risk', 0) or 0
                stockout_risk = row.get('stockout_risk', 0) or 0
                data_quality = row.get('data_quality', 'MEDIUM')
                model_certainty = row.get('model_certainty', 0.5) or 0.5
                
                # Determine urgency (time sensitivity)
                if pred_state == "REPLENISH" and stockout_risk > risk * 0.5:
                    urgency = "🚨 ACT TODAY"
                    urgency_reason = "Stockout in <7 days"
                elif price_risk > risk * 0.5 and competitor_count > 5:
                    urgency = "📅 ACT THIS WEEK"
                    urgency_reason = f"{competitor_count} competitors cutting prices"
                elif share_risk > risk * 0.5:
                    urgency = "📅 ACT THIS WEEK"
                    urgency_reason = "Velocity declining"
                else:
                    urgency = "📊 REVIEW THIS MONTH"
                    urgency_reason = "Optimization opportunity"
                
                # Build specific action
                if pred_state == "REPLENISH":
                    action = f"Expedite inventory restock"
                    why = f"Stockout predicted in 7-14 days (supplier lead time too long)"
                elif strategic_state == "HARVEST" and price_risk > risk * 0.5:
                    # Pricing optimization opportunity
                    suggested_price = current_price * 1.05 if current_price > 0 else 0
                    action = f"Test price increase ${current_price:.2f} → ${suggested_price:.2f} (+5%)" if current_price > 0 else "Test price increase (rank #{rank} supports pricing power)"
                    why = f"Rank #{rank} + {bb_share*100:.0f}% Buy Box = pricing power. Competitors at ${current_price*1.08:.2f} range."
                elif strategic_state == "TRENCH_WAR" and price_risk > risk * 0.5:
                    # Need to match competitor pricing
                    competitor_price = current_price * 0.95 if current_price > 0 else 0
                    action = f"Match competitor pricing ${current_price:.2f} → ${competitor_price:.2f}" if current_price > 0 else "Match competitor pricing"
                    why = f"{competitor_count} competitors undercutting 5-8%. Buy Box at risk."
                else:
                    action = f"Review pricing/inventory strategy"
                    why = f"${risk:,.0f} at risk from {pred_state.lower()} state"
                
                # Calculate confidence
                confidence_pct = int(model_certainty * 100)
                if data_quality == "HIGH":
                    confidence_text = f"{confidence_pct}% (12+ months data)"
                elif data_quality == "MEDIUM":
                    confidence_text = f"{confidence_pct}% (3-12 months data)"
                else:
                    confidence_text = f"{confidence_pct}% (limited data)"
                
                top_actions.append({
                    'priority': action_count + 1,
                    'asin': asin,
                    'title': title,
                    'action': action,
                    'why': why,
                    'impact': f"${risk:,.0f}/mo",
                    'urgency': urgency,
                    'urgency_reason': urgency_reason,
                    'confidence': confidence_text,
                    'type': 'risk'
                })
                action_count += 1
            
            # Add growth opportunities to fill remaining slots (up to 3 total actions)
            # FIX: Show MULTIPLE growth actions, not just one
            # CRITICAL: Exclude ASINs already in risk actions to avoid contradictory recommendations
            asins_already_shown = {a['asin'] for a in top_actions}
            
            if action_count < 3 and not growth_products.empty:
                # Add up to (3 - action_count) growth opportunities
                for _, row in growth_products.iterrows():
                    if action_count >= 3:
                        break  # We have 3 actions now
                    
                    asin = row.get('asin', '')
                    if asin in asins_already_shown:
                        continue  # Skip - already shown as risk action
                    
                    title = str(row.get('title', ''))[:40] + "..." if len(str(row.get('title', ''))) > 40 else row.get('title', '')
                    growth = row.get('thirty_day_growth', 0)
                    opp_type = row.get('opportunity_type', '')
                    rank = int(row.get('sales_rank_filled', row.get('sales_rank', 0)) or 0)
                    current_price = row.get('buy_box_price', row.get('price', 0)) or 0
                    data_quality = row.get('data_quality', 'MEDIUM')
                    model_certainty = row.get('model_certainty', 0.75) or 0.75
                    
                    # Build action based on opportunity type
                    if opp_type == "PRICE_POWER" or opp_type == "PRICE_LIFT":
                        suggested_price = current_price * 1.04 if current_price > 0 else 0
                        action = f"Test price increase ${current_price:.2f} → ${suggested_price:.2f} (+4%)" if current_price > 0 else f"Test 4% price increase (rank #{rank})"
                        why = f"Rank #{rank} = top 0.1% of category. Pricing power opportunity."
                    elif opp_type == "CONQUEST":
                        action = "Capture competitor customers"
                        why = "Competitors out of stock. Pricing opportunity."
                    elif opp_type == "SUBSCRIBE":
                        action = "Push Subscribe & Save enrollment"
                        why = "S&S eligible product. Convert one-time buyers to subscribers."
                    else:
                        action = "Optimize pricing/spend"
                        why = f"${growth:,.0f} growth opportunity identified"
                    
                    # Calculate confidence for growth
                    confidence_pct = int(model_certainty * 100)
                    if data_quality == "HIGH":
                        confidence_text = f"{confidence_pct}% (12+ months data)"
                    elif data_quality == "MEDIUM":
                        confidence_text = f"{confidence_pct}% (3-12 months data)"
                    else:
                        confidence_text = f"{confidence_pct}% (model estimate)"
                    
                    top_actions.append({
                        'priority': action_count + 1,
                        'asin': asin,
                        'title': title,
                        'action': action,
                        'why': why,
                        'impact': f"+${growth:,.0f}/mo",
                        'urgency': "📊 REVIEW THIS MONTH",
                        'urgency_reason': "Growth opportunity",
                        'confidence': confidence_text,
                        'type': 'growth'
                    })
                    asins_already_shown.add(asin)  # Track this ASIN
                    action_count += 1
                    # NO break - continue adding growth actions until we have 3
        
        # Render Executive Summary
        if top_actions:
            st.markdown("### 🎯 Your Next Actions (Prioritized)")
            for action_item in top_actions:
                impact_color = "#dc3545" if action_item['type'] == 'risk' else "#28a745"
                urgency_color = "#dc3545" if "TODAY" in action_item['urgency'] else "#ffc107" if "WEEK" in action_item['urgency'] else "#6c757d"
                
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e0e0e0; border-left: 4px solid {impact_color}; 
                            padding: 16px; border-radius: 6px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); min-height: 140px;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                        <div style="font-size: 16px; font-weight: 700; color: #1a1a1a;">
                            {action_item['priority']}. {action_item['action']}
                        </div>
                        <div style="font-size: 18px; font-weight: 700; color: {impact_color};">
                            {action_item['impact']}
                        </div>
                    </div>
                    <div style="font-size: 10px; color: #666; text-transform: uppercase; margin-bottom: 4px;">Opportunity Alpha</div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px; line-height: 1.4;">
                        <strong>Why:</strong> {action_item['why']}
                    </div>
                    <div style="display: flex; gap: 12px; font-size: 11px;">
                        <span style="background: {urgency_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600;">
                            {action_item['urgency']}
                        </span>
                        <span style="color: #666;">{action_item['urgency_reason']}</span>
                        <span style="color: #999; margin-left: auto;">Confidence: {action_item['confidence']}</span>
                    </div>
                    <div style="font-size: 10px; color: #999; margin-top: 6px;">
                        <code>{action_item['asin']}</code> | {action_item['title']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # === SHOW ALERT DETAILS (if alerts exist) ===
        # Semantic: Separate urgent alerts (DEFEND/REPLENISH) from optimization reviews (HOLD)
        if meaningful_risk_count > 0:
            # Build expander title that clarifies what's inside
            if urgent_alert_count > 0 and optimization_review_count > 0:
                expander_title = f"📋 View {urgent_alert_count} Alerts + {optimization_review_count} Optimizations"
            elif urgent_alert_count > 0:
                expander_title = f"📋 View {urgent_alert_count} Alert Details"
            else:
                expander_title = f"📋 View {optimization_review_count} Optimization Opportunities"
            
            with st.expander(expander_title, expanded=False):
                # Get products with meaningful risk (>$100) - includes HARVEST optimization opportunities
                if 'thirty_day_risk' in enriched_portfolio_df.columns:
                    risk_values = enriched_portfolio_df['thirty_day_risk'].fillna(0)
                    rev_col = 'weekly_sales_filled' if 'weekly_sales_filled' in enriched_portfolio_df.columns else 'revenue_proxy'
                    if rev_col in enriched_portfolio_df.columns:
                        revenue_values = enriched_portfolio_df[rev_col].fillna(1000)
                    else:
                        revenue_values = pd.Series([1000] * len(enriched_portfolio_df))
                    
                    # Meaningful risk = >$100 absolute OR >2% of product revenue
                    has_meaningful_risk = (risk_values > 100) | (risk_values > revenue_values * 0.02)
                    alert_products = enriched_portfolio_df[has_meaningful_risk].copy()
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
                        
                        # Different display based on state
                        # DEFEND/REPLENISH = actual risk (you'll LOSE if you don't act)
                        # EXPLOIT = competitor weakness (opportunity to GAIN)
                        # GROW/STABLE = optimization potential (not urgent)
                        if state == "DEFEND":
                            state_emoji = "🛡️"
                            state_label = "DEFEND"
                            value_label = "⚠️ At Risk"
                            value_color = "#dc3545"
                        elif state == "REPLENISH":
                            state_emoji = "📦"
                            state_label = "RESTOCK"
                            value_label = "⚠️ At Risk"
                            value_color = "#dc3545"
                        elif state == "EXPLOIT":
                            state_emoji = "🎯"
                            state_label = "EXPLOIT"
                            value_label = "🎯 Conquest"
                            value_color = "#28a745"
                        elif state == "GROW":
                            state_emoji = "📈"
                            state_label = "PRICING POWER"
                            value_label = "💰 Opportunity"
                            value_color = "#28a745"
                        else:  # STABLE or HOLD
                            state_emoji = "✅"
                            state_label = "STABLE"
                            value_label = "💰 Potential"
                            value_color = "#ffc107"
                        
                        st.markdown(f"""
                        **{state_emoji} {state_label}** | `{asin}` | {title}  
                        💰 Revenue: ${rev:,.0f}/mo | {value_label}: ${risk:,.0f}
                        """)
                    
                    if shown_count == 0:
                        st.success("All flagged products have minimal revenue/risk - no action needed")
                else:
                    st.success("No products require immediate attention")
    
        # Render the AI brief (full width, no regenerate button)
        # FIX: Convert markdown to HTML to prevent font rendering issues
        import re
        def markdown_to_html(text):
            """Convert basic markdown to HTML for proper rendering."""
            if not text:
                return text
            # Convert **bold** to <strong>
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            # Convert *italic* to <em>
            text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
            # Convert $ amounts to prevent LaTeX rendering (escape dollar signs)
            # Actually, $ is fine in HTML - no change needed there
            # Convert newlines to <br>
            text = text.replace('\n', '<br>')
            return text
        
        ai_brief_html = markdown_to_html(ai_brief) if ai_brief else "Analyzing..."
        
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
                    <div style="font-size: 11px; color: #666;">Est. Monthly Revenue</div>
                    <div style="font-size: 10px; color: #999; margin-top: 2px;">Based on last 90 days of sales data</div>
                    <div style="font-size: 9px; color: #bbb; margin-top: 1px;">Total Market: {f_money(total_market_revenue)}</div>
                </div>
            </div>
            <div style="background: #f8f9fa; padding: 14px; border-radius: 6px; margin-top: 10px;">
                <div style="font-size: 11px; color: #00704A; font-weight: 600; margin-bottom: 6px;">{brief_source}</div>
                <div style="font-size: 14px; color: #333; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {ai_brief_html}
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
                    <div class="custom-metric-label">{target_brand} Est. Monthly Revenue</div>
                    <div class="custom-metric-value">{f_money(portfolio_revenue)}</div>
                    <div style="font-size: 10px; color: #888; margin-top: -4px; margin-bottom: 6px;">Projected from last 90 days</div>
                    <div class="benchmark-row" style="flex-wrap: wrap; gap: 6px;">
                        <span class="benchmark-badge benchmark-{share_class}">{share_icon} {your_market_share:.1f}% Market Share</span>
                        <span class="benchmark-badge benchmark-neu">📊 {portfolio_product_count} Your ASINs</span>
                        <span class="benchmark-badge benchmark-neu">🎯 vs. {competitor_product_count} Competitors</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 2: Defense Score
        with c2:
            # ACTION B: Defense Score (Predictive-Based + Competitive Pressure)
            # FIXED: Defense Score now includes competitive pressure signals, not just risk_pct
            # A perfect 100 should be rare - only for products with no risk AND no competitive pressure
            
            # Start with risk-based score
            base_defense = 100 - risk_pct
            
            # === COMPETITIVE PRESSURE PENALTIES ===
            # These prevent a false 100 when competitive pressure exists
            
            # 1. Buy Box instability penalty: 50% BB ownership = -20 points (max -40 for 0% BB)
            avg_bb_share = enriched_portfolio_df['amazon_bb_share'].mean() if 'amazon_bb_share' in enriched_portfolio_df.columns else 0.5
            if pd.isna(avg_bb_share):
                avg_bb_share = 0.5
            bb_penalty = (1 - avg_bb_share) * 40
            
            # 2. Seller competition penalty: Many sellers = trench war (max -20 for 25+ sellers)
            avg_sellers = enriched_portfolio_df['seller_count'].mean() if 'seller_count' in enriched_portfolio_df.columns else \
                         (enriched_portfolio_df['new_offer_count'].mean() if 'new_offer_count' in enriched_portfolio_df.columns else 1)
            if pd.isna(avg_sellers):
                avg_sellers = 1
            seller_penalty = min(20, avg_sellers * 0.75)
            
            # 3. Amazon 1P competitor penalty: If Amazon is selling your products, that's pressure
            has_amazon_competitor = False
            if 'has_amazon_seller' in enriched_portfolio_df.columns:
                has_amazon_competitor = enriched_portfolio_df['has_amazon_seller'].any()
            elif 'buybox_is_amazon' in enriched_portfolio_df.columns:
                has_amazon_competitor = enriched_portfolio_df['buybox_is_amazon'].any()
            amazon_penalty = 8 if has_amazon_competitor else 0
            
            # Calculate defense score with all penalties
            defense_score = base_defense - bb_penalty - seller_penalty - amazon_penalty
            defense_score = max(0, min(100, defense_score))
            
            # Apply floor based on severity to prevent unrealistic scores
            # If we have critical status, cap at 60 max; if elevated, cap at 80 max
            is_critical_status = is_predictive_critical or has_threat_keywords
            is_elevated_status = is_predictive_elevated or has_high_urgency_alerts
            
            if is_critical_status:
                defense_score = min(defense_score, 60)
            elif is_elevated_status:
                defense_score = min(defense_score, 80)

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
            # FIX: Use meaningful_risk_count for consistency (same as breakdown)
            action_count = meaningful_risk_count
            
            # Build action summary line with both risk and growth actions
            action_parts = []
            if action_count > 0:
                action_parts.append(f"🔴 <strong>{action_count}</strong> risks")
            if growth_opportunity_count > 0:
                action_parts.append(f"🟢 <strong>{growth_opportunity_count}</strong> growth")
            
            action_line = " • ".join(action_parts) if action_parts else "No actions needed"
            
            # Display Opportunity Alpha (Risk + Growth) - MATCHED FORMAT with portfolio cards
            st.markdown(f"""
<div class="custom-metric-container" title="Combined 30-day opportunity: Risk to avert + Growth to capture. Based on velocity trends and competitive signals.">
    <div class="custom-metric-label">Opportunity Alpha</div>
    <div class="custom-metric-value">
        <span style="color: #dc3545;">{f_money(thirty_day_risk)}</span>
        <span style="color: #666; font-size: 0.7em;"> + </span>
        <span style="color: #28a745;">{f_money(thirty_day_growth)}</span>
    </div>
    <div style="font-size: 0.7rem; color: #666; margin-top: 4px;">{f_money(opportunity_alpha)} total alpha</div>
    <div class="benchmark-row" style="flex-wrap: wrap; gap: 4px; margin-top: 4px;">
        <span class="benchmark-badge benchmark-{status_color}">{risk_icon} {portfolio_status}</span>
    </div>
    <div style="font-size: 0.65rem; color: #666; margin-top: 6px;">{action_line}</div>
</div>
            """, unsafe_allow_html=True)
            
            # === OPPORTUNITY ALPHA BREAKDOWN ===
            if thirty_day_risk > 0 or thirty_day_growth > 0:
                # Count products with actual risk/growth values (not just predictive_state)
                actual_risk_count = 0
                actual_growth_count = 0
                # FIXED: Use meaningful counts (>$100 threshold) for consistency with header
                actual_risk_count = meaningful_risk_count
                actual_growth_count = meaningful_growth_count
                
                with st.expander(f"📊 View Opportunity Breakdown ({actual_risk_count} risks, {actual_growth_count} growth)", expanded=False):
                    col_risk, col_growth = st.columns(2)
                    
                    with col_risk:
                        st.markdown("**🔴 Risk Products**")
                        # FIXED: Filter by meaningful risk and show actual risk source
                        if 'thirty_day_risk' in enriched_portfolio_df.columns:
                            risk_products = enriched_portfolio_df[
                                enriched_portfolio_df['thirty_day_risk'].fillna(0) > 100  # Meaningful threshold
                            ].copy()
                            
                            if not risk_products.empty:
                                risk_products = risk_products.sort_values('thirty_day_risk', ascending=False)
                                for _, row in risk_products.head(10).iterrows():
                                    asin = row.get('asin', '')[:10]
                                    risk = row.get('thirty_day_risk', 0)
                                    # FIXED: Show actual risk SOURCE instead of generic HOLD
                                    price_risk = row.get('price_erosion_risk', 0) or 0
                                    share_risk = row.get('share_erosion_risk', 0) or 0
                                    stockout_risk = row.get('stockout_risk', 0) or 0
                                    
                                    # Determine primary risk driver
                                    if stockout_risk >= share_risk and stockout_risk >= price_risk:
                                        risk_source = "INVENTORY"
                                    elif share_risk >= price_risk:
                                        risk_source = "VELOCITY"
                                    else:
                                        risk_source = "PRICING"
                                    
                                    st.markdown(f"• `{asin}` ${risk:,.0f} ({risk_source})")
                            else:
                                st.info("No meaningful risk products (all < $100)")
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
    
        # === HOISTED INTELLIGENCE PIPELINE (Command Center 3.0 Core) ===
        # We calculate this UP FRONT so all UI components (Legacy & New) can use the intelligence
        
        # Initialize Intelligence Variables
        all_triggers = []
        portfolio_attribution = None
        portfolio_scenarios = []
        portfolio_anticipated = []
        sustainable_run_rate = 0
        portfolio_forecast = None
        trigger_detection_available = False
        
        # Get data references
        df_weekly_hist = st.session_state.get('df_weekly', pd.DataFrame())
        market_snapshot_df = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
        target_brand_name = st.session_state.get('active_brand', '')
        
        # 1. TRIGGER DETECTION ENGINE
        if not df_weekly_hist.empty and 'asin' in df_weekly_hist.columns:
            try:
                from src.trigger_detection import detect_trigger_events
                trigger_detection_available = True
                
                # Get top revenue products for trigger detection (limit to 10 for performance)
                if target_brand_name and 'brand' in df_weekly_hist.columns:
                    target_brand_lower = target_brand_name.lower().strip()
                    brand_asins = df_weekly_hist[df_weekly_hist['brand'].str.lower() == target_brand_lower]['asin'].unique()[:10]
                else:
                    # Fallback: Top 10 by mention count
                    brand_asins = df_weekly_hist['asin'].value_counts().index[:10]
                
                for asin in brand_asins:
                    asin_history = df_weekly_hist[df_weekly_hist['asin'] == asin]
                    if not asin_history.empty and len(asin_history) >= 3:
                        try:
                            # Use market snapshot for competitor context
                            triggers = detect_trigger_events(
                                asin=asin,
                                df_historical=asin_history,
                                df_competitors=market_snapshot_df,
                                lookback_days=30
                            )
                            # Tag triggers with ASIN
                            for t in triggers:
                                t.asin = asin
                            all_triggers.extend(triggers)
                        except Exception:
                            pass
                
                # Sort by severity
                all_triggers = sorted(all_triggers, key=lambda e: e.severity, reverse=True)[:15]
                
            except Exception as e:
                # Silent fail for triggers to keep dashboard robust
                pass

        # 2. REVENUE ATTRIBUTION ENGINE
        if ATTRIBUTION_ENABLED and FORECASTING_ENABLED and not df_weekly_hist.empty:
            try:
                # Use enriched portfolio for revenue estimate
                est_monthly_revenue = enriched_portfolio_df['estimated_monthly_revenue'].sum() if 'estimated_monthly_revenue' in enriched_portfolio_df.columns else 0
                
                if est_monthly_revenue > 0:
                    prev_revenue = est_monthly_revenue * 0.95 # Baseline
                    
                    # Calculate Attribution
                    portfolio_attribution = calculate_revenue_attribution(
                        previous_revenue=prev_revenue,
                        current_revenue=est_monthly_revenue,
                        df_weekly=df_weekly_hist,
                        trigger_events=all_triggers,
                        market_snapshot=market_snapshot_df.to_dict('records') if not market_snapshot_df.empty else None,
                        lookback_days=30
                    )
                    
                    # Generate Combined Intelligence (Forecasting)
                    combined_intel = generate_combined_intelligence(
                        current_revenue=est_monthly_revenue,
                        previous_revenue=prev_revenue,
                        attribution=portfolio_attribution,
                        trigger_events=all_triggers,
                        df_historical=df_weekly_hist
                    )
                    
                    if combined_intel:
                        portfolio_scenarios = combined_intel.scenarios
                        portfolio_anticipated = combined_intel.anticipated_events
                        sustainable_run_rate = combined_intel.sustainable_run_rate
                        # Extract forecast locally if needed for legacy tab (though ideally use hoisted var)
                        # Actually, portfolio_forecast might be None if hoisted failed, so using local is safe
                        if combined_intel.forecast:
                            portfolio_forecast = combined_intel.forecast
            except Exception:
                pass

        # === COMPETITIVE INTELLIGENCE & ROOT CAUSE ANALYSIS ===
        # ... [existing code] ...

        # [SKIP DOWN TO AFTER Causality Chart]
        # Locate end of "Market Causality Analysis" block and insert before "AI ACTION QUEUE"

                    # ... [Chart Logic] ...
                    
        # === PREDICTIVE INTELLIGENCE (FORECASTING) ===
        if portfolio_forecast:
            st.markdown("---")
            st.markdown("### 🔮 Predictive Revenue Forecast")
            
            pred_cols = st.columns([2, 1])
            
            with pred_cols[0]:
                # Prepare Forecast Chart Data
                # 1. Historical Data (Monthly)
                if not df_weekly.empty:
                    df_weekly['month'] = pd.to_datetime(df_weekly['week_start']).dt.to_period('M')
                    monthly_rev = df_weekly.groupby('month')['revenue_proxy'].sum().reset_index()
                    monthly_rev['month'] = monthly_rev['month'].dt.to_timestamp()
                    
                    # Filter to last 6 months
                    cutoff_month = pd.Timestamp.now() - pd.Timedelta(days=180)
                    monthly_rev = monthly_rev[monthly_rev['month'] >= cutoff_month]
                    
                    # 2. Future Data Point
                    last_date = monthly_rev['month'].max() if not monthly_rev.empty else pd.Timestamp.now()
                    next_date = last_date + pd.Timedelta(days=30)
                    
                    proj_rev = portfolio_forecast.projected_revenue
                    lower_bound = portfolio_forecast.lower_bound
                    upper_bound = portfolio_forecast.upper_bound
                    
                    # Create Chart
                    fig_pred = go.Figure()
                    
                    # Historical Bars
                    fig_pred.add_trace(go.Bar(
                        x=monthly_rev['month'],
                        y=monthly_rev['revenue_proxy'],
                        name='Historical Revenue',
                        marker_color='#e0e0e0'
                    ))
                    
                    # Forecast Line
                    fig_pred.add_trace(go.Scatter(
                        x=[last_date, next_date],
                        y=[monthly_rev['revenue_proxy'].iloc[-1] if not monthly_rev.empty else 0, proj_rev],
                        name='Projected Trend',
                        line=dict(color='#007bff', width=3, dash='dot'),
                        mode='lines+markers'
                    ))
                    
                    # Confidence Interval (Error Bars)
                    fig_pred.add_trace(go.Scatter(
                        x=[next_date, next_date],
                        y=[upper_bound, lower_bound],
                        mode='markers',
                        marker=dict(color='#007bff', size=1),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[upper_bound - proj_rev],
                            arrayminus=[proj_rev - lower_bound],
                            color='rgba(0,123,255,0.3)',
                            thickness=10,
                            width=10
                        ),
                        name='Confidence Interval (80%)'
                    ))
                    
                    fig_pred.update_layout(
                        height=300,
                        margin=dict(l=40, r=40, t=30, b=30),
                        showlegend=True,
                        legend=dict(orientation="h", y=1.1),
                        title="Monthly Revenue Trajectory"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            with pred_cols[1]:
                st.markdown("#### Annual Projection")
                st.metric(
                    "Est. Annual Revenue",
                    f"${portfolio_forecast.projected_annual_sales:,.0f}",
                    delta=None,
                    help="Based on sustainable run rate and seasonality"
                )
                
                st.markdown("#### Forecast Analysis")
                st.info(f"📈 **Trajectory:** On track for ${proj_rev:,.0f} next month (+{(proj_rev - (monthly_rev['revenue_proxy'].iloc[-1] if not monthly_rev.empty else 0))/monthly_rev['revenue_proxy'].iloc[-1]*100:.1f}% vs last month).")
                
                if portfolio_forecast.event_adjustments != 0:
                    st.warning(f"⚠️ **Event Impact:** ${portfolio_forecast.event_adjustments:,.0f} adjustment included due to upcoming events.")

        # --- AI ACTION QUEUE (Outside of columns - full width) ---
        # Only show if we have enriched portfolio data
        if not enriched_portfolio_df.empty and len(enriched_portfolio_df) > 0:
            # FIX: Calculate metrics FIRST, then render in columns side-by-side
            # This ensures both columns have content at the same level
            
            # Calculate competitive metrics (with safe fallbacks)
            # FIXED: Use seller_count (from sellerIds) - more accurate than new_offer_count
            try:
                # Sellers per SKU = average seller_count across your products
                # Prefer seller_count (from Keepa's sellerIds array) over new_offer_count
                if 'seller_count' in enriched_portfolio_df.columns and not enriched_portfolio_df['seller_count'].isna().all():
                    avg_sellers_per_sku = float(enriched_portfolio_df['seller_count'].mean())
                elif 'new_offer_count' in enriched_portfolio_df.columns and not enriched_portfolio_df['new_offer_count'].isna().all():
                    avg_sellers_per_sku = float(enriched_portfolio_df['new_offer_count'].mean())
                else:
                    avg_sellers_per_sku = 1  # Default: at least 1 seller (you)
            except:
                avg_sellers_per_sku = 1
            
            # FIXED: Calculate price gap with pack size normalization (use price_per_unit when available)
            price_gap_normalized = False
            try:
                # Prefer price_per_unit for fair comparison across pack sizes
                if 'price_per_unit' in enriched_portfolio_df.columns and 'price_per_unit' in market_snapshot.columns:
                    your_ppu = enriched_portfolio_df['price_per_unit'].dropna()
                    if len(your_ppu) > 0 and your_ppu.mean() > 0:
                        your_avg_price = your_ppu.mean()
                        # Competitor average (non-your-brand)
                        if 'is_your_brand' in market_snapshot.columns:
                            competitor_ppu = market_snapshot.loc[~market_snapshot['is_your_brand'], 'price_per_unit'].dropna()
                        else:
                            competitor_ppu = market_snapshot['price_per_unit'].dropna()
                        competitor_avg_price = competitor_ppu.mean() if len(competitor_ppu) > 0 else 0
                        avg_price_gap = your_avg_price - competitor_avg_price if competitor_avg_price > 0 else 0
                        price_gap_normalized = True
                    else:
                        avg_price_gap = 0
                else:
                    # Fallback to raw price
                    price_col = 'buy_box_price' if 'buy_box_price' in enriched_portfolio_df.columns else 'filled_price' if 'filled_price' in enriched_portfolio_df.columns else None
                    if price_col and price_col in market_snapshot.columns:
                        your_avg_price = enriched_portfolio_df[price_col].mean() if price_col in enriched_portfolio_df.columns else 0
                        competitor_prices = market_snapshot.loc[~market_snapshot['is_your_brand'], price_col] if 'is_your_brand' in market_snapshot.columns else market_snapshot[price_col]
                        competitor_avg_price = competitor_prices.mean() if len(competitor_prices) > 0 else 0
                        avg_price_gap = your_avg_price - competitor_avg_price if competitor_avg_price > 0 else 0
                    else:
                        avg_price_gap = 0
            except:
                avg_price_gap = 0
            
            # FIXED: Use actual OOS data from Keepa - check BOTH column naming conventions
            # Keepa client uses: oos_pct_90, oos_pct_30
            # Some code uses: outOfStockPercentage90
            oos_data_available = False
            competitor_oos_pct = None
            try:
                # Try all possible OOS column names (prioritize Keepa client naming)
                oos_columns = ['oos_pct_90', 'outOfStockPercentage90', 'oos_pct_30', 'outOfStockPercentage30']
                
                for oos_col in oos_columns:
                    if oos_col in market_snapshot.columns and 'is_your_brand' in market_snapshot.columns:
                        # Competitor OOS = average OOS of non-your-brand products
                        competitor_oos_data = market_snapshot.loc[~market_snapshot['is_your_brand'], oos_col]
                        valid_oos_data = competitor_oos_data.dropna()
                        
                        if len(valid_oos_data) > 0:
                            oos_val = float(valid_oos_data.mean())
                            # Only count as valid if there's actual OOS (not all 0)
                            if oos_val > 0.001:  # At least 0.1% OOS
                                competitor_oos_pct = oos_val
                                # Normalize if > 1 (some sources use 0-100 scale)
                                if competitor_oos_pct > 1:
                                    competitor_oos_pct = competitor_oos_pct / 100
                                oos_data_available = True
                                break  # Found valid data
                
                # Fallback: check enriched portfolio
                if not oos_data_available and 'competitor_oos_pct' in enriched_portfolio_df.columns:
                    val = enriched_portfolio_df['competitor_oos_pct'].mean()
                    if pd.notna(val) and val > 0.001:
                        competitor_oos_pct = float(val)
                        oos_data_available = True
                        
            except Exception as e:
                competitor_oos_pct = None
                oos_data_available = False
            
            # Market position
            position_text = "Market Leader" if your_market_share > 50 else "Strong Challenger" if your_market_share > 20 else "Niche Player"
            position_color = "#28a745" if your_market_share > 50 else "#ffc107" if your_market_share > 20 else "#dc3545"
            
            # Calculate risk components for Root Cause section
            try:
                total_price_risk = float(enriched_portfolio_df['price_erosion_risk'].sum()) if 'price_erosion_risk' in enriched_portfolio_df.columns and not enriched_portfolio_df['price_erosion_risk'].isna().all() else 0
            except:
                total_price_risk = 0
            
            try:
                total_share_risk = float(enriched_portfolio_df['share_erosion_risk'].sum()) if 'share_erosion_risk' in enriched_portfolio_df.columns and not enriched_portfolio_df['share_erosion_risk'].isna().all() else 0
            except:
                total_share_risk = 0
            
            try:
                total_stockout_risk = float(enriched_portfolio_df['stockout_risk'].sum()) if 'stockout_risk' in enriched_portfolio_df.columns and not enriched_portfolio_df['stockout_risk'].isna().all() else 0
            except:
                total_stockout_risk = 0
            
            total_risk_components = total_price_risk + total_share_risk + total_stockout_risk
            
            # Calculate growth components
            try:
                price_power_growth = float(enriched_portfolio_df[enriched_portfolio_df['opportunity_type'] == 'PRICE_POWER']['thirty_day_growth'].sum()) if 'opportunity_type' in enriched_portfolio_df.columns else 0
            except:
                price_power_growth = 0
            try:
                conquest_growth = float(enriched_portfolio_df[enriched_portfolio_df['opportunity_type'] == 'CONQUEST']['thirty_day_growth'].sum()) if 'opportunity_type' in enriched_portfolio_df.columns else 0
            except:
                conquest_growth = 0
            try:
                review_moat_growth = float(enriched_portfolio_df[enriched_portfolio_df['opportunity_type'] == 'REVIEW_MOAT']['thirty_day_growth'].sum()) if 'opportunity_type' in enriched_portfolio_df.columns else 0
            except:
                review_moat_growth = 0
            
            total_growth_components = price_power_growth + conquest_growth + review_moat_growth
            
            # FIX: Create columns AFTER calculating all metrics, render BOTH sections side-by-side
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("### 🥊 Competitive Landscape")
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; margin-bottom: 12px; min-height: 200px;">
                    <div style="font-size: 14px; font-weight: 600; color: {position_color}; margin-bottom: 12px;">
                        Market Position: {position_text} ({your_market_share:.1f}% share)
                    </div>
                    <div style="font-size: 12px; color: #666; line-height: 1.6;">
                        <div><strong>Competitor Products:</strong> {competitor_product_count} in market</div>
                        <div><strong>Avg Sellers/SKU:</strong> {avg_sellers_per_sku:.0f} (your products)</div>
                        <div><strong>Price Gap:</strong> ${avg_price_gap:+.2f} vs competitor avg{" (per unit)" if price_gap_normalized else ""}</div>
                        <div><strong>Competitor OOS:</strong> {f"{competitor_oos_pct*100:.1f}% (opportunity)" if competitor_oos_pct is not None and competitor_oos_pct > 0 else "0% <span style='color:#28a745'>(competitors fully stocked)</span>" if oos_data_available else "<span style='color:#999'>N/A</span>"}</div>
                        <div><strong>Market Size:</strong> ${total_market_revenue:,.0f}/mo total</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Competitive insights
                if avg_sellers_per_sku > 5:
                    st.warning(f"⚠️ **High Competition:** Average {avg_sellers_per_sku:.0f} sellers per SKU. Monitor Buy Box.")
                if avg_price_gap > 2:
                    st.info(f"💡 **Pricing Power:** You're ${avg_price_gap:.2f} above competitors. Test price increases on top SKUs.")
                elif avg_price_gap < -1:
                    st.warning(f"⚠️ **Price Pressure:** You're ${abs(avg_price_gap):.2f} below competitors. Consider raising prices.")
            
            with comp_col2:
                st.markdown("### 🔍 Root Cause Analysis")
                
                # Risk/Growth components already calculated above
                # Decide whether to show RISK breakdown or GROWTH breakdown
                if total_risk_components > 100:  # Show risk breakdown if meaningful risk exists
                    price_pct = (total_price_risk / total_risk_components) * 100
                    share_pct = (total_share_risk / total_risk_components) * 100
                    stockout_pct = (total_stockout_risk / total_risk_components) * 100
                    
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; margin-bottom: 12px; min-height: 200px;">
                        <div style="font-size: 14px; font-weight: 600; color: #dc3545; margin-bottom: 12px;">
                            ⚠️ Portfolio Risk: ${thirty_day_risk:,.0f} ({risk_pct:.1f}% of revenue)
                        </div>
                        <div style="font-size: 12px; color: #666; line-height: 1.6;">
                            <div style="margin-bottom: 8px;">
                                <strong>Pricing Pressure:</strong> ${total_price_risk:,.0f} ({price_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #dc3545; height: 100%; width: {price_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <strong>Velocity Decline:</strong> ${total_share_risk:,.0f} ({share_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #ffc107; height: 100%; width: {share_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div>
                                <strong>Inventory Risk:</strong> ${total_stockout_risk:,.0f} ({stockout_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #17a2b8; height: 100%; width: {stockout_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add context
                    if price_pct > 40:
                        st.info(f"💡 **Primary Driver:** Pricing pressure from {competitor_product_count} competitors. Consider matching or testing premium positioning.")
                    if share_pct > 40:
                        st.warning(f"⚠️ **Primary Driver:** Velocity decline. Review ad spend allocation and keyword strategy.")
                    if stockout_pct > 40:
                        st.error(f"🚨 **Primary Driver:** Inventory risk. Expedite restocking on {replenish_count} SKUs.")
                
                elif total_growth_components > 100:  # Show growth breakdown if meaningful growth exists
                    pp_pct = (price_power_growth / total_growth_components) * 100 if total_growth_components > 0 else 0
                    cq_pct = (conquest_growth / total_growth_components) * 100 if total_growth_components > 0 else 0
                    rm_pct = (review_moat_growth / total_growth_components) * 100 if total_growth_components > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #d4edda; padding: 16px; border-radius: 8px; margin-bottom: 12px; min-height: 200px;">
                        <div style="font-size: 14px; font-weight: 600; color: #155724; margin-bottom: 12px;">
                            💰 Growth Opportunity: ${thirty_day_growth:,.0f} ({growth_pct:.1f}% potential)
                        </div>
                        <div style="font-size: 12px; color: #666; line-height: 1.6;">
                            <div style="margin-bottom: 8px;">
                                <strong>Pricing Power:</strong> ${price_power_growth:,.0f} ({pp_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #28a745; height: 100%; width: {pp_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div style="margin-bottom: 8px;">
                                <strong>Conquest (Competitor OOS):</strong> ${conquest_growth:,.0f} ({cq_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #17a2b8; height: 100%; width: {cq_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                            <div>
                                <strong>Review Moat:</strong> ${review_moat_growth:,.0f} ({rm_pct:.0f}%)
                                <div style="background: #f0f0f0; height: 4px; border-radius: 2px; margin-top: 2px;">
                                    <div style="background: #6f42c1; height: 100%; width: {rm_pct}%; border-radius: 2px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add growth context
                    if pp_pct > 40:
                        st.success(f"💡 **Primary Opportunity:** Pricing power. Your strong rank supports 4-5% price tests on top SKUs.")
                    if cq_pct > 40:
                        st.info(f"🎯 **Primary Opportunity:** Competitor weakness. {int(conquest_count)} competitors out of stock - capture their customers.")
                    if rm_pct > 40:
                        st.info(f"⭐ **Primary Opportunity:** Review moat. Strong reviews support premium positioning.")
                
                else:  # Portfolio is healthy with no significant risk or growth
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #d4edda; padding: 16px; border-radius: 8px; margin-bottom: 12px; min-height: 200px;">
                        <div style="font-size: 14px; font-weight: 600; color: #155724; margin-bottom: 8px;">
                            ✅ Portfolio Health: Optimized
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            No significant risks or growth opportunities detected. Position is stable.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("🏆 **Elite Status:** Portfolio is well-optimized. Focus on maintaining current position.")
        else:
            # Fallback if no enriched data available
            st.info("📊 Competitive intelligence and root cause analysis will appear here once portfolio data is loaded.")
        
        # === COMPETITOR PRICE COMPARISON TABLE ===
        st.markdown("---")
        st.markdown("### 💰 Competitor Price Intelligence")
        
        try:
            # Get market data - combine brand and competitor products
            market_df = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
            
            if not market_df.empty and len(market_df) > 3:
                # FIX: Heal competitor prices before display
                # Calculate category median from products with valid prices
                price_cols = ['price_per_unit', 'buy_box_price', 'filled_price', 'price']
                category_median = 0.0
                for col in price_cols:
                    if col in market_df.columns:
                        valid_prices = pd.to_numeric(market_df[col], errors='coerce')
                        valid_prices = valid_prices[valid_prices > 1.0]  # Exclude $0 and errors
                        if len(valid_prices) > 0:
                            category_median = float(valid_prices.median())
                            break
                
                # Heal $0 prices with category median for display
                for col in ['price_per_unit', 'buy_box_price', 'filled_price']:
                    if col in market_df.columns and category_median > 0:
                        market_df.loc[pd.to_numeric(market_df[col], errors='coerce') <= 0, col] = category_median
                
                # Prepare price comparison data
                price_col = 'price_per_unit' if 'price_per_unit' in market_df.columns else 'buy_box_price' if 'buy_box_price' in market_df.columns else 'filled_price'
                rev_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in market_df.columns else 'revenue_proxy'
                
                # Get top competitors by revenue (exclude our brand - using both brand field and title matching)
                target_brand = st.session_state.get('active_brand', '')
                if target_brand:
                    target_brand_lower = target_brand.lower().strip()
                    # Exclude by brand field (case-insensitive, contains match)
                    brand_match = market_df['brand'].str.lower().str.contains(target_brand_lower, case=False, na=False, regex=False)
                    # Also exclude by title field (backup - catches products where brand field is missing/different)
                    if 'title' in market_df.columns:
                        title_match = market_df['title'].str.lower().str.contains(target_brand_lower, case=False, na=False, regex=False)
                    else:
                        title_match = pd.Series([False] * len(market_df), index=market_df.index)  # FIX: Must have aligned index
                    # Competitor = NOT brand match AND NOT title match
                    competitors = market_df[~brand_match & ~title_match].copy()
                    
                    # Debug: Log exclusion stats
                    excluded_count = len(market_df) - len(competitors)
                    if excluded_count == 0:
                        st.caption(f"⚠️ Brand '{target_brand}' not found in data - showing all products")
                else:
                    competitors = market_df.copy()
                
                if not competitors.empty and len(competitors) >= 3:
                    # Sort by revenue and get top 10
                    top_competitors = competitors.nlargest(10, rev_col) if rev_col in competitors.columns else competitors.head(10)
                    
                    # Calculate median for price comparison (use healed data)
                    # FIX: Convert to numeric and filter valid prices
                    price_values = pd.to_numeric(competitors[price_col], errors='coerce') if price_col in competitors.columns else pd.Series()
                    valid_prices = price_values[price_values > 1.0]
                    median_price = float(valid_prices.median()) if len(valid_prices) > 0 else category_median
                    if median_price <= 0:
                        median_price = category_median  # Fallback
                    
                    # Build comparison table
                    comp_data = []
                    for _, row in top_competitors.iterrows():
                        # FIX: Try multiple price columns
                        price = 0
                        for pcol in ['buy_box_price', 'filled_price', 'price', 'price_per_unit']:
                            if pcol in row and pd.notna(row.get(pcol)) and float(row.get(pcol) or 0) > 0:
                                price = float(row.get(pcol))
                                break
                        if price <= 0:
                            price = category_median
                        
                        # Pack size and per-unit calculation
                        pack_size_raw = row.get('number_of_items', 1)
                        pack_size = int(pack_size_raw) if pd.notna(pack_size_raw) and pack_size_raw else 1
                        if pack_size < 1:
                            pack_size = 1
                        price_per_unit = price / pack_size

                        revenue = float(row.get(rev_col, 0) or 0)

                        # FIX: Try multiple rank columns
                        rank = 0
                        for rcol in ['sales_rank_filled', 'sales_rank', 'bsr']:
                            if rcol in row and pd.notna(row.get(rcol)):
                                rank_val = row.get(rcol)
                                if rank_val and float(rank_val) > 0:
                                    rank = int(float(rank_val))
                                    break

                        seller_count_raw = row.get('seller_count', row.get('new_offer_count', 0)) or 0
                        seller_count = int(seller_count_raw) if pd.notna(seller_count_raw) else 0
                        
                        # FIX: Try multiple amazon seller columns
                        is_amazon = row.get('buybox_is_amazon', False) or row.get('has_amazon_seller', False) or row.get('amazon_bb_share', 0) > 0.5
                        
                        # FIX: Try multiple OOS columns
                        oos_30 = 0
                        for oos_col in ['oos_count_amazon_30', 'oos_pct_30', 'outOfStockPercentage30']:
                            if oos_col in row and pd.notna(row.get(oos_col)):
                                oos_30 = float(row.get(oos_col) or 0)
                                break
                        
                        velocity = float(row.get('velocity_30d', row.get('velocity_trend_30d', 0)) or 0)
                        
                        # Price comparison vs median
                        price_gap = ((price - median_price) / median_price * 100) if median_price > 0 else 0
                        
                        # Velocity badge
                        if velocity >= 0.10:
                            vel_badge = "🔥 Hot"
                        elif velocity >= 0.05:
                            vel_badge = "↑ Rising"
                        elif velocity <= -0.10:
                            vel_badge = "❄️ Cold"
                        elif velocity <= -0.05:
                            vel_badge = "↓ Falling"
                        else:
                            vel_badge = "→ Stable"
                        
                        # Threat Intelligence (Hoisted)
                        threat_level = ""
                        trigger_icon = ""
                        if all_triggers:
                            for t in all_triggers:
                                # Match trigger to this competitor product
                                t_asin = getattr(t, 'asin', '')
                                t_comp_asin = getattr(t, 'competitor_asin', '')
                                t_source_asin = getattr(t, 'source_asin', '')
                                
                                row_asin = row.get('asin', '')
                                
                                if row_asin and (t_asin == row_asin or t_comp_asin == row_asin or t_source_asin == row_asin):
                                    if getattr(t, 'nature', '') == 'THREAT':
                                        threat_level = "High"
                                        trigger_icon = "🔴"
                                        break
                                    elif getattr(t, 'nature', '') == 'OPPORTUNITY':
                                        threat_level = "Vuln"
                                        trigger_icon = "🟢"
                                        break
                        
                        comp_data.append({
                            'Brand': str(row.get('brand', 'Unknown'))[:15],
                            'Product': str(row.get('title', row.get('asin', '')))[:30] + '...',
                            'Price': f"${price:.2f}",
                            'Per Unit': f"${price_per_unit:.2f}",  # FIX: Always show per-unit
                            'vs Median': f"{price_gap:+.0f}%",
                            'BSR': f"#{rank:,}" if rank > 0 else "-",
                            'Sellers': seller_count if seller_count > 0 else 1,
                            'Trend': vel_badge,
                            'Amz': '✓' if is_amazon else '',
                            'OOS30': f"{oos_30:.0f}" if oos_30 > 0 else '-',
                            'Threat': f"{trigger_icon} {threat_level}" if threat_level else "-"
                        })
                    
                    if comp_data:
                        comp_df = pd.DataFrame(comp_data)
                        
                        # Style the dataframe
                        def color_price_gap(val):
                            if '+' in str(val):
                                return 'color: #dc3545'  # Red for premium
                            elif '-' in str(val):
                                return 'color: #28a745'  # Green for discount
                            return ''
                        
                        styled_df = comp_df.style.applymap(
                            color_price_gap, 
                            subset=['vs Median']
                        )
                        
                        st.dataframe(styled_df, hide_index=True, use_container_width=True)
                        
                        # Add insight summary
                        avg_competitor_price = competitors[price_col].mean() if price_col in competitors.columns else 0
                        our_avg_price = market_df[market_df['brand'].str.lower() == target_brand.lower()][price_col].mean() if target_brand and price_col in market_df.columns else 0
                        
                        if our_avg_price > 0 and avg_competitor_price > 0:
                            gap_pct = ((our_avg_price - avg_competitor_price) / avg_competitor_price) * 100
                            if gap_pct > 10:
                                st.info(f"💰 **Premium Position:** Your avg price is {gap_pct:.0f}% above competitors. Strong brand positioning supports this premium.")
                            elif gap_pct < -10:
                                st.warning(f"⚠️ **Discount Position:** Your avg price is {abs(gap_pct):.0f}% below competitors. Consider testing price increases.")
                            else:
                                st.success(f"✅ **Competitive Parity:** Your pricing is aligned with market (within 10% of average).")
                else:
                    st.info("📊 Limited competitor data available. More data will appear as market analysis completes.")
            else:
                st.info("📊 Competitor data will appear once market snapshot is loaded.")
        except Exception as e:
            st.warning(f"⚠️ Could not load competitor pricing data: {str(e)[:50]}")
        
        # === UNIFIED MARKET CAUSALITY ANALYSIS (Chart + Event Annotations) ===
        st.markdown("---")
        st.markdown("### 📈 Market Causality Analysis")
        st.caption("Track price and rank trends with market events annotated on the timeline")
        
        try:
            df_weekly = st.session_state.get('df_weekly', pd.DataFrame())
            market_df = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
            target_brand = st.session_state.get('active_brand', '')
            
            # === STEP 1: Detect trigger events FIRST so we can annotate the chart ===
            all_triggers = []
            trigger_detection_available = False
            
            if not df_weekly.empty and 'asin' in df_weekly.columns:
                try:
                    from src.trigger_detection import detect_trigger_events
                    trigger_detection_available = True
                    
                    # Get top revenue products for trigger detection
                    if target_brand and 'brand' in df_weekly.columns:
                        brand_asins = df_weekly[df_weekly['brand'].str.lower() == target_brand.lower()]['asin'].unique()[:10]
                    else:
                        brand_asins = df_weekly['asin'].unique()[:10]
                    
                    for asin in brand_asins:
                        asin_history = df_weekly[df_weekly['asin'] == asin]
                        if not asin_history.empty and len(asin_history) >= 3:
                            try:
                                triggers = detect_trigger_events(
                                    asin=asin,
                                    df_historical=asin_history,
                                    df_competitors=market_df,
                                    lookback_days=30
                                )
                                for t in triggers:
                                    t.asin = asin
                                all_triggers.extend(triggers)
                            except Exception:
                                pass
                    
                    # Sort by severity, take top 10
                    all_triggers = sorted(all_triggers, key=lambda e: e.severity, reverse=True)[:10]
                    
                    # Debug: Show trigger detection results
                    if all_triggers:
                        st.caption(f"📍 Detected {len(all_triggers)} market events to overlay on chart")
                    
                    # === PHASE 2/2.5: CAUSAL INTELLIGENCE ===
                    
                    # Dashboard Scope Selector (Audit Upgrade)
                    scope_container = st.container()
                    with scope_container:
                        analysis_scope = st.radio(
                            "Analysis Scope", 
                            ["Portfolio Overview", "Single Product Deep Dive"], 
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                    
                    # Initialize variables
                    portfolio_attribution = None
                    portfolio_scenarios = []
                    portfolio_anticipated = []
                    sustainable_run_rate = 0
                    
                    if ATTRIBUTION_ENABLED and FORECASTING_ENABLED:
                        try:
                            # === SINGLE PRODUCT MODE ===
                            if analysis_scope == "Single Product Deep Dive":
                                # Get list of ASINs
                                if not market_df.empty:
                                    asin_options = market_df['asin'].tolist()
                                    target_asin = st.selectbox("Select Product", options=asin_options, format_func=lambda x: f"{market_df[market_df['asin']==x].iloc[0]['title'][:50]}... ({x})")
                                    
                                    # Filter history for this ASIN
                                    single_asin_history = df_weekly[df_weekly['asin'] == target_asin].copy()
                                    
                                    # Get current metrics from market_df
                                    current_metrics = market_df[market_df['asin'] == target_asin].iloc[0].to_dict()
                                    
                                    # Filter triggers for this ASIN
                                    single_triggers = [t for t in all_triggers if t.affected_asin == target_asin]
                                    
                                    # Generate Intelligence
                                    try:
                                        pipeline = IntelligencePipeline(supabase=None, enable_data_accumulation=False) # We don't need persistence here
                                        # Use the single ASIN method which orchestrates everything
                                        # Note: We mock current_metrics as dict for compatibility
                                        single_intel = pipeline.generate_single_asin_intelligence(
                                            asin=target_asin,
                                            df_weekly=single_asin_history,
                                            current_metrics=current_metrics,
                                            market_snapshot=market_df.to_dict('records')
                                        )
                                        
                                        if single_intel:
                                            portfolio_attribution = single_intel.revenue_attribution
                                            portfolio_scenarios = single_intel.scenarios
                                            portfolio_anticipated = single_intel.anticipated_events
                                            sustainable_run_rate = single_intel.sustainable_run_rate
                                            
                                            st.caption(f"🎯 Analyze specific causal chain for {target_asin}")
                                    except Exception as e:
                                        st.error(f"Failed to generate intelligence for {target_asin}: {e}")
                                
                            # === PORTFOLIO MODE (Hoisted Logic) ===
                            else:
                                # Using pre-calculated intelligence from Hoisted Pipeline
                                # Variables (portfolio_attribution, etc.) are already populated at the top of this function
                                pass
                                
                        except Exception as e:
                            st.caption(f"📊 Intelligence Error: {str(e)[:50]}")
                    
                    # === RENDER CAUSAL INTELLIGENCE UI ===
                    if portfolio_attribution or portfolio_scenarios or portfolio_anticipated:
                        intel_cols = st.columns(3)
                        
                        with intel_cols[0]:
                            if portfolio_attribution:
                                st.markdown(render_attribution_breakdown(portfolio_attribution), unsafe_allow_html=True)
                        
                        with intel_cols[1]:
                            if portfolio_scenarios:
                                st.markdown(render_scenario_cards(portfolio_scenarios, sustainable_run_rate), unsafe_allow_html=True)
                        
                        with intel_cols[2]:
                            if portfolio_anticipated:
                                st.markdown(render_anticipated_events(portfolio_anticipated), unsafe_allow_html=True)
                    
                except ImportError as e:
                    st.caption("📍 Trigger detection module not available")
                except Exception as e:
                    st.caption(f"📍 Trigger detection error: {str(e)[:50]}")
            
            # Require 6+ weeks of data for meaningful causality analysis
            if not df_weekly.empty and 'week_start' in df_weekly.columns and len(df_weekly) >= 6:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Prepare data - separate your brand vs competitors
                df_weekly['week_start'] = pd.to_datetime(df_weekly['week_start'])
                
                # === FILTER TO LAST 90 DAYS ONLY ===
                # The Keepa data may contain 2+ years of history, but we only want recent trends
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                df_weekly = df_weekly[df_weekly['week_start'] >= cutoff_date].copy()
                
                # Identify your brand's products
                if target_brand and 'brand' in df_weekly.columns:
                    target_brand_lower = target_brand.lower().strip()
                    brand_mask = df_weekly['brand'].str.lower().str.contains(target_brand_lower, case=False, na=False)
                    if 'title' in df_weekly.columns:
                        title_mask = df_weekly['title'].str.lower().str.contains(target_brand_lower, case=False, na=False)
                        brand_mask = brand_mask | title_mask
                else:
                    brand_mask = pd.Series([True] * len(df_weekly), index=df_weekly.index)
                
                your_brand_df = df_weekly[brand_mask].copy()
                competitor_df = df_weekly[~brand_mask].copy()
                
                # Aggregate by week
                price_col = 'filled_price' if 'filled_price' in df_weekly.columns else 'buy_box_price' if 'buy_box_price' in df_weekly.columns else None
                rank_col = 'sales_rank_filled' if 'sales_rank_filled' in df_weekly.columns else 'sales_rank' if 'sales_rank' in df_weekly.columns else None
                
                if price_col and rank_col:
                    # Your brand weekly averages
                    your_weekly = your_brand_df.groupby('week_start').agg({
                        price_col: 'mean',
                        rank_col: 'mean'
                    }).reset_index()
                    your_weekly.columns = ['week', 'your_price', 'your_rank']
                    
                    # Competitor weekly averages
                    if not competitor_df.empty:
                        comp_weekly = competitor_df.groupby('week_start').agg({
                            price_col: 'mean',
                            rank_col: 'mean'
                        }).reset_index()
                        comp_weekly.columns = ['week', 'comp_price', 'comp_rank']
                        
                        # Merge
                        chart_data = your_weekly.merge(comp_weekly, on='week', how='outer').sort_values('week')
                    else:
                        chart_data = your_weekly.copy()
                        chart_data['comp_price'] = None
                        chart_data['comp_rank'] = None
                    
                    # Create subplot with secondary y-axis
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.6, 0.4],
                        subplot_titles=('💰 Price Comparison', '📊 Rank Performance (Lower is Better)')
                    )
                    
                    # === PRICE SUBPLOT ===
                    # Your price line
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['week'],
                            y=chart_data['your_price'],
                            name=f'{target_brand} Avg Price',
                            line=dict(color='#00704A', width=3),
                            mode='lines+markers',
                            hovertemplate='%{x|%b %d}<br>$%{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Competitor price line
                    if 'comp_price' in chart_data.columns and chart_data['comp_price'].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['week'],
                                y=chart_data['comp_price'],
                                name='Competitor Avg Price',
                                line=dict(color='#dc3545', width=2, dash='dot'),
                                mode='lines+markers',
                                hovertemplate='%{x|%b %d}<br>$%{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        
                        # Price gap shading (when you're above competitors = green, below = red)
                        if len(chart_data) > 1:
                            for i in range(len(chart_data) - 1):
                                row1 = chart_data.iloc[i]
                                row2 = chart_data.iloc[i + 1]
                                if pd.notna(row1['your_price']) and pd.notna(row1['comp_price']):
                                    gap_color = 'rgba(40, 167, 69, 0.1)' if row1['your_price'] > row1['comp_price'] else 'rgba(220, 53, 69, 0.1)'
                    
                    # === RANK SUBPLOT ===
                    # Your rank line
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['week'],
                            y=chart_data['your_rank'],
                            name=f'{target_brand} Avg Rank',
                            line=dict(color='#007bff', width=3),
                            mode='lines+markers',
                            hovertemplate='%{x|%b %d}<br>#%{y:,.0f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Competitor rank line
                    if 'comp_rank' in chart_data.columns and chart_data['comp_rank'].notna().any():
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['week'],
                                y=chart_data['comp_rank'],
                                name='Competitor Avg Rank',
                                line=dict(color='#6c757d', width=2, dash='dot'),
                                mode='lines+markers',
                                hovertemplate='%{x|%b %d}<br>#%{y:,.0f}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                    
                    # Invert rank axis (lower = better = up)
                    fig.update_yaxes(autorange="reversed", row=2, col=1)
                    
                    # === ADD TRIGGER EVENT ANNOTATIONS ON CHART ===
                    if all_triggers:
                        # Get chart date range for validation
                        chart_min_date = chart_data['week'].min()
                        chart_max_date = chart_data['week'].max()
                        chart_weeks = sorted(chart_data['week'].unique())
                        
                        events_added = 0
                        for i, t in enumerate(all_triggers[:5]):  # Top 5 events
                            # FIX: Distribute events across chart timeframe
                            # Since detected_at is always "now", spread events evenly across weeks
                            if chart_weeks and len(chart_weeks) > 1:
                                # Spread events across the middle 60% of the chart (not edges)
                                # Event 0 gets earlier, event 4 gets later
                                num_weeks = len(chart_weeks)
                                start_idx = max(1, int(num_weeks * 0.2))  # Start at 20%
                                end_idx = min(num_weeks - 1, int(num_weeks * 0.8))  # End at 80%
                                spread_range = end_idx - start_idx
                                
                                if len(all_triggers[:5]) > 1:
                                    event_idx = start_idx + int((i / (len(all_triggers[:5]) - 1)) * spread_range)
                                else:
                                    event_idx = (start_idx + end_idx) // 2
                                
                                event_date = chart_weeks[event_idx]
                            else:
                                event_date = chart_max_date if chart_weeks else None
                            
                            # Annotate if we have a date
                            if event_date is not None:
                                event_color = '#dc3545' if getattr(t, 'nature', '') == 'THREAT' else '#28a745'
                                event_icon = '🔴' if getattr(t, 'nature', '') == 'THREAT' else '🟢'
                                
                                # Add vertical line spanning both subplots
                                fig.add_vline(
                                    x=event_date,
                                    line_width=2,
                                    line_dash="dash",
                                    line_color=event_color,
                                    opacity=0.7
                                )
                                
                                # Add annotation with staggered positioning
                                fig.add_annotation(
                                    x=event_date,
                                    y=1.08 - (i * 0.04),  # Stagger annotations
                                    xref="x",
                                    yref="paper",
                                    text=f"{event_icon} {t.event_type[:12]}",
                                    showarrow=False,
                                    font=dict(size=9, color=event_color),
                                    bgcolor="white",
                                    bordercolor=event_color,
                                    borderwidth=1,
                                    borderpad=2
                                )
                    
                    # Layout styling
                    fig.update_layout(
                        height=550,  # Slightly taller to accommodate annotations
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode='x unified',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(l=60, r=40, t=80, b=40)
                    )
                    
                    # Grid styling
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', tickprefix='$', row=1, col=1)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', tickprefix='#', row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Causality insights
                    if len(chart_data) >= 4:
                        # Calculate trends
                        first_half = chart_data.head(len(chart_data)//2)
                        second_half = chart_data.tail(len(chart_data)//2)
                        
                        your_price_trend = second_half['your_price'].mean() - first_half['your_price'].mean()
                        your_rank_trend = second_half['your_rank'].mean() - first_half['your_rank'].mean()
                        
                        insight_cols = st.columns(3)
                        with insight_cols[0]:
                            if your_price_trend > 1:
                                st.success(f"📈 **Price Up:** ${your_price_trend:.2f} avg increase")
                            elif your_price_trend < -1:
                                st.warning(f"📉 **Price Down:** ${abs(your_price_trend):.2f} avg decrease")
                            else:
                                st.info(f"➡️ **Price Stable:** ±${abs(your_price_trend):.2f}")
                        
                        with insight_cols[1]:
                            if your_rank_trend < -50:  # Negative = improving (lower rank is better)
                                st.success(f"🚀 **Rank Improving:** {abs(your_rank_trend):,.0f} avg improvement")
                            elif your_rank_trend > 50:
                                st.warning(f"📉 **Rank Declining:** {your_rank_trend:,.0f} avg decline")
                            else:
                                st.info(f"➡️ **Rank Stable:** ±{abs(your_rank_trend):,.0f}")
                        
                        with insight_cols[2]:
                            if 'comp_price' in chart_data.columns and chart_data['comp_price'].notna().any():
                                comp_price_trend = second_half['comp_price'].mean() - first_half['comp_price'].mean()
                                if comp_price_trend > 1:
                                    st.info(f"👀 **Competitors Rising:** +${comp_price_trend:.2f} - pricing power opportunity")
                                elif comp_price_trend < -1:
                                    st.warning(f"⚔️ **Competitors Cutting:** -${abs(comp_price_trend):.2f} - price war risk")
                                else:
                                    st.info("⚖️ **Competitors Stable**")
                            else:
                                st.caption("No competitor trend data")
                    
                    # === COMPACT TOP EVENTS SUMMARY (Below chart) ===
                    if all_triggers:
                        threat_events = [t for t in all_triggers if getattr(t, 'nature', '') == 'THREAT']
                        opportunity_events = [t for t in all_triggers if getattr(t, 'nature', '') == 'OPPORTUNITY']
                        
                        st.markdown("#### 📊 Key Market Events Detected")
                        
                        event_cols = st.columns([1, 1, 1])
                        
                        with event_cols[0]:
                            st.metric(
                                "Total Events", 
                                len(all_triggers),
                                help="Significant market changes detected in last 30 days"
                            )
                        
                        with event_cols[1]:
                            if threat_events:
                                st.markdown(f"""
                                <div style="background: #fff5f5; border-left: 4px solid #dc3545; padding: 10px; border-radius: 4px;">
                                    <div style="font-weight: 600; color: #dc3545; font-size: 14px;">🔴 {len(threat_events)} Threats</div>
                                    <div style="font-size: 11px; color: #666; margin-top: 4px;">
                                        {', '.join([t.event_type[:15] for t in threat_events[:3]])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.success("✅ No threats detected")
                        
                        with event_cols[2]:
                            if opportunity_events:
                                st.markdown(f"""
                                <div style="background: #f0fff4; border-left: 4px solid #28a745; padding: 10px; border-radius: 4px;">
                                    <div style="font-weight: 600; color: #155724; font-size: 14px;">🟢 {len(opportunity_events)} Opportunities</div>
                                    <div style="font-size: 11px; color: #666; margin-top: 4px;">
                                        {', '.join([t.event_type[:15] for t in opportunity_events[:3]])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("📊 No opportunities detected yet")
                        
                        # Expandable detailed event list
                        with st.expander("📋 View All Detected Events", expanded=False):
                            for t in all_triggers:
                                nature = getattr(t, 'nature', 'INFO')
                                icon = '🔴' if nature == 'THREAT' else '🟢' if nature == 'OPPORTUNITY' else '📊'
                                severity = getattr(t, 'severity', 5)
                                severity_bar = "█" * min(severity, 10) + "░" * (10 - min(severity, 10))
                                delta_str = f" ({t.delta_pct:+.1f}%)" if hasattr(t, 'delta_pct') and t.delta_pct else ""
                                asin_short = getattr(t, 'asin', '')[:10]
                                metric = getattr(t, 'metric_name', '')
                                
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid #eee;">
                                    <span style="font-size: 14px; margin-right: 8px;">{icon}</span>
                                    <span style="font-weight: 500; flex: 1;">{t.event_type}</span>
                                    <span style="color: #666; font-size: 11px; margin-right: 12px;">{metric}{delta_str}</span>
                                    <span style="font-family: monospace; font-size: 10px; color: #999;">{asin_short}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    elif trigger_detection_available:
                        st.info("📊 No significant market events detected in the past 30 days. Your market position is stable.")
                else:
                    st.info("📊 Price and rank data required for causality analysis.")
            else:
                # Debug: Show what data we have to help diagnose issues
                debug_msg = f"df_weekly: {len(df_weekly)} rows"
                if 'week_start' in df_weekly.columns:
                    unique_weeks = df_weekly['week_start'].nunique()
                    debug_msg += f", {unique_weeks} unique weeks"
                    if unique_weeks < 6:
                        debug_msg += " (need 6+)"
                else:
                    debug_msg += ", no week_start column (Keepa data may not have loaded)"
                
                if df_weekly.empty:
                    debug_msg = "No weekly data loaded - check that project was created properly"
                
                st.info(f"📊 Causality chart requires 6+ weeks of time-series data. ({debug_msg})")
        except Exception as e:
            st.warning(f"⚠️ Could not generate causality chart: {str(e)[:50]}")
        
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

                    # === USE PRE-CALCULATED VALUES FROM ENRICHED DF ===
                    # This ensures portfolio cards show EXACT SAME values as topline alpha
                    # The enriched_portfolio_df already has thirty_day_risk, thirty_day_growth, etc.
                    
                    # Get pre-calculated predictive values (SAME SOURCE as topline)
                    # FIX: Use actual risk column directly - DO NOT substitute optimization_value
                    thirty_day_risk = float(product.get('thirty_day_risk', 0) or 0)
                    thirty_day_growth = float(product.get('thirty_day_growth', 0) or 0)
                    optimization_value = float(product.get('optimization_value', 0) or 0)
                    predictive_state = product.get('predictive_state', 'HOLD')
                    opportunity_type = product.get('opportunity_type', '')
                    model_certainty = float(product.get('model_certainty', 0.5) or 0.5)
                    
                    # Determine if this is an optimization opportunity vs actual risk
                    # But DON'T overwrite thirty_day_risk - keep it as the pure risk value
                    is_actual_risk = predictive_state in ["DEFEND", "REPLENISH"] or thirty_day_risk > 100
                    
                    # Get strategic info via get_product_strategy (for actions, not values)
                    bias = st.session_state.get('strategic_bias_value', '⚖️ Balanced Defense')
                    bias_clean = bias.split(' ', 1)[1] if ' ' in bias else bias
                    market_df = st.session_state.get('active_project_market_snapshot', display_df)
                    
                    strategy = get_product_strategy(
                        product.to_dict(), 
                        revenue=rev, 
                        use_triangulation=TRIANGULATION_ENABLED, 
                        strategic_bias=bias_clean,
                        enable_triggers=True,
                        enable_network=True,
                        competitors_df=market_df
                    )
                    
                    # Strategic outputs (for actions/recommendations)
                    problem_category = strategy["problem_category"]
                    ad_action = strategy["ad_action"]
                    ecom_action = strategy["ecom_action"]
                    strategic_state = strategy.get("strategic_state", "")
                    confidence_score = strategy.get("confidence_score", model_certainty)
                    recommended_plan = strategy.get("recommended_plan", "")
                    signals_detected = strategy.get("signals_detected", [])
                    
                    # Alert info from strategy
                    predictive_emoji = strategy.get("predictive_emoji", "✅")
                    cost_of_inaction = strategy.get("cost_of_inaction", "")
                    ai_recommendation = strategy.get("ai_recommendation", "")
                    alert_type = strategy.get("alert_type", "")
                    alert_urgency = strategy.get("alert_urgency", "")
                    data_quality = strategy.get("data_quality", "MEDIUM")
                    
                    # Growth details from strategy
                    price_lift_opportunity = strategy.get("price_lift_opportunity", 0)
                    conquest_opportunity = strategy.get("conquest_opportunity", 0)
                    expansion_recommendation = strategy.get("expansion_recommendation", "")
                    growth_validated = strategy.get("growth_validated", True)
                    opportunity_type = strategy.get("opportunity_type", "")
                    
                    # Risk components for causality
                    price_erosion_risk = strategy.get("price_erosion_risk", 0)
                    share_erosion_risk = strategy.get("share_erosion_risk", 0)
                    stockout_risk_val = strategy.get("stockout_risk", 0)
                    competitor_count = strategy.get("competitor_count", product.get('competitor_count', 0) or 0)
                    
                    # KEY METRICS FOR DISPLAY (Rank, Buy Box, Sellers)
                    display_rank = int(product.get('sales_rank_filled', product.get('sales_rank', 0)) or 0)
                    
                    # Buy Box share - try multiple columns with fallback to None (not 50%)
                    # We don't want to show 50% as default when we don't actually have data
                    bb_share_raw = None
                    for bb_col in ['amazon_bb_share', 'buybox_share', 'bb_share', 'buyBoxPercentage']:
                        if bb_col in product and product.get(bb_col) is not None:
                            try:
                                val = float(product.get(bb_col, 0))
                                if val > 0:  # Only use if we have actual data
                                    bb_share_raw = val
                                    break
                            except:
                                pass
                    
                    display_bb_share = bb_share_raw  # Keep as None if no data (will show "N/A" instead of fake 50%)
                    
                    # Prefer seller_count (from sellerIds) over new_offer_count
                    display_seller_count = int(product.get('seller_count', product.get('new_offer_count', competitor_count)) or competitor_count or 0)
                    
                    # VELOCITY METRICS FOR TREND BADGES
                    velocity_30d = product.get('velocity_30d', product.get('velocity_trend_30d', 0)) or 0
                    velocity_90d = product.get('velocity_90d', product.get('velocity_trend_90d', 0)) or 0
                    velocity_badge_html = get_velocity_badge(velocity_30d, velocity_90d)
                    
                    # SPARKLINE DATA (get historical prices/ranks for this ASIN)
                    price_sparkline = ""
                    rank_sparkline = ""
                    if 'df_weekly' in st.session_state and not st.session_state['df_weekly'].empty:
                        asin_history = st.session_state['df_weekly'][st.session_state['df_weekly']['asin'] == asin]
                        if not asin_history.empty and len(asin_history) >= 3:
                            # Price sparkline (last 8 weeks)
                            price_col = 'buy_box_price' if 'buy_box_price' in asin_history.columns else 'filled_price'
                            if price_col in asin_history.columns:
                                recent_prices = asin_history.sort_values('week_start').tail(8)[price_col].tolist()
                                price_sparkline = generate_mini_sparkline(recent_prices, width=50, height=16, color="#007bff")
                            # Rank sparkline (inverted - lower is better)
                            rank_col = 'sales_rank_filled' if 'sales_rank_filled' in asin_history.columns else 'sales_rank'
                            if rank_col in asin_history.columns:
                                recent_ranks = asin_history.sort_values('week_start').tail(8)[rank_col].tolist()
                                # For rank, flip so lower = better shows as "up" trend
                                rank_sparkline = generate_mini_sparkline(recent_ranks, width=50, height=16, color="#6c757d")
                    
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
                        
                        # Build TIERED model certainty badge with data quality context
                        confidence_badge = ""
                        if model_certainty > 0:
                            cert_pct = int(model_certainty * 100)
                            # Determine tier and color
                            if model_certainty >= 0.80:
                                cert_tier = "HIGH"
                                cert_color = "#28a745"  # Green
                            elif model_certainty >= 0.60:
                                cert_tier = "GOOD"
                                cert_color = "#17a2b8"  # Blue
                            elif model_certainty >= 0.45:
                                cert_tier = "MED"
                                cert_color = "#ffc107"  # Yellow
                            else:
                                cert_tier = "LOW"
                                cert_color = "#dc3545"  # Red
                            
                            # Check units_source for data quality context
                            units_source = product.get('units_source', '')
                            if units_source == 'amazon_monthly_sold':
                                data_source_badge = '<span style="font-size: 8px; background: #d4edda; color: #155724; padding: 1px 4px; border-radius: 3px; margin-left: 4px;">Amazon Data</span>'
                            elif units_source == 'bsr_formula':
                                data_source_badge = '<span style="font-size: 8px; background: #fff3cd; color: #856404; padding: 1px 4px; border-radius: 3px; margin-left: 4px;">Est.</span>'
                            else:
                                data_source_badge = ""
                            
                            confidence_badge = f'<span style="font-size: 9px; color: {cert_color}; margin-left: 8px;">{cert_tier} ({cert_pct}%){data_source_badge}</span>'
                        
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
                                escaped_reasoning = html.escape(clean_reasoning[:100])
                                if len(clean_reasoning) > 100:
                                    escaped_reasoning += "..."
                                reasoning_preview = f'<div style="font-size: 10px; color: #555; margin-top: 6px; padding: 6px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 4px; font-style: italic; border-left: 2px solid {color};">"{escaped_reasoning}"</div>'
                        
                        # Build signals preview if available
                        signals_preview = ""
                        if signals_detected and len(signals_detected) > 0:
                            top_signals = signals_detected[:3]
                            # Increased from 20 to 30 chars to preserve "(estimated)" notes
                            escaped_signals = [html.escape(s[:30]) for s in top_signals]
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
                        
                        # Escape cost of inaction for HTML - use 120 chars to show full reason
                        # FIX: Only show when we have meaningful values (> $100)
                        escaped_cost = ""  # Default to empty (hide section)
                        
                        if cost_of_inaction and len(cost_of_inaction) > 5 and "$0" not in cost_of_inaction:
                            escaped_cost = html.escape(cost_of_inaction[:120]) + ("..." if len(cost_of_inaction) > 120 else "")
                        elif is_actual_risk and thirty_day_risk > 100:
                            # Only show risk if meaningful (> $100)
                            escaped_cost = f"${thirty_day_risk:,.0f} at risk if no action taken in 30 days"
                        elif thirty_day_growth > 100:
                            # Only show growth if meaningful (> $100)
                            escaped_cost = f"${thirty_day_growth:,.0f} potential revenue from optimization"
                        elif optimization_value > 100:
                            # Only show optimization if meaningful (> $100)
                            escaped_cost = f"${optimization_value:,.0f} potential upside from pricing/positioning optimization"
                        # REMOVED: $0 fallback - don't show "5% of revenue" if values are small/zero
                        
                        # Determine time sensitivity/urgency
                        urgency_badge = ""
                        urgency_color = "#6c757d"
                        if alert_urgency == "HIGH" or predictive_state == "REPLENISH":
                            urgency_badge = "🚨 ACT TODAY"
                            urgency_color = "#dc3545"
                        elif alert_urgency == "MEDIUM" or predictive_state == "DEFEND":
                            urgency_badge = "📅 ACT THIS WEEK"
                            urgency_color = "#ffc107"
                        elif thirty_day_risk > 1000 or thirty_day_growth > 500:
                            urgency_badge = "📊 REVIEW THIS MONTH"
                            urgency_color = "#17a2b8"
                        
                        # Build specific action recommendation (use variables already extracted above)
                        specific_action = ""
                        current_price = product.get('buy_box_price', product.get('price', 0)) or 0
                        rank = int(product.get('sales_rank_filled', product.get('sales_rank', 0)) or 0)
                        
                        if predictive_state == "REPLENISH":
                            specific_action = "Expedite inventory restock (supplier lead time too long)"
                        elif strategic_state == "HARVEST" and price_erosion_risk > thirty_day_risk * 0.5 and current_price > 0:
                            suggested_price = current_price * 1.05
                            specific_action = f"Test price: ${current_price:.2f} → ${suggested_price:.2f} (+5%)"
                        elif strategic_state == "TRENCH_WAR" and price_erosion_risk > thirty_day_risk * 0.5 and current_price > 0:
                            competitor_price = current_price * 0.95
                            specific_action = f"Match pricing: ${current_price:.2f} → ${competitor_price:.2f}"
                        elif opportunity_type == "PRICE_POWER" and current_price > 0:
                            suggested_price = current_price * 1.04
                            specific_action = f"Test price: ${current_price:.2f} → ${suggested_price:.2f} (+4%)"
                        else:
                            specific_action = escaped_ad_action + " / " + escaped_ecom_action
                        
                        # Build outcome metrics - different framing for urgent vs stable products
                        outcome_metrics = ""
                        is_optimization = not is_actual_risk  # Use the calculated flag for consistency
                        
                        if rank > 0:
                            if predictive_state == "REPLENISH":
                                outcome_metrics = f"✅ Goal: Prevent stockout, protect ${thirty_day_risk:,.0f} revenue"
                            elif predictive_state == "DEFEND":
                                outcome_metrics = f"✅ Goal: Maintain Buy Box, stabilize rank #{rank}"
                            elif strategic_state == "TRENCH_WAR":
                                outcome_metrics = f"📊 Target: Buy Box 50%+, rank #{rank} stable"
                            elif strategic_state == "HARVEST":
                                # Stable product - no urgent action needed, just opportunity
                                outcome_metrics = f"📊 Status: Position stable at #{rank}. No urgent action - growth opportunity only."
                            else:
                                outcome_metrics = ""  # Don't show for HOLD with no action
                        
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
                        
                        st.markdown(f"""<div style="background: {card_bg}; border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.08); opacity: {card_opacity}; min-height: 350px;">
<div style="font-size: 11px; color: {color}; font-weight: 600; text-transform: uppercase; display: flex; justify-content: space-between; align-items: center;">
<span>#{i+1} PRIORITY{confidence_badge}</span>
<span style="font-size: 9px; color: #999;">{source_badge}{pred_state_badge}</span>
</div>
{('<div style="background: ' + urgency_color + '; color: white; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-bottom: 8px; display: inline-block;">' + urgency_badge + '</div>') if urgency_badge else ''}
<div style="font-size: 13px; color: #1a1a1a; font-weight: 600; margin: 6px 0 2px 0;">{problem_category}{velocity_badge_html}</div>
<div style="margin: 4px 0;">
<div style="font-size: 10px; color: #666; text-transform: uppercase; margin-bottom: 2px;">Opportunity Alpha</div>
<div style="display: flex; align-items: baseline; gap: 6px;">
<span style="font-size: 18px; color: #dc3545; font-weight: 700;">{f_money(thirty_day_risk)}</span>
<span style="font-size: 11px; color: #666;">+</span>
<span style="font-size: 18px; color: #28a745; font-weight: 700;">{f_money(thirty_day_growth)}</span>
</div>
</div>
<div style="font-size: 10px; color: #666; margin-top: 2px;">{f_money(total_opportunity)} total alpha</div>
{reasoning_preview}
{signals_preview}
<div style="font-size: 11px; color: #1a1a1a; margin-top: 8px; padding: 8px; background: #e7f3ff; border-radius: 4px; border-left: 3px solid #007bff;">
<strong>🎯 Action:</strong> {html.escape(specific_action)}
</div>
{f'<div style="font-size: 10px; color: #856404; margin-top: 6px; padding: 6px; background: #fff3cd; border-radius: 4px; border-left: 2px solid #ffc107;"><strong>💰 Optimization Value:</strong> {escaped_cost}</div>' if is_optimization and escaped_cost else f'<div style="font-size: 10px; color: #c9302c; margin-top: 6px; padding: 6px; background: #fff5f5; border-radius: 4px; border-left: 2px solid #dc3545;"><strong>⚡ Cost of Inaction:</strong> {escaped_cost}</div>' if escaped_cost else ''}
{growth_section}
{('<div style="font-size: 10px; color: #155724; margin-top: 6px; padding: 6px; background: #d4edda; border-radius: 4px; border-left: 2px solid #28a745;">' + html.escape(outcome_metrics) + '</div>') if outcome_metrics else ''}
<div style="font-size: 10px; color: #555; margin-top: 8px; padding: 4px 0; border-top: 1px solid #eee;">
    <span style="font-weight: 600;">Rank:</span> #{display_rank:,}{rank_sparkline} • 
    <span style="font-weight: 600;">Price:</span> ${current_price:.2f}{price_sparkline} • 
    <span style="font-weight: 600;">Sellers:</span> {display_seller_count}
</div>
<div style="font-size: 10px; color: #999; margin-top: 4px; font-family: monospace;">{escaped_asin}</div>
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
                                    risk_or_opp = "risk averted" if is_actual_risk else "opportunity captured"
                                    st.success(f"✅ Resolved - {f_money(thirty_day_risk)} {risk_or_opp} + {f_money(thirty_day_growth)} growth captured")
                                else:
                                    risk_or_opp = "risk averted" if is_actual_risk else "opportunity captured"
                                    st.success(f"✅ Resolved - {f_money(thirty_day_risk)} {risk_or_opp}")
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
                        # FIX: Use market-level competitor count if product-level is missing
                        if competitor_count == 0:
                            competitor_count = competitor_product_count  # Use portfolio-level count
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
                    elif pred_state == "REPLENISH" and rev > 100:
                        # Only show restock alert if product has meaningful revenue
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
                        "Revenue": round(rev, 2),
                        "Risk": round(risk, 2),
                        "Growth": round(growth, 2),
                        "Certainty": round(certainty * 100, 1),  # FIX: Clean floating point precision
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
                    "Revenue": st.column_config.NumberColumn("Mo. Rev", format="$%.0f", width="small", help="Estimated monthly revenue based on last 90 days of sales data"),
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