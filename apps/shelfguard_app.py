import streamlit as st
import pandas as pd
from openai import OpenAI
from data import get_all_data
from engine import run_weekly_analysis, run_date_range_analysis
from finance import analyze_capital_efficiency, f_money, f_pct
from demo_data import render_asin_upload_ui, get_demo_data, clear_demo_data
from search_to_state_ui import render_discovery_ui, render_project_dashboard, render_project_selector

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])
except Exception:
    openai_client = None

import hashlib
import re

# Pre-compile regex for performance
_METRICS_PATTERN = re.compile(r'\$[\d,]+|\d+\.\d+%|\d+ products')

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

    Cached by data_hash (portfolio metrics), not by date, to avoid
    unnecessary API calls when only the date range changes.

    Performance: Cached results reduce API calls and latency.
    """
    if openai_client is None:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are ShelfGuard's AI strategist, powered by 36 months of historical data analysis. Generate a brief, actionable executive summary for an e-commerce portfolio manager.

Rules:
- Be direct and specific. No fluff.
- Lead with the most urgent issue.
- Quantify everything ($ amounts, counts, percentages).
- When relevant, reference historical context (e.g., "lowest margin in 36 months" or "top 10% revenue historically").
- End with one clear action to take this week.
- Keep it under 100 words.
- Use plain language, not jargon."""
                },
                {
                    "role": "user", 
                    "content": f"""Here's this week's portfolio data:

{portfolio_summary}

Generate a brief executive summary. What's the #1 priority and why?"""
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

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

# --- DATA SOURCE TOGGLE ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Source")

# Check if demo data is loaded
demo_data = get_demo_data()
is_demo_mode = demo_data is not None

if is_demo_mode:
    st.sidebar.success("🚀 **Demo Mode Active**")
    st.sidebar.caption(f"{len(demo_data['asin'].unique())} ASINs loaded")
    if st.sidebar.button("❌ Exit Demo Mode"):
        clear_demo_data()
        st.rerun()
else:
    with st.sidebar.expander("🚀 Try Your Own ASINs", expanded=False):
        render_asin_upload_ui()

st.sidebar.markdown("---")

# === TOP LEVEL NAVIGATION ===
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 Current Dashboard", "🔍 Market Discovery", "📂 My Projects"])

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
    # Existing dashboard (requires data)
    try:
        # 2. DATA INGESTION
        if is_demo_mode:
            df_raw = demo_data.copy()
            st.info("🚀 **Demo Mode:** Analyzing your uploaded ASINs. [Exit Demo Mode via sidebar]")
        else:
            with st.spinner("🔄 Loading Global Data..."):
                df_raw = get_all_data()

        if df_raw.empty:
            st.warning("⚠️ No data found. Please upload ASINs or check your Supabase connection.")
            st.stop()

        # Normalize dates and handle empty states (optimized)
        df_raw["week_start"] = pd.to_datetime(df_raw["week_start"], utc=True)
        # Vectorized operation - get unique weeks directly
        unique_weeks = df_raw["week_start"].dropna().dt.date.unique()
        all_weeks = sorted(unique_weeks, reverse=True)

        if not all_weeks:
            st.error("❌ No valid weeks found in dataset.")
            st.stop()
    
        # === DATE RANGE SELECTOR ===
        st.sidebar.markdown("### 📅 Date Range")
    
        # View mode selector
        view_mode = st.sidebar.radio(
            "View Mode",
            ["📊 Weekly View", "📈 Date Range"],
            index=0,
            help="Weekly View: Single week snapshot | Date Range: Aggregate across multiple weeks"
        )
    
        selected_week = None
        date_range = None
    
        if view_mode == "📊 Weekly View":
            # Default to most recent week
            default_week_idx = 0
            selected_week = st.sidebar.selectbox(
                "Select Week",
                all_weeks,
                index=default_week_idx,
                format_func=lambda x: f"{x.strftime('%b %d, %Y')} (Most Recent)" if x == all_weeks[0] else x.strftime('%b %d, %Y')
            )
        else:
            # Date Range View
            min_date = min(all_weeks)
            max_date = max(all_weeks)
        
            # Quick presets
            st.sidebar.markdown("**Quick Presets:**")
            col1, col2 = st.sidebar.columns(2)
        
            preset = None
            with col1:
                if st.button("Last 4 Weeks", use_container_width=True):
                    preset = "4w"
                if st.button("Last 3 Months", use_container_width=True):
                    preset = "3m"
            with col2:
                if st.button("Last 6 Months", use_container_width=True):
                    preset = "6m"
                if st.button("YTD", use_container_width=True):
                    preset = "ytd"
        
            # Calculate preset dates
            if preset == "4w":
                end_date = max_date
                start_date = max_date - pd.Timedelta(days=28)
            elif preset == "3m":
                end_date = max_date
                start_date = max_date - pd.Timedelta(days=90)
            elif preset == "6m":
                end_date = max_date
                start_date = max_date - pd.Timedelta(days=180)
            elif preset == "ytd":
                end_date = max_date
                start_date = pd.Timestamp(max_date.year, 1, 1).date()
            else:
                start_date = min_date
                end_date = max_date
        
            # Date range picker
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(start_date, end_date),
                min_value=min_date,
                max_value=max_date,
                help="Select start and end dates to analyze aggregate performance"
            )
        
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                # Single date selected, use as end date
                end_date = date_range if date_range else max_date
                start_date = end_date - pd.Timedelta(days=28)  # Default to 4 weeks back
    
        # Initialize chat state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_open" not in st.session_state:
            st.session_state.chat_open = False
    
        # 3. ANALYSIS EXECUTION
        with st.spinner("🧠 Executing Predictive Intelligence..."):
            if view_mode == "📊 Weekly View":
                res = run_weekly_analysis(df_raw, selected_week)
            else:
                # Date Range Mode: Aggregate data across the entire range
                range_weeks = [w for w in all_weeks if start_date <= w <= end_date]
                if not range_weeks:
                    st.error(f"❌ No data found for date range {start_date} to {end_date}.")
                    st.stop()

                # Run aggregated analysis across the date range
                res = run_date_range_analysis(df_raw, start_date, end_date)
        
        if res["data"].empty:
            if view_mode == "📊 Weekly View":
                st.info(f"📅 No Starbucks activity recorded for the week of {selected_week}.")
            else:
                st.info(f"📅 No Starbucks activity recorded for the date range {start_date} to {end_date}.")
            st.stop()

        # Financial and Efficiency Diagnostics
        fin = analyze_capital_efficiency(res["capital_flow"], res_data=res["data"])

        # Determine date label for display
        if view_mode == "📊 Weekly View":
            date_label = f"Week of {selected_week}"
        else:
            date_label = f"{start_date} to {end_date}"

        # Build AI context
        portfolio_context = f"""
        CURRENT PORTFOLIO SNAPSHOT ({date_label}):
        - Total Revenue: {f_money(res.get('total_rev', 0))}
        - YoY Growth: {res.get('yoy_delta', 0)*100:.1f}%
        - Share Velocity: {res.get('share_delta', 0)*100:.1f}% vs 6% category benchmark
        - Portfolio Efficiency: {fin.get('efficiency_score', 0):.0f}/100
        - Portfolio Status: {fin.get('portfolio_status', 'Unknown')}

        CAPITAL ZONES:
        {chr(10).join([f"- {zone}: {f_money(rev)}" for zone, rev in res.get('capital_flow', {}).items()])}

        KEY METRICS:
        - Avg Velocity Decay: {fin.get('avg_velocity_decay', 1.0):.2f}x
        - Annualized Waste: {f_money(fin.get('annualized_waste', 0))}
        - Growth Allocation: {fin.get('growth_alloc', 0)*100:.1f}%

        TOP FLAVORS BY REVENUE:
        {chr(10).join([f"- {f}: {f_money(d.get('total_revenue', 0))} ({d.get('health_status', 'N/A')})" for f, d in sorted(res.get('hierarchy', {}).items(), key=lambda x: x[1].get('total_revenue', 0), reverse=True)[:5]])}
        """
    
        # 4. HEADER WITH FLOATING AI CHAT
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.title("🛡️ ShelfGuard: Unified Command Center")
            st.caption(f"Strategy & Analytics Dashboard | Predictive Intelligence Active (36M Lookback)")
    
        with header_col2:
            with st.popover("🤖 AI Assistant", use_container_width=True):
                st.markdown("#### Strategy Assistant")
                st.caption("Ask questions about your portfolio")
            
                # Display last response if exists
                if st.session_state.chat_messages:
                    last_msgs = st.session_state.chat_messages[-4:]
                    for msg in last_msgs:
                        if msg["role"] == "user":
                            st.markdown(f'<div class="user-query-box">💬 {msg["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="ai-response-box">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            
                st.markdown("---")
            
                # Quick action buttons
                q_col1, q_col2 = st.columns(2)
                if q_col1.button("📊 Summary", key="ai_q1", use_container_width=True):
                    prompt = "Give me a 2-sentence portfolio summary with the key action I should take."
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    try:
                        response = openai_client.chat.completions.create(
                            model=st.secrets["openai"]["model"],
                            messages=[
                                {"role": "system", "content": f"You are ShelfGuard AI. Be concise (2 sentences max). {portfolio_context}"},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=150
                        )
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    except Exception as e:
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
            
                if q_col2.button("🚨 Risks", key="ai_q2", use_container_width=True):
                    prompt = "What are my top 2 portfolio risks right now?"
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    try:
                        response = openai_client.chat.completions.create(
                            model=st.secrets["openai"]["model"],
                            messages=[
                                {"role": "system", "content": f"You are ShelfGuard AI. Be concise (2 sentences max). {portfolio_context}"},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=150
                        )
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    except Exception as e:
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
            
                if q_col1.button("🚀 Actions", key="ai_q3", use_container_width=True):
                    prompt = "What's the #1 action I should take this week?"
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    try:
                        response = openai_client.chat.completions.create(
                            model=st.secrets["openai"]["model"],
                            messages=[
                                {"role": "system", "content": f"You are ShelfGuard AI. Be concise (2 sentences max). {portfolio_context}"},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=150
                        )
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    except Exception as e:
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
            
                if q_col2.button("🗑️ Clear", key="ai_q4", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
            
                # Custom question input
                custom_q = st.text_input("Or ask anything:", placeholder="Why is share velocity negative?", key="custom_ai_q")
                if custom_q:
                    st.session_state.chat_messages.append({"role": "user", "content": custom_q})
                    try:
                        response = openai_client.chat.completions.create(
                            model=st.secrets["openai"]["model"],
                            messages=[
                                {"role": "system", "content": f"You are ShelfGuard AI. Be concise and actionable. {portfolio_context}"},
                                {"role": "user", "content": custom_q}
                            ],
                            max_tokens=200
                        )
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.choices[0].message.content})
                    except Exception as e:
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
    
        # === COMPACT EXECUTIVE BRIEFING ===
        total_rev_curr = res.get("total_rev", 0)
        yoy_delta = res.get("yoy_delta", 0)
        share_delta = res.get("share_delta", 0)
        capital_flow = res.get("capital_flow", {})
    
        # Calculate zone metrics
        bleed_rev = sum(v for k, v in capital_flow.items() if "BLEED" in k)
        drag_rev = sum(v for k, v in capital_flow.items() if "DRAG" in k)
        fortress_rev = sum(v for k, v in capital_flow.items() if "FORTRESS" in k)
        waste_rev = bleed_rev + drag_rev
        waste_pct = (waste_rev / total_rev_curr * 100) if total_rev_curr > 0 else 0
    
        # Status determination
        if waste_pct > 30:
            status_emoji, status_text, status_color = "🔴", "CRITICAL", "#dc3545"
        elif waste_pct > 15:
            status_emoji, status_text, status_color = "🟡", "ATTENTION", "#ffc107"
        else:
            status_emoji, status_text, status_color = "🟢", "HEALTHY", "#28a745"
    
        # Top action
        if bleed_rev > total_rev_curr * 0.1:
            top_action = f"EXIT {f_money(bleed_rev)} in bleeding products"
        elif drag_rev > total_rev_curr * 0.25:
            top_action = f"FIX {f_money(drag_rev)} underperformers"
        elif share_delta < -0.05:
            top_action = "DEFEND market share"
        elif fortress_rev > total_rev_curr * 0.5:
            top_action = "EXPAND — test price increases"
        else:
            top_action = "OPTIMIZE efficiency"
    
        # === CALCULATE FORECAST METRICS (for AI Brief) ===
        demand_forecast = res.get("demand_forecast", {})
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
        view_mode = res.get("view_mode", "weekly")
        if view_mode == "range":
            date_range = res.get("date_range", (None, None))
            date_display = f"{date_range[0].strftime('%b %d')} - {date_range[1].strftime('%b %d, %Y')}" if date_range[0] and date_range[1] else "Date Range"
            date_label = "Date Range"
        else:
            date_display = selected_week.strftime('%b %d, %Y') if selected_week else "Week"
            date_label = "Week of"
    
        # === AI-GENERATED WEEKLY BRIEF (LLM-Powered) ===
        data_df = res.get("data", pd.DataFrame())
    
        # Build portfolio summary for LLM
        if not data_df.empty and 'problem_category' in data_df.columns:
            problem_counts = data_df.groupby('problem_category').agg({
                'weekly_sales_filled': 'sum',
                'asin': 'count'
            }).to_dict('index')
        
            losing_money = problem_counts.get('🔥 Losing Money', {'weekly_sales_filled': 0, 'asin': 0})
            losing_share = problem_counts.get('📉 Losing Share', {'weekly_sales_filled': 0, 'asin': 0})
            price_problem = problem_counts.get('💰 Price Problem', {'weekly_sales_filled': 0, 'asin': 0})
            scale_winners = problem_counts.get('🚀 Scale Winner', {'weekly_sales_filled': 0, 'asin': 0})
            healthy = problem_counts.get('✅ Healthy', {'weekly_sales_filled': 0, 'asin': 0})
        
            # Build summary for LLM (with 36M intelligence)
            portfolio_summary = f"""
    PORTFOLIO SNAPSHOT ({date_label} {date_display}):
    - Total Revenue: {f_money(total_rev_curr)}/week
    - YoY Change: {yoy_delta*100:+.1f}%
    - Market Share Change: {share_delta*100:+.1f}%
    - 8W Forecast: {portfolio_forecast*100:+.1f}% (Confidence: {forecast_confidence}, based on {max_years} year(s) of historical data)

    PROBLEM BREAKDOWN:
    - Losing Money: {int(losing_money['asin'])} products, {f_money(losing_money['weekly_sales_filled'])}/wk (negative margin - every sale loses money)
    - Losing Share: {int(losing_share['asin'])} products, {f_money(losing_share['weekly_sales_filled'])}/wk (velocity decay >1.5x - competitors winning)
    - Price Problems: {int(price_problem['asin'])} products, {f_money(price_problem['weekly_sales_filled'])}/wk (price gap or Buy Box issues)
    - Scale Winners: {int(scale_winners['asin'])} products, {f_money(scale_winners['weekly_sales_filled'])}/wk (high margin, strong BB, growing)
    - Healthy: {int(healthy['asin'])} products, {f_money(healthy['weekly_sales_filled'])}/wk (on track)

    CONTEXT:
    - Waste (Losing Money + Losing Share): {waste_pct:.1f}% of portfolio
    - Portfolio Status: {status_text}
    - Analysis based on 36 months of historical performance data
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
                brief_source = "🤖 AI STRATEGIST"
            else:
                # Fallback to rule-based brief
                brief_parts = []
                if losing_money['asin'] > 0:
                    brief_parts.append(f"**{int(losing_money['asin'])} products losing money** ({f_money(losing_money['weekly_sales_filled'])}/wk) — exit immediately.")
                if losing_share['asin'] > 0:
                    brief_parts.append(f"**{int(losing_share['asin'])} products losing share** ({f_money(losing_share['weekly_sales_filled'])}/wk) — fix visibility.")
                if price_problem['asin'] > 0:
                    brief_parts.append(f"**{int(price_problem['asin'])} products with pricing issues** ({f_money(price_problem['weekly_sales_filled'])}/wk) — reprice now.")
                if scale_winners['asin'] > 0:
                    brief_parts.append(f"**{int(scale_winners['asin'])} winners ready to scale** ({f_money(scale_winners['weekly_sales_filled'])}/wk).")
            
                ai_brief = " ".join(brief_parts[:2]) if brief_parts else "Portfolio stable. Focus on optimization."
                brief_source = "📊 ANALYSIS"
        else:
            ai_brief = "Analyzing portfolio data..."
            brief_source = "⏳ LOADING"
    
        # YoY context
        if yoy_delta > 0.05:
            yoy_context = f"Revenue is up {yoy_delta*100:.0f}% vs last year."
        elif yoy_delta < -0.05:
            yoy_context = f"Revenue is down {abs(yoy_delta)*100:.0f}% vs last year — needs attention."
        else:
            yoy_context = "Revenue is flat vs last year."
    
        # Share context  
        if share_delta > 0:
            share_context = f"You're gaining market share (+{share_delta*100:.1f}%)."
        elif share_delta < -0.03:
            share_context = f"You're losing market share ({share_delta*100:.1f}%) — competitors are winning."
        else:
            share_context = "Market share is stable."
    
        # 36M Intelligence context
        if 'velocity_decay' in data_df.columns:
            accelerating_count = len(data_df[data_df['velocity_decay'] < 0.9])
            decaying_count = len(data_df[data_df['velocity_decay'] > 1.2])
            total_products = len(data_df)
        
            if accelerating_count > decaying_count:
                intel_36m = f"📊 36M: {accelerating_count}/{total_products} products accelerating vs historical avg."
            elif decaying_count > 3:
                intel_36m = f"⚠️ 36M: {decaying_count}/{total_products} products decaying vs historical avg."
            else:
                intel_36m = f"📊 36M: Velocity stable across portfolio."
        else:
            intel_36m = ""
    
        # Render the AI brief with refresh button
        brief_col1, brief_col2 = st.columns([4, 1])
        with brief_col1:
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
                        <div style="font-size: 11px; color: #666;">Weekly Revenue</div>
                    </div>
                </div>
                <div style="background: #f8f9fa; padding: 14px; border-radius: 6px; margin-top: 10px;">
                    <div style="font-size: 11px; color: #00704A; font-weight: 600; margin-bottom: 6px;">{brief_source}</div>
                    <div style="font-size: 14px; color: #333; line-height: 1.5;">
                        {ai_brief}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 10px;">
                        {yoy_context} {share_context} {intel_36m}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with brief_col2:
            if brief_source == "🤖 AI STRATEGIST":
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("🔄 Regenerate", key="refresh_brief", help="Force regenerate AI brief (bypasses cache)"):
                    st.session_state.force_refresh_brief = True
                    st.rerun()
    
        # --- STRATEGIC TILES ---
        c1, c2, c3, c4 = st.columns(4)
    
        # TILE 1: Weekly Portfolio Rev (Stacked Performance Deltas + Forecast)
        with c1:
            share_delta = res.get("share_delta", 0) # Relative: vs 6% Category
            yoy_delta = res.get("yoy_delta", 0)     # Absolute: vs LY
        
            # Forecast metrics already calculated above for AI brief
        
            # Define color classes and icons
            yoy_class = "pos" if yoy_delta > 0 else "neg" if yoy_delta < 0 else "neu"
            share_class = "pos" if share_delta > 0 else "neg" if share_delta < 0 else "neu"
            forecast_class = "pos" if portfolio_forecast > 0.05 else "neg" if portfolio_forecast < -0.05 else "neu"
            yoy_icon = "↑" if yoy_delta > 0 else "↓" if yoy_delta < 0 else "→"
            share_icon = "↑" if share_delta > 0 else "↓" if share_delta < 0 else "→"
            forecast_icon = "📈" if portfolio_forecast > 0.05 else "📉" if portfolio_forecast < -0.05 else "→"

            # Display custom metric with integrated deltas inside the tile
            st.markdown(f"""
                <div class="custom-metric-container">
                    <div class="custom-metric-label">Weekly Portfolio Rev</div>
                    <div class="custom-metric-value">{f_money(total_rev_curr)}</div>
                    <div class="benchmark-row" style="flex-wrap: wrap; gap: 6px;">
                        <span class="benchmark-badge benchmark-{yoy_class}">{yoy_icon} {yoy_delta:+.1%} vs 1Y</span>
                        <span class="benchmark-badge benchmark-{share_class}">{share_icon} {share_delta:+.1%} Share</span>
                        <span class="benchmark-badge benchmark-{forecast_class}" title="Based on {max_years} year(s) of data | Confidence: {forecast_confidence}">{forecast_icon} {portfolio_forecast:+.1%} 8W Forecast</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 2: Portfolio Integrity %
        with c2:
            healthy_zones = ["🏰 FORTRESS (Cash Flow)", "🚀 FRONTIER (Growth)"]
            integrity_score = (res["data"][res["data"]["capital_zone"].isin(healthy_zones)]["weekly_sales_filled"].sum() / total_rev_curr) * 100 if total_rev_curr > 0 else 0
        
            # Determine benchmark status
            if integrity_score >= 85:
                benchmark_status = "Elite"
                benchmark_class = "benchmark-elite"
                benchmark_icon = "🏆"
            elif integrity_score >= 60:
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
                    <div class="custom-metric-label">Portfolio Integrity %</div>
                    <div class="custom-metric-value">{integrity_score:.1f}%</div>
                    <div class="benchmark-row">
                        <span class="benchmark-badge {benchmark_class}">{benchmark_icon} {benchmark_status}</span>
                        <span class="benchmark-target">Target: 80%+</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 3: Contribution Leak (Benchmark + CI Update)
        with c3:
            bad_zones = ["📉 DRAG (Waste)", "📉 DRAG (Terminal Decay)", "🩸 BLEED (Negative Margin)"]
            leak_total = res["data"][res["data"]["capital_zone"].isin(bad_zones)]["weekly_sales_filled"].sum()
            leak_exposure = (leak_total / total_rev_curr) if total_rev_curr > 0 else 0
        
            # 1. Benchmark Logic: Define the Severity
            if leak_exposure > 0.25:
                leak_status, leak_icon, status_color = "Critical", "🚨", "neg"
            elif leak_exposure > 0.15:
                leak_status, leak_icon, status_color = "Attention", "⚠️", "neu"
            else:
                leak_status, leak_icon, status_color = "Optimized", "✅", "pos"
            
            # 2. CI Logic: Correlate with Share Velocity
            ci_context = "CI Alert: Market Share loss is accelerating." if share_delta < 0 else "CI Note: Leak may be due to aggressive share acquisition."

            # Display custom metric with benchmark badge
            st.markdown(f"""
                <div class="custom-metric-container" title="Revenue at Risk from Bleed/Drag zones. {ci_context} Target: <15%.">
                    <div class="custom-metric-label">Contribution Leak</div>
                    <div class="custom-metric-value">{f_money(leak_total)}</div>
                    <div class="benchmark-row">
                        <span class="benchmark-badge benchmark-{status_color}">{leak_icon} {leak_status}</span>
                        <span class="benchmark-target">{leak_exposure:.1%} Exposure</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 4: Net Efficiency Delta (Strategic Alpha Scorecard)
        with c4:
            efficiency_score = fin.get('efficiency_score', 0)
        
            # WoW Validation: Calculate from session state or use placeholder
            # This will be populated by comparing to previous week's efficiency
            prev_efficiency = st.session_state.get('prev_efficiency_score', efficiency_score)
            wow_delta = efficiency_score - prev_efficiency
            st.session_state['prev_efficiency_score'] = efficiency_score
        
            # Saved Revenue: Track overrides from Directive Action Log
            # Placeholder until override tracking is implemented in sidebar
            saved_revenue = st.session_state.get('saved_revenue', 0)
            override_count = st.session_state.get('override_count', 0)
        
            # Validation Status
            if wow_delta > 0:
                validation_class = "validated"
                validation_icon = "↑"
                validation_text = f"{wow_delta:+.1f} Validated"
            elif wow_delta < 0:
                validation_class = "rejected"
                validation_icon = "↓"
                validation_text = f"{wow_delta:+.1f} Rejected"
            else:
                validation_class = "neutral"
                validation_icon = "→"
                validation_text = "0.0 Baseline"
        
            # Alpha Status: Human Alpha vs System Optimized
            if saved_revenue > 0 and override_count > 0:
                alpha_status = "Human Alpha"
                alpha_class = "human"
            else:
                alpha_status = "System Optimized"
                alpha_class = "system"
        
            # Render the Strategic Alpha Card
            st.markdown(f"""
                <div class="alpha-card">
                    <div class="alpha-label">Net Efficiency Delta</div>
                    <div>
                        <span class="alpha-score">{efficiency_score:.0f}</span>
                        <span class="alpha-score-suffix">/ 100</span>
                    </div>
                    <div class="alpha-validation {validation_class}">
                        {validation_icon} {validation_text} WoW
                    </div>
                    <div class="alpha-divider"></div>
                    <div class="alpha-saved">
                        <span>💰</span>
                        <span class="alpha-saved-value">{f_money(saved_revenue)}</span>
                        <span style="color:#666;">Saved Revenue</span>
                        <span class="alpha-status-badge {alpha_class}">{alpha_status}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
            # --- AI ACTION QUEUE ---
            tab1, tab2 = st.tabs(["🎯 AI Action Queue", "🖼️ Visual Audit"])

            with tab1:
                # AI ACTION QUEUE - Unified view with AI-prioritized actions
                display_df = res["data"].copy()
        
                # === TOP PRIORITIES (AI-ranked by $ impact) ===
                if 'problem_category' in display_df.columns:
                    # Calculate impact per problem category
                    problem_summary = display_df.groupby('problem_category').agg({
                        'weekly_sales_filled': 'sum',
                        'asin': 'count'
                    }).rename(columns={'asin': 'count', 'weekly_sales_filled': 'revenue'})
                    problem_summary = problem_summary.sort_values('revenue', ascending=False)
            
                    # Define action priority order (most urgent first)
                    priority_order = ["🔥 Losing Money", "📉 Losing Share", "💰 Price Problem", "🚀 Scale Winner", "✅ Healthy", "📊 Monitor"]
            
                    # Get top 3 actionable problems (exclude Healthy/Monitor)
                    actionable = problem_summary[~problem_summary.index.str.contains('Healthy|Monitor', na=False)]
                    top_3 = actionable.head(3)
            
                    st.markdown("#### 🎯 Top Priorities")
            
                    if not top_3.empty:
                        priority_cols = st.columns(len(top_3))
                
                        for i, (problem, data) in enumerate(top_3.iterrows()):
                            count = int(data['count'])
                            rev = data['revenue']
                    
                            # Determine color and action text
                            if "Losing Money" in problem:
                                color, action = "#dc3545", "Exit immediately"
                            elif "Losing Share" in problem:
                                color, action = "#fd7e14", "Fix visibility"
                            elif "Price Problem" in problem:
                                color, action = "#ffc107", "Reprice now"
                            elif "Scale Winner" in problem:
                                color, action = "#28a745", "Invest more"
                            else:
                                color, action = "#6c757d", "Review"
                    
                            with priority_cols[i]:
                                st.markdown(f"""
                                <div style="background: white; border: 1px solid #e0e0e0; padding: 16px; 
                                            border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                                    <div style="font-size: 11px; color: {color}; font-weight: 600; text-transform: uppercase;">#{i+1} PRIORITY</div>
                                    <div style="font-size: 12px; color: #666; margin-top: 2px;">{problem}</div>
                                    <div style="font-size: 24px; color: #1a1a1a; font-weight: 700; margin: 8px 0 4px 0;">{count} products</div>
                                    <div style="font-size: 14px; color: #00704A; font-weight: 600;">{f_money(rev)}/week</div>
                                    <div style="font-size: 11px; color: #666; margin-top: 6px;">→ {action}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No urgent issues. Portfolio is healthy!")
            
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
                    # Streamlined columns: Product, Context, 36M Intelligence, Action, Revenue
                    cols_to_show = ["asin", "Flavor", "Count", "weekly_sales_filled"]
            
                    # Add 36M velocity decay (key historical metric)
                    if "velocity_decay" in display_df.columns:
                        cols_to_show.append("velocity_decay")
            
                    # Add forecast signal
                    if "forecast_signal" in display_df.columns:
                        cols_to_show.append("forecast_signal")
            
                    # Add action columns if available
                    if "ecom_action" in display_df.columns:
                        cols_to_show.append("ecom_action")
                    if "ad_action" in display_df.columns:
                        cols_to_show.append("ad_action")
            
                    final_df = display_df[cols_to_show].rename(columns={
                        "asin": "ASIN", 
                        "weekly_sales_filled": "Revenue",
                        "velocity_decay": "📊 36M Decay",
                        "forecast_signal": "📈 8W Forecast",
                        "ecom_action": "Action",
                        "ad_action": "Media"
                    }).drop_duplicates(subset=["ASIN"]).sort_values("Revenue", ascending=False)

                    st.dataframe(
                        final_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                            "📊 36M Decay": st.column_config.NumberColumn(
                                format="%.2fx",
                                help="Velocity Decay: <1.0 = accelerating, >1.0 = slowing (based on 36M average)"
                            ),
                        }
                    )
            
                    # Problem-specific playbook
                    if selected_problem and selected_problem != "All Products":
                        st.markdown("---")
                
                        if "Losing Money" in selected_problem:
                            st.error("""**🔥 Losing Money — Exit Protocol**
                    
        **Why:** These products have negative margin. Every sale costs you money.

        **Action:** 
        1. PAUSE all ad spend immediately  
        2. Liquidate remaining inventory (fire sale pricing)  
        3. Exit SKU within 2 weeks""")
                    
                        elif "Losing Share" in selected_problem:
                            st.warning("""**📉 Losing Share — Recovery Protocol**
                    
        **Why:** Velocity is decaying faster than market. You're losing to competitors.

        **Action:**
        1. Audit Buy Box — are you being undercut?
        2. Increase ad visibility (+25% budget)
        3. Review competitor pricing and match if needed""")
                    
                        elif "Price Problem" in selected_problem:
                            st.info("""**💰 Price Problem — Reprice Protocol**
                    
        **Why:** Your price gap is too wide or Buy Box share is dropping.

        **Action:**
        1. Clip coupon to close price gap
        2. If BB% < 50%, consider matching competitor price
        3. Hold ad spend until pricing is fixed""")
                    
                        elif "Scale Winner" in selected_problem:
                            st.success("""**🚀 Scale Winner — Accelerate Protocol**
                    
        **Why:** High margin, strong Buy Box, growing velocity. These are your best products.

        **Action:**
        1. Test price increase (+5-10%)
        2. Scale ad spend (+25-50%)
        3. Ensure inventory is fully stocked""")
                    
                        elif "Healthy" in selected_problem:
                            st.success("""**✅ Healthy — Maintain Protocol**
                    
        **Why:** On track. No immediate action required.

        **Action:**
        1. Monitor weekly for changes
        2. Optimize ROAS on existing campaigns
        3. Protect margins — don't discount unnecessarily""")
                    
                except Exception as table_err:
                    st.error("Unable to render Execution Queue.")
                    st.exception(table_err)

            with tab2:
                st.markdown("### 🖼️ Visual Portfolio Audit")
                st.caption("Top 16 products by revenue")
        
                # Use full data for visual audit (not filtered)
                full_df = res["data"].copy()
                gallery_df = full_df[full_df["main_image"] != ""].sort_values("weekly_sales_filled", ascending=False).head(16)
        
                if not gallery_df.empty:
                    cols = st.columns(4)
                    for i, (_, row) in enumerate(gallery_df.iterrows()):
                        with cols[i % 4]:
                            clean_title = (row['title'][:45] + '...') if len(row['title']) > 45 else row['title']
                            problem = row.get('problem_category', row.get('capital_zone', 'N/A'))
                    
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
                                        {f_money(row['weekly_sales_filled'])}
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