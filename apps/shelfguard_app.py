import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from data import get_all_data
from engine import run_weekly_analysis, run_date_range_analysis
from finance import analyze_capital_efficiency, f_money, f_pct
from demo_data import render_asin_upload_ui, get_demo_data, clear_demo_data
from search_to_state_ui import render_discovery_ui, render_project_dashboard, render_project_selector, render_user_dashboard

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
                    "content": """You are ShelfGuard's AI strategist, powered by 36 months of historical data analysis. Generate a prescriptive strategic brief for an e-commerce portfolio manager.

Rules:
- Be prescriptive, not descriptive. Use action language ("Protocol Activated:", "Execute immediately:", "Deploy:")
- Lead with the most urgent threat detection.
- Quantify everything ($ amounts, counts, percentages).
- Reference historical intelligence (e.g., "threat level exceeds 36M baseline" or "velocity decay pattern detected").
- End with one executable command for this session.
- Keep it under 100 words.
- Use tactical language: threats, protocols, defense perimeter, alpha capture."""
                },
                {
                    "role": "user",
                    "content": f"""Here's the current defense perimeter status:

{portfolio_summary}

Generate a prescriptive strategic brief. What protocol must be activated immediately?"""
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
    # === STATE MACHINE ROUTER ===
    # Check if user has activated a project
    active_project_asin = st.session_state.get('active_project_asin', None)

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
        # 2. DATA INGESTION FROM DISCOVERY
        df_weekly = st.session_state.get('active_project_data', pd.DataFrame())
        market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
        project_name = st.session_state.get('active_project_name', 'Unknown Project')
        seed_asin = st.session_state.get('active_project_asin', None)

        if df_weekly.empty or market_snapshot.empty:
            st.warning("⚠️ No project data found. Please create a project in Market Discovery.")
            st.stop()

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
        st.caption(f"🎯 Analyzing: **{target_brand}** vs. Market")

        # Step 2: Create two dataframes
        portfolio_df = market_snapshot[market_snapshot['brand'] == target_brand].copy()
        market_df = market_snapshot.copy()  # Full market for comparison

        # Add is_your_brand flag for backward compatibility
        market_snapshot['is_your_brand'] = market_snapshot['brand'] == target_brand

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
        portfolio_snapshot_df['weekly_sales_filled'] = portfolio_snapshot_df['revenue_proxy']
        portfolio_snapshot_df['asin'] = portfolio_snapshot_df.get('asin', '')

        # All products in portfolio_df are "Your Brand - Healthy"
        portfolio_snapshot_df['problem_category'] = '✅ Your Brand - Healthy'
        portfolio_snapshot_df['capital_zone'] = '🏰 FORTRESS (Cash Flow)'

        # Create res object that dashboard expects (scoped to YOUR BRAND ONLY)
        res = {
            'data': portfolio_snapshot_df,  # ONLY your brand's products
            'total_rev': portfolio_revenue,  # YOUR revenue, not market revenue
            'yoy_delta': 0.0,
            'share_delta': 0.0,
            'capital_flow': {
                '🏰 FORTRESS (Cash Flow)': portfolio_revenue,
                '📉 DRAG (Waste)': 0  # No drag in our own portfolio for Discovery
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

        # Build AI context for Strategic Brief (Brand vs Market)
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

        # 4. SYSTEM STATUS BANNER + HEADER
        # Determine system status based on revenue performance
        total_rev_curr = res.get("total_rev", 0)
        target_revenue = total_rev_curr * 1.0  # Placeholder - would come from user's target
        revenue_gap = total_rev_curr - target_revenue

        if revenue_gap < 0:
            system_status = "🔴 SYSTEM STATUS: CRITICAL THREATS DETECTED"
            status_bg = "#dc3545"
        else:
            system_status = "🟢 SYSTEM STATUS: OPTIMIZED"
            status_bg = "#28a745"

        # System Status Banner
        st.markdown(f"""
        <div style="background: {status_bg}; color: white; padding: 12px 20px; border-radius: 8px;
                    margin-bottom: 20px; text-align: center; font-weight: 700; font-size: 14px;">
            {system_status}
        </div>
        """, unsafe_allow_html=True)

        # Header with AI chat
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.title("🛡️ Command Center")
            st.caption(f"Workflow OS | Predictive Intelligence Active (36M Lookback)")
    
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

        # Build market summary for LLM (Brand vs Market)
        if not data_df.empty:
            # Build summary for LLM showing brand performance vs market
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
                # Fallback to rule-based brief (Brand vs Market)
                brief_parts = []
                if portfolio_product_count > 0 and your_market_share < 30:
                    brief_parts.append(f"**{portfolio_product_count} {target_brand} products** controlling {your_market_share:.1f}% share — growth opportunity available.")
                if competitor_revenue > portfolio_revenue * 2:
                    brief_parts.append(f"**Competitors control {f_money(competitor_revenue)}/month** vs. your {f_money(portfolio_revenue)}/month — defend and expand.")

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
                        {share_context} {competitive_context}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with brief_col2:
            if brief_source == "🤖 STRATEGIC BRIEF":
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("🔄 Regenerate", key="refresh_brief", help="Force regenerate AI brief (bypasses cache)"):
                    st.session_state.force_refresh_brief = True
                    st.rerun()
    
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
            healthy_zones = ["🏰 FORTRESS (Cash Flow)", "🚀 FRONTIER (Growth)"]
            defense_score = (res["data"][res["data"]["capital_zone"].isin(healthy_zones)]["weekly_sales_filled"].sum() / total_rev_curr) * 100 if total_rev_curr > 0 else 0

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
    
        # TILE 3: Recoverable Alpha
        with c3:
            bad_zones = ["📉 DRAG (Waste)", "📉 DRAG (Terminal Decay)", "🩸 BLEED (Negative Margin)"]
            recoverable_alpha = res["data"][res["data"]["capital_zone"].isin(bad_zones)]["weekly_sales_filled"].sum()
            leak_exposure = (recoverable_alpha / total_rev_curr) if total_rev_curr > 0 else 0

            # 1. Benchmark Logic: Define the Severity
            if leak_exposure > 0.25:
                leak_status, leak_icon, status_color = "Critical", "🚨", "neg"
            elif leak_exposure > 0.15:
                leak_status, leak_icon, status_color = "Attention", "⚠️", "neu"
            else:
                leak_status, leak_icon, status_color = "Optimized", "✅", "pos"

            # 2. CI Logic: Correlate with Share Velocity
            ci_context = "Value currently at risk" if share_delta < 0 else "Opportunity for optimization"

            # Display custom metric with benchmark badge
            st.markdown(f"""
                <div class="custom-metric-container" title="{ci_context}. Target: <15%.">
                    <div class="custom-metric-label">Recoverable Alpha</div>
                    <div class="custom-metric-value" style="color: #dc3545;">{f_money(recoverable_alpha)}</div>
                    <div class="benchmark-row">
                        <span class="benchmark-badge benchmark-{status_color}">{leak_icon} {leak_status}</span>
                        <span class="benchmark-target">{leak_exposure:.1%} at Risk</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
        # TILE 4: Banked Alpha (Strategic Alpha Scorecard)
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
                alpha_status = "Value Captured"
                alpha_class = "human"
                alpha_icon = "💰"
            else:
                alpha_status = "Awaiting Action"
                alpha_class = "system"
                alpha_icon = "⏳"

            # Render the Banked Alpha Card
            st.markdown(f"""
                <div class="alpha-card">
                    <div class="alpha-label">Banked Alpha</div>
                    <div>
                        <span class="alpha-score" style="color: #28a745;">{f_money(banked_alpha)}</span>
                    </div>
                    <div class="alpha-validation validated">
                        {alpha_icon} {task_count} Tasks Completed
                    </div>
                    <div class="alpha-divider"></div>
                    <div class="alpha-saved">
                        <span class="alpha-status-badge {alpha_class}">{alpha_status}</span>
                        <span style="color:#666; font-size: 0.75rem;">Value captured this session</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
            # --- AI ACTION QUEUE ---
            tab1, tab2 = st.tabs(["🎯 AI Action Queue", "🖼️ Visual Audit"])

            with tab1:
                # AI ACTION QUEUE - Unified view with AI-prioritized actions
                display_df = res["data"].copy()
        
                # === TOP PRIORITIES (Brand Portfolio Analysis) ===
                # Note: display_df now contains ONLY the target brand's products
                st.markdown(f"#### 🎯 {target_brand} Portfolio Actions")

                # For single-brand portfolio, show top products by revenue
                top_products = display_df.nlargest(3, 'weekly_sales_filled')

                if not top_products.empty:
                    priority_cols = st.columns(len(top_products))

                    for i, (idx, product) in enumerate(top_products.iterrows()):
                        rev = product['weekly_sales_filled']
                        asin = product['asin']
                        title = product.get('title', asin)[:40] + "..." if len(product.get('title', asin)) > 40 else product.get('title', asin)

                        # All products are "your brand" - color code by revenue contribution
                        revenue_share = (rev / portfolio_revenue * 100) if portfolio_revenue > 0 else 0
                        if revenue_share > 20:
                            color, action = "#28a745", "Protect - Top Performer"
                        elif revenue_share > 10:
                            color, action = "#00704A", "Scale - Strong Product"
                        else:
                            color, action = "#ffc107", "Optimize - Growth Opportunity"

                        # Check if task is completed
                        task_id = f"priority_{i}_{asin}"
                        is_completed = task_id in st.session_state.get('completed_tasks', set())

                        with priority_cols[i]:
                            # Card styling changes based on completion
                            card_opacity = "0.5" if is_completed else "1.0"
                            card_bg = "#f0f0f0" if is_completed else "white"

                            st.markdown(f"""
                            <div style="background: {card_bg}; border: 1px solid #e0e0e0; padding: 16px;
                                        border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                                        opacity: {card_opacity};">
                                <div style="font-size: 11px; color: {color}; font-weight: 600; text-transform: uppercase;">#{i+1} PRIORITY</div>
                                <div style="font-size: 12px; color: #666; margin-top: 2px;">{title}</div>
                                <div style="font-size: 24px; color: #1a1a1a; font-weight: 700; margin: 8px 0 4px 0;">{revenue_share:.1f}%</div>
                                <div style="font-size: 14px; color: #00704A; font-weight: 600;">{f_money(rev)}/month</div>
                                <div style="font-size: 11px; color: #666; margin-top: 6px;">→ {action}</div>
                                <div style="font-size: 10px; color: #999; margin-top: 4px; font-family: monospace;">{asin}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Execute button
                            if is_completed:
                                st.success("✅ Analyzed", icon="✅")
                            else:
                                if st.button(f"Analyze", key=f"exec_{task_id}", use_container_width=True, type="primary"):
                                    # Mark task as completed
                                    if 'completed_tasks' not in st.session_state:
                                        st.session_state['completed_tasks'] = set()
                                    st.session_state['completed_tasks'].add(task_id)

                                    # Increment Banked Alpha
                                    st.session_state['banked_alpha'] = st.session_state.get('banked_alpha', 0) + rev

                                    st.success(f"✅ Product analyzed - {revenue_share:.1f}% of portfolio revenue")
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
                    # Streamlined columns for Discovery data (brand portfolio)
                    cols_to_show = ["asin"]

                    # Add title if available
                    if "title" in display_df.columns:
                        cols_to_show.append("title")

                    # Add brand if available
                    if "brand" in display_df.columns:
                        cols_to_show.append("brand")

                    # Add price if available
                    if "price" in display_df.columns:
                        cols_to_show.append("price")

                    # Always show revenue
                    cols_to_show.append("weekly_sales_filled")

                    # Add BSR if available
                    if "bsr" in display_df.columns:
                        cols_to_show.append("bsr")

                    # Build column rename mapping
                    rename_map = {
                        "asin": "ASIN",
                        "title": "Product Title",
                        "brand": "Brand",
                        "price": "Price",
                        "weekly_sales_filled": "Revenue",
                        "bsr": "Sales Rank"
                    }

                    final_df = display_df[cols_to_show].copy()
                    final_df = final_df.rename(columns=rename_map)
                    final_df = final_df.drop_duplicates(subset=["ASIN"]).sort_values("Revenue", ascending=False)

                    # Configure column display
                    column_config = {
                        "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                    }

                    # Add price formatting if column exists
                    if "Price" in final_df.columns:
                        column_config["Price"] = st.column_config.NumberColumn(format="$%.2f")

                    # Add BSR formatting if column exists
                    if "Sales Rank" in final_df.columns:
                        column_config["Sales Rank"] = st.column_config.NumberColumn(format="%d")

                    st.dataframe(
                        final_df,
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