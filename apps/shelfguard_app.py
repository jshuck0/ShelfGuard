import streamlit as st
import pandas as pd
from data import get_all_data
from engine import run_weekly_analysis
from finance import analyze_capital_efficiency, f_money, f_pct

# 1. PAGE CONFIGURATION
# Must be the first Streamlit command. 
st.set_page_config(page_title="ShelfGuard OS", layout="wide", page_icon="🛡️")

# --- UI STYLING (Full CSS Block) ---
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-left: 6px solid #00704A !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    [data-testid="stMetricValue"] div {
        color: #00704A !important;
    }
    .product-card {
        background-color: white;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        height: 320px;
    }
    .product-card:hover {
        border-color: #00704A;
        transform: translateY(-5px);
    }
    .product-img {
        max-height: 140px;
        max-width: 100%;
        margin: 0 auto;
        display: block;
        object-fit: contain;
    }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

try:
    # 2. DATA INGESTION
    with st.spinner("🔄 Loading Global Data..."):
        df_raw = get_all_data()
    
    if df_raw.empty:
        st.warning("⚠️ No data found in the database. Please check your Supabase connection.")
        st.stop()

    # Normalize dates and handle empty states
    df_raw["week_start"] = pd.to_datetime(df_raw["week_start"])
    all_weeks = sorted(df_raw["week_start"].dt.date.dropna().unique(), reverse=True)
    
    if not all_weeks:
        st.error("❌ No valid weeks found in dataset.")
        st.stop()
        
    selected_week = st.sidebar.selectbox("Fiscal Period", all_weeks)
    
    # 3. ANALYSIS EXECUTION
    # This executes the 36-month trend math and the Category Growth benchmarks in engine.py
    with st.spinner("🧠 Executing Predictive Intelligence..."):
        res = run_weekly_analysis(df_raw, selected_week)
        
    if res["data"].empty:
        st.info(f"📅 No Starbucks activity recorded for the week of {selected_week}.")
        st.stop()

    # Financial and Efficiency Diagnostics
    fin = analyze_capital_efficiency(res["capital_flow"], res_data=res["data"])
    
    # 4. HEADER
    st.title("🛡️ ShelfGuard: Unified Command Center")
    st.caption(f"Strategy & Analytics Dashboard | Predictive Intelligence Active (36M Lookback)")
    
    # --- ROW 1: THE DIRECTOR-LEVEL STRATEGIC TILES ---
    c1, c2, c3, c4 = st.columns(4)
    
    # TILE 1: Weekly Portfolio Rev (Now including Share Velocity Benchmark)
    total_rev_curr = res.get("total_rev", 0)
    share_delta = res.get("share_delta", 0) # Calculated in engine.py vs 6% Category Growth
    delta_color = "normal" if share_delta >= 0 else "inverse"
    
    c1.metric(
        label="Weekly Portfolio Rev", 
        value=f_money(total_rev_curr), 
        delta=f"{share_delta:+.1%} Share Velocity",
        delta_color=delta_color,
        help="Current weekly revenue benchmarked against a 6.0% category growth rate baseline."
    )
    
    # TILE 2: Portfolio Integrity % (Revenue Quality)
    # Tracks the percentage of total revenue coming from 'Healthy' Strategic Zones
    healthy_zones = ["🏰 FORTRESS (Cash Flow)", "🚀 FRONTIER (Growth)"]
    integrity_score = (res["data"][res["data"]["capital_zone"].isin(healthy_zones)]["weekly_sales_filled"].sum() / total_rev_curr) * 100
    c2.metric("Portfolio Integrity %", f"{integrity_score:.1f}%", 
              help="The percentage of total revenue coming from Healthy (Fortress/Frontier) SKUs.")
    
    # TILE 3: Contribution Leak (Immediate Waste)
    # Aggregates operational loss and ad waste from 'Shaky' zones
    bad_zones = ["📉 DRAG (Waste)", "📉 DRAG (Terminal Decay)", "🩸 BLEED (Negative Margin)"]
    leak_total = res["data"][res["data"]["capital_zone"].isin(bad_zones)]["weekly_sales_filled"].sum()
    c3.metric("Contribution Leak", f_money(leak_total), 
              delta=f"{(leak_total/total_rev_curr):.1%} Exposure", delta_color="inverse",
              help="Immediate weekly dollar loss (Ads + Margin) from operationally broken ASINs.")
    
    # TILE 4: Net Efficiency Delta (Validation Placeholder)
    # Will be updated to compare WoW efficiency movement to prove strategic impact
    c4.metric("Net Efficiency Delta", f"{fin.get('efficiency_score', 0):.0f}/100", 
              delta="WoW Validation", delta_color="normal",
              help="The Week-over-Week change in total Portfolio Efficiency.")
    
    st.divider()

    # --- ROW 2: STRATEGIC ZONE FILTERING ---
    available_zones = sorted(res["data"]["capital_zone"].unique())
    selected_zone = st.radio(
        "Strategic Filter", 
        options=available_zones, 
        horizontal=True
    )

    display_df = res["data"].copy()
    if selected_zone:
        display_df = display_df[display_df["capital_zone"] == selected_zone]

    # --- ROW 3: PERFORMANCE MATRIX & VISUAL AUDIT ---
    tab1, tab2 = st.tabs(["📊 Performance Matrix", "🖼️ Visual Audit"])

    with tab1:
        try:
            # Preparing DataFrame using the clean 'Flavor' and 'Count' columns
            final_df = display_df[[
                "Flavor", "Count", "capital_zone", "efficiency_score", "velocity_decay", 
                "Trend (36M)", "net_margin", "weekly_sales_filled", "ad_action", "ecom_action"
            ]].rename(columns={
                "capital_zone": "Status",
                "efficiency_score": "Score",
                "velocity_decay": "Decay",
                "Trend (36M)": "3-Year Velocity",
                "net_margin": "Margin %",
                "weekly_sales_filled": "Revenue",
                "ad_action": "Media Directive",
                "ecom_action": "Ops Lever"
            }).sort_values("Revenue", ascending=False)

            st.dataframe(
                final_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Revenue": st.column_config.NumberColumn(format="$%.2f"),
                    "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Decay": st.column_config.NumberColumn(
                        "Decay", help=">1.0 = Rank is worsening vs 3-yr average.", format="%.2fx"
                    ),
                    "3-Year Velocity": st.column_config.LineChartColumn(
                        "3-Year Velocity", help="Historical Sales Rank Trend (Lower is better)"
                    ),
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%d", color="#00704A"
                    )
                }
            )
        except Exception as table_err:
            st.error("Unable to render Performance Matrix.")
            st.exception(table_err)

    with tab2:
        st.write("### Starbucks Visual Portfolio Audit")
        # Pull products for gallery view sorted by revenue
        gallery_df = display_df[display_df["main_image"] != ""].sort_values("weekly_sales_filled", ascending=False).head(16)
        
        if not gallery_df.empty:
            cols = st.columns(4)
            for i, (_, row) in enumerate(gallery_df.iterrows()):
                with cols[i % 4]:
                    clean_title = (row['title'][:45] + '...') if len(row['title']) > 45 else row['title']
                    st.markdown(f"""
                        <div class="product-card">
                            <img src="{row['main_image']}" class="product-img">
                            <div style="margin-top:10px; height:50px; overflow:hidden;">
                                <b style="font-size:0.85rem; color:#333;">{clean_title}</b>
                            </div>
                            <div style="font-size:1.1rem; color:#00704A; font-weight:bold; margin-top:5px;">
                                {f_money(row['weekly_sales_filled'])}
                            </div>
                            <div style="font-size:0.75rem; color:#666; font-weight:600;">{row['capital_zone']}</div>
                            <div style="font-size:0.7rem; color:#999;">ASIN: {row['asin']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No product images available for the selected filter.")

except Exception as e:
    st.error(f"🛡️ Command Center Offline: {e}")
    st.exception(e)