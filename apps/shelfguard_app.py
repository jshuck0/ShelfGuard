import streamlit as st
st.set_page_config(page_title="Analytics Pro", page_icon="📊") 
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)
import pandas as pd
from data import get_all_data
from engine import run_weekly_analysis
from finance import analyze_capital_efficiency, f_money, f_pct

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ShelfGuard OS", layout="wide", page_icon="🛡️")

# --- CSS CONSOLIDATION (Unified Styling) ---
st.markdown("""
    <style>
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
    """, unsafe_allow_html=True)

try:
    # 2. DATA INGESTION
    with st.spinner("🔄Loading..."):
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
    # We use a spinner here because the 36-month trend math is computationally heavy
    with st.spinner("🧠 Executing Predictive Intelligence..."):
        res = run_weekly_analysis(df_raw, selected_week)
        
    # --- DATA INTEGRITY CHECK ---
    # Ensure the engine returned the required dataframes
    if res["data"].empty:
        st.info(f"📅 No Starbucks activity recorded for the week of {selected_week}.")
        st.stop()

    fin = analyze_capital_efficiency(res["capital_flow"], res_data=res["data"])
    
    # 4. HEADER
    st.title("🛡️ ShelfGuard: Unified Command Center")
    st.caption(f"Strategy & Analytics Dashboard | Predictive Intelligence Active (36M Lookback)")
    
    # --- ROW 1: STRATEGIC CAPITAL METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Weekly Portfolio Rev", f_money(fin["total_rev"]))
    
    c2.metric("Efficiency Score", f"{fin.get('efficiency_score', 0):.0f}/100", 
              help="Composite score of margin health and inventory velocity.")
    
    # Calculate Inefficient Capital dynamically
    bad_zones = ["📉 DRAG (Waste)", "📉 DRAG (Terminal Decay)", "🩸 BLEED (Negative Margin)"]
    bad_spend = sum(res["capital_flow"].get(zone, 0) for zone in bad_zones)
    
    c3.metric("Inefficient Capital", f_money(bad_spend), 
              delta=f"{fin.get('drag_pct', 0):.1%} Exposure", delta_color="inverse")
    
    c4.metric("Annualized Risk", f_money(fin.get("annualized_waste", 0)), 
              delta=f"{fin.get('avg_velocity_decay', 1.0):.2f}x Velocity Decay", delta_color="inverse")
    
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
        # We wrap the dataframe in a try-block because LineChartColumn is sensitive to data types
        try:
            # Prepare DataFrame for display
            final_df = display_df[[
                "variation_attributes", "capital_zone", "efficiency_score", "velocity_decay", 
                "Trend (36M)", "net_margin", "weekly_sales_filled", "ad_action", "ecom_action"
            ]].rename(columns={
                "variation_attributes": "Product Details",
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
            st.error("Unable to render Performance Matrix. This is usually due to corrupted historical rank data.")
            st.exception(table_err)

    with tab2:
        st.write("### Starbucks Visual Portfolio Audit")
        # Filter for rows that actually have images
        gallery_df = display_df[display_df["main_image"] != ""].sort_values("weekly_sales_filled", ascending=False).head(16)
        
        if not gallery_df.empty:
            cols = st.columns(4)
            for i, (_, row) in enumerate(gallery_df.iterrows()):
                with cols[i % 4]:
                    # Truncate title for UI consistency
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