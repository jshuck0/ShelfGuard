import streamlit as st
import pandas as pd
from data import get_all_data
from engine import run_weekly_analysis
from finance import analyze_capital_efficiency, f_money, f_pct

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="ShelfGuard OS", layout="wide", page_icon="🛡️")

# --- CSS CONSOLIDATION (Unified Styling) ---
st.markdown("""
    <style>
    /* 1. FORCE METRIC CARDS - Targeting the specific container */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-left: 6px solid #00704A !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        width: 100% !important;
    }

    /* 2. FORCE METRIC LABELS & VALUES */
    [data-testid="stMetricLabel"] p {
        font-size: 1rem !important;
        color: #555555 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricValue"] div {
        font-size: 1.8rem !important;
        color: #00704A !important;
        font-weight: 800 !important;
    }

    /* 3. VISUAL AUDIT CARDS */
    .product-card {
        background-color: white !important;
        border: 1px solid #e6e9ef !important;
        border-radius: 12px !important;
        padding: 15px !important;
        text-align: center !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s ease !important;
    }

    .product-card:hover {
        border-color: #00704A !important;
        box-shadow: 0 8px 16px rgba(0,112,74,0.15) !important;
        transform: translateY(-5px) !important;
    }

    /* Force images to be centered and constrained */
    .product-img {
        max-height: 140px !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    # 2. DATA INGESTION
    df = get_all_data()
    
    # --- FIX: Force conversion to datetime before using .dt accessor ---
    df["week_start"] = pd.to_datetime(df["week_start"], errors='coerce')
    # --------------------------------------------------------------------
    
    # Ensure dates are normalized for selection
    all_weeks = sorted(df["week_start"].dt.date.dropna().unique(), reverse=True)
    selected_week = st.sidebar.selectbox("Fiscal Period", all_weeks)
    
    # 3. ANALYSIS EXECUTION
    res = run_weekly_analysis(df, selected_week)
    fin = analyze_capital_efficiency(res["capital_flow"])
    
    st.title("🛡️ ShelfGuard: Unified Command Center")
    st.caption(f"Strategy & Analytics Dashboard | Period Ending: {selected_week}")
    
    # --- ROW 1: STRATEGIC CAPITAL METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Weekly Portfolio Rev", f_money(fin["total_rev"]))
    c2.metric("Efficiency Score", f"{fin['efficiency_score']:.0f}/100")
    
    bad_spend = res["capital_flow"].get("📉 DRAG (Waste)", 0) + res["capital_flow"].get("🩸 BLEED (Negative Margin)", 0)
    c3.metric("Inefficient Capital", f_money(bad_spend), 
              delta=f"{fin['drag_pct']:.1%} of Rev", delta_color="inverse")
              
    c4.metric("Annualized Waste Risk", f_money(fin["annualized_waste"]), 
              delta="Projected Leakage", delta_color="inverse")
    
    st.divider()

    # --- ROW 2: STRATEGIC ZONE FILTERING ---
    selected_zone = st.radio(
        "Strategic Filter", 
        options=["🏰 FORTRESS (Cash Flow)", "🚀 FRONTIER (Growth)", "📉 DRAG (Waste)", "🩸 BLEED (Negative Margin)"], 
        horizontal=True
    )

    display_df = res["data"].copy()
    if selected_zone:
        display_df = display_df[display_df["capital_zone"] == selected_zone]

    # --- VARIATION CLEANING ---
    def clean_var_string(s):
        val = str(s).strip()
        if not s or val in ["", "nan", "None", "Standard", "Standard Product"]:
            return "Standard SKU"
        return val

    display_df["Clean Details"] = display_df["variation_attributes"].apply(clean_var_string)

    # --- ROW 3: PRODUCT GALLERY & TABLE ---
    tab1, tab2 = st.tabs(["📊 Performance Matrix", "🖼️ Visual Audit"])

    with tab1:
        st.dataframe(
            display_df[[
                "Clean Details", "capital_zone", "efficiency_score", "net_margin", 
                "weekly_sales_filled", "ad_action", "ecom_action", "asin"
            ]].rename(columns={
                "Clean Details": "Product / Variation",
                "capital_zone": "Status",
                "efficiency_score": "Score",
                "net_margin": "Net Margin %",
                "weekly_sales_filled": "Revenue",
                "ad_action": "Media Directive",
                "ecom_action": "Ops Lever"
            }).sort_values("Revenue", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Revenue": st.column_config.NumberColumn(
                    "Revenue",
                    format="$%.2f"  # REMOVED COMMA: Streamlit handles thousands sep automatically
                ),
                "Net Margin %": st.column_config.NumberColumn(
                    "Net Margin %",
                    format="%.2f%%"  # Standardized percentage display
                ),
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0, 
                    max_value=100, 
                    format="%d",
                    color="#00704A"
                )
            }
        )

    with tab2:
        st.write("### Starbucks Visual Portfolio Audit")
        gallery_df = display_df[display_df["main_image"] != ""].sort_values("weekly_sales_filled", ascending=False).head(12)
    
        if not gallery_df.empty:
            cols = st.columns(4)
            for i, (_, row) in enumerate(gallery_df.iterrows()):
                with cols[i % 4]:
                    st.markdown(f"""
                        <div class="product-card">
                            <img src="{row['main_image']}" class="product-img">
                            <div style="margin-top:10px; height:45px; overflow:hidden;">
                                <b style="font-size:0.85rem; color:#333;">{row['Clean Details']}</b>
                            </div>
                            <div style="font-size:1.1rem; color:#00704A; font-weight:bold; margin:5px 0;">
                                {f_money(row['weekly_sales_filled'])}
                            </div>
                            <div style="font-size:0.7rem; color:#999;">ASIN: {row['asin']}</div>
                        </div>
                    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Command Center Offline: {e}")
    st.exception(e) 
    st.info("Check if sync_supabase.py was run after the new Keepa batches were fetched.")