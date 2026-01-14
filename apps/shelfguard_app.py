import streamlit as st
import pandas as pd
from data import get_all_data
from engine import run_weekly_analysis
from finance import analyze_capital_efficiency, f_money, f_pct

st.set_page_config(page_title="ShelfGuard OS", layout="wide", page_icon="🛡️")

# Professional Styling: Green accents for Starbucks brand alignment
st.markdown("""<style>
    .stMetric { background: #ffffff; padding: 15px; border-left: 5px solid #00704A; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    h3 { padding-top: 1rem; padding-bottom: 0.5rem; }
    </style>""", unsafe_allow_html=True)

try:
    # 1. Load Data
    df = get_all_data()
    all_weeks = sorted(df["week_start"].unique(), reverse=True)
    selected_week = st.sidebar.selectbox("Fiscal Period", all_weeks)
    
    # 2. Run Strategic Analysis (Engine)
    res = run_weekly_analysis(df, selected_week)
    
    # 3. Calculate Board-Level Metrics (Finance)
    fin_metrics = analyze_capital_efficiency(res["capital_flow"])
    
    st.title("🛡️ ShelfGuard: Unified Command Center")
    
    # ==========================================================
    # STRATEGIC LAYER: THE CFO & VP VIEW
    # Goal: Ensure Capital Efficiency. "Are we funding the right things?"
    # ==========================================================
    st.header("1. Strategic Capital Allocation")
    
    # The "Board Room" Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Weekly Revenue", f_money(fin_metrics["total_rev"]))
    
    c2.metric("Growth Allocation", f_pct(fin_metrics["growth_alloc"]), 
              delta="Target: 30%")
    
    c3.metric("Inefficient Capital (Drag)", f_money(res["capital_flow"].get("📉 DRAG (Waste)", 0)), 
              delta=f"{fin_metrics['drag_pct']:.1%} of Portfolio", delta_color="inverse")
              
    c4.metric("Annualized Waste Risk", f_money(fin_metrics["annualized_waste"]), 
              delta="-Cash Bleed", delta_color="inverse")
    
    # Visual Health Bar
    st.markdown("#### Portfolio Efficiency Health")
    st.progress(fin_metrics["efficiency_score"] / 100, 
                text=f"Efficiency Score: {fin_metrics['efficiency_score']:.0f}/100")
    
    st.divider()

    # ==========================================================
    # TACTICAL LAYER: THE MANAGER VIEW
    # Goal: Operational Execution. "What do I do right now?"
    # ==========================================================
    st.header("2. Operational Response Matrix")
    
    col_filter, col_kpi = st.columns([1, 3])
    with col_filter:
        st.info("🎯 **Directives for Ad & Ecom Teams**")
        # Default to showing 'Drag' items because that is the immediate fire to put out
        zone_filter = st.multiselect("Filter by Strategic Zone", 
                                     options=["📉 DRAG (Waste)", "🚀 FRONTIER (Growth)", "🏰 FORTRESS (Cash Flow)"], 
                                     default=["📉 DRAG (Waste)"])
    
    # Filter Data based on selection
    display_df = res["data"].copy()
    if zone_filter:
        display_df = display_df[display_df["capital_zone"].isin(zone_filter)]
    
    # THE "ACTION" TABLE
    st.dataframe(
        display_df[["title", "capital_zone", "weekly_sales_filled", "ad_action", "ecom_action"]].rename(columns={
            "title": "Product Line",
            "capital_zone": "Status Zone",
            "weekly_sales_filled": "Revenue Vol",
            "ad_action": "📣 Media Buyer Directive",   # Explicitly labeled for Ad Manager
            "ecom_action": "📦 Ops Manager Lever"     # Explicitly labeled for Ecom Manager
        }).sort_values("Revenue Vol", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "Revenue Vol": st.column_config.NumberColumn(format="$%,.0f"),
        }
    )

except Exception as e:
    st.error(f"System Loading Error: {e}")