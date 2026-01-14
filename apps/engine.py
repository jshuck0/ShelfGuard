import pandas as pd
import numpy as np

def _safe_float(val, default=0.0):
    """Prevents crashes from dirty data (NoneType > int errors)."""
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default

def analyze_strategic_matrix(row):
    """
    The Unified Brain: Generates directives for ALL 3 Personas.
    """
    # 1. Sanitize & Normalize Data
    bb = _safe_float(row.get('amazon_bb_share'), default=1.0)
    price = _safe_float(row.get('filled_price'), default=0.0)
    raw_comp = row.get('new_fba_price')
    comp = _safe_float(raw_comp, default=price) 
    gap = (price - comp) / comp if comp > 0 else 0
    weeks_cover = _safe_float(row.get('weeks_of_cover'), default=4.0)
    
    # --- STRATEGIC LAYER (CFO) ---
    # Classify the asset into a Capital Allocation Zone
    if bb > 0.90 and gap < 0.05:
        capital_zone = "🏰 FORTRESS (Cash Flow)" # Protect at all costs
    elif bb > 0.60 and bb < 0.90:
        capital_zone = "🚀 FRONTIER (Growth)"    # Invest aggressively to win
    else:
        capital_zone = "📉 DRAG (Waste)"         # Divest / Fix immediately

    # --- TACTICAL LAYER: AD MANAGER (Media Execution) ---
    if bb < 0.50: 
        ad_action = "🛑 HARD PAUSE (Leakage)"
    elif bb < 0.85 and gap > 0.05: 
        ad_action = "🛡️ DEFENSIVE BIDDING"
    elif bb > 0.95: 
        ad_action = "🚀 SCALE (High Confidence)"
    else: 
        ad_action = "⚖️ ROAS OPTIMIZATION"

    # --- TACTICAL LAYER: ECOM MANAGER (Ops Levers) ---
    if gap > 0.10 and bb < 0.70: 
        ecom_action = "🎫 CLIP COUPON (Price War)"
    elif bb < 0.90 and gap <= 0.02: 
        ecom_action = "📦 AUDIT INVENTORY"
    elif bb > 0.98:
        ecom_action = "📈 TEST PRICE (+3%)"
    else: 
        ecom_action = "✅ MAINTAIN MSRP"

    return pd.Series([ad_action, ecom_action, capital_zone, gap])

def run_weekly_analysis(rows, week):
    target_date = pd.to_datetime(week).normalize()
    df_w = rows[rows["week_start"] == target_date].copy()
    sbux = df_w[df_w["is_starbucks"] == 1].copy()

    if sbux.empty: 
        return {"data": pd.DataFrame(), "capital_flow": {}, "total_rev": 0}

    # Clean Revenue/Sales columns before math
    sbux['weekly_sales_filled'] = sbux['weekly_sales_filled'].apply(lambda x: _safe_float(x))
    sbux['amazon_bb_share'] = sbux['amazon_bb_share'].apply(lambda x: _safe_float(x, 1.0))

    # RUN THE UNIFIED ANALYSIS
    sbux[['ad_action', 'ecom_action', 'capital_zone', 'price_gap']] = sbux.apply(analyze_strategic_matrix, axis=1)
    
    # AGGREGATE FOR CFO (Summing revenue by zone)
    capital_flow = sbux.groupby("capital_zone")['weekly_sales_filled'].sum().to_dict()
    
    return {
        "data": sbux,
        "capital_flow": capital_flow,
        "total_rev": sbux['weekly_sales_filled'].sum()
    }