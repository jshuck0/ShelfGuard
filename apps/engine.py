import pandas as pd
import numpy as np

def _safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default

def calculate_net_realization(row):
    """Calibrated for 2026 Amazon Fee structures."""
    price = _safe_float(row.get('filled_price'))
    if price <= 0: return 0.0
    
    # Fees & Logistics
    fba_fee = _safe_float(row.get('fba_fees'), default=4.50) 
    referral_fee = price * 0.15 
    storage_cost = _safe_float(row.get('package_vol_cf'), default=0.05) * 0.87 
    cogs = price * 0.25 # Estimated COGS for Starbucks
    
    net_profit = price - (fba_fee + referral_fee + storage_cost + cogs)
    return net_profit / price

def calculate_row_efficiency(bb, gap, weeks_cover, net_margin):
    """Portfolio efficiency scoring (0-100)."""
    base_score = bb * 40
    price_score = max(0, 30 - (gap * 300))
    margin_score = min(30, max(0, (net_margin - 0.10) * 100))
    stock_factor = 1.0 if weeks_cover >= 2.0 else 0.2
    return min(100, int((base_score + price_score + margin_score) * stock_factor))

def analyze_strategic_matrix(row):
    """Assigns products to Strategic Zones and provides directives."""
    bb = _safe_float(row.get('amazon_bb_share'), default=1.0)
    price = _safe_float(row.get('filled_price'), default=0.0)
    comp = _safe_float(row.get('new_fba_price'), default=price) 
    gap = (price - comp) / comp if comp > 0 else 0
    weeks_cover = _safe_float(row.get('weeks_of_cover'), default=4.0)
    
    net_margin = calculate_net_realization(row)
    efficiency_score = calculate_row_efficiency(bb, gap, weeks_cover, net_margin)

    # 1. SEGMENTATION
    if net_margin < 0.05:
        capital_zone = "🩸 BLEED (Negative Margin)"
    elif bb > 0.85 and net_margin > 0.15:
        capital_zone = "🏰 FORTRESS (Cash Flow)"
    elif bb > 0.60 and net_margin > 0.10:
        capital_zone = "🚀 FRONTIER (Growth)"
    else:
        capital_zone = "📉 DRAG (Waste)"

    # 2. ACTIONABLE DIRECTIVES
    if capital_zone == "🩸 BLEED (Negative Margin)":
        ecom_action = "💀 KILL SKU / REPRICE"
        ad_action = "🛑 HARD PAUSE"
    elif gap > 0.08:
        ecom_action = "🎫 CLIP COUPON"
        ad_action = "⚖️ MAINTAIN ROAS"
    elif bb > 0.95 and net_margin > 0.20:
        ecom_action = "📈 TEST PRICE (+5%)"
        ad_action = "🚀 SCALE AD SPEND"
    else:
        ecom_action = "✅ MAINTAIN"
        ad_action = "⚖️ ROAS OPTIMIZATION"

    return pd.Series([ad_action, ecom_action, capital_zone, gap, efficiency_score, net_margin])

def run_weekly_analysis(rows, week):
    """Filters data for the selected week and executes the matrix."""
    # --- THE DATE FIX: Ensuring exact matches between DB and Sidebar ---
    target_date = pd.to_datetime(week).date()
    rows = rows.copy()
    rows["week_start_dt"] = pd.to_datetime(rows["week_start"]).dt.date
    
    # Filter for the week
    df_w = rows[rows["week_start_dt"] == target_date].copy()
    
    # Filter for Starbucks Portfolio
    sbux = df_w[df_w["is_starbucks"] == 1].copy()

    if sbux.empty: 
        return {"data": pd.DataFrame(), "capital_flow": {}, "total_rev": 0}

    # Apply calculations
    sbux[['ad_action', 'ecom_action', 'capital_zone', 'price_gap', 'efficiency_score', 'net_margin']] = sbux.apply(analyze_strategic_matrix, axis=1)
    
    # Group for the charts
    capital_flow = sbux.groupby("capital_zone")['weekly_sales_filled'].sum().to_dict()
    
    return {
        "data": sbux, 
        "capital_flow": capital_flow, 
        "total_rev": sbux['weekly_sales_filled'].sum()
    }