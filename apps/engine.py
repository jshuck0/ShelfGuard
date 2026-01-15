import pandas as pd
import numpy as np

def _safe_float(val, default=0.0):
    """Bulletproof float conversion for financial calculations."""
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default

def calculate_net_realization(row):
    """Calibrated for 2026 Amazon Fee structures (Referral 15% + FBA Hikes)."""
    price = _safe_float(row.get('filled_price'))
    if price <= 0: return 0.0
    
    # Updated Fee Logic
    fba_fee = _safe_float(row.get('fba_fees'), default=4.50) 
    referral_fee = price * 0.15 
    storage_cost = _safe_float(row.get('package_vol_cf'), default=0.05) * 0.87 
    cogs = price * 0.25 # Estimated COGS for CPG/Starbucks portfolio
    
    net_profit = price - (fba_fee + referral_fee + storage_cost + cogs)
    return net_profit / price

def calculate_row_efficiency(bb, gap, weeks_cover, net_margin, velocity_factor):
    """Portfolio efficiency scoring (0-100) with Predictive Multi-Year Weighting."""
    base_score = bb * 40
    price_score = max(0, 30 - (gap * 300))
    margin_score = min(30, max(0, (net_margin - 0.10) * 100))
    
    # Momentum Penalty: If velocity is decaying (factor > 1.1), penalize the capital efficiency
    momentum_penalty = 1 / velocity_factor if velocity_factor > 1.1 else 1.0
    stock_factor = 1.0 if weeks_cover >= 2.0 else 0.2
    
    return min(100, int((base_score + price_score + margin_score) * stock_factor * momentum_penalty))

def analyze_strategic_matrix(row):
    """Assigns products to Strategic Zones using 36-month Predictive Signals."""
    bb = _safe_float(row.get('amazon_bb_share'), default=1.0)
    price = _safe_float(row.get('filled_price'), default=0.0)
    comp = _safe_float(row.get('new_fba_price'), default=price) 
    gap = (price - comp) / comp if comp > 0 else 0
    weeks_cover = _safe_float(row.get('weeks_of_cover'), default=4.0)
    velocity_decay = _safe_float(row.get('velocity_decay'), default=1.0)
    
    net_margin = calculate_net_realization(row)
    efficiency_score = calculate_row_efficiency(bb, gap, weeks_cover, net_margin, velocity_decay)

    # 1. PREDICTIVE SEGMENTATION LOGIC
    if net_margin < 0.05:
        capital_zone = "🩸 BLEED (Negative Margin)"
    elif velocity_decay > 1.5:
        # High Decay (worsening rank) triggers Terminal Status regardless of current margin
        capital_zone = "📉 DRAG (Terminal Decay)"
    elif bb > 0.85 and net_margin > 0.15 and velocity_decay < 1.1:
        capital_zone = "🏰 FORTRESS (Cash Flow)"
    elif bb > 0.60 and net_margin > 0.10:
        capital_zone = "🚀 FRONTIER (Growth)"
    else:
        capital_zone = "📉 DRAG (Waste)"

    # 2. ACTIONABLE DIRECTIVES (Agentic Execution)
    if "BLEED" in capital_zone or velocity_decay > 1.8:
        ecom_action = "💀 LIQUIDATE / EXIT"
        ad_action = "🛑 HARD PAUSE"
    elif gap > 0.08:
        ecom_action = "🎫 CLIP COUPON"
        ad_action = "⚖️ MAINTAIN ROAS"
    elif "FORTRESS" in capital_zone and velocity_decay < 0.9:
        ecom_action = "📈 TEST PRICE (+5%)"
        ad_action = "🚀 SCALE AD SPEND"
    else:
        ecom_action = "✅ MAINTAIN"
        ad_action = "⚖️ ROAS OPTIMIZATION"

    return pd.Series([ad_action, ecom_action, capital_zone, gap, efficiency_score, net_margin])

def run_weekly_analysis(all_rows, selected_week):
    """
    Executes the 36-month trend analysis and generates the Strategic Matrix.
    """
    target_date = pd.to_datetime(selected_week).date()
    
    # 1. PREDICTIVE ENGINE: Filter for all history up to the target date
    history = all_rows[all_rows["week_start"].dt.date <= target_date].copy()
    
    # Force Numeric for rank calculations
    history['sales_rank_filled'] = pd.to_numeric(history['sales_rank_filled'], errors='coerce').fillna(0)
    
    # Calculate Long-Term Average (3-Year Baseline)
    lt_avg = history.groupby('asin')['sales_rank_filled'].mean()
    
    # Calculate Recent Average (Last 8 weeks of available data)
    recent = history.sort_values(['asin', 'week_start'], ascending=[True, False])
    rt_avg = recent.groupby('asin').head(8).groupby('asin')['sales_rank_filled'].mean()
    
    # THE SPARKLINE FIX: Ensure data is a pure list of floats for st.LineChartColumn
    trend_arrays = history.sort_values('week_start').groupby('asin')['sales_rank_filled'].apply(lambda x: [float(v) for v in x])
    
    # Create the Intelligence Layer
    velocity_intel = pd.DataFrame({
        'velocity_decay': (rt_avg / lt_avg).fillna(1.0).round(2),
        'Trend (36M)': trend_arrays
    }).reset_index()

    # 2. SNAPSHOT: Filter for the target week
    df_snapshot = all_rows[all_rows["week_start"].dt.date == target_date].copy()
    sbux = df_snapshot[df_snapshot["is_starbucks"] == 1].copy()

    if sbux.empty: 
        return {"data": pd.DataFrame(), "capital_flow": {}, "total_rev": 0}

    # 3. MERGE & EXECUTE
    sbux = sbux.merge(velocity_intel, on='asin', how='left')
    
    sbux[[
        'ad_action', 'ecom_action', 'capital_zone', 'price_gap', 
        'efficiency_score', 'net_margin'
    ]] = sbux.apply(analyze_strategic_matrix, axis=1)
    
    # Aggregate Portfolio Metrics
    capital_flow = sbux.groupby("capital_zone")['weekly_sales_filled'].sum().to_dict()
    
    return {
        "data": sbux, 
        "capital_flow": capital_flow, 
        "total_rev": sbux['weekly_sales_filled'].sum()
    }