import pandas as pd

# Constants for calculations
BLEED_MULTIPLIER = 1.5
TERMINAL_DECAY_MULTIPLIER = 1.2
ANNUALIZATION_WEEKS = 52

def analyze_capital_efficiency(predictive_zones, res_data=None):
    """
    The CFO's Calculator: Predictive Edition.
    
    NOW USES PREDICTIVE ZONES instead of legacy capital_flow:
    - DEFEND: Products needing immediate action (replaces BLEED/DRAG)
    - EXPLOIT: Growth opportunities (replaces FRONTIER)
    - REPLENISH: Inventory alerts
    - HOLD: Healthy products (replaces FORTRESS)

    Performance optimizations:
    - Pre-computed constants
    - Reduced redundant calculations
    - Optimized conditional logic
    """
    # Extract revenue by predictive zone (with defaults)
    total_rev = sum(predictive_zones.values())
    
    # NEW: Predictive zone mappings
    defend_rev = predictive_zones.get("🛡️ DEFEND", 0)
    replenish_rev = predictive_zones.get("🔄 REPLENISH", 0)
    exploit_rev = predictive_zones.get("⚡ EXPLOIT", 0)
    hold_rev = predictive_zones.get("✅ HOLD", 0)
    
    # 1. PREDICTIVE RISK CALCULATION
    # Calculate average velocity decay if data available
    avg_decay = 1.0
    if res_data is not None and not res_data.empty:
        # Use predictive_zone instead of capital_zone
        if 'predictive_zone' in res_data.columns:
            risk_df = res_data[res_data['predictive_zone'].str.contains("DEFEND|REPLENISH", na=False)]
        elif 'is_healthy' in res_data.columns:
            risk_df = res_data[res_data['is_healthy'] == False]
        else:
            risk_df = pd.DataFrame()
        
        if not risk_df.empty and 'velocity_decay' in risk_df.columns:
            avg_decay = risk_df['velocity_decay'].mean()

    # Boardroom Logic: Products in DEFEND/REPLENISH are at risk
    predictive_multiplier = max(1.0, avg_decay)
    weighted_risk = (defend_rev * BLEED_MULTIPLIER) + replenish_rev
    annualized_waste = weighted_risk * ANNUALIZATION_WEEKS * predictive_multiplier

    # 2. STRATEGIC RATIOS (optimized calculation)
    if total_rev > 0:
        # Efficiency based on healthy products (HOLD + EXPLOIT)
        healthy_ratio = (hold_rev + exploit_rev) / total_rev
        efficiency_score = max(0, healthy_ratio * 100)
        growth_alloc = exploit_rev / total_rev
        risk_pct = (defend_rev + replenish_rev) / total_rev
    else:
        efficiency_score, growth_alloc, risk_pct = 100, 0, 0

    # 3. 2026 BENCHMARK CHECK (optimized conditional)
    status = "🟢 HEALTHY" if efficiency_score > 85 else ("🟡 OPTIMIZE" if efficiency_score > 70 else "🔴 CRITICAL")

    return {
        "total_rev": total_rev,
        "efficiency_score": efficiency_score,
        "risk_pct": risk_pct,  # Renamed from drag_pct
        "growth_alloc": growth_alloc,
        "annualized_waste": annualized_waste,
        "portfolio_status": status,
        "avg_velocity_decay": avg_decay
    }

# --- Formatting Helpers (Unchanged) ---
def f_money(v): 
    if v is None: return "$0"
    if v >= 1_000_000_000: return f"${v/1_000_000_000:.2f}B"
    if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
    if v >= 1_000: return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def f_pct(v): 
    if v is None: return "0.0%"
    return f"{v:.1%}"