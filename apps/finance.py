import pandas as pd

# Constants for calculations
BLEED_MULTIPLIER = 1.5
TERMINAL_DECAY_MULTIPLIER = 1.2
ANNUALIZATION_WEEKS = 52

def analyze_capital_efficiency(capital_flow, res_data=None):
    """
    The CFO's Calculator: Predictive Edition.
    Calibrated for 2026 Amazon Fee structures & 36-Month Velocity Trends.

    Performance optimizations:
    - Pre-computed constants
    - Reduced redundant calculations
    - Optimized conditional logic
    """
    # Extract revenue by zone (with defaults)
    total_rev = sum(capital_flow.values())
    drag_rev = capital_flow.get("📉 DRAG (Waste)", 0)
    terminal_drag_rev = capital_flow.get("📉 DRAG (Terminal Decay)", 0)
    bleed_rev = capital_flow.get("🩸 BLEED (Negative Margin)", 0)
    frontier_rev = capital_flow.get("🚀 FRONTIER (Growth)", 0)
    fortress_rev = capital_flow.get("🏰 FORTRESS (Cash Flow)", 0)

    # 1. PREDICTIVE WASTE CALCULATION
    # Calculate average velocity decay of underperforming segments (optimized)
    avg_decay = 1.0
    if res_data is not None and not res_data.empty:
        risk_df = res_data[res_data['capital_zone'].str.contains("DRAG|BLEED", na=False)]
        if not risk_df.empty:
            avg_decay = risk_df['velocity_decay'].mean()

    # Boardroom Logic: If velocity is decaying, waste is harder to recover
    predictive_multiplier = max(1.0, avg_decay)
    weighted_waste = drag_rev + terminal_drag_rev + (bleed_rev * BLEED_MULTIPLIER)
    annualized_waste = weighted_waste * ANNUALIZATION_WEEKS * predictive_multiplier

    # 2. STRATEGIC RATIOS (optimized calculation)
    if total_rev > 0:
        # Penalize Bleed (1.5x) and Terminal Decay (1.2x) in the efficiency score
        inefficiency_ratio = (drag_rev + (terminal_drag_rev * TERMINAL_DECAY_MULTIPLIER) +
                             (bleed_rev * BLEED_MULTIPLIER)) / total_rev
        efficiency_score = max(0, (1 - inefficiency_ratio) * 100)
        growth_alloc = frontier_rev / total_rev
        drag_pct = (drag_rev + terminal_drag_rev + bleed_rev) / total_rev
    else:
        efficiency_score, growth_alloc, inefficiency_ratio, drag_pct = 100, 0, 0, 0

    # 3. 2026 BENCHMARK CHECK (optimized conditional)
    status = "🟢 HEALTHY" if efficiency_score > 85 else ("🟡 OPTIMIZE" if efficiency_score > 70 else "🔴 CRITICAL")

    return {
        "total_rev": total_rev,
        "efficiency_score": efficiency_score,
        "drag_pct": drag_pct,
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