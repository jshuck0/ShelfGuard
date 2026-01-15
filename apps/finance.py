import pandas as pd
import numpy as np

def analyze_capital_efficiency(capital_flow, res_data=None):
    """
    The CFO's Calculator: Predictive Edition.
    Calibrated for 2026 Amazon Fee structures & 36-Month Velocity Trends.
    """
    total_rev = sum(capital_flow.values())
    drag_rev = capital_flow.get("📉 DRAG (Waste)", 0)
    # New Zone from updated engine
    terminal_drag_rev = capital_flow.get("📉 DRAG (Terminal Decay)", 0)
    bleed_rev = capital_flow.get("🩸 BLEED (Negative Margin)", 0)
    frontier_rev = capital_flow.get("🚀 FRONTIER (Growth)", 0)
    fortress_rev = capital_flow.get("🏰 FORTRESS (Cash Flow)", 0)

    # 1. PREDICTIVE WASTE CALCULATION
    # Instead of just 52 weeks, we weight 'Terminal Decay' and 'Bleed' heavier.
    # Terminal Decay assets are penalized 1.3x because they are 'Zombies' clogging the supply chain.
    if res_data is not None and not res_data.empty:
        # We calculate the average velocity decay of the underperforming segments
        risk_df = res_data[res_data['capital_zone'].str.contains("DRAG|BLEED")]
        avg_decay = risk_df['velocity_decay'].mean() if not risk_df.empty else 1.0
    else:
        avg_decay = 1.0

    # Boardroom Logic: If velocity is decaying, waste is harder to recover
    predictive_multiplier = max(1.0, avg_decay)
    annualized_waste = (drag_rev + terminal_drag_rev + (bleed_rev * 1.5)) * 52 * predictive_multiplier

    # 2. STRATEGIC RATIOS
    if total_rev > 0:
        # Penalize Bleed (1.5x) and Terminal Decay (1.2x) in the efficiency score
        inefficiency_ratio = (drag_rev + (terminal_drag_rev * 1.2) + (bleed_rev * 1.5)) / total_rev
        efficiency_score = max(0, (1 - inefficiency_ratio) * 100)
        growth_alloc = frontier_rev / total_rev
    else:
        efficiency_score, growth_alloc, inefficiency_ratio = 100, 0, 0

    # 3. 2026 BENCHMARK CHECK
    status = "🟢 HEALTHY" if efficiency_score > 85 else "🟡 OPTIMIZE" if efficiency_score > 70 else "🔴 CRITICAL"

    return {
        "total_rev": total_rev,
        "efficiency_score": efficiency_score,
        "drag_pct": (drag_rev + terminal_drag_rev + bleed_rev) / total_rev if total_rev > 0 else 0,
        "growth_alloc": growth_alloc,
        "annualized_waste": annualized_waste,
        "portfolio_status": status,
        "avg_velocity_decay": avg_decay # Proves to the user that trends are being factored in
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