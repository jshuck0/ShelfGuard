import pandas as pd

def analyze_capital_efficiency(capital_flow, res_data=None):
    """
    The CFO's Calculator. 
    Calibrated for 2026 Amazon Fee structures (Referral 15% + FBA Hikes).
    """
    total_rev = sum(capital_flow.values())
    drag_rev = capital_flow.get("📉 DRAG (Waste)", 0)
    bleed_rev = capital_flow.get("🩸 BLEED (Negative Margin)", 0)
    frontier_rev = capital_flow.get("🚀 FRONTIER (Growth)", 0)
    fortress_rev = capital_flow.get("🏰 FORTRESS (Cash Flow)", 0)

    if total_rev > 0:
        # STRATEGIC RATIOS
        # We weight 'Bleed' heavier (1.5x) because it actively destroys capital
        inefficiency_ratio = (drag_rev + (bleed_rev * 1.5)) / total_rev
        growth_alloc = frontier_rev / total_rev
        
        # EFFICIENCY SCORE (0-100)
        # Penalizes the portfolio heavily if 'Bleed' SKUs are high revenue
        efficiency_score = max(0, (1 - inefficiency_ratio) * 100)
    else:
        efficiency_score, growth_alloc, inefficiency_ratio = 100, 0, 0

    # ANNUALIZED IMPACT (The "Boardroom" Number)
    # This represents the cash that could be recovered by killing 'Bleed' 
    # and optimizing 'Drag' SKUs.
    annualized_waste = (drag_rev + bleed_rev) * 52 
    
    # 2026 BENCHMARK CHECK
    status = "🟢 HEALTHY" if efficiency_score > 85 else "🟡 OPTIMIZE" if efficiency_score > 70 else "🔴 CRITICAL"

    return {
        "total_rev": total_rev,
        "efficiency_score": efficiency_score,
        "drag_pct": (drag_rev + bleed_rev) / total_rev if total_rev > 0 else 0,
        "growth_alloc": growth_alloc,
        "annualized_waste": annualized_waste,
        "portfolio_status": status
    }

# --- Formatting Helpers ---
# Use these in Streamlit to keep the UI clean and scannable

def f_money(v): 
    """Formats large numbers into B, M, or K strings."""
    if v is None: return "$0"
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def f_pct(v): 
    """Formats floats into clean 1-decimal percentages."""
    if v is None: return "0.0%"
    return f"{v:.1%}"