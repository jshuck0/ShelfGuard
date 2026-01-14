import pandas as pd

def analyze_capital_efficiency(capital_flow):
    """
    The CFO's Calculator. 
    Converts raw revenue zones into Board-Level Efficiency Metrics.
    """
    # 1. Extract Totals
    total_rev = sum(capital_flow.values())
    drag_rev = capital_flow.get("📉 DRAG (Waste)", 0)
    frontier_rev = capital_flow.get("🚀 FRONTIER (Growth)", 0)
    fortress_rev = capital_flow.get("🏰 FORTRESS (Cash Flow)", 0)

    # 2. Calculate Ratios (The "Health" Score)
    # If total_rev is 0, prevent division by zero errors
    if total_rev > 0:
        drag_pct = drag_rev / total_rev
        growth_alloc = frontier_rev / total_rev
        efficiency_score = (1 - drag_pct) * 100 # 100 is perfect, 0 is all waste
    else:
        drag_pct = 0
        growth_alloc = 0
        efficiency_score = 100

    # 3. Project Annualized Impact (The "So What")
    # "If we fix this Drag, here is how much cash we unlock per year"
    annualized_waste = drag_rev * 52 
    
    return {
        "total_rev": total_rev,
        "efficiency_score": efficiency_score,
        "drag_pct": drag_pct,
        "growth_alloc": growth_alloc,
        "annualized_waste": annualized_waste
    }

# --- Formatting Helpers (DRY Principle) ---
def f_money(v): 
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    return f"${v:,.0f}"

def f_pct(v): 
    return f"{v:.1%}"