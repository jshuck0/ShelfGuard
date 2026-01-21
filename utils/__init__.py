"""
ShelfGuard Utils Module
========================
AI-powered strategic intelligence for Amazon portfolio management.

This module contains:
- ai_engine: UNIFIED AI ENGINE (Strategic Classification + Predictive Intelligence)
- data_healer: Universal data filling and interpolation for all metrics

The AI Engine combines:
1. LLM-powered strategic state classification (FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL)
2. Predictive 30-day risk forecasting with velocity trends
3. Actionable alerts (Inventory, Pricing, Rank protection)
4. Model certainty based on data quality
"""

from utils.ai_engine import (
    # Unified AI Engine
    StrategicTriangulator,
    StrategicBrief,  # Now includes strategic + predictive + growth outputs
    StrategicState,
    triangulate_portfolio,
    get_portfolio_state_summary,
    analyze_strategy_with_llm,
    analyze_portfolio_async,
    STATE_DEFINITIONS,
    generate_portfolio_brief,
    generate_portfolio_brief_sync,
    # Predictive Components (integrated into StrategicBrief, but exposed for advanced use)
    PredictiveAlpha,
    calculate_predictive_alpha,
    calculate_portfolio_predictive_risk,
    calculate_portfolio_intelligence_vectorized,  # HIGH PERFORMANCE vectorized calculation
    # Velocity Extraction (for backfill integration)
    extract_velocity_trends,
    extract_portfolio_velocity,
    # Growth Intelligence (offensive layer)
    ExpansionAlpha,
    calculate_expansion_alpha,
    is_growth_eligible,
)

from utils.data_healer import (
    clean_and_interpolate_metrics,
    heal_price_metrics,
    heal_rank_metrics,
    heal_review_metrics,
    heal_competitive_metrics,
    generate_data_quality_report,
    validate_healing,
    MetricGroup,
    ALL_METRIC_GROUPS,
    # Variation deduplication
    apply_variation_deduplication,
    heal_market_snapshot,
)

__all__ = [
    # Unified AI Engine (Strategic + Predictive)
    "StrategicTriangulator",        # Main entry point - unified analysis
    "StrategicBrief",               # Unified output with strategic + predictive fields
    "StrategicState",               # Strategic state enum
    "triangulate_portfolio",        # Batch portfolio analysis
    "get_portfolio_state_summary",  # Portfolio summary
    "analyze_strategy_with_llm",    # Async LLM analysis
    "analyze_portfolio_async",      # Async batch analysis
    "STATE_DEFINITIONS",            # State visual properties
    "generate_portfolio_brief",     # Portfolio brief generation
    "generate_portfolio_brief_sync",# Sync portfolio brief
    # Predictive Components (for advanced use - integrated into StrategicBrief)
    "PredictiveAlpha",              # Predictive output dataclass
    "calculate_predictive_alpha",   # Direct predictive calculation
    "calculate_portfolio_predictive_risk",  # Portfolio risk + growth aggregation
    "calculate_portfolio_intelligence_vectorized",  # Vectorized calculation (100x faster)
    # Velocity Extraction (for backfill integration)
    "extract_velocity_trends",      # Extract velocity from weekly backfill for one ASIN
    "extract_portfolio_velocity",   # Batch extract velocity for all ASINs
    # Growth Intelligence (offensive layer)
    "ExpansionAlpha",               # Growth opportunity dataclass
    "calculate_expansion_alpha",    # Calculate growth potential
    "is_growth_eligible",           # Velocity validation gate
    
    # Data Healer
    "clean_and_interpolate_metrics",
    "heal_price_metrics",
    "heal_rank_metrics",
    "heal_review_metrics",
    "heal_competitive_metrics",
    "generate_data_quality_report",
    "validate_healing",
    "MetricGroup",
    "ALL_METRIC_GROUPS",
    # Variation deduplication
    "apply_variation_deduplication",
    "heal_market_snapshot",
]
