"""
ShelfGuard Utils Module
========================
AI-powered strategic intelligence for Amazon portfolio management.

This module contains:
- ai_engine: UNIFIED AI ENGINE (Strategic Classification + Predictive Intelligence)
- keepa_extended_fields: Extended Keepa Product Finder field definitions
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
    StrategicBrief,  # Now includes both strategic + predictive outputs
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
    # Velocity Extraction (for backfill integration)
    extract_velocity_trends,
    extract_portfolio_velocity,
)

from utils.keepa_extended_fields import (
    KeepaField,
    AmazonDomain,
    ALL_EXTENDED_FIELDS,
    get_recommended_fields_for_triangulation,
    build_product_finder_query,
    transform_product_finder_row,
    normalize_keepa_price,
    keepa_minutes_to_days,
    keepa_rating_to_stars,
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
    "calculate_portfolio_predictive_risk",  # Portfolio risk aggregation
    # Velocity Extraction (for backfill integration)
    "extract_velocity_trends",      # Extract velocity from weekly backfill for one ASIN
    "extract_portfolio_velocity",   # Batch extract velocity for all ASINs
    
    # Keepa Extended Fields
    "KeepaField",
    "AmazonDomain",
    "ALL_EXTENDED_FIELDS",
    "get_recommended_fields_for_triangulation",
    "build_product_finder_query",
    "transform_product_finder_row",
    "normalize_keepa_price",
    "keepa_minutes_to_days",
    "keepa_rating_to_stars",
    
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
]
