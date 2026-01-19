"""
ShelfGuard Utils Module
========================
AI-powered strategic intelligence for Amazon portfolio management.

This module contains:
- ai_engine: The Strategic LLM Classifier for product classification
- keepa_extended_fields: Extended Keepa Product Finder field definitions
- data_healer: Universal data filling and interpolation for all metrics
"""

from utils.ai_engine import (
    StrategicTriangulator,
    StrategicBrief,
    StrategicState,
    triangulate_portfolio,
    get_portfolio_state_summary,
    analyze_strategy_with_llm,
    analyze_portfolio_async,
    STATE_DEFINITIONS,
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
    # LLM Classifier
    "StrategicTriangulator",
    "StrategicBrief",
    "StrategicState",
    "triangulate_portfolio",
    "get_portfolio_state_summary",
    "analyze_strategy_with_llm",
    "analyze_portfolio_async",
    "STATE_DEFINITIONS",
    
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
