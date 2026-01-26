"""
ShelfGuard Strategic Intelligence Agent - Generation 3/4

The Self-Driving Analyst with a Causal Reasoning Engine.

Architecture (The Semantic Stack):
- config.py: The Physics of Amazon (Causal Graph / Prior)
- profiler.py: The Triage Nurse (Vitals / State Observer)
- tools/: The Specialists (Calibrator, Volatility, Prediction, Causal, Cluster)
- brief.py: The Handoff Protocol (Diagnostic Brief)
- orchestrator.py: The Brain (Sherlock / LLM Synthesizer)

Legacy Components (for backward compatibility):
- models: Data structures for narratives, briefs, and events
- event_stream: Sparse event transformer
- sherlock_engine: Original 4-step LLM critic loop
- journal: Persistent memory for predictions
- world_context: Calendar, seasonality, external context
"""

# =============================================================================
# NEW ARCHITECTURE (Generation 3/4) - The Causal Reasoning Engine
# =============================================================================

# Config - The Physics
from .config import (
    KEEPA_CONFIG,
    get_target_direction,
    get_threshold,
    get_causal_chain,
    is_improvement,
    calculate_change_severity,
)

# Profiler - The Triage Nurse
from .profiler import (
    run_profiler,
    ProfilerVitals,
    MetricVital,
    DataHealth,
    TrendResult,
)

# Brief - The Handoff Protocol
from .brief import (
    DiagnosticBrief,
    build_diagnostic_brief,
    MarketContext,
    ProductIdentity,
    ExecutiveSummary,
)

# Orchestrator - The Brain
from .orchestrator import (
    Orchestrator,
    OrchestratorOutput,
    StrategicNarrative,
    RedTeamAnalysis,
    run_sherlock_analysis,
    get_quick_vitals,
)

# Tools - The Specialists
from .tools import (
    calibrate_physics,
    CalibratedPhysics,
    detect_anomalies,
    AnomalySignal,
    forecast_metrics,
    ForecastSignal,
    analyze_causality,
    CausalSignal,
    segment_products,
    ClusterSignal,
)


# =============================================================================
# LEGACY COMPONENTS (for backward compatibility)
# =============================================================================

def __getattr__(name):
    """Lazy imports for legacy components."""
    # Legacy models
    if name in ("DailyBrief", "JournalEntry", 
                "EnrichedEvent", "ReviewSignal", "SearchSignal",
                "NarrativeType", "ActionUrgency", "EventOwner"):
        from src.analyst import models
        return getattr(models, name)
    
    # Legacy event stream
    if name in ("transform_to_event_stream", "format_events_for_llm", "get_event_summary"):
        from src.analyst import event_stream
        return getattr(event_stream, name)
    
    # Legacy sherlock (now redirects to orchestrator for run_sherlock_analysis)
    if name == "run_sherlock_sync":
        from src.analyst import sherlock_engine
        return getattr(sherlock_engine, name)
    
    # World context
    if name in ("get_world_context", "format_world_context_for_llm", "get_search_trends"):
        from src.analyst import world_context
        return getattr(world_context, name)
    
    # Journal
    if name in ("save_prediction", "get_recent_journal", "format_journal_context"):
        from src.analyst import journal
        return getattr(journal, name)
    
    raise AttributeError(f"module 'src.analyst' has no attribute '{name}'")


__all__ = [
    # === NEW ARCHITECTURE ===
    # Config
    "KEEPA_CONFIG",
    "get_target_direction",
    "get_threshold",
    "get_causal_chain",
    "is_improvement",
    "calculate_change_severity",
    # Profiler
    "run_profiler",
    "ProfilerVitals",
    "MetricVital",
    "DataHealth",
    "TrendResult",
    # Brief
    "DiagnosticBrief",
    "build_diagnostic_brief",
    "MarketContext",
    "ProductIdentity",
    "ExecutiveSummary",
    # Orchestrator
    "Orchestrator",
    "OrchestratorOutput",
    "StrategicNarrative",
    "RedTeamAnalysis",
    "run_sherlock_analysis",
    "get_quick_vitals",
    # Tools
    "calibrate_physics",
    "CalibratedPhysics",
    "detect_anomalies",
    "AnomalySignal",
    "forecast_metrics",
    "ForecastSignal",
    "analyze_causality",
    "CausalSignal",
    "segment_products",
    "ClusterSignal",
    
    # === LEGACY (Backward Compatibility) ===
    "DailyBrief",
    "JournalEntry",
    "EnrichedEvent",
    "ReviewSignal",
    "SearchSignal",
    "NarrativeType",
    "ActionUrgency",
    "EventOwner",
    "transform_to_event_stream",
    "format_events_for_llm",
    "get_event_summary",
    "run_sherlock_sync",
    "get_world_context",
    "format_world_context_for_llm",
    "get_search_trends",
    "save_prediction",
    "get_recent_journal",
    "format_journal_context",
]
