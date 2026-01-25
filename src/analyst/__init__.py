"""
ShelfGuard Strategic Intelligence Agent

The AI Analyst that wakes up at 3 AM, investigates market changes,
and delivers strategic narratives before you log in.

Components:
- models: Data structures for narratives, briefs, and events
- event_stream: Sparse event transformer (18K rows → 2K events)
- sherlock_engine: 4-step LLM critic loop (Analyst → Skeptic → Oracle → Red Team)
- journal: Persistent memory for predictions and outcomes
- world_context: Calendar, seasonality, and external context
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("StrategicNarrative", "DailyBrief", "JournalEntry", 
                "EnrichedEvent", "ReviewSignal", "SearchSignal",
                "NarrativeType", "ActionUrgency", "EventOwner"):
        from src.analyst import models
        return getattr(models, name)
    
    if name in ("transform_to_event_stream", "format_events_for_llm", "get_event_summary"):
        from src.analyst import event_stream
        return getattr(event_stream, name)
    
    if name in ("run_sherlock_analysis", "run_sherlock_sync"):
        from src.analyst import sherlock_engine
        return getattr(sherlock_engine, name)
    
    if name in ("get_world_context", "format_world_context_for_llm", "get_search_trends"):
        from src.analyst import world_context
        return getattr(world_context, name)
    
    if name in ("save_prediction", "get_recent_journal", "format_journal_context"):
        from src.analyst import journal
        return getattr(journal, name)
    
    raise AttributeError(f"module 'src.analyst' has no attribute '{name}'")

__all__ = [
    # Models
    "StrategicNarrative",
    "DailyBrief", 
    "JournalEntry",
    "EnrichedEvent",
    "ReviewSignal",
    "SearchSignal",
    "NarrativeType",
    "ActionUrgency",
    "EventOwner",
    # Event Stream
    "transform_to_event_stream",
    "format_events_for_llm",
    "get_event_summary",
    # Sherlock Engine
    "run_sherlock_analysis",
    "run_sherlock_sync",
    # World Context
    "get_world_context",
    "format_world_context_for_llm",
    "get_search_trends",
    # Journal
    "save_prediction",
    "get_recent_journal",
    "format_journal_context",
]
