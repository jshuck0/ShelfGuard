"""
Journal: Persistent Memory for the AI Analyst

The Journal tracks predictions and their outcomes so the AI can learn
from its mistakes. It prevents the same bad suggestions from repeating.

Key behaviors:
1. Save predictions from each Daily Brief
2. Score predictions after their time horizon expires
3. Inject recent prediction history into prompts
4. Calculate model accuracy over time
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from src.analyst.models import JournalEntry, StrategicNarrative, DailyBrief


# =============================================================================
# IN-MEMORY JOURNAL (for MVP / testing)
# =============================================================================

_memory_journal: List[JournalEntry] = []


def save_prediction(
    project_id: str,
    narrative: StrategicNarrative,
    prediction_date: datetime = None
) -> JournalEntry:
    """
    Save a prediction from a narrative to the journal.
    
    This is called after generating a Daily Brief to track predictions.
    """
    entry = JournalEntry(
        project_id=project_id,
        prediction_date=prediction_date or datetime.now(),
        narrative_title=narrative.title,
        prediction=narrative.prediction,
        confidence=narrative.confidence,
        expected_outcome=narrative.trigger_to_watch or "Check metrics",
        time_horizon_days=narrative.time_horizon_days,
    )
    
    _memory_journal.append(entry)
    return entry


def save_predictions_from_brief(
    project_id: str,
    brief: DailyBrief
) -> List[JournalEntry]:
    """Save all predictions from a Daily Brief."""
    entries = []
    for narrative in brief.narratives:
        if narrative.prediction:  # Only save if there's a prediction
            entry = save_prediction(project_id, narrative, brief.generated_at)
            entries.append(entry)
    return entries


def get_recent_journal(
    project_id: str,
    days: int = 14
) -> List[JournalEntry]:
    """Get journal entries from the last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    return [
        entry for entry in _memory_journal
        if entry.project_id == project_id and entry.prediction_date >= cutoff
    ]


def get_pending_predictions(
    project_id: str
) -> List[JournalEntry]:
    """Get predictions that haven't been scored yet."""
    now = datetime.now()
    pending = []
    
    for entry in _memory_journal:
        if entry.project_id != project_id:
            continue
        if entry.was_correct is not None:
            continue  # Already scored
        
        # Check if time horizon has passed
        expected_date = entry.prediction_date + timedelta(days=entry.time_horizon_days)
        if now >= expected_date:
            pending.append(entry)
    
    return pending


def score_prediction(
    entry: JournalEntry,
    actual_outcome: str,
    was_correct: bool,
    lesson_learned: Optional[str] = None
) -> JournalEntry:
    """
    Score a prediction after its time horizon has passed.
    
    This updates the journal entry with the actual outcome.
    """
    entry.outcome_date = datetime.now()
    entry.actual_outcome = actual_outcome
    entry.was_correct = was_correct
    entry.lesson_learned = lesson_learned
    return entry


def calculate_accuracy(
    project_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Calculate prediction accuracy over time.
    
    Returns accuracy metrics for the journal context.
    """
    cutoff = datetime.now() - timedelta(days=days)
    
    scored = [
        entry for entry in _memory_journal
        if entry.project_id == project_id 
        and entry.was_correct is not None
        and entry.prediction_date >= cutoff
    ]
    
    if not scored:
        return {
            "total_predictions": 0,
            "accuracy": 0.0,
            "correct": 0,
            "incorrect": 0,
        }
    
    correct = sum(1 for e in scored if e.was_correct)
    
    return {
        "total_predictions": len(scored),
        "accuracy": correct / len(scored),
        "correct": correct,
        "incorrect": len(scored) - correct,
    }


def format_journal_context(
    project_id: str,
    max_entries: int = 5
) -> str:
    """
    Format recent journal entries for LLM prompt injection.
    
    This gives the AI memory of its past predictions and outcomes.
    """
    recent = get_recent_journal(project_id, days=14)
    accuracy = calculate_accuracy(project_id, days=30)
    
    if not recent:
        return "No prior predictions in journal. This is the first analysis."
    
    lines = []
    lines.append("=== JOURNAL: YOUR PREDICTION HISTORY ===")
    lines.append(f"Model Accuracy (30d): {accuracy['accuracy']*100:.0f}% ({accuracy['correct']}/{accuracy['total_predictions']})")
    lines.append("")
    
    # Show recent entries (most recent first)
    recent_sorted = sorted(recent, key=lambda e: e.prediction_date, reverse=True)[:max_entries]
    
    for entry in recent_sorted:
        lines.append(entry.to_llm_context())
    
    # Add learning instruction
    lines.append("")
    lines.append("INSTRUCTION: Learn from past mistakes. Do NOT repeat failed strategies.")
    
    return "\n".join(lines)


# =============================================================================
# SUPABASE JOURNAL (for production)
# =============================================================================

def save_prediction_to_supabase(
    supabase_client,
    project_id: str,
    narrative: StrategicNarrative,
    prediction_date: datetime = None
) -> Optional[Dict]:
    """
    Save a prediction to Supabase analyst_journal table.
    
    Requires the analyst_journal table from the migration.
    """
    try:
        data = {
            "project_id": project_id,
            "prediction_date": (prediction_date or datetime.now()).date().isoformat(),
            "narrative_title": narrative.title,
            "prediction": narrative.prediction,
            "confidence": narrative.confidence,
            "expected_outcome": narrative.trigger_to_watch or "Check metrics",
            "time_horizon_days": narrative.time_horizon_days,
        }
        
        result = supabase_client.table("analyst_journal").upsert(
            data,
            on_conflict="project_id,prediction_date,narrative_title"
        ).execute()
        
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error saving to journal: {e}")
        return None


def get_journal_from_supabase(
    supabase_client,
    project_id: str,
    days: int = 14
) -> List[JournalEntry]:
    """Load journal entries from Supabase."""
    try:
        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        result = supabase_client.table("analyst_journal").select("*").eq(
            "project_id", project_id
        ).gte(
            "prediction_date", cutoff
        ).order(
            "prediction_date", desc=True
        ).execute()
        
        entries = []
        for row in result.data or []:
            entries.append(JournalEntry(
                id=row.get("id"),
                project_id=row.get("project_id"),
                prediction_date=datetime.fromisoformat(row["prediction_date"]) if row.get("prediction_date") else datetime.now(),
                narrative_title=row.get("narrative_title", ""),
                prediction=row.get("prediction", ""),
                confidence=row.get("confidence", 0.7),
                expected_outcome=row.get("expected_outcome", ""),
                time_horizon_days=row.get("time_horizon_days", 7),
                outcome_date=datetime.fromisoformat(row["outcome_date"]) if row.get("outcome_date") else None,
                actual_outcome=row.get("actual_outcome"),
                was_correct=row.get("was_correct"),
                lesson_learned=row.get("lesson_learned"),
            ))
        
        return entries
    except Exception as e:
        print(f"Error loading journal: {e}")
        return []


def score_prediction_in_supabase(
    supabase_client,
    entry_id: str,
    actual_outcome: str,
    was_correct: bool,
    lesson_learned: Optional[str] = None
) -> bool:
    """Update a prediction with its outcome in Supabase."""
    try:
        data = {
            "actual_outcome": actual_outcome,
            "was_correct": was_correct,
            "lesson_learned": lesson_learned,
            "scored_at": datetime.now().isoformat(),
        }
        
        supabase_client.table("analyst_journal").update(data).eq("id", entry_id).execute()
        return True
    except Exception as e:
        print(f"Error scoring prediction: {e}")
        return False
