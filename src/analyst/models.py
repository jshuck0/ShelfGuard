"""
Data Models for the Strategic Intelligence Agent

These dataclasses define the output schema for the AI Analyst:
- EnrichedEvent: A sparse event with semantic tags and derivatives
- StrategicNarrative: A single insight with prediction and action
- DailyBrief: The morning report containing 3 narratives
- JournalEntry: A prediction tracked over time for learning
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class NarrativeType(Enum):
    """Classification of strategic narratives."""
    CONQUEST = "conquest"           # Opportunity to capture market share
    THREAT = "threat"               # Defensive priority
    OPTIMIZATION = "optimization"   # Efficiency improvement
    VULNERABILITY = "vulnerability" # Red Team finding


class ActionUrgency(Enum):
    """When to act on a recommendation."""
    NOW = "now"                     # Act immediately (< 24 hrs)
    THIS_WEEK = "this_week"         # Act within 7 days
    MONITOR = "monitor"             # Track and reassess


class EventOwner(Enum):
    """Whose event is this?"""
    PORTFOLIO = "portfolio"         # Your product
    COMPETITOR = "competitor"       # Competitor product


@dataclass
class EnrichedEvent:
    """
    A single market event in the sparse event stream.
    
    This is the compressed format: only significant changes are captured,
    not every day of stable data.
    """
    # Core identity
    date: str                           # ISO date string
    asin: str
    event_type: str                     # BASELINE, PRICE_DROP, RANK_SPIKE, etc.
    owner: EventOwner = EventOwner.COMPETITOR
    
    # Product context
    brand: str = ""
    title: str = ""
    
    # Semantic tags (extracted from title)
    tags: List[str] = field(default_factory=list)  # ["aluminum-free", "3-pack", "tropical"]
    
    # The change (old → new)
    metric_name: str = ""               # "price", "rank", "reviews"
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    change_pct: Optional[float] = None
    
    # Pre-computed derivatives (so LLM doesn't do math)
    price_gap_vs_category: str = ""     # "+17% (Premium)"
    rank_velocity_7d: str = ""          # "-20% (Crash)"
    competitive_context: str = ""       # "3 competitors OOS"
    
    # NEW: Additional Keepa metrics for richer context
    monthly_sold: Optional[int] = None          # Amazon's actual units/month estimate
    velocity_30d: Optional[float] = None        # Pre-calculated sales trend (-15% = declining)
    oos_pct_30: Optional[float] = None          # Out-of-stock frequency (0.3 = 30% OOS)
    seller_count: Optional[int] = None          # Number of active sellers
    
    # Chain linking
    prior_event: Optional[str] = None   # "Followed PRICE_DROP by 24hrs"
    
    # Full state snapshot (for BASELINE events)
    state_snapshot: Optional[Dict] = None
    
    def to_llm_string(self) -> str:
        """Format for LLM consumption."""
        owner_tag = "📍YOUR" if self.owner == EventOwner.PORTFOLIO else "🎯COMP"
        
        if self.event_type == "BASELINE":
            # Include key metrics in baseline
            baseline_parts = [', '.join(self.tags)] if self.tags else []
            if self.monthly_sold:
                baseline_parts.append(f"{self.monthly_sold} units/mo")
            if self.seller_count:
                baseline_parts.append(f"{self.seller_count} sellers")
            if self.oos_pct_30 and self.oos_pct_30 > 0.1:
                baseline_parts.append(f"OOS {self.oos_pct_30*100:.0f}%")
            baseline_str = " | ".join(baseline_parts) if baseline_parts else ""
            return f"[{self.date}] {owner_tag} {self.brand} | BASELINE | {baseline_str}"
        
        change_str = ""
        if self.old_value is not None and self.new_value is not None:
            change_str = f"{self.old_value} → {self.new_value}"
            if self.change_pct:
                change_str += f" ({self.change_pct:+.1f}%)"
        
        context_parts = []
        if self.price_gap_vs_category:
            context_parts.append(self.price_gap_vs_category)
        if self.rank_velocity_7d:
            context_parts.append(self.rank_velocity_7d)
        if self.velocity_30d is not None and abs(self.velocity_30d) > 10:
            trend = "↑" if self.velocity_30d > 0 else "↓"
            context_parts.append(f"Sales {trend}{abs(self.velocity_30d):.0f}%")
        if self.oos_pct_30 and self.oos_pct_30 > 0.2:
            context_parts.append(f"OOS risk {self.oos_pct_30*100:.0f}%")
        if self.prior_event:
            context_parts.append(self.prior_event)
        
        context_str = " | ".join(context_parts) if context_parts else ""
        
        return f"[{self.date}] {owner_tag} {self.brand} | {self.event_type} | {change_str} | {context_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "asin": self.asin,
            "event_type": self.event_type,
            "owner": self.owner.value,
            "brand": self.brand,
            "title": self.title,
            "tags": self.tags,
            "metric_name": self.metric_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_pct": self.change_pct,
            "price_gap_vs_category": self.price_gap_vs_category,
            "rank_velocity_7d": self.rank_velocity_7d,
            "competitive_context": self.competitive_context,
            "monthly_sold": self.monthly_sold,
            "velocity_30d": self.velocity_30d,
            "oos_pct_30": self.oos_pct_30,
            "seller_count": self.seller_count,
            "prior_event": self.prior_event,
            "state_snapshot": self.state_snapshot,
        }


@dataclass
class StrategicNarrative:
    """
    A single strategic insight with prediction and recommended action.
    
    The Daily Brief contains up to 3 of these narratives.
    """
    # Identity
    title: str                          # "The Salt & Stone Opportunity"
    narrative_type: NarrativeType = NarrativeType.OPTIMIZATION
    
    # The Pattern (What Happened)
    pattern_summary: str = ""           # "Salt & Stone went OOS on Jan 18..."
    evidence_events: List[str] = field(default_factory=list)  # Event IDs that support this
    confidence: float = 0.7             # 0.0 - 1.0
    
    # The Prediction (What Will Happen)
    prediction: str = ""                # "If they remain OOS 7+ days..."
    expected_impact: float = 0.0        # +$12,000 or -$5,000
    trigger_to_watch: str = ""          # "Salt & Stone inventory restock"
    reversal_risk: str = ""             # "Gains will reverse when they restock"
    time_horizon_days: int = 7
    
    # The Action (What To Do)
    recommended_action: str = ""        # "Increase conquest PPC by 40%"
    action_urgency: ActionUrgency = ActionUrgency.MONITOR
    action_rationale: str = ""          # "Capture reviews while traffic high"
    
    # Red Team (for VULNERABILITY type)
    attacker_perspective: Optional[str] = None  # "If I were Native, I would..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "narrative_type": self.narrative_type.value,
            "pattern_summary": self.pattern_summary,
            "evidence_events": self.evidence_events,
            "confidence": self.confidence,
            "prediction": self.prediction,
            "expected_impact": self.expected_impact,
            "trigger_to_watch": self.trigger_to_watch,
            "reversal_risk": self.reversal_risk,
            "time_horizon_days": self.time_horizon_days,
            "recommended_action": self.recommended_action,
            "action_urgency": self.action_urgency.value,
            "action_rationale": self.action_rationale,
            "attacker_perspective": self.attacker_perspective,
        }
    
    def get_urgency_emoji(self) -> str:
        """Get emoji for urgency level."""
        if self.action_urgency == ActionUrgency.NOW:
            return "🚨"
        elif self.action_urgency == ActionUrgency.THIS_WEEK:
            return "⚠️"
        return "👀"
    
    def get_type_emoji(self) -> str:
        """Get emoji for narrative type."""
        emojis = {
            NarrativeType.CONQUEST: "📍",
            NarrativeType.THREAT: "⚔️",
            NarrativeType.OPTIMIZATION: "📈",
            NarrativeType.VULNERABILITY: "🛡️",
        }
        return emojis.get(self.narrative_type, "📊")


@dataclass
class DailyBrief:
    """
    The morning intelligence report.
    
    This is what the Commander sees at 9 AM - the result of
    the overnight Sherlock analysis.
    """
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    project_id: Optional[str] = None
    
    # Context injected into analysis
    world_context: Dict[str, Any] = field(default_factory=dict)
    journal_context: str = ""           # "Last week you predicted X, outcome was Y"
    
    # The Headlines
    market_summary: str = ""            # "Your market is stable with 1 major opportunity"
    
    # The 3 Strategic Narratives
    narratives: List[StrategicNarrative] = field(default_factory=list)
    
    # Red Team Finding
    red_team_insight: str = ""          # "Your biggest vulnerability is..."
    
    # Editor's Desk (NEW) - Track what ideas were killed
    editor_killed_count: int = 0        # How many ideas the Editor killed
    editor_kill_reasons: List[Dict[str, str]] = field(default_factory=list)  # {"insight": "", "reason": "", "test_failed": ""}
    product_identity: Dict[str, str] = field(default_factory=dict)  # category, brand, product_types
    
    # Aggregate Metrics
    predicted_revenue_30d: float = 0.0
    total_opportunity_value: float = 0.0
    total_risk_value: float = 0.0
    
    # Key Lists
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    
    # Quality Metrics
    event_count: int = 0                # How many events were analyzed
    model_accuracy_30d: float = 0.0     # Journal-based accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "project_id": self.project_id,
            "world_context": self.world_context,
            "journal_context": self.journal_context,
            "market_summary": self.market_summary,
            "narratives": [n.to_dict() for n in self.narratives],
            "red_team_insight": self.red_team_insight,
            "editor_killed_count": self.editor_killed_count,
            "editor_kill_reasons": self.editor_kill_reasons,
            "product_identity": self.product_identity,
            "predicted_revenue_30d": self.predicted_revenue_30d,
            "total_opportunity_value": self.total_opportunity_value,
            "total_risk_value": self.total_risk_value,
            "key_risks": self.key_risks,
            "key_opportunities": self.key_opportunities,
            "event_count": self.event_count,
            "model_accuracy_30d": self.model_accuracy_30d,
        }
    
    def get_headline(self) -> str:
        """Generate email subject line."""
        if not self.narratives:
            return "Strategy Brief: Market Stable"
        
        top = self.narratives[0]
        impact_str = f"{top.expected_impact:+,.0f}" if top.expected_impact else ""
        return f"Strategy Brief: {top.title} ({impact_str})"


@dataclass
class JournalEntry:
    """
    A prediction tracked over time for learning.
    
    The Journal is the AI's memory - it remembers what it predicted
    and learns from whether those predictions were correct.
    """
    # Identity
    id: Optional[str] = None
    project_id: Optional[str] = None
    
    # The Prediction
    prediction_date: datetime = field(default_factory=datetime.now)
    narrative_title: str = ""           # "The Salt & Stone Opportunity"
    prediction: str = ""                # "Price drop will boost rank 100 spots"
    confidence: float = 0.7
    expected_outcome: str = ""          # "Rank improvement by Jan 20"
    time_horizon_days: int = 7
    
    # The Outcome (filled in later)
    outcome_date: Optional[datetime] = None
    actual_outcome: Optional[str] = None    # "Rank only improved 10 spots"
    was_correct: Optional[bool] = None
    lesson_learned: Optional[str] = None    # "Price drops don't work for mature SKUs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None,
            "narrative_title": self.narrative_title,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "expected_outcome": self.expected_outcome,
            "time_horizon_days": self.time_horizon_days,
            "outcome_date": self.outcome_date.isoformat() if self.outcome_date else None,
            "actual_outcome": self.actual_outcome,
            "was_correct": self.was_correct,
            "lesson_learned": self.lesson_learned,
        }
    
    def to_llm_context(self) -> str:
        """Format for prompt injection."""
        if self.was_correct is None:
            return f"PENDING: '{self.prediction}' (check by {self.expected_outcome})"
        
        result = "CORRECT ✓" if self.was_correct else "WRONG ✗"
        lesson = f" Lesson: {self.lesson_learned}" if self.lesson_learned else ""
        return f"{result}: Predicted '{self.prediction}' → Actual: '{self.actual_outcome}'.{lesson}"


@dataclass
class ReviewSignal:
    """
    Sentiment and theme signals extracted from product reviews.
    
    This is the "Language" layer that complements Keepa's "Numbers".
    """
    asin: str
    date_range: str = "Last 30 days"
    review_count_new: int = 0
    avg_sentiment: float = 0.5          # 0.0 (negative) to 1.0 (positive)
    
    # Theme extraction (LLM-generated)
    positive_themes: List[str] = field(default_factory=list)
    negative_themes: List[str] = field(default_factory=list)
    emerging_complaints: List[str] = field(default_factory=list)
    
    # Competitive comparison
    vs_competitor_sentiment: str = ""   # "+0.15 vs Native"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asin": self.asin,
            "date_range": self.date_range,
            "review_count_new": self.review_count_new,
            "avg_sentiment": self.avg_sentiment,
            "positive_themes": self.positive_themes,
            "negative_themes": self.negative_themes,
            "emerging_complaints": self.emerging_complaints,
            "vs_competitor_sentiment": self.vs_competitor_sentiment,
        }


@dataclass
class SearchSignal:
    """
    Search volume and trend signals from Google Trends.
    
    This is the "Demand" layer that shows category health.
    """
    keyword: str
    search_volume_index: int = 50       # 0-100 relative index
    trend_direction: str = "stable"     # "growing", "stable", "declining"
    trend_pct_change: float = 0.0       # MoM change
    
    # Related queries
    rising_queries: List[str] = field(default_factory=list)
    
    # Competitive ranks (if available)
    your_organic_rank: Optional[int] = None
    competitor_ranks: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keyword": self.keyword,
            "search_volume_index": self.search_volume_index,
            "trend_direction": self.trend_direction,
            "trend_pct_change": self.trend_pct_change,
            "rising_queries": self.rising_queries,
            "your_organic_rank": self.your_organic_rank,
            "competitor_ranks": self.competitor_ranks,
        }
