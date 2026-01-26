# src/analyst/brief.py

"""
DIAGNOSTIC BRIEF - The Handoff Protocol

This module defines the structured communication format between
Layer 2 (Python Tools) and Layer 3 (LLM Orchestrator).

The DiagnosticBrief is the "Single Source of Truth" that:
1. Aggregates all tool outputs
2. Pre-computes the "So What" variables
3. Provides LLM-ready prompt strings
4. Reduces token cost by 90% vs raw data

Architecture:
    Tools → DiagnosticBrief → Orchestrator (LLM) → Narrative
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .profiler import ProfilerVitals
from .tools.calibrator import CalibratedPhysics
from .tools.volatility import AnomalySignal
from .tools.prediction import ForecastSignal
from .tools.causal import CausalSignal
from .tools.cluster import ClusterSignal


@dataclass
class MarketContext:
    """External market context (world state)."""
    season: str = "UNKNOWN"  # "WINTER", "SPRING", "SUMMER", "FALL"
    upcoming_events: List[str] = field(default_factory=list)
    holidays: List[str] = field(default_factory=list)
    economic_factors: List[str] = field(default_factory=list)
    category_trends: Dict[str, str] = field(default_factory=dict)
    
    def to_prompt_string(self) -> str:
        lines = ["MARKET CONTEXT:"]
        lines.append(f"  Season: {self.season}")
        if self.upcoming_events:
            lines.append(f"  Upcoming Events: {', '.join(self.upcoming_events)}")
        if self.holidays:
            lines.append(f"  Holidays: {', '.join(self.holidays)}")
        if self.economic_factors:
            lines.append(f"  Economic Factors: {', '.join(self.economic_factors)}")
        return "\n".join(lines)


@dataclass
class ProductIdentity:
    """Product metadata for context."""
    asin: str
    title: str = ""
    brand: str = ""
    category: str = ""
    product_type: str = ""
    price_tier: str = ""  # "BUDGET", "MID", "PREMIUM"
    lifecycle_stage: str = ""  # "LAUNCH", "GROWTH", "MATURITY", "DECLINE"
    is_own_product: bool = True  # vs competitor
    
    def to_prompt_string(self) -> str:
        lines = [f"PRODUCT: {self.title or self.asin}"]
        if self.brand:
            lines.append(f"  Brand: {self.brand}")
        if self.category:
            lines.append(f"  Category: {self.category}")
        lines.append(f"  Price Tier: {self.price_tier}, Stage: {self.lifecycle_stage}")
        return "\n".join(lines)


@dataclass
class ExecutiveSummary:
    """High-level summary for quick consumption."""
    overall_health: str = "UNKNOWN"  # From profiler
    confidence_score: float = 0.0
    
    # Key numbers
    current_rank: Optional[int] = None
    rank_trend: str = "STABLE"
    current_revenue: Optional[float] = None
    revenue_trend: str = "STABLE"
    
    # Counts
    critical_alerts: int = 0
    opportunities: int = 0
    threats: int = 0
    
    # One-liner
    headline: str = ""
    
    def to_prompt_string(self) -> str:
        lines = ["EXECUTIVE SUMMARY:"]
        lines.append(f"  Health: {self.overall_health} (Confidence: {self.confidence_score:.0%})")
        if self.current_rank:
            lines.append(f"  Rank: #{self.current_rank:,} ({self.rank_trend})")
        if self.current_revenue:
            lines.append(f"  Weekly Revenue: ${self.current_revenue:,.0f} ({self.revenue_trend})")
        lines.append(f"  Alerts: {self.critical_alerts} | Opportunities: {self.opportunities} | Threats: {self.threats}")
        if self.headline:
            lines.append(f"  → {self.headline}")
        return "\n".join(lines)


@dataclass
class DiagnosticBrief:
    """
    The complete diagnostic package passed to the LLM.
    
    This is the ONLY input the LLM receives - it should never see raw data.
    All math, filtering, and significance testing has already been done.
    The LLM's job is to synthesize narratives and strategy.
    """
    # Metadata
    asin: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Context
    product_identity: ProductIdentity = field(default_factory=lambda: ProductIdentity(asin="UNKNOWN"))
    market_context: MarketContext = field(default_factory=MarketContext)
    
    # Summary
    executive_summary: ExecutiveSummary = field(default_factory=ExecutiveSummary)
    
    # Tool Outputs
    profiler_vitals: Optional[ProfilerVitals] = None
    calibrated_physics: Optional[CalibratedPhysics] = None
    anomaly_signal: Optional[AnomalySignal] = None
    forecast_signal: Optional[ForecastSignal] = None
    causal_signal: Optional[CausalSignal] = None
    cluster_signal: Optional[ClusterSignal] = None
    
    # Journal context (past predictions)
    past_predictions: List[Dict[str, Any]] = field(default_factory=list)
    journal_lessons: List[str] = field(default_factory=list)
    
    # Aggregated insights
    all_alerts: List[str] = field(default_factory=list)
    all_opportunities: List[str] = field(default_factory=list)
    all_threats: List[str] = field(default_factory=list)
    
    # Warnings from tools
    tool_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "generated_at": self.generated_at,
            "product": {
                "title": self.product_identity.title,
                "brand": self.product_identity.brand,
                "category": self.product_identity.category,
                "price_tier": self.product_identity.price_tier,
                "lifecycle": self.product_identity.lifecycle_stage,
            },
            "market_context": {
                "season": self.market_context.season,
                "events": self.market_context.upcoming_events,
                "holidays": self.market_context.holidays,
            },
            "executive_summary": {
                "health": self.executive_summary.overall_health,
                "confidence": self.executive_summary.confidence_score,
                "rank": self.executive_summary.current_rank,
                "rank_trend": self.executive_summary.rank_trend,
                "revenue": self.executive_summary.current_revenue,
                "revenue_trend": self.executive_summary.revenue_trend,
                "headline": self.executive_summary.headline,
            },
            "profiler": self.profiler_vitals.to_dict() if self.profiler_vitals else None,
            "calibration": self.calibrated_physics.to_dict() if self.calibrated_physics else None,
            "anomalies": self.anomaly_signal.to_dict() if self.anomaly_signal else None,
            "forecast": self.forecast_signal.to_dict() if self.forecast_signal else None,
            "causal": self.causal_signal.to_dict() if self.causal_signal else None,
            "clusters": self.cluster_signal.to_dict() if self.cluster_signal else None,
            "alerts": self.all_alerts,
            "opportunities": self.all_opportunities,
            "threats": self.all_threats,
            "journal": {
                "past_predictions": self.past_predictions,
                "lessons": self.journal_lessons,
            },
            "warnings": self.tool_warnings,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_prompt_string(self) -> str:
        """
        Generate the complete prompt string for the LLM.
        
        This is the primary output format - a structured text document
        that gives the LLM everything it needs to generate strategic narratives.
        """
        sections = []
        
        # Header
        sections.append("=" * 60)
        sections.append(f"DIAGNOSTIC BRIEF: {self.asin}")
        sections.append(f"Generated: {self.generated_at}")
        sections.append("=" * 60)
        
        # Product Identity
        sections.append("")
        sections.append(self.product_identity.to_prompt_string())
        
        # Executive Summary
        sections.append("")
        sections.append(self.executive_summary.to_prompt_string())
        
        # Market Context
        sections.append("")
        sections.append(self.market_context.to_prompt_string())
        
        # Profiler Vitals
        if self.profiler_vitals:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.profiler_vitals.to_prompt_string())
        
        # Calibrated Physics
        if self.calibrated_physics:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.calibrated_physics.to_prompt_string())
        
        # Anomalies
        if self.anomaly_signal:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.anomaly_signal.to_prompt_string())
        
        # Forecast
        if self.forecast_signal:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.forecast_signal.to_prompt_string())
        
        # Causal Analysis
        if self.causal_signal:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.causal_signal.to_prompt_string())
        
        # Clusters
        if self.cluster_signal:
            sections.append("")
            sections.append("-" * 40)
            sections.append(self.cluster_signal.to_prompt_string())
        
        # Journal / Historical Context
        if self.past_predictions or self.journal_lessons:
            sections.append("")
            sections.append("-" * 40)
            sections.append("HISTORICAL CONTEXT (From Journal):")
            for pred in self.past_predictions[-3:]:  # Last 3 predictions
                outcome = pred.get("outcome", "PENDING")
                sections.append(f"  Past Prediction: {pred.get('prediction', 'N/A')}")
                sections.append(f"  Outcome: {outcome}")
            if self.journal_lessons:
                sections.append("  Lessons Learned:")
                for lesson in self.journal_lessons[-3:]:
                    sections.append(f"    - {lesson}")
        
        # Aggregated Alerts & Opportunities
        if self.all_alerts:
            sections.append("")
            sections.append("-" * 40)
            sections.append("🚨 CRITICAL ALERTS:")
            for alert in self.all_alerts:
                sections.append(f"  • {alert}")
        
        if self.all_opportunities:
            sections.append("")
            sections.append("💰 OPPORTUNITIES:")
            for opp in self.all_opportunities:
                sections.append(f"  • {opp}")
        
        if self.all_threats:
            sections.append("")
            sections.append("⚠️ THREATS:")
            for threat in self.all_threats:
                sections.append(f"  • {threat}")
        
        # Footer
        sections.append("")
        sections.append("=" * 60)
        sections.append("END OF DIAGNOSTIC BRIEF")
        sections.append("=" * 60)
        
        return "\n".join(sections)
    
    def token_estimate(self) -> int:
        """Estimate token count for the prompt string."""
        # Rough estimate: ~4 characters per token
        prompt = self.to_prompt_string()
        return len(prompt) // 4


def build_diagnostic_brief(
    asin: str,
    profiler_vitals: Optional[ProfilerVitals] = None,
    calibrated_physics: Optional[CalibratedPhysics] = None,
    anomaly_signal: Optional[AnomalySignal] = None,
    forecast_signal: Optional[ForecastSignal] = None,
    causal_signal: Optional[CausalSignal] = None,
    cluster_signal: Optional[ClusterSignal] = None,
    product_info: Optional[Dict] = None,
    market_context: Optional[MarketContext] = None,
    journal_entries: Optional[List[Dict]] = None
) -> DiagnosticBrief:
    """
    Factory function to build a DiagnosticBrief from tool outputs.
    
    This is the main entry point for creating briefs.
    It aggregates all tool outputs and computes the executive summary.
    """
    brief = DiagnosticBrief(asin=asin)
    
    # Set tool outputs
    brief.profiler_vitals = profiler_vitals
    brief.calibrated_physics = calibrated_physics
    brief.anomaly_signal = anomaly_signal
    brief.forecast_signal = forecast_signal
    brief.causal_signal = causal_signal
    brief.cluster_signal = cluster_signal
    
    # Build product identity
    if product_info:
        brief.product_identity = ProductIdentity(
            asin=asin,
            title=product_info.get("title", ""),
            brand=product_info.get("brand", ""),
            category=product_info.get("category", ""),
            product_type=product_info.get("product_type", ""),
            price_tier=_infer_price_tier(product_info.get("price")),
            lifecycle_stage=_infer_lifecycle(profiler_vitals),
            is_own_product=product_info.get("is_own", True),
        )
    else:
        brief.product_identity = ProductIdentity(asin=asin)
    
    # Set market context
    if market_context:
        brief.market_context = market_context
    
    # Build executive summary
    brief.executive_summary = _build_executive_summary(brief)
    
    # Aggregate alerts, opportunities, threats
    brief.all_alerts, brief.all_opportunities, brief.all_threats = _aggregate_insights(brief)
    
    # Process journal entries
    if journal_entries:
        brief.past_predictions = journal_entries[-5:]  # Last 5
        brief.journal_lessons = _extract_lessons(journal_entries)
    
    # Collect tool warnings
    brief.tool_warnings = _collect_warnings(brief)
    
    return brief


def _infer_price_tier(price: Optional[float]) -> str:
    """Infer price tier from price."""
    if price is None:
        return "UNKNOWN"
    if price < 10:
        return "BUDGET"
    elif price < 25:
        return "MID"
    else:
        return "PREMIUM"


def _infer_lifecycle(vitals: Optional[ProfilerVitals]) -> str:
    """Infer product lifecycle stage from vitals."""
    if not vitals:
        return "UNKNOWN"
    
    # Use review count and trend to infer stage
    review_vital = vitals.target_vitals.get("review_count")
    rank_trend = vitals.trends.get("sales_rank", None)
    
    if review_vital and review_vital.current_value:
        reviews = review_vital.current_value
        if reviews < 50:
            return "LAUNCH"
        elif reviews < 200:
            if rank_trend and rank_trend.direction == "GROWING":
                return "GROWTH"
            return "EMERGING"
        else:
            if rank_trend and rank_trend.direction == "DECLINING":
                return "DECLINE"
            return "MATURITY"
    
    return "UNKNOWN"


def _build_executive_summary(brief: DiagnosticBrief) -> ExecutiveSummary:
    """Build executive summary from brief components."""
    summary = ExecutiveSummary()
    
    # From profiler
    if brief.profiler_vitals:
        summary.overall_health = brief.profiler_vitals.overall_health
        summary.confidence_score = brief.profiler_vitals.confidence_score
        summary.critical_alerts = len(brief.profiler_vitals.critical_alerts)
        
        # Get rank
        rank_vital = brief.profiler_vitals.target_vitals.get("sales_rank")
        if rank_vital and rank_vital.current_value:
            summary.current_rank = int(rank_vital.current_value)
        
        rank_trend = brief.profiler_vitals.trends.get("sales_rank")
        if rank_trend:
            summary.rank_trend = rank_trend.direction
        
        # Get revenue
        rev_vital = brief.profiler_vitals.target_vitals.get("revenue")
        if rev_vital and rev_vital.current_value:
            summary.current_revenue = rev_vital.current_value
    
    # From forecast
    if brief.forecast_signal:
        summary.opportunities = len([
            f for f in brief.forecast_signal.forecasts.values()
            if f.expected_change_pct() and f.expected_change_pct() > 0.1
        ])
        summary.threats = len([
            f for f in brief.forecast_signal.forecasts.values()
            if f.expected_change_pct() and f.expected_change_pct() < -0.1
        ])
    
    # Generate headline
    summary.headline = _generate_headline(summary, brief)
    
    return summary


def _generate_headline(summary: ExecutiveSummary, brief: DiagnosticBrief) -> str:
    """Generate a one-line headline for the brief."""
    if summary.overall_health == "CRITICAL":
        return "URGENT: Product requires immediate attention"
    elif summary.overall_health == "AT_RISK":
        if summary.rank_trend == "DECLINING":
            return "Performance declining - intervention recommended"
        return "Warning signs detected - monitoring advised"
    elif summary.overall_health == "HEALTHY":
        if summary.opportunities > 0:
            return f"Strong performance with {summary.opportunities} growth opportunities identified"
        return "Stable performance - maintain current strategy"
    else:
        return "Analysis complete - review details below"


def _aggregate_insights(brief: DiagnosticBrief) -> tuple:
    """Aggregate all alerts, opportunities, and threats from tools."""
    alerts = []
    opportunities = []
    threats = []
    
    # From profiler
    if brief.profiler_vitals:
        alerts.extend(brief.profiler_vitals.critical_alerts)
    
    # From anomalies
    if brief.anomaly_signal:
        for anomaly in brief.anomaly_signal.anomalies:
            if anomaly.severity in ["MAJOR", "EXTREME"]:
                if anomaly.is_good:
                    opportunities.append(f"{anomaly.metric} {anomaly.direction}: {anomaly.context}")
                else:
                    threats.append(f"{anomaly.metric} {anomaly.direction}: {anomaly.context}")
    
    # From forecast
    if brief.forecast_signal:
        for pred in brief.forecast_signal.key_predictions:
            if "improve" in pred.lower() or "increase" in pred.lower():
                opportunities.append(pred)
            elif "decline" in pred.lower() or "decrease" in pred.lower():
                threats.append(pred)
    
    # From causal
    if brief.causal_signal:
        for finding in brief.causal_signal.key_findings:
            if "broken" in finding.lower():
                alerts.append(finding)
    
    return alerts, opportunities, threats


def _extract_lessons(journal_entries: List[Dict]) -> List[str]:
    """Extract lessons learned from journal entries."""
    lessons = []
    
    for entry in journal_entries:
        if entry.get("outcome") == "WRONG":
            prediction = entry.get("prediction", "")
            actual = entry.get("actual", "")
            lessons.append(f"Predicted '{prediction}' but actually '{actual}'")
    
    return lessons[-3:]  # Last 3 lessons


def _collect_warnings(brief: DiagnosticBrief) -> List[str]:
    """Collect all warnings from tools."""
    warnings = []
    
    if brief.profiler_vitals:
        warnings.extend(brief.profiler_vitals.data_health.warnings)
    
    if brief.calibrated_physics:
        warnings.extend(brief.calibrated_physics.warnings)
    
    if brief.anomaly_signal:
        warnings.extend(brief.anomaly_signal.warnings)
    
    if brief.forecast_signal:
        warnings.extend(brief.forecast_signal.warnings)
    
    if brief.causal_signal:
        warnings.extend(brief.causal_signal.warnings)
    
    if brief.cluster_signal:
        warnings.extend(brief.cluster_signal.warnings)
    
    return list(set(warnings))  # Deduplicate
