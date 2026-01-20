"""
AI Engine V2 - Trigger-Aware Strategic Classification

Enhanced LLM prompts that inject trigger events for causal reasoning.
First stage of two-stage LLM architecture (Classification → Insight).

This engine:
1. Takes trigger events as input
2. Classifies product strategic state with causal reasoning
3. Calculates risk scores amplified by triggers
4. Provides foundation for insight generation (v3)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import openai
import os
import json

from src.models.product_status import ProductStatus
from src.models.trigger_event import TriggerEvent


class TriggerAwareAIEngine:
    """
    Strategic classification engine with trigger event awareness.

    Classifies products into strategic states based on:
    - Current metrics
    - Historical trends
    - Trigger events (causal reasoning)
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize AI engine with OpenAI API key."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key

    def classify_strategic_state(
        self,
        asin: str,
        current_metrics: Dict[str, Any],
        historical_trends: Dict[str, Any],
        trigger_events: List[TriggerEvent],
        competitor_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify product's strategic state using trigger-aware LLM.

        Args:
            asin: Product ASIN
            current_metrics: Current snapshot (price, BSR, reviews, etc.)
            historical_trends: 30/90 day trends
            trigger_events: Detected trigger events (sorted by severity)
            competitor_data: Optional competitor metrics

        Returns:
            Dict with:
                - product_status: ProductStatus enum
                - strategic_state: Human-readable state
                - confidence: 0-100 confidence score
                - reasoning: LLM explanation with trigger citations
                - risk_score: 0-100 (amplified by triggers)
                - growth_score: 0-100 (dampened by triggers)
        """

        # Build LLM prompt with trigger injection
        prompt = self._build_classification_prompt(
            asin=asin,
            current_metrics=current_metrics,
            historical_trends=historical_trends,
            trigger_events=trigger_events,
            competitor_data=competitor_data
        )

        # Call LLM for classification
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for consistent classification
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and parse result
            classification = self._parse_classification_result(result, trigger_events)

            return classification

        except Exception as e:
            print(f"⚠️  AI classification error: {str(e)}")
            return self._get_fallback_classification(trigger_events)

    def calculate_trigger_amplified_risk(
        self,
        base_risk: float,
        trigger_events: List[TriggerEvent]
    ) -> float:
        """
        Amplify risk score based on trigger severity.

        Args:
            base_risk: Base risk score (0-100)
            trigger_events: Detected triggers

        Returns:
            Amplified risk score (0-100)
        """

        if not trigger_events:
            return base_risk

        # Get highest severity trigger
        max_severity = max(t.severity for t in trigger_events)

        # Amplification factor based on severity
        # Severity 10 = 2x amplification
        # Severity 5 = 1.25x amplification
        amplification = 1 + (max_severity / 20)  # 1.0 to 1.5x

        amplified_risk = min(100, base_risk * amplification)

        return amplified_risk

    def calculate_trigger_dampened_growth(
        self,
        base_growth: float,
        trigger_events: List[TriggerEvent]
    ) -> float:
        """
        Dampen growth score based on critical triggers.

        Args:
            base_growth: Base growth score (0-100)
            trigger_events: Detected triggers

        Returns:
            Dampened growth score (0-100)
        """

        if not trigger_events:
            return base_growth

        # Count critical triggers (severity >= 8)
        critical_count = sum(1 for t in trigger_events if t.severity >= 8)

        # Dampening factor
        dampening = 1 - (critical_count * 0.15)  # -15% per critical trigger
        dampening = max(0.4, dampening)  # Cap at 60% reduction

        dampened_growth = base_growth * dampening

        return dampened_growth

    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================

    def _build_classification_prompt(
        self,
        asin: str,
        current_metrics: Dict[str, Any],
        historical_trends: Dict[str, Any],
        trigger_events: List[TriggerEvent],
        competitor_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive prompt with trigger injection."""

        # Format trigger events for LLM
        trigger_context = self._format_triggers_for_llm(trigger_events)

        # Format metrics
        metrics_context = self._format_metrics_for_llm(current_metrics, historical_trends)

        # Format competitor data
        competitor_context = self._format_competitors_for_llm(competitor_data) if competitor_data else ""

        prompt = f"""Analyze this Amazon product and classify its strategic state.

PRODUCT: {asin}

{metrics_context}

DETECTED TRIGGER EVENTS (Sorted by Severity):
{trigger_context}

{competitor_context}

CLASSIFICATION TASK:
Based on the metrics and trigger events above, classify this product's strategic state.

AVAILABLE STATUSES (Choose ONE):

CRITICAL (Priority 100) - Immediate action required:
- critical_margin_collapse: Margins dropping rapidly, profitability at risk
- critical_inventory_risk: Stockout imminent or already occurred
- critical_buybox_loss: Lost BuyBox to competitors
- critical_velocity_crash: Sales velocity dropped dramatically

OPPORTUNITY (Priority 75) - Capitalize on opportunity:
- opportunity_price_power: Can raise prices without losing sales
- opportunity_ad_waste: High ad spend with low ROI
- opportunity_review_gap: Competitors have review advantage
- opportunity_competitor_weakness: Competitor showing weakness (OOS, price issues)

WATCH (Priority 50) - Monitor closely:
- watch_new_competitor: New strong competitor entered market
- watch_price_war: Active price war detected
- watch_seasonal_anomaly: Unusual seasonal pattern
- watch_rank_volatility: BSR unstable

STABLE (Priority 0) - No immediate action:
- stable_fortress: Strong moat, excellent metrics
- stable_cash_cow: Consistent performer
- stable_niche: Small but reliable market

IMPORTANT RULES:
1. You MUST cite specific trigger events in your reasoning
2. Higher severity triggers = higher priority status
3. Multiple triggers can compound into critical status
4. If no critical triggers exist, don't force a critical classification
5. Provide confidence score (0-100) based on data quality

Return ONLY valid JSON in this exact format:
{{
    "product_status": "critical_margin_collapse",
    "strategic_state": "Margin Collapse - Immediate Attention Required",
    "confidence": 85,
    "reasoning": "Margin collapsed from 40% to 15% due to TRIGGER: price_war_active (severity 8). Three competitors dropped prices 20%+ in 7 days, forcing defensive price cuts. Current price of $24.99 is 35% below optimal. Immediate repricing needed to stabilize margins.",
    "risk_score": 85,
    "growth_score": 30
}}
"""

        return prompt

    def _format_triggers_for_llm(self, trigger_events: List[TriggerEvent]) -> str:
        """Format trigger events for LLM consumption."""

        if not trigger_events:
            return "⚠️  No trigger events detected. Product appears stable."

        trigger_lines = []
        for i, trigger in enumerate(trigger_events[:5], 1):  # Top 5 triggers
            trigger_lines.append(
                f"{i}. [{trigger.event_type}] (Severity: {trigger.severity}/10)\n"
                f"   Metric: {trigger.metric_name}\n"
                f"   Change: {trigger.baseline_value:.2f} → {trigger.current_value:.2f} "
                f"({trigger.delta_pct:+.1f}%)\n"
                f"   Detected: {trigger.detected_at.strftime('%Y-%m-%d %H:%M')}"
            )

        return "\n\n".join(trigger_lines)

    def _format_metrics_for_llm(
        self,
        current: Dict[str, Any],
        trends: Dict[str, Any]
    ) -> str:
        """Format metrics for LLM."""

        return f"""CURRENT METRICS:
- Price: ${current.get('price', 0):.2f}
- BSR: {current.get('bsr', 'N/A')}
- Review Count: {current.get('review_count', 0)}
- Rating: {current.get('rating', 0):.1f}★
- BuyBox %: {current.get('buybox_share', 0) * 100:.0f}%
- Inventory: {current.get('inventory', 'N/A')} units

30-DAY TRENDS:
- Price Change: {trends.get('price_change_30d', 0):+.1f}%
- BSR Change: {trends.get('bsr_change_30d', 0):+.1f}%
- Review Velocity: {trends.get('reviews_gained_30d', 0)} new reviews
- BuyBox Change: {trends.get('buybox_change_30d', 0):+.1f}%
"""

    def _format_competitors_for_llm(self, competitor_data: Dict[str, Any]) -> str:
        """Format competitor data for LLM."""

        if not competitor_data:
            return ""

        return f"""COMPETITIVE CONTEXT:
- Competitors in Category: {competitor_data.get('competitor_count', 0)}
- Your Price vs Median: {competitor_data.get('price_vs_median', 0):+.1f}%
- Your Reviews vs Median: {competitor_data.get('reviews_vs_median', 0):+.1f}%
- Competitors OOS: {competitor_data.get('competitors_oos', 0)}
"""

    def _get_system_prompt(self) -> str:
        """System prompt defining AI behavior."""

        return """You are an expert Amazon FBA strategist specializing in strategic product classification.

Your role:
1. Analyze product metrics and trigger events
2. Classify products into strategic states (CRITICAL/OPPORTUNITY/WATCH/STABLE)
3. Provide clear reasoning citing specific trigger events
4. Assign confidence scores based on data quality

Key principles:
- Trigger events are FACTS - always cite them in reasoning
- Severity matters - higher severity = higher priority
- Multiple triggers compound risk
- Be precise with numbers (include $ amounts, percentages)
- Don't exaggerate - if stable, say so

Return ONLY valid JSON. No markdown, no explanations outside JSON."""

    def _parse_classification_result(
        self,
        result: Dict[str, Any],
        trigger_events: List[TriggerEvent]
    ) -> Dict[str, Any]:
        """Parse and validate LLM classification result."""

        try:
            # Parse ProductStatus
            status_str = result.get('product_status', 'stable_fortress')
            product_status = ProductStatus[status_str.upper()]

            # Validate scores
            confidence = max(0, min(100, result.get('confidence', 70)))
            risk_score = max(0, min(100, result.get('risk_score', 50)))
            growth_score = max(0, min(100, result.get('growth_score', 50)))

            # Apply trigger amplification
            risk_score = self.calculate_trigger_amplified_risk(risk_score, trigger_events)
            growth_score = self.calculate_trigger_dampened_growth(growth_score, trigger_events)

            return {
                'product_status': product_status,
                'strategic_state': result.get('strategic_state', 'Unknown State'),
                'confidence': confidence,
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'risk_score': risk_score,
                'growth_score': growth_score,
                'llm_raw_response': result
            }

        except Exception as e:
            print(f"⚠️  Error parsing classification: {str(e)}")
            return self._get_fallback_classification(trigger_events)

    def _get_fallback_classification(self, trigger_events: List[TriggerEvent]) -> Dict[str, Any]:
        """Fallback classification if LLM fails."""

        # Simple heuristic: classify by highest trigger severity
        if not trigger_events:
            status = ProductStatus.STABLE_FORTRESS
            risk_score = 20
        elif trigger_events[0].severity >= 9:
            status = ProductStatus.CRITICAL_MARGIN_COLLAPSE
            risk_score = 90
        elif trigger_events[0].severity >= 7:
            status = ProductStatus.WATCH_PRICE_WAR
            risk_score = 60
        else:
            status = ProductStatus.STABLE_FORTRESS
            risk_score = 30

        return {
            'product_status': status,
            'strategic_state': 'Fallback Classification',
            'confidence': 50,
            'reasoning': 'LLM classification failed - using fallback heuristic',
            'risk_score': risk_score,
            'growth_score': 50,
            'llm_raw_response': None
        }


def validate_classification_quality(classification: Dict[str, Any]) -> bool:
    """
    Validate classification quality.

    Quality gates:
    1. Must have reasoning
    2. Reasoning must be >50 characters
    3. Must have confidence >30
    4. Must cite at least one specific metric/trigger

    Returns:
        True if classification passes quality gates
    """

    reasoning = classification.get('reasoning', '')
    confidence = classification.get('confidence', 0)

    # Gate 1: Must have reasoning
    if not reasoning or len(reasoning) < 50:
        return False

    # Gate 2: Confidence threshold
    if confidence < 30:
        return False

    # Gate 3: Must cite something specific (contains number or %)
    has_numbers = any(char.isdigit() or char == '%' for char in reasoning)
    if not has_numbers:
        return False

    return True
