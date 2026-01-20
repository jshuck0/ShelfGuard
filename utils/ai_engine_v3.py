"""
AI Engine V3 - Network-Aware Insight Generation

Enhanced LLM prompts that inject network intelligence (category benchmarks, competitive position).
Second stage of two-stage LLM architecture (Classification → Insight).

This engine:
1. Takes classification from v2 + network intelligence
2. Generates actionable insights with specific $ amounts
3. Cites both trigger events AND network context
4. Includes projected upside/downside in dollars
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import openai
import os
import json

from src.models.product_status import ProductStatus
from src.models.trigger_event import TriggerEvent


class NetworkAwareInsightEngine:
    """
    Insight generation engine with network intelligence.

    Generates actionable recommendations based on:
    - Strategic classification (from v2)
    - Trigger events (causal reasoning)
    - Network intelligence (category benchmarks, competitive position)
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize insight engine with OpenAI API key."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key

    def generate_insight(
        self,
        asin: str,
        classification: Dict[str, Any],
        trigger_events: List[TriggerEvent],
        network_intelligence: Dict[str, Any],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate actionable insight using network-aware LLM.

        Args:
            asin: Product ASIN
            classification: Strategic classification from v2
            trigger_events: Detected trigger events
            network_intelligence: Category benchmarks + competitive position
            current_metrics: Current product metrics

        Returns:
            Dict with:
                - recommendation: Specific action to take
                - projected_upside_monthly: $ amount upside
                - downside_risk_monthly: $ amount downside
                - action_type: repair/optimize/harvest/defend/expand
                - time_horizon_days: How urgent (1-90 days)
                - confidence: 0-100
                - reasoning: Full explanation with trigger + network citations
        """

        # Build LLM prompt with network intelligence injection
        prompt = self._build_insight_prompt(
            asin=asin,
            classification=classification,
            trigger_events=trigger_events,
            network_intelligence=network_intelligence,
            current_metrics=current_metrics
        )

        # Call LLM for insight generation
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
                temperature=0.5,  # Slightly higher for creative recommendations
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and parse result
            insight = self._parse_insight_result(result, current_metrics)

            return insight

        except Exception as e:
            print(f"⚠️  AI insight generation error: {str(e)}")
            return self._get_fallback_insight(classification, current_metrics)

    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================

    def _build_insight_prompt(
        self,
        asin: str,
        classification: Dict[str, Any],
        trigger_events: List[TriggerEvent],
        network_intelligence: Dict[str, Any],
        current_metrics: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt with network intelligence injection."""

        # Format classification
        classification_context = f"""STRATEGIC CLASSIFICATION:
Status: {classification['product_status'].value}
State: {classification['strategic_state']}
Risk Score: {classification['risk_score']:.0f}/100
Growth Score: {classification['growth_score']:.0f}/100
Reasoning: {classification['reasoning']}
"""

        # Format trigger events
        trigger_context = self._format_triggers_for_llm(trigger_events)

        # Format network intelligence
        network_context = self._format_network_intelligence_for_llm(network_intelligence)

        # Format current metrics
        metrics_context = self._format_current_metrics(current_metrics)

        prompt = f"""Generate an actionable insight for this Amazon product.

PRODUCT: {asin}

{classification_context}

{trigger_context}

{network_context}

{metrics_context}

INSIGHT GENERATION TASK:
Based on the classification, triggers, and network intelligence above, generate a specific, actionable recommendation.

ACTION TYPES (Choose ONE):
- repair: Fix critical issue (margins, inventory, BuyBox)
- optimize: Improve efficiency (pricing, ads, reviews)
- harvest: Maximize profit from stable position
- defend: Protect against competitive threats
- expand: Capitalize on growth opportunity

REQUIREMENTS:
1. MUST include specific dollar amounts for projected upside/downside
2. MUST cite at least one trigger event in reasoning
3. MUST reference network intelligence (category benchmarks or competitive position)
4. MUST be actionable (specific steps, not vague advice)
5. MUST include time horizon (how urgent?)

CALCULATION GUIDANCE:
- Current monthly revenue ≈ (30-day sales estimate based on BSR) × Price
- Margin impact = Revenue × Margin % change
- For pricing: If suggest raising price 10%, upside = Current Revenue × 0.10 × Current Margin
- For inventory: If OOS risk, downside = Daily Revenue × Days OOS
- For BuyBox: If lost BuyBox, downside = Revenue × (1 - New BuyBox %)

Return ONLY valid JSON in this exact format:
{{
    "recommendation": "Raise price from $24.99 to $27.99 (+12%). Category median is $28.50 and you have a review advantage (150 vs 85 median). Test $27.99 for 7 days, monitor conversion rate.",
    "projected_upside_monthly": 450.00,
    "downside_risk_monthly": 150.00,
    "action_type": "optimize",
    "time_horizon_days": 7,
    "confidence": 80,
    "reasoning": "TRIGGER: opportunity_price_power detected (reviews 76% above median). NETWORK: Your price is 12% below category median ($24.99 vs $28.50) despite having 150 reviews vs 85 median - clear pricing power. Upside: $3 price increase × 150 units/mo = $450/mo. Risk: May lose 10% of sales = $150/mo downside. Net EV: +$300/mo.",
    "key_metrics_cited": ["price_vs_median: -12%", "review_count_vs_median: +76%", "category_median_price: $28.50"]
}}
"""

        return prompt

    def _format_triggers_for_llm(self, trigger_events: List[TriggerEvent]) -> str:
        """Format trigger events for LLM."""

        if not trigger_events:
            return "TRIGGER EVENTS:\n⚠️  No trigger events detected."

        trigger_lines = ["TRIGGER EVENTS (Top 3):"]
        for i, trigger in enumerate(trigger_events[:3], 1):
            trigger_lines.append(
                f"{i}. {trigger.event_type} (Severity {trigger.severity}/10)\n"
                f"   {trigger.metric_name}: {trigger.baseline_value:.2f} → {trigger.current_value:.2f} "
                f"({trigger.delta_pct:+.1f}%)"
            )

        return "\n".join(trigger_lines)

    def _format_network_intelligence_for_llm(self, network_intel: Dict[str, Any]) -> str:
        """Format network intelligence for LLM."""

        category_benchmarks = network_intel.get('category_benchmarks', {})
        competitive_position = network_intel.get('competitive_position', {})
        brand_intel = network_intel.get('brand_intelligence', {})

        output = ["NETWORK INTELLIGENCE:"]

        # Category benchmarks
        if category_benchmarks:
            output.append("\nCategory Benchmarks:")
            output.append(f"- Median Price: ${category_benchmarks.get('median_price', 0):.2f}")
            output.append(f"- Median Reviews: {category_benchmarks.get('median_review_count', 0):.0f}")
            output.append(f"- Median Rating: {category_benchmarks.get('median_rating', 0):.2f}★")
            output.append(f"- Median BSR: {category_benchmarks.get('median_bsr', 'N/A')}")
            output.append(f"- Data Quality: {category_benchmarks.get('data_quality', 'UNKNOWN')}")
            output.append(f"- ASINs Tracked: {category_benchmarks.get('total_asins_tracked', 0)}")

        # Competitive position
        if competitive_position:
            output.append("\nYour Competitive Position:")
            output.append(f"- Price vs Median: {competitive_position.get('price_vs_median', 0):+.1f}%")
            output.append(f"- Reviews vs Median: {competitive_position.get('reviews_vs_median', 0):+.1f}%")
            output.append(f"- Rating vs Median: {competitive_position.get('rating_vs_median', 0):+.2f}★")
            output.append(f"- Price Percentile: {competitive_position.get('price_percentile', 50):.0f}th")
            output.append(f"- Review Percentile: {competitive_position.get('review_percentile', 50):.0f}th")

            # Competitive advantages
            advantages = competitive_position.get('competitive_advantages', [])
            if advantages:
                output.append("\nCompetitive Advantages:")
                for adv in advantages:
                    output.append(f"  ✓ {adv}")

            # Competitive weaknesses
            weaknesses = competitive_position.get('competitive_weaknesses', [])
            if weaknesses:
                output.append("\nCompetitive Weaknesses:")
                for weak in weaknesses:
                    output.append(f"  ✗ {weak}")

        # Brand intelligence
        if brand_intel and brand_intel.get('data_available'):
            output.append("\nBrand Intelligence:")
            output.append(f"- Brand: {brand_intel.get('brand', 'Unknown')}")
            output.append(f"- Avg Price: ${brand_intel.get('avg_price', 0):.2f}")
            output.append(f"- Avg Reviews: {brand_intel.get('avg_review_count', 0):.0f}")
            output.append(f"- Product Count: {brand_intel.get('product_count', 0)}")

        if len(output) == 1:  # Only header
            output.append("⚠️  No network intelligence available yet.")

        return "\n".join(output)

    def _format_current_metrics(self, current_metrics: Dict[str, Any]) -> str:
        """Format current metrics for context."""

        return f"""CURRENT PRODUCT METRICS:
- Price: ${current_metrics.get('price', 0):.2f}
- BSR: {current_metrics.get('bsr', 'N/A')}
- Review Count: {current_metrics.get('review_count', 0)}
- Rating: {current_metrics.get('rating', 0):.1f}★
- BuyBox Share: {current_metrics.get('buybox_share', 0) * 100:.0f}%
- Estimated Monthly Sales: {current_metrics.get('estimated_monthly_sales', 0):.0f} units
- Estimated Monthly Revenue: ${current_metrics.get('estimated_monthly_revenue', 0):.2f}
"""

    def _get_system_prompt(self) -> str:
        """System prompt for insight generation."""

        return """You are an expert Amazon FBA strategist specializing in actionable insights.

Your role:
1. Generate specific, actionable recommendations (not vague advice)
2. Include exact dollar amounts for projected upside and downside
3. Cite trigger events and network intelligence in reasoning
4. Provide clear implementation steps
5. Assess realistic time horizons

Key principles:
- BE SPECIFIC: Include exact prices, percentages, dollar amounts
- CITE EVIDENCE: Reference triggers and network benchmarks
- BE REALISTIC: Don't promise unrealistic gains
- SHOW YOUR WORK: Explain calculations for upside/downside
- BE ACTIONABLE: "Raise price to $X" not "consider pricing"

Quality gates (your output MUST have):
✓ Specific dollar amount for upside (projected_upside_monthly)
✓ Specific dollar amount for downside (downside_risk_monthly)
✓ At least one trigger event cited in reasoning
✓ At least one network metric cited (category median, percentile, etc.)
✓ Clear action type (repair/optimize/harvest/defend/expand)

Return ONLY valid JSON. No markdown, no explanations outside JSON."""

    def _parse_insight_result(
        self,
        result: Dict[str, Any],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse and validate LLM insight result."""

        try:
            # Validate required fields
            recommendation = result.get('recommendation', '')
            upside = float(result.get('projected_upside_monthly', 0))
            downside = float(result.get('downside_risk_monthly', 0))
            action_type = result.get('action_type', 'optimize')
            time_horizon = int(result.get('time_horizon_days', 30))
            confidence = max(0, min(100, result.get('confidence', 70)))
            reasoning = result.get('reasoning', '')

            # Sanity checks
            if upside < 0:
                upside = 0
            if downside < 0:
                downside = 0

            # Cap upside/downside at 10x monthly revenue (sanity check)
            max_impact = current_metrics.get('estimated_monthly_revenue', 1000) * 10
            if upside > max_impact:
                upside = max_impact
            if downside > max_impact:
                downside = max_impact

            return {
                'recommendation': recommendation,
                'projected_upside_monthly': round(upside, 2),
                'downside_risk_monthly': round(downside, 2),
                'action_type': action_type,
                'time_horizon_days': time_horizon,
                'confidence': confidence,
                'reasoning': reasoning,
                'key_metrics_cited': result.get('key_metrics_cited', []),
                'llm_raw_response': result
            }

        except Exception as e:
            print(f"⚠️  Error parsing insight: {str(e)}")
            return self._get_fallback_insight(None, current_metrics)

    def _get_fallback_insight(
        self,
        classification: Optional[Dict[str, Any]],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback insight if LLM fails."""

        return {
            'recommendation': 'Monitor product metrics and wait for more data to generate specific recommendations.',
            'projected_upside_monthly': 0.0,
            'downside_risk_monthly': 0.0,
            'action_type': 'harvest',
            'time_horizon_days': 30,
            'confidence': 30,
            'reasoning': 'LLM insight generation failed - using fallback recommendation. More data needed for specific guidance.',
            'key_metrics_cited': [],
            'llm_raw_response': None
        }


def validate_insight_quality(insight: Dict[str, Any]) -> bool:
    """
    Validate insight quality using strict gates.

    Quality gates:
    1. Must have recommendation (>50 chars)
    2. Must have specific dollar amounts (upside OR downside > 0)
    3. Must cite trigger events in reasoning (contains "TRIGGER:")
    4. Must cite network intelligence in reasoning (contains "NETWORK:" or "median" or "percentile")
    5. Must have confidence >40

    Returns:
        True if insight passes ALL quality gates
    """

    recommendation = insight.get('recommendation', '')
    upside = insight.get('projected_upside_monthly', 0)
    downside = insight.get('downside_risk_monthly', 0)
    reasoning = insight.get('reasoning', '')
    confidence = insight.get('confidence', 0)

    # Gate 1: Must have meaningful recommendation
    if not recommendation or len(recommendation) < 50:
        print("❌ Quality gate failed: Recommendation too short")
        return False

    # Gate 2: Must have dollar amounts
    if upside <= 0 and downside <= 0:
        print("❌ Quality gate failed: No dollar amounts provided")
        return False

    # Gate 3: Must cite trigger events
    if 'TRIGGER:' not in reasoning.upper():
        print("❌ Quality gate failed: No trigger event cited")
        return False

    # Gate 4: Must cite network intelligence
    network_keywords = ['NETWORK:', 'median', 'percentile', 'benchmark', 'category']
    has_network_citation = any(keyword in reasoning.lower() for keyword in network_keywords)
    if not has_network_citation:
        print("❌ Quality gate failed: No network intelligence cited")
        return False

    # Gate 5: Confidence threshold
    if confidence < 40:
        print("❌ Quality gate failed: Confidence too low")
        return False

    return True


def calculate_net_expected_value(insight: Dict[str, Any]) -> float:
    """
    Calculate net expected value (EV) of recommendation.

    EV = (Upside × Success Probability) - (Downside × Failure Probability)

    Success probability based on confidence score.
    """

    upside = insight.get('projected_upside_monthly', 0)
    downside = insight.get('downside_risk_monthly', 0)
    confidence = insight.get('confidence', 50)

    # Convert confidence to probability
    success_prob = confidence / 100
    failure_prob = 1 - success_prob

    # Calculate EV
    ev = (upside * success_prob) - (downside * failure_prob)

    return round(ev, 2)
