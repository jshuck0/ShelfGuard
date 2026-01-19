"""
ShelfGuard Strategic LLM Classifier
=====================================
LLM-Powered AI Engine for Product Strategic Classification

This module uses an LLM (GPT-4o-mini) to analyze complex Keepa signals and classify
products into Strategic States based on nuanced signal interpretation.

The LLM approach handles "gray areas" that deterministic rules cannot:
- Seasonal dips vs actual distress
- Premium pricing power vs overpricing
- Competitive pressure vs healthy market dynamics

Strategic States:
1. FORTRESS    - Dominant position, high margins, stable velocity
2. HARVEST     - Strong cash flow, can raise prices, minimal investment needed
3. TRENCH_WAR  - Under competitive attack, defending market share
4. DISTRESS    - Margin compression, velocity decay, needs intervention
5. TERMINAL    - Exit required, liquidation protocol

Features:
- Async parallel processing for 20-50 products in <3 seconds
- Fallback to deterministic logic if LLM fails
- Human-readable reasoning for each classification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd
import numpy as np
import asyncio
import json
import os
from datetime import datetime

# LLM imports - try OpenAI first, then Anthropic
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# =============================================================================
# DEBUG TRACKING
# =============================================================================

def _track_llm_call(success: bool = True):
    """Track LLM calls for debugging."""
    try:
        if STREAMLIT_AVAILABLE:
            import streamlit as st
            if 'llm_stats' not in st.session_state:
                st.session_state.llm_stats = {'llm_calls': 0, 'fallback_calls': 0}
            
            if success:
                st.session_state.llm_stats['llm_calls'] += 1
            else:
                st.session_state.llm_stats['fallback_calls'] += 1
    except:
        pass  # Silently fail if not in Streamlit context


# =============================================================================
# STRATEGIC STATES (Simplified to 5)
# =============================================================================

class StrategicState(Enum):
    """
    The 5 Strategic States for CPG product classification.
    Each state represents a distinct strategic position requiring specific actions.
    """
    FORTRESS = "FORTRESS"       # Dominant position, high margins, stable velocity
    HARVEST = "HARVEST"         # Strong cash flow, can raise prices
    TRENCH_WAR = "TRENCH_WAR"   # Under competitive attack, defending share
    DISTRESS = "DISTRESS"       # Margin compression, velocity decay
    TERMINAL = "TERMINAL"       # Exit required, liquidation protocol


# State definitions with visual properties
STATE_DEFINITIONS = {
    StrategicState.FORTRESS: {
        "emoji": "🏰",
        "primary_outcome": "Protect & Expand",
        "default_action": "Defend position with defensive keywords. Test price increases.",
        "color": "#00704A",  # Starbucks green
    },
    StrategicState.HARVEST: {
        "emoji": "🌾",
        "primary_outcome": "Extract Maximum Value",
        "default_action": "Raise prices. Reduce ad spend. Maximize profit extraction.",
        "color": "#28a745",  # Success green
    },
    StrategicState.TRENCH_WAR: {
        "emoji": "⚔️",
        "primary_outcome": "Defend Market Share",
        "default_action": "Match competitor pricing. Increase visibility spend. Hold position.",
        "color": "#fd7e14",  # Warning orange
    },
    StrategicState.DISTRESS: {
        "emoji": "🚨",
        "primary_outcome": "Stabilize & Fix",
        "default_action": "Pause non-essential spend. Fix pricing. Evaluate root cause.",
        "color": "#dc3545",  # Danger red
    },
    StrategicState.TERMINAL: {
        "emoji": "💀",
        "primary_outcome": "Execute Exit",
        "default_action": "Liquidation protocol. Clear inventory at breakeven. No new investment.",
        "color": "#343a40",  # Dark gray
    },
}


# =============================================================================
# STRATEGIC BRIEF OUTPUT
# =============================================================================

@dataclass
class StrategicBrief:
    """
    UNIFIED AI ENGINE OUTPUT
    
    Combines:
    1. Strategic Classification (FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL)
    2. Predictive Intelligence (30-day risk forecast, alerts, cost of inaction)
    
    This is the single output from the consolidated AI engine.
    """
    # === STRATEGIC CLASSIFICATION ===
    strategic_state: str              # FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL
    confidence: float                 # Model certainty (0.0 - 1.0)
    reasoning: str                    # AI-generated explanation
    recommended_action: str           # AI-generated specific action
    
    # Visual properties (from STATE_DEFINITIONS)
    state_emoji: str = ""
    state_color: str = ""
    primary_outcome: str = ""
    
    # Source tracking
    source: str = "llm"               # "llm" or "fallback"
    signals_detected: List[str] = field(default_factory=list)
    asin: str = ""
    
    # === PREDICTIVE INTELLIGENCE ===
    thirty_day_risk: float = 0.0      # Projected $ at risk over next 30 days
    daily_burn_rate: float = 0.0      # Current daily loss rate
    velocity_multiplier: float = 1.0  # Trend adjustment (>1 = accelerating loss)
    
    # Risk components
    price_erosion_risk: float = 0.0   # Risk from pricing pressure
    share_erosion_risk: float = 0.0   # Risk from market share loss
    stockout_risk: float = 0.0        # Risk from inventory stockout
    
    # Predictive state (forward-looking)
    predictive_state: str = "HOLD"    # DEFEND, EXPLOIT, REPLENISH, HOLD
    predictive_emoji: str = "✅"
    predictive_description: str = ""
    
    # Cost of inaction
    cost_of_inaction: str = ""        # Human-readable consequence
    ai_recommendation: str = ""       # Predictive alert for UI
    alert_type: str = ""              # INVENTORY, PRICING, RANK, or empty
    alert_urgency: str = ""           # HIGH, MEDIUM, LOW
    predicted_event_date: str = ""    # When the event will occur
    action_deadline: str = ""         # When user must act
    
    # Model quality
    data_quality: str = "MEDIUM"      # HIGH, MEDIUM, LOW, VERY_LOW
    
    # === GROWTH INTELLIGENCE ===
    thirty_day_growth: float = 0.0        # Predicted expansion alpha
    price_lift_opportunity: float = 0.0   # Revenue from price optimization
    conquest_opportunity: float = 0.0     # Revenue from competitor vulnerabilities
    expansion_recommendation: str = ""    # Growth-specific AI recommendation
    growth_validated: bool = True         # False if velocity declining (blocks growth recs)
    opportunity_type: str = ""            # PRICE_LIFT, CONQUEST, EXPAND, or empty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dashboard rendering."""
        return {
            # Strategic outputs
            "strategic_state": self.strategic_state,
            "confidence_score": self.confidence,
            "primary_outcome": self.primary_outcome,
            "recommended_plan": self.recommended_action,
            "reasoning": self.reasoning,
            "signals_detected": self.signals_detected,
            "state_emoji": self.state_emoji,
            "state_color": self.state_color,
            "source": self.source,
            "asin": self.asin,
            # Predictive outputs
            "thirty_day_risk": self.thirty_day_risk,
            "daily_burn_rate": self.daily_burn_rate,
            "predictive_state": self.predictive_state,
            "predictive_emoji": self.predictive_emoji,
            "cost_of_inaction": self.cost_of_inaction,
            "ai_recommendation": self.ai_recommendation,
            "alert_type": self.alert_type,
            "alert_urgency": self.alert_urgency,
            "model_certainty": self.confidence,
            "data_quality": self.data_quality,
        }


# =============================================================================
# LLM SYSTEM PROMPT
# =============================================================================

STRATEGIST_SYSTEM_PROMPT = """You are a Senior CPG Strategist with 20 years of Amazon experience.
Analyze the following product metrics and classify it into ONE of 5 Strategic States.

## Strategic States

1. **FORTRESS** - Dominant market position
   - Strong Buy Box ownership (>80%)
   - Healthy margins (>15%)
   - Stable or improving rank
   - Low competitive pressure
   - Example: Market leader with pricing power

2. **HARVEST** - Cash cow, maximize extraction
   - Stable rank (not growing, not declining significantly)
   - Good margins
   - Low ad spend efficiency (can reduce spend)
   - Premium price holding despite competition
   - Example: Mature product with implicit pricing power

3. **TRENCH_WAR** - Competitive battle, defend share
   - Increasing competitor count
   - Price pressure (competitors undercutting)
   - Rank volatility
   - Buy Box rotation/loss
   - Example: Category under attack from new entrants

4. **DISTRESS** - Needs intervention, value at risk
   - Margin compression (<10%)
   - Velocity decay (rank worsening)
   - Review velocity stagnant or negative
   - Stock issues or pricing problems
   - Example: Product losing relevance, needs fixing

5. **TERMINAL** - Exit required, cut losses
   - Severely negative or zero margins
   - Sustained rank decline (>30% over 90 days)
   - Zero path to profitability
   - Example: End of lifecycle, liquidate inventory

## Analysis Guidelines

- Focus on the 90-DAY TREND, not daily noise
- Look for NUANCED COMBINATIONS:
  * High Price + Low Competition + Stable Rank = FORTRESS (luxury positioning)
  * Declining Rank + Healthy Margin = DISTRESS (fixable problem)
  * Price War + High Volume = TRENCH_WAR (defend share)
  * Negative Margin + Any Velocity = TERMINAL (exit immediately)
- Consider CONTEXT: A rank drop from #100 to #150 is different from #10,000 to #15,000
- Trust your CPG instincts about brand dynamics and category health

## Output Format

Return ONLY valid JSON with this exact structure:
{
    "strategic_state": "STATE_NAME",
    "confidence": 0.95,
    "reasoning": "1-2 clear sentences explaining WHY. Be specific and direct.",
    "recommended_action": "One specific action (e.g., 'Reduce ad spend 20%', 'Increase price to $X', 'Improve listing images')"
}

Important:
- Keep reasoning under 80 characters
- Make recommended_action specific and measurable
- Use business language, not military jargon
- Return ONLY the JSON object, no other text."""


def _get_strategic_bias_instructions(strategic_bias: str) -> str:
    """
    Generate additional LLM instructions based on the user's strategic focus.
    
    This is the "magic" that makes the entire dashboard shift based on one selector.
    """
    bias_instructions = {
        "Profit Maximization": """
## 🎯 STRATEGIC BIAS: PROFIT MAXIMIZATION

The user has set their priority to **Profit**. Adjust your analysis accordingly:

- **Heavily penalize low margins** (<10%): Classify as DISTRESS even if velocity is good
- **Reward price increases**: If price is up and margin improved, prefer FORTRESS or HARVEST
- **Be aggressive on cost**: Recommend "Cut ad spend" or "Raise price" before "Scale ads"
- **Question growth spending**: If ad spend is high but margin is thin, recommend pullback
- **Example**: Product with 8% margin and growing rank → DISTRESS ("Unsustainable growth")
""",
        "Balanced Defense": """
## 🎯 STRATEGIC BIAS: BALANCED DEFENSE

The user has set their priority to **Balanced**. Use standard strategic logic:

- Evaluate all factors equally (margin, velocity, competition, reviews)
- Apply the standard state definitions without heavy bias
- Recommend balanced actions that consider both profitability and market position
""",
        "Aggressive Growth": """
## 🎯 STRATEGIC BIAS: AGGRESSIVE GROWTH

The user has set their priority to **Growth**. Adjust your analysis accordingly:

- **Forgive margin compression** if rank is improving: 5% margin + improving rank = TRENCH_WAR, not TERMINAL
- **Reward velocity gains**: Prioritize products with improving sales rank
- **Encourage investment**: Recommend "Increase ad spend" or "Scale campaigns" for products with momentum
- **Be patient with new launches**: Don't classify as TERMINAL unless rank is catastrophic (>100k) AND declining
- **Example**: Product with 3% margin but rank improving 20% → TRENCH_WAR ("Acceptable sacrifice for share gain")

**CONQUEST INTELLIGENCE:**
When you detect competitor vulnerabilities (OOS, price cuts, declining reviews):
- Recommend specific conquest actions with estimated revenue capture
- Prioritize share gain over margin protection
- Format recommendations as: "Conquest Opportunity: [Competitor] vulnerable. Redirect $X ad spend to capture $Y revenue."
- If competitor_oos_pct > 30%, strongly recommend aggressive keyword bidding
- If competitor is cutting prices, recommend holding price and increasing ad visibility
"""
    }
    
    return bias_instructions.get(strategic_bias, bias_instructions["Balanced Defense"])


# =============================================================================
# LLM CLIENT
# =============================================================================

def _get_openai_client() -> Optional[AsyncOpenAI]:
    """Get async OpenAI client with API key from environment or secrets."""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = None
    
    # Try Streamlit secrets first
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY")
        except Exception:
            pass
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    return AsyncOpenAI(api_key=api_key)


def _get_model_name() -> str:
    """Get LLM model name from secrets or default."""
    # Try Streamlit secrets first
    if STREAMLIT_AVAILABLE:
        try:
            model = st.secrets.get("openai", {}).get("model")
            if model:
                return model
        except Exception:
            pass
    
    # Default to GPT-4o-mini for cost efficiency
    return "gpt-4o-mini"


# =============================================================================
# CORE LLM CLASSIFIER
# =============================================================================

async def analyze_strategy_with_llm(
    row_data: Dict[str, Any],
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
    strategic_bias: str = "Balanced Defense"
) -> StrategicBrief:
    """
    Analyze a product row using LLM and return strategic classification.
    
    Args:
        row_data: Product metrics dictionary
        client: Optional AsyncOpenAI client (will create if not provided)
        model: Optional model name (defaults to gpt-4o-mini)
        timeout: Request timeout in seconds
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        
    Returns:
        StrategicBrief with LLM-generated classification and reasoning
    """
    # Get client and model
    if client is None:
        client = _get_openai_client()
    
    if client is None:
        # No LLM available, use fallback
        return _determine_state_fallback(row_data, strategic_bias=strategic_bias)
    
    if model is None:
        model = _get_model_name()
    
    # Prepare data for LLM (clean and format)
    clean_data = _prepare_row_for_llm(row_data)
    
    # Build system prompt with strategic bias
    bias_instructions = _get_strategic_bias_instructions(strategic_bias)
    full_system_prompt = f"{STRATEGIST_SYSTEM_PROMPT}\n\n{bias_instructions}"
    
    try:
        # Call LLM with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": f"Analyze this product:\n\n```json\n{json.dumps(clean_data, indent=2)}\n```"}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=300,
                response_format={"type": "json_object"}
            ),
            timeout=timeout
        )
        
        # Parse response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Validate and build StrategicBrief
        state_str = result.get("strategic_state", "DISTRESS").upper()
        
        # Validate state is one of our 5 states
        valid_states = [s.value for s in StrategicState]
        if state_str not in valid_states:
            state_str = "DISTRESS"  # Default to DISTRESS if invalid
        
        state = StrategicState(state_str)
        state_def = STATE_DEFINITIONS[state]
        
        # Normalize confidence: handle both decimal (0.95) and percentage (95) formats
        raw_confidence = result.get("confidence", 0.7)
        if isinstance(raw_confidence, str):
            # Remove % sign if present and convert
            raw_confidence = raw_confidence.replace("%", "").strip()
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            # If parsing fails, use default
            confidence = 0.7
        
        # If confidence > 1 and <= 100, assume it's a percentage and convert to decimal
        # Most LLMs following the prompt will return 0.0-1.0, so values > 1 are likely percentages
        if 1.0 < confidence <= 100.0:
            confidence = confidence / 100.0
        
        # Clamp to valid range (0.0 to 1.0)
        confidence = max(0.0, min(1.0, confidence))
        
        # Ensure minimum reasonable confidence for LLM responses (avoid showing 1% for valid analyses)
        # If confidence is suspiciously low (< 0.3), it's likely a parsing error - use default
        if confidence < 0.3:
            confidence = 0.7  # Default to medium-high confidence for LLM responses
        
        # Track successful LLM call
        _track_llm_call(success=True)
        
        return StrategicBrief(
            strategic_state=state_str,
            confidence=confidence,
            reasoning=result.get("reasoning", "LLM analysis complete."),
            recommended_action=result.get("recommended_action", state_def["default_action"]),
            state_emoji=state_def["emoji"],
            state_color=state_def["color"],
            primary_outcome=state_def["primary_outcome"],
            source="llm",
            signals_detected=_extract_signal_summary(clean_data),
            asin=row_data.get("asin", ""),
        )
        
    except asyncio.TimeoutError:
        # Timeout - use fallback
        _track_llm_call(success=False)
        return _determine_state_fallback(row_data, reason="LLM timeout", strategic_bias=strategic_bias)
    except json.JSONDecodeError:
        # Invalid JSON response - use fallback
        _track_llm_call(success=False)
        return _determine_state_fallback(row_data, reason="Invalid LLM response", strategic_bias=strategic_bias)
    except Exception as e:
        # Any other error - use fallback
        _track_llm_call(success=False)
        return _determine_state_fallback(row_data, reason=f"LLM error: {str(e)[:50]}", strategic_bias=strategic_bias)


def _prepare_row_for_llm(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and format row data for LLM consumption.
    
    Removes noise, formats numbers nicely, focuses on key metrics.
    """
    clean = {}
    
    # Identity
    if "asin" in row_data:
        clean["asin"] = row_data["asin"]
    if "title" in row_data:
        title = str(row_data["title"])[:80]
        clean["title"] = title + "..." if len(str(row_data.get("title", ""))) > 80 else title
    
    # Pricing metrics
    price_fields = ["filled_price", "buy_box_price", "amazon_price", "current_AMAZON"]
    for field in price_fields:
        if field in row_data and row_data[field] is not None:
            val = _safe_float(row_data[field])
            if val and val > 0:
                # Convert cents to dollars if needed
                if val > 500:
                    val = val / 100
                clean["current_price"] = f"${val:.2f}"
                break
    
    # Price trend
    if "avg90_AMAZON" in row_data or "price_90d_avg" in row_data:
        avg = _safe_float(row_data.get("avg90_AMAZON") or row_data.get("price_90d_avg"))
        if avg and avg > 0:
            if avg > 500:
                avg = avg / 100
            clean["price_90d_avg"] = f"${avg:.2f}"
    
    # Margin
    if "net_margin" in row_data:
        margin = _safe_float(row_data["net_margin"])
        if margin is not None:
            clean["net_margin"] = f"{margin*100:.1f}%"
    
    # Rank metrics
    rank_fields = ["sales_rank_filled", "sales_rank", "current_SALES"]
    for field in rank_fields:
        if field in row_data and row_data[field] is not None:
            rank = _safe_float(row_data[field])
            if rank and rank > 0:
                clean["current_sales_rank"] = int(rank)
                break
    
    # Rank trends
    if "deltaPercent30_SALES" in row_data or "rank_delta_30d_pct" in row_data:
        delta = _safe_float(row_data.get("deltaPercent30_SALES") or row_data.get("rank_delta_30d_pct"))
        if delta is not None:
            clean["rank_change_30d"] = f"{delta:+.1f}%"
    
    if "deltaPercent90_SALES" in row_data or "rank_delta_90d_pct" in row_data:
        delta = _safe_float(row_data.get("deltaPercent90_SALES") or row_data.get("rank_delta_90d_pct"))
        if delta is not None:
            clean["rank_change_90d"] = f"{delta:+.1f}%"
    
    if "velocity_decay" in row_data:
        decay = _safe_float(row_data["velocity_decay"], default=1.0)
        if decay != 1.0:
            if decay < 0.9:
                clean["velocity_trend"] = f"Accelerating ({decay:.2f}x)"
            elif decay > 1.1:
                clean["velocity_trend"] = f"Decaying ({decay:.2f}x)"
            else:
                clean["velocity_trend"] = "Stable"
    
    # Competition
    offer_fields = ["new_offer_count", "current_COUNT_NEW"]
    for field in offer_fields:
        if field in row_data and row_data[field] is not None:
            count = _safe_float(row_data[field])
            if count is not None:
                clean["competitor_count"] = int(count)
                break
    
    if "delta30_COUNT_NEW" in row_data:
        delta = _safe_float(row_data["delta30_COUNT_NEW"])
        if delta is not None and delta != 0:
            clean["competitor_change_30d"] = f"{delta:+.0f} sellers"
    
    # Buy Box
    if "amazon_bb_share" in row_data:
        bb = _safe_float(row_data["amazon_bb_share"])
        if bb is not None:
            clean["buybox_ownership"] = f"{bb*100:.0f}%"
    
    if "buyBoxStatsAmazon30" in row_data:
        bb = _safe_float(row_data["buyBoxStatsAmazon30"])
        if bb is not None:
            clean["amazon_buybox_30d"] = f"{bb:.0f}%"
    
    # Reviews
    review_fields = ["review_count", "current_COUNT_REVIEWS"]
    for field in review_fields:
        if field in row_data and row_data[field] is not None:
            count = _safe_float(row_data[field])
            if count is not None:
                clean["review_count"] = int(count)
                break
    
    if "delta30_COUNT_REVIEWS" in row_data:
        delta = _safe_float(row_data["delta30_COUNT_REVIEWS"])
        if delta is not None and delta != 0:
            clean["reviews_added_30d"] = int(delta)
    
    # Rating
    rating_fields = ["rating", "current_RATING"]
    for field in rating_fields:
        if field in row_data and row_data[field] is not None:
            rating = _safe_float(row_data[field])
            if rating is not None and rating > 0:
                # Keepa stores as rating*10
                if rating > 10:
                    rating = rating / 10
                clean["rating"] = f"{rating:.1f}★"
                break
    
    # Revenue (for context)
    if "weekly_sales_filled" in row_data:
        rev = _safe_float(row_data["weekly_sales_filled"])
        if rev is not None and rev > 0:
            clean["weekly_revenue"] = f"${rev:,.0f}"
    
    # Monthly sold (if available)
    if "monthlySold" in row_data:
        sold = _safe_float(row_data["monthlySold"])
        if sold is not None and sold > 0:
            clean["monthly_units_sold"] = f"{int(sold):,}+"
    
    # Price gap
    if "price_gap" in row_data:
        gap = _safe_float(row_data["price_gap"])
        if gap is not None and abs(gap) > 0.03:
            clean["price_vs_competitor"] = f"{gap*100:+.0f}%"
    
    return clean


def _extract_signal_summary(clean_data: Dict[str, Any]) -> List[str]:
    """Extract key signals as a summary list for display."""
    signals = []
    
    if "current_sales_rank" in clean_data:
        signals.append(f"Rank: #{clean_data['current_sales_rank']:,}")
    
    if "rank_change_90d" in clean_data:
        signals.append(f"90d Rank Δ: {clean_data['rank_change_90d']}")
    
    if "net_margin" in clean_data:
        signals.append(f"Margin: {clean_data['net_margin']}")
    
    if "buybox_ownership" in clean_data:
        signals.append(f"Buy Box: {clean_data['buybox_ownership']}")
    
    if "competitor_count" in clean_data:
        signals.append(f"Sellers: {clean_data['competitor_count']}")
    
    if "review_count" in clean_data:
        signals.append(f"Reviews: {clean_data['review_count']:,}")
    
    return signals


# =============================================================================
# FALLBACK DETERMINISTIC LOGIC
# =============================================================================

def _determine_state_fallback(
    row_data: Dict[str, Any], 
    reason: str = "No LLM available",
    strategic_bias: str = "Balanced Defense"
) -> StrategicBrief:
    """
    Fallback deterministic logic when LLM is unavailable or fails.
    
    Uses simple rules to classify products into strategic states.
    Adjusts thresholds based on user's strategic bias.
    This ensures the dashboard never crashes even without LLM access.
    """
    # Extract key metrics
    margin = _safe_float(row_data.get("net_margin"), default=0.10)
    bb_share = _safe_float(row_data.get("amazon_bb_share"), default=0.80)
    velocity_decay = _safe_float(row_data.get("velocity_decay"), default=1.0)
    offer_count = _safe_float(row_data.get("new_offer_count") or row_data.get("current_COUNT_NEW"), default=5)
    rank_delta_90 = _safe_float(row_data.get("deltaPercent90_SALES") or row_data.get("rank_delta_90d_pct"), default=0)
    price_gap = _safe_float(row_data.get("price_gap"), default=0)
    
    # Adjust thresholds based on strategic bias
    if strategic_bias == "Profit Maximization":
        # More strict on margins, less forgiving
        margin_terminal = 0.03  # Stricter
        margin_distress = 0.10  # Stricter
        margin_harvest = 0.15
        margin_fortress = 0.20  # Higher bar
    elif strategic_bias == "Aggressive Growth":
        # More forgiving on margins if velocity is good
        margin_terminal = 0.00  # Only truly zero margin is terminal
        margin_distress = 0.05  # More forgiving
        margin_harvest = 0.12
        margin_fortress = 0.15  # Lower bar
    else:  # Balanced Defense
        margin_terminal = 0.02
        margin_distress = 0.08
        margin_harvest = 0.15
        margin_fortress = 0.18
    
    # Decision tree
    signals = []
    
    # TERMINAL: Severe margin issues or sustained decline
    # Growth mode is more forgiving if rank is improving
    is_rank_improving = rank_delta_90 < -10  # Negative = improving rank
    if strategic_bias == "Aggressive Growth" and is_rank_improving and margin > 0:
        # Don't classify as TERMINAL if we're growing (even with low margin)
        pass
    elif margin < margin_terminal or (rank_delta_90 and rank_delta_90 > 30 and margin < margin_distress):
        state = StrategicState.TERMINAL
        confidence = 0.85
        reasoning = f"Critical metrics: Margin {margin*100:.1f}%, 90d rank change {rank_delta_90:+.0f}%."
        action = "Initiate exit protocol. Clear inventory at breakeven."
        signals.append(f"Margin CRITICAL ({margin*100:.1f}%)")
        if rank_delta_90 > 30:
            signals.append(f"Rank DECLINING ({rank_delta_90:+.0f}%)")
    
    # DISTRESS: Margin compression or velocity decay
    elif margin < margin_distress or velocity_decay > 1.3:
        state = StrategicState.DISTRESS
        confidence = 0.75
        reasoning = f"Product showing stress signals. Margin {margin*100:.1f}%, velocity decay {velocity_decay:.2f}x."
        if strategic_bias == "Profit Maximization":
            action = "Cut all discretionary spend. Raise price immediately."
        elif strategic_bias == "Aggressive Growth":
            action = "Optimize spend efficiency. Maintain market position."
        else:
            action = "Pause non-essential spend. Investigate root cause. Fix pricing if needed."
        if margin < margin_distress:
            signals.append(f"Margin LOW ({margin*100:.1f}%)")
        if velocity_decay > 1.3:
            signals.append(f"Velocity DECAYING ({velocity_decay:.2f}x)")
    
    # TRENCH_WAR: High competition or BB pressure
    elif offer_count > 10 or bb_share < 0.60 or price_gap > 0.10:
        state = StrategicState.TRENCH_WAR
        confidence = 0.70
        reasoning = f"Competitive pressure detected. {int(offer_count)} sellers, BB share {bb_share*100:.0f}%."
        if strategic_bias == "Profit Maximization":
            action = "Avoid price war. Focus on differentiation. Consider raising price."
        elif strategic_bias == "Aggressive Growth":
            action = "Defend aggressively. Match pricing. Scale defensive ads."
        else:
            action = "Defend position. Match competitor pricing. Increase visibility spend."
        if offer_count > 10:
            signals.append(f"Competition HIGH ({int(offer_count)} sellers)")
        if bb_share < 0.60:
            signals.append(f"Buy Box WEAK ({bb_share*100:.0f}%)")
        if price_gap > 0.10:
            signals.append(f"Price gap {price_gap*100:+.0f}%")
    
    # HARVEST: Stable, good margins, can extract value
    elif margin > margin_harvest and velocity_decay < 1.1 and bb_share > 0.75:
        state = StrategicState.HARVEST
        confidence = 0.80
        reasoning = f"Strong fundamentals for value extraction. Margin {margin*100:.1f}%, stable velocity."
        if strategic_bias == "Profit Maximization":
            action = "Maximize extraction. Raise price +10%. Cut ad spend 30%."
        elif strategic_bias == "Aggressive Growth":
            action = "Invest for scale. Test ad expansion to adjacent keywords."
        else:
            action = "Test price increase +5%. Reduce ad spend. Maximize profit."
        signals.append(f"Margin HEALTHY ({margin*100:.1f}%)")
        signals.append(f"Velocity STABLE ({velocity_decay:.2f}x)")
        signals.append(f"Buy Box STRONG ({bb_share*100:.0f}%)")
    
    # FORTRESS: Dominant position
    elif margin > margin_fortress and bb_share > 0.85 and velocity_decay < 0.95 and offer_count < 8:
        state = StrategicState.FORTRESS
        confidence = 0.85
        reasoning = f"Dominant market position. High margins ({margin*100:.1f}%), strong BB ({bb_share*100:.0f}%), accelerating."
        if strategic_bias == "Profit Maximization":
            action = "Maximize margins. Premium pricing strategy. Reduce spend."
        elif strategic_bias == "Aggressive Growth":
            action = "Leverage dominance. Scale into adjacent categories."
        else:
            action = "Defend with premium positioning. Consider price increase."
        signals.append(f"Margin EXCELLENT ({margin*100:.1f}%)")
        signals.append(f"Buy Box DOMINANT ({bb_share*100:.0f}%)")
        signals.append(f"Velocity ACCELERATING ({velocity_decay:.2f}x)")
    
    # Default to HARVEST if nothing else matches
    else:
        state = StrategicState.HARVEST
        confidence = 0.60
        reasoning = "Metrics within normal range. Product appears stable."
        if strategic_bias == "Profit Maximization":
            action = "Focus on efficiency. Look for margin improvement opportunities."
        elif strategic_bias == "Aggressive Growth":
            action = "Test expansion. Look for growth levers (ads, content, A/B tests)."
        else:
            action = "Monitor performance. Maintain current strategy."
        signals.append("Metrics within normal range")
    
    state_def = STATE_DEFINITIONS[state]
    
    # Create base brief
    brief = StrategicBrief(
        strategic_state=state.value,
        confidence=confidence,
        reasoning=f"{reasoning} [Fallback: {reason}]",
        recommended_action=action,
        state_emoji=state_def["emoji"],
        state_color=state_def["color"],
        primary_outcome=state_def["primary_outcome"],
        source="fallback",
        signals_detected=signals,
        asin=row_data.get("asin", ""),
    )
    
    # Calculate growth intelligence for fallback as well
    # This ensures growth is always available even without LLM
    try:
        v90 = _safe_float(row_data.get('velocity_trend_90d', rank_delta_90 / 100 if rank_delta_90 else 0), 0.0)
        competitor_oos = _safe_float(row_data.get('competitor_oos_pct', row_data.get('outOfStockPercentage90', 0)), 0.0)
        if competitor_oos > 1:  # Normalize if percentage
            competitor_oos = competitor_oos / 100
        
        revenue = _safe_float(row_data.get('weekly_sales_filled', row_data.get('revenue_proxy', 1000)), 1000.0)
        
        expansion = calculate_expansion_alpha(
            row_data=row_data,
            revenue=revenue,
            velocity_trend_90d=v90,
            price_gap_vs_competitor=price_gap,
            competitor_oos_pct=competitor_oos,
            strategic_state=state.value,
            strategic_bias=strategic_bias,
        )
        
        # Enrich brief with growth intelligence
        brief.thirty_day_growth = expansion.thirty_day_growth
        brief.price_lift_opportunity = expansion.price_optimization_gain
        brief.conquest_opportunity = expansion.conquest_revenue
        brief.expansion_recommendation = expansion.ai_recommendation
        brief.growth_validated = expansion.velocity_validated
        brief.opportunity_type = expansion.opportunity_type
    except Exception:
        # If growth calculation fails, defaults are already set in dataclass
        pass
    
    return brief


# =============================================================================
# PARALLEL BATCH PROCESSING
# =============================================================================

async def analyze_portfolio_async(
    rows: List[Dict[str, Any]],
    max_concurrent: int = 10,
    timeout_per_item: float = 10.0
) -> List[StrategicBrief]:
    """
    Analyze multiple products in parallel using asyncio.gather.
    
    This allows processing 20-50 products in under 3 seconds by running
    LLM calls concurrently.
    
    Args:
        rows: List of product data dictionaries
        max_concurrent: Maximum concurrent LLM calls (default 10)
        timeout_per_item: Timeout per product in seconds
        
    Returns:
        List of StrategicBrief objects in same order as input
    """
    # Get shared client for all calls
    client = _get_openai_client()
    model = _get_model_name()
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def analyze_with_semaphore(row: Dict[str, Any]) -> StrategicBrief:
        async with semaphore:
            return await analyze_strategy_with_llm(row, client, model, timeout_per_item)
    
    # Run all analyses in parallel
    tasks = [analyze_with_semaphore(row) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions by using fallback
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append(_determine_state_fallback(rows[i], reason=f"Exception: {str(result)[:30]}"))
        else:
            final_results.append(result)
    
    return final_results


def triangulate_portfolio(
    df: pd.DataFrame,
    use_llm: bool = True,
    max_concurrent: int = 10
) -> pd.DataFrame:
    """
    Process an entire portfolio DataFrame through the LLM classifier.
    
    This is the main entry point for synchronous code (like Streamlit).
    Uses asyncio.run() to execute the async LLM calls.
    
    Args:
        df: DataFrame with product rows
        use_llm: Whether to use LLM (True) or fallback only (False)
        max_concurrent: Maximum concurrent LLM calls
        
    Returns:
        DataFrame with new columns for strategic classification
    """
    # Convert DataFrame to list of dicts
    rows = df.to_dict('records')
    
    if use_llm:
        # Run async analysis (same pattern as analyze and generate_portfolio_brief_sync)
        try:
            # Create new event loop explicitly to work in Streamlit's async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(analyze_portfolio_async(rows, max_concurrent))
            loop.close()
        except Exception as e:
            # Fallback on any error
            results = [_determine_state_fallback(row, f"Error: {str(e)[:30]}") for row in rows]
    else:
        # Use fallback only
        results = [_determine_state_fallback(row, "LLM disabled") for row in rows]
    
    # Build result DataFrame
    df_enhanced = df.copy()
    
    df_enhanced["strategic_state"] = [r.strategic_state for r in results]
    df_enhanced["strategic_confidence"] = [r.confidence for r in results]
    df_enhanced["strategic_outcome"] = [r.primary_outcome for r in results]
    df_enhanced["strategic_plan"] = [r.recommended_action for r in results]
    df_enhanced["strategic_reasoning"] = [r.reasoning for r in results]
    df_enhanced["strategic_signals"] = [r.signals_detected for r in results]
    df_enhanced["strategic_emoji"] = [r.state_emoji for r in results]
    df_enhanced["strategic_color"] = [r.state_color for r in results]
    df_enhanced["strategic_source"] = [r.source for r in results]
    
    return df_enhanced


# =============================================================================
# PREDICTIVE ALPHA CALCULATION
# =============================================================================

@dataclass
class PredictiveAlpha:
    """
    Predictive Alpha calculation result.
    
    Transforms static "leakage" metrics into forward-looking 30-day risk forecasts.
    
    Formula: (Current_Daily_Leakage * 30 days) * (1 + 90_day_sales_velocity_trend)
    
    Inventory Logic: If days_until_stockout < supplier_lead_time, calculate as
    total projected profit lost during the predicted stockout window.
    """
    # Core prediction
    thirty_day_risk: float              # Predicted $ at risk over next 30 days
    daily_burn_rate: float              # Current daily loss rate
    velocity_multiplier: float          # Trend-based adjustment (>1 = accelerating, <1 = decelerating)
    
    # Risk components
    price_erosion_risk: float           # Risk from competitor pricing pressure
    share_erosion_risk: float           # Risk from market share loss
    stockout_risk: float                # Risk from inventory stockout
    
    # Predictive state
    predictive_state: str               # DEFEND, EXPLOIT, REPLENISH, HOLD
    state_emoji: str                    # Visual indicator
    state_description: str              # Human-readable explanation
    
    # Model confidence
    model_certainty: float              # R-squared / confidence (0-1)
    data_quality: str                   # "HIGH" (12+ months), "MEDIUM" (3-12), "LOW" (<3)
    
    # Actionable insight
    cost_of_inaction: str               # Human-readable consequence
    
    # Predictive Alerts (for AI Recommendation field)
    ai_recommendation: str = ""         # Full predictive recommendation for UI
    alert_type: str = ""                # INVENTORY, PRICING, RANK, or empty
    alert_urgency: str = ""             # HIGH, MEDIUM, LOW
    predicted_event_date: str = ""      # When the predicted event will occur
    action_deadline: str = ""           # When user must act to prevent loss


@dataclass
class ExpansionAlpha:
    """
    Growth Intelligence calculation result.
    
    Calculates predicted revenue gains from offensive actions:
    - Price Optimization: Raising price without volume loss
    - Market Share Conquest: Capturing competitor vulnerabilities
    - Keyword Expansion: Scaling into adjacent search terms
    
    CRITICAL: Uses 90-day velocity validation to block growth recs on dying ASINs.
    """
    # Core growth prediction
    thirty_day_growth: float            # Predicted $ gain over next 30 days
    price_optimization_gain: float      # Revenue from price increase opportunity
    conquest_revenue: float             # Revenue from competitor vulnerabilities
    keyword_expansion_gain: float       # Revenue from adjacent keyword scaling
    
    # Opportunity classification
    opportunity_type: str               # PRICE_LIFT, CONQUEST, EXPAND, or empty
    opportunity_urgency: str            # HIGH, MEDIUM, LOW
    target_competitor_asin: str = ""    # Vulnerable competitor ASIN if applicable
    
    # Velocity validation gate
    velocity_validated: bool = True     # False if 90d trend is declining
    blocked_reason: str = ""            # Why growth is blocked (if applicable)
    
    # Actionable insight
    ai_recommendation: str = ""         # Growth-specific recommendation for UI


# =============================================================================
# VELOCITY EXTRACTION FROM BACKFILL DATA
# =============================================================================

def extract_velocity_trends(df_weekly: pd.DataFrame, asin: str) -> Dict[str, float]:
    """
    Extract 30-day and 90-day velocity trends from weekly backfill data.
    
    This is the critical link between historical data and predictive intelligence.
    Velocity trends power the 30-day risk forecast.
    
    Args:
        df_weekly: DataFrame with weekly backfill data (must have 'asin', 'week_start', 'sales_rank_filled')
        asin: The ASIN to extract velocity for
        
    Returns:
        Dict with velocity_trend_30d and velocity_trend_90d (negative = improving rank, positive = declining)
    """
    result = {
        'velocity_trend_30d': 0.0,
        'velocity_trend_90d': 0.0,
        'data_weeks': 0,
        'data_quality': 'VERY_LOW'
    }
    
    if df_weekly is None or df_weekly.empty:
        return result
    
    # Filter to this ASIN and sort by date
    asin_data = df_weekly[df_weekly['asin'] == asin].copy()
    
    if asin_data.empty:
        return result
    
    # Ensure we have the right columns
    if 'week_start' not in asin_data.columns and 'date' not in asin_data.columns:
        return result
        
    date_col = 'week_start' if 'week_start' in asin_data.columns else 'date'
    rank_col = 'sales_rank_filled' if 'sales_rank_filled' in asin_data.columns else 'sales_rank'
    
    if rank_col not in asin_data.columns:
        return result
    
    asin_data = asin_data.sort_values(date_col)
    result['data_weeks'] = len(asin_data)
    
    # Calculate 30-day velocity (last 4 weeks vs previous 4 weeks)
    if len(asin_data) >= 4:
        recent_4w = asin_data.tail(4)[rank_col].mean()
        older_4w = asin_data.iloc[-8:-4][rank_col].mean() if len(asin_data) >= 8 else recent_4w
        
        # Velocity = (recent - older) / older
        # Positive = rank getting worse (higher number), Negative = rank improving
        if older_4w > 0:
            result['velocity_trend_30d'] = (recent_4w - older_4w) / older_4w
        
        result['data_quality'] = 'LOW' if len(asin_data) < 8 else 'MEDIUM'
    
    # Calculate 90-day velocity (entire dataset, weighted toward recent)
    if len(asin_data) >= 8:
        # Compare first third vs last third
        third = len(asin_data) // 3
        early = asin_data.head(third)[rank_col].mean()
        late = asin_data.tail(third)[rank_col].mean()
        
        if early > 0:
            result['velocity_trend_90d'] = (late - early) / early
        
        result['data_quality'] = 'MEDIUM' if len(asin_data) < 12 else 'HIGH'
    
    return result


def extract_portfolio_velocity(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Batch extract velocity trends for all ASINs in the backfill data.
    
    Args:
        df_weekly: DataFrame with weekly backfill data
        
    Returns:
        DataFrame with ASIN, velocity_trend_30d, velocity_trend_90d, data_quality
    """
    if df_weekly is None or df_weekly.empty:
        return pd.DataFrame(columns=['asin', 'velocity_trend_30d', 'velocity_trend_90d', 'data_quality'])
    
    asins = df_weekly['asin'].unique()
    results = []
    
    for asin in asins:
        velocity = extract_velocity_trends(df_weekly, asin)
        results.append({
            'asin': asin,
            'velocity_trend_30d': velocity['velocity_trend_30d'],
            'velocity_trend_90d': velocity['velocity_trend_90d'],
            'data_quality': velocity['data_quality'],
            'data_weeks': velocity['data_weeks']
        })
    
    return pd.DataFrame(results)


def calculate_predictive_alpha(
    row_data: Dict[str, Any],
    revenue: float,
    velocity_trend_30d: float = 0.0,
    velocity_trend_90d: float = 0.0,
    competitor_review_velocity: float = 1.0,
    your_review_velocity: float = 1.0,
    days_to_stockout: Optional[float] = None,
    supplier_lead_time: int = 7,  # Default 7-day lead time
    competitor_oos_pct: float = 0.0,
    competitor_price_momentum: float = 0.0,  # -1 to 1, negative = competitor cutting prices
    price_gap_vs_competitor: float = 0.0,
    bsr_trend_30d: float = 0.0,  # BSR change over 30 days (positive = declining rank)
    current_bsr: int = 100000,
    months_of_data: int = 3,
    strategic_state: str = "HARVEST",
    strategic_bias: str = "Balanced Defense"  # User's strategic focus
) -> PredictiveAlpha:
    """
    Calculate Predictive Alpha - forward-looking 30-day risk forecast.
    
    FORMULA: (Current_Daily_Leakage * 30 days) * (1 + 90_day_sales_velocity_trend)
    
    INVENTORY LOGIC: If days_until_stockout < supplier_lead_time, calculate as
    total projected profit lost during the predicted stockout window.
    
    Args:
        row_data: Product data dictionary
        revenue: Current monthly revenue
        velocity_trend_30d: 30-day sales velocity change (-1 to 1, negative = declining)
        velocity_trend_90d: 90-day sales velocity change (-1 to 1)
        competitor_review_velocity: Competitor's review growth rate
        your_review_velocity: Your review growth rate
        days_to_stockout: Days until predicted stockout (None if not applicable)
        supplier_lead_time: Days needed to replenish inventory
        competitor_oos_pct: Competitor out-of-stock percentage (0-1)
        competitor_price_momentum: Competitor pricing trend (-1 to 1, negative = cutting)
        price_gap_vs_competitor: Your price vs competitor median (-1 to 1, positive = you're higher)
        bsr_trend_30d: BSR change over 30 days (positive = rank declining, e.g., 0.2 = 20% worse)
        current_bsr: Current Best Seller Rank
        months_of_data: Months of historical data available
        strategic_state: Current strategic classification
        
    Returns:
        PredictiveAlpha with 30-day risk forecast and AI recommendations
    """
    from datetime import datetime, timedelta
    
    # ========== BASE CALCULATION ==========
    # Formula: (Current_Daily_Leakage * 30 days) * (1 + 90_day_sales_velocity_trend)
    base_opportunity_rate = 0.15
    daily_leakage = (revenue * base_opportunity_rate) / 30.0
    
    # Core formula implementation
    velocity_adjustment = 1.0 + velocity_trend_90d  # 90-day trend as per formula
    base_30day_risk = daily_leakage * 30 * velocity_adjustment
    
    # Velocity multiplier for UI display
    if velocity_trend_30d < -0.2:
        velocity_multiplier = 1.5 + abs(velocity_trend_30d)
    elif velocity_trend_30d < 0:
        velocity_multiplier = 1.0 + abs(velocity_trend_30d) * 0.5
    elif velocity_trend_30d > 0.1:
        velocity_multiplier = max(0.5, 1.0 - velocity_trend_30d * 0.3)
    else:
        velocity_multiplier = 1.0
    
    # ========== STRATEGIC BIAS WEIGHTS ==========
    # Apply user's strategic focus to weight different risk components
    # These weights adjust which risks are prioritized in the final calculation
    strategic_bias = strategic_bias or "Balanced Defense"
    if "Profit" in strategic_bias:
        # PROFIT MODE: Prioritize margin protection over growth
        price_weight = 1.5      # Weight pricing defense higher
        inventory_weight = 0.8  # Inventory is secondary
        rank_weight = 0.7       # Rank is tertiary
    elif "Growth" in strategic_bias:
        # GROWTH MODE: Prioritize market position over margin
        price_weight = 0.7      # Accept margin compression
        inventory_weight = 1.0  # Inventory still important
        rank_weight = 1.5       # Weight rank defense highest
    else:
        # BALANCED MODE: Standard weights
        price_weight = 1.0
        inventory_weight = 1.0
        rank_weight = 1.0
    
    # Initialize alert variables
    alert_type = ""
    alert_urgency = ""
    predicted_event_date = ""
    action_deadline = ""
    ai_recommendation = ""
    
    # ========== INVENTORY STOCKOUT ANALYSIS ==========
    # If days_until_stockout < supplier_lead_time, it's too late to prevent
    stockout_risk = 0.0
    stockout_window = 0
    
    if days_to_stockout is not None:
        if days_to_stockout < supplier_lead_time:
            # CRITICAL: Already too late to prevent stockout with standard shipping
            stockout_window = 30 - days_to_stockout  # Days OOS in next 30 days
            daily_profit = (revenue / 30.0) * 0.25  # 25% margin
            stockout_risk = stockout_window * daily_profit
            
            stockout_date = datetime.now() + timedelta(days=int(days_to_stockout))
            predicted_event_date = stockout_date.strftime("%b %d")
            action_deadline = "IMMEDIATE - expedite shipping"
            alert_type = "INVENTORY"
            alert_urgency = "HIGH"
            ai_recommendation = f"🚨 Inventory Alert: Stockout predicted by {predicted_event_date}. Expedite shipment NOW to save ${stockout_risk:,.0f} in projected loss."
            
        elif days_to_stockout < 14:
            # URGENT: Within 2 weeks, standard lead time may not be enough
            stockout_window = max(0, 30 - days_to_stockout - supplier_lead_time)
            daily_profit = (revenue / 30.0) * 0.25
            stockout_risk = stockout_window * daily_profit
            
            stockout_date = datetime.now() + timedelta(days=int(days_to_stockout))
            ship_by_date = datetime.now() + timedelta(days=int(days_to_stockout - supplier_lead_time))
            predicted_event_date = stockout_date.strftime("%b %d")
            action_deadline = ship_by_date.strftime("%b %d")
            alert_type = "INVENTORY"
            alert_urgency = "HIGH"
            ai_recommendation = f"📦 Inventory Alert: Stockout predicted by {predicted_event_date}. Ship by {action_deadline} to save ${stockout_risk:,.0f} in projected loss."
            
        elif days_to_stockout < 30:
            # Monitor: Plan replenishment
            stockout_date = datetime.now() + timedelta(days=int(days_to_stockout))
            predicted_event_date = stockout_date.strftime("%b %d")
            alert_type = "INVENTORY"
            alert_urgency = "MEDIUM"
    
    # ========== PRICING DEFENSE ANALYSIS ==========
    price_erosion_risk = 0.0
    
    if competitor_price_momentum < -0.1 and price_gap_vs_competitor > 0:
        # Competitor aggressively cutting prices while you're higher
        # Probability of Buy Box loss based on price momentum
        buybox_loss_probability = min(0.95, 0.5 + abs(competitor_price_momentum))
        hours_to_loss = max(12, int(48 * (1 - abs(competitor_price_momentum))))
        
        price_erosion_risk = revenue * 0.15 * buybox_loss_probability
        
        if not alert_type and buybox_loss_probability > 0.7:
            alert_type = "PRICING"
            alert_urgency = "HIGH"
            predicted_event_date = f"{hours_to_loss}h"
            action_deadline = "IMMEDIATE"
            ai_recommendation = f"⚡ Price Defense: {int(buybox_loss_probability*100)}% probability of Buy Box loss within {hours_to_loss}h. Adjust price now to protect ${price_erosion_risk:,.0f} in 30-day revenue."
    
    elif competitor_oos_pct > 0.3:
        # Competitor out of stock - opportunity to raise price
        price_erosion_risk = -revenue * 0.05  # Negative = opportunity
        if not alert_type:
            alert_type = "PRICING"
            alert_urgency = "LOW"
            ai_recommendation = f"🎯 Price Opportunity: Competitor {int(competitor_oos_pct*100)}% OOS. Raise price to capture ${abs(price_erosion_risk):,.0f} additional margin."
    
    # ========== RANK PROTECTION ANALYSIS ==========
    share_erosion_risk = 0.0
    days_to_overtake = None
    review_velocity_ratio = competitor_review_velocity / max(your_review_velocity, 0.01)
    
    if bsr_trend_30d > 0.15 or review_velocity_ratio > 2.0:
        # BSR declining significantly or competitor growing faster
        share_erosion_risk = revenue * 0.20 * max(bsr_trend_30d, 0.1)
        
        if bsr_trend_30d > 0:
            # Predict when rank #1 lost (if applicable)
            days_to_rank_loss = max(7, int(30 / (bsr_trend_30d * 2)))
            rank_loss_date = datetime.now() + timedelta(days=days_to_rank_loss)
            
            if not alert_type and current_bsr < 100:  # Top 100 product
                alert_type = "RANK"
                alert_urgency = "HIGH" if bsr_trend_30d > 0.25 else "MEDIUM"
                predicted_event_date = rank_loss_date.strftime("%b %d")
                action_deadline = "This week"
                ai_recommendation = f"📉 BSR Defense: Predicted loss of Category Rank #{current_bsr} by {predicted_event_date}. Increase ad spend to defend ${share_erosion_risk:,.0f} in visibility value."
    
    if review_velocity_ratio > 2.0:
        days_to_overtake = max(14, int(60 / review_velocity_ratio))
        if not alert_type:
            overtake_date = datetime.now() + timedelta(days=days_to_overtake)
            alert_type = "RANK"
            alert_urgency = "MEDIUM"
            predicted_event_date = overtake_date.strftime("%b %d")
            ai_recommendation = f"📊 Competitive Alert: Competitor review velocity {review_velocity_ratio:.1f}x yours. Market share at risk by {predicted_event_date}."
    
    # ========== TOTAL 30-DAY RISK ==========
    # Apply strategic bias weights to risk components
    weighted_stockout_risk = stockout_risk * inventory_weight
    weighted_price_risk = max(0, price_erosion_risk) * price_weight
    weighted_share_risk = share_erosion_risk * rank_weight
    
    thirty_day_risk = max(0, (
        base_30day_risk +
        weighted_stockout_risk +
        weighted_price_risk +
        weighted_share_risk
    ))
    
    # Daily burn rate for trending
    daily_burn_rate = thirty_day_risk / 30.0
    
    # ========== PREDICTIVE STATE ==========
    if stockout_risk > 0 and alert_urgency == "HIGH":
        predictive_state = "REPLENISH"
        state_emoji = "📦"
        state_description = f"Stockout predicted in {int(days_to_stockout)} days"
        cost_of_inaction = f"${stockout_risk:,.0f} profit loss during stockout window"
    elif competitor_oos_pct > 0.3:
        predictive_state = "EXPLOIT"
        state_emoji = "🎯"
        state_description = f"Competitor {int(competitor_oos_pct*100)}% OOS - pricing opportunity"
        cost_of_inaction = f"${abs(price_erosion_risk):,.0f} margin opportunity if price raised"
    elif alert_type == "PRICING" and alert_urgency == "HIGH":
        predictive_state = "DEFEND"
        state_emoji = "🛡️"
        state_description = "Buy Box at risk - immediate action required"
        cost_of_inaction = f"${price_erosion_risk:,.0f} revenue at risk from Buy Box loss"
    elif alert_type == "RANK" and share_erosion_risk > revenue * 0.1:
        predictive_state = "DEFEND"
        state_emoji = "🛡️"
        state_description = "Organic visibility declining"
        cost_of_inaction = f"${share_erosion_risk:,.0f} market share erosion by end of month"
    else:
        predictive_state = "HOLD"
        state_emoji = "✅"
        state_description = "Position stable - monitor"
        cost_of_inaction = f"${thirty_day_risk:,.0f} projected optimization opportunity"
    
    # Default AI recommendation if no alert triggered
    if not ai_recommendation:
        if thirty_day_risk > revenue * 0.2:
            ai_recommendation = f"Monitor closely: ${thirty_day_risk:,.0f} at risk over next 30 days based on current trajectory."
        else:
            ai_recommendation = f"Position stable. ${thirty_day_risk:,.0f} optimization opportunity available."
    
    # ========== MODEL CERTAINTY ==========
    if months_of_data >= 12:
        model_certainty = 0.90
        data_quality = "HIGH"
    elif months_of_data >= 6:
        model_certainty = 0.75
        data_quality = "MEDIUM"
    elif months_of_data >= 3:
        model_certainty = 0.60
        data_quality = "LOW"
    else:
        model_certainty = 0.40
        data_quality = "VERY_LOW"
    
    # Adjust certainty based on velocity consistency
    if abs(velocity_trend_30d - velocity_trend_90d) < 0.1:
        model_certainty = min(0.95, model_certainty + 0.05)
    else:
        model_certainty = max(0.40, model_certainty - 0.10)
    
    return PredictiveAlpha(
        thirty_day_risk=thirty_day_risk,
        daily_burn_rate=daily_burn_rate,
        velocity_multiplier=velocity_multiplier,
        price_erosion_risk=max(0, price_erosion_risk),
        share_erosion_risk=share_erosion_risk,
        stockout_risk=stockout_risk,
        predictive_state=predictive_state,
        state_emoji=state_emoji,
        state_description=state_description,
        model_certainty=model_certainty,
        data_quality=data_quality,
        cost_of_inaction=cost_of_inaction,
        ai_recommendation=ai_recommendation,
        alert_type=alert_type,
        alert_urgency=alert_urgency,
        predicted_event_date=predicted_event_date,
        action_deadline=action_deadline,
    )


# =============================================================================
# EXPANSION ALPHA CALCULATION (GROWTH INTELLIGENCE)
# =============================================================================

def is_growth_eligible(velocity_trend_90d: float, strategic_state: str) -> bool:
    """
    Velocity validation gate for growth recommendations.
    
    Block growth recommendations for dying ASINs to ensure the model
    does not recommend expansion on products that should be in defense/exit mode.
    
    Returns False if:
    - 90-day velocity is declining more than 15%
    - Strategic state is TERMINAL or DISTRESS
    """
    if velocity_trend_90d > 0.15:  # Rank getting worse (positive = declining)
        return False
    if strategic_state in ["TERMINAL", "DISTRESS"]:
        return False
    return True


def calculate_expansion_alpha(
    row_data: Dict[str, Any],
    revenue: float,
    velocity_trend_90d: float = 0.0,
    price_gap_vs_competitor: float = 0.0,
    competitor_oos_pct: float = 0.0,
    competitor_monthly_rev: float = 0.0,
    your_market_share: float = 0.0,
    strategic_state: str = "HARVEST",
    strategic_bias: str = "Balanced Defense"
) -> ExpansionAlpha:
    """
    Calculate Expansion Alpha - forward-looking 30-day growth forecast.
    
    FORMULA:
    - Price Optimization: revenue * price_headroom * (1 - volume_sensitivity)
    - Conquest: competitor_vulnerable_revenue * capture_rate
    - Expansion: revenue * 0.10 * velocity_multiplier (if velocity improving)
    
    CRITICAL: Uses 90-day velocity validation gate to block growth recs on dying ASINs.
    
    Args:
        row_data: Product data dictionary
        revenue: Current monthly revenue
        velocity_trend_90d: 90-day velocity change (positive = declining, negative = improving)
        price_gap_vs_competitor: Your price vs median (positive = you're higher)
        competitor_oos_pct: Competitor out-of-stock percentage (0-1)
        competitor_monthly_rev: Competitor's monthly revenue (for conquest calculation)
        your_market_share: Your share of the market (0-1)
        strategic_state: Current strategic classification
        strategic_bias: User's strategic focus
        
    Returns:
        ExpansionAlpha with 30-day growth forecast and recommendations
    """
    strategic_bias = strategic_bias or "Balanced Defense"
    
    # ========== VELOCITY VALIDATION GATE ==========
    # Block growth recommendations for declining ASINs
    if not is_growth_eligible(velocity_trend_90d, strategic_state):
        blocked_reason = ""
        if velocity_trend_90d > 0.15:
            blocked_reason = f"90-day velocity declining {velocity_trend_90d*100:.0f}%. Focus on stabilization first."
        elif strategic_state in ["TERMINAL", "DISTRESS"]:
            blocked_reason = f"Product in {strategic_state} state. Address fundamentals before growth."
        
        return ExpansionAlpha(
            thirty_day_growth=0.0,
            price_optimization_gain=0.0,
            conquest_revenue=0.0,
            keyword_expansion_gain=0.0,
            opportunity_type="",
            opportunity_urgency="",
            velocity_validated=False,
            blocked_reason=blocked_reason,
            ai_recommendation=f"Growth blocked: {blocked_reason}"
        )
    
    # ========== STRATEGIC BIAS THRESHOLDS ==========
    if "Growth" in strategic_bias:
        # More aggressive growth recommendations
        min_price_lift_threshold = 0.02   # Recommend at 2% headroom
        conquest_capture_rate = 0.20      # Assume 20% capture
        expansion_rate = 0.12             # 12% expansion opportunity
    elif "Profit" in strategic_bias:
        # Focus on price optimization
        min_price_lift_threshold = 0.03   # Recommend at 3% headroom
        conquest_capture_rate = 0.10      # Conservative 10%
        expansion_rate = 0.05             # 5% expansion
    else:
        # Balanced mode
        min_price_lift_threshold = 0.05   # Only recommend at 5%+ headroom
        conquest_capture_rate = 0.15      # Moderate 15%
        expansion_rate = 0.08             # 8% expansion
    
    # Initialize outputs
    price_optimization_gain = 0.0
    conquest_revenue = 0.0
    keyword_expansion_gain = 0.0
    opportunity_type = ""
    opportunity_urgency = ""
    target_competitor_asin = ""
    ai_recommendation = ""
    
    # ========== PRICE OPTIMIZATION ANALYSIS ==========
    # If you're priced below competitors, there's headroom to raise price
    # price_gap_vs_competitor: negative = you're cheaper, positive = you're higher
    price_headroom = -price_gap_vs_competitor  # Invert: negative gap = positive headroom
    
    if price_headroom > min_price_lift_threshold:
        # Volume sensitivity based on velocity (declining velocity = more price sensitive)
        volume_sensitivity = max(0.2, min(0.6, 0.3 + velocity_trend_90d))
        price_optimization_gain = revenue * price_headroom * (1 - volume_sensitivity)
        
        if not opportunity_type:
            opportunity_type = "PRICE_LIFT"
            if price_headroom > 0.10:
                opportunity_urgency = "HIGH"
            elif price_headroom > 0.05:
                opportunity_urgency = "MEDIUM"
            else:
                opportunity_urgency = "LOW"
            
            ai_recommendation = f"💰 Price Opportunity: You're priced {price_headroom*100:.0f}% below market. Raise price to capture ${price_optimization_gain:,.0f} additional margin."
    
    # ========== CONQUEST ANALYSIS ==========
    # If competitor is vulnerable (OOS or price cutting aggressively)
    if competitor_oos_pct > 0.30:
        target_competitor_asin = row_data.get('top_competitor_asin', '')
        effective_competitor_rev = competitor_monthly_rev if competitor_monthly_rev > 0 else revenue * 2
        conquest_revenue = effective_competitor_rev * competitor_oos_pct * conquest_capture_rate
        
        if not opportunity_type or conquest_revenue > price_optimization_gain:
            opportunity_type = "CONQUEST"
            opportunity_urgency = "HIGH" if competitor_oos_pct > 0.50 else "MEDIUM"
            
            if target_competitor_asin:
                ai_recommendation = f"🎯 Conquest Opportunity: Competitor {target_competitor_asin} is {int(competitor_oos_pct*100)}% OOS. Redirect ad budget to their keywords to capture ${conquest_revenue:,.0f} in 30-day revenue."
            else:
                ai_recommendation = f"🎯 Conquest Opportunity: Competitor {int(competitor_oos_pct*100)}% OOS. Increase ad spend on category keywords to capture ${conquest_revenue:,.0f}."
    
    # ========== KEYWORD EXPANSION ANALYSIS ==========
    # If velocity is improving (negative = rank improving), recommend expansion
    if velocity_trend_90d < -0.05:  # Improving by more than 5%
        velocity_multiplier = 1.0 + abs(velocity_trend_90d)
        keyword_expansion_gain = revenue * expansion_rate * velocity_multiplier
        
        if not opportunity_type:
            opportunity_type = "EXPAND"
            opportunity_urgency = "MEDIUM" if velocity_trend_90d < -0.10 else "LOW"
            ai_recommendation = f"🚀 Expansion Opportunity: Momentum detected (velocity +{abs(velocity_trend_90d)*100:.0f}%). Scale keyword coverage to capture ${keyword_expansion_gain:,.0f} in new revenue."
    
    # ========== TOTAL 30-DAY GROWTH ==========
    thirty_day_growth = price_optimization_gain + conquest_revenue + keyword_expansion_gain
    
    # Default recommendation if no specific opportunity
    if not ai_recommendation and thirty_day_growth > 0:
        ai_recommendation = f"📈 Growth potential: ${thirty_day_growth:,.0f} available through optimization."
    
    return ExpansionAlpha(
        thirty_day_growth=thirty_day_growth,
        price_optimization_gain=price_optimization_gain,
        conquest_revenue=conquest_revenue,
        keyword_expansion_gain=keyword_expansion_gain,
        opportunity_type=opportunity_type,
        opportunity_urgency=opportunity_urgency,
        target_competitor_asin=target_competitor_asin,
        velocity_validated=True,
        blocked_reason="",
        ai_recommendation=ai_recommendation
    )


# =============================================================================
# VECTORIZED INTELLIGENCE LAYER (HIGH PERFORMANCE)
# =============================================================================

def calculate_portfolio_intelligence_vectorized(
    df: pd.DataFrame,
    strategic_bias: str = "Balanced Defense"
) -> pd.DataFrame:
    """
    VECTORIZED calculation of all intelligence metrics for the portfolio.
    
    This is the HIGH-PERFORMANCE alternative to row-wise iteration.
    Calculates Risk, Growth, and Opportunity Alpha for ALL ASINs in one pass.
    
    Performance: ~100x faster than iterrows() for 100+ ASINs.
    
    Args:
        df: Portfolio DataFrame with required columns
        strategic_bias: User's strategic focus
        
    Returns:
        DataFrame with intelligence columns added:
        - thirty_day_risk, predictive_state, price_erosion_risk
        - thirty_day_growth, opportunity_type, growth_validated
        - opportunity_alpha (risk + growth combined)
    """
    if df.empty:
        return df
    
    # Work with a view, not a copy (memory efficient)
    result = df
    
    # === EXTRACT BASE METRICS (vectorized) ===
    revenue = result.get('weekly_sales_filled', result.get('revenue_proxy', pd.Series([1000] * len(result)))).fillna(1000)
    v30 = result.get('velocity_trend_30d', pd.Series([0.0] * len(result))).fillna(0.0)
    v90 = result.get('velocity_trend_90d', pd.Series([0.0] * len(result))).fillna(0.0)
    competitor_oos = result.get('competitor_oos_pct', result.get('outOfStockPercentage90', pd.Series([0.0] * len(result)))).fillna(0.0)
    price_gap = result.get('price_gap_vs_competitor', result.get('price_delta', pd.Series([0.0] * len(result)))).fillna(0.0)
    
    # Normalize competitor OOS to 0-1 range
    competitor_oos = np.where(competitor_oos > 1, competitor_oos / 100, competitor_oos)
    
    # === STRATEGIC BIAS WEIGHTS ===
    if "Profit" in strategic_bias:
        price_weight, inventory_weight, rank_weight = 1.5, 0.8, 0.7
        min_price_lift = 0.03
        conquest_rate = 0.10
    elif "Growth" in strategic_bias:
        price_weight, inventory_weight, rank_weight = 0.7, 1.0, 1.5
        min_price_lift = 0.02
        conquest_rate = 0.20
    else:  # Balanced
        price_weight, inventory_weight, rank_weight = 1.0, 1.0, 1.0
        min_price_lift = 0.05
        conquest_rate = 0.15
    
    # === RISK CALCULATION (vectorized) ===
    base_opportunity_rate = 0.15
    daily_leakage = (revenue * base_opportunity_rate) / 30.0
    velocity_adjustment = 1.0 + v90
    base_30day_risk = daily_leakage * 30 * velocity_adjustment
    
    # Price erosion (vectorized)
    price_erosion = np.where(price_gap > 0.05, revenue * price_gap * 0.3 * price_weight, 0)
    
    # Share erosion (vectorized)  
    share_erosion = np.where(v90 > 0.15, revenue * 0.20 * v90 * rank_weight, 0)
    
    # Total risk
    thirty_day_risk = np.maximum(0, base_30day_risk + price_erosion + share_erosion)
    
    # Predictive state (vectorized)
    predictive_state = np.where(
        v90 > 0.20, "DEFEND",
        np.where(
            competitor_oos > 0.30, "EXPLOIT",
            np.where(
                price_erosion > revenue * 0.10, "DEFEND",
                "HOLD"
            )
        )
    )
    
    # === GROWTH CALCULATION (vectorized) ===
    # Velocity gate: block growth for declining ASINs
    growth_validated = (v90 <= 0.15)
    
    # Price headroom (negative gap = you're cheaper = headroom)
    price_headroom = np.maximum(0, -price_gap)
    volume_sensitivity = np.clip(0.3 + v90, 0.2, 0.6)
    price_lift_gain = np.where(
        (price_headroom > min_price_lift) & growth_validated,
        revenue * price_headroom * (1 - volume_sensitivity),
        0
    )
    
    # Conquest opportunity
    conquest_gain = np.where(
        (competitor_oos > 0.30) & growth_validated,
        revenue * 2 * competitor_oos * conquest_rate,  # Assume competitor rev = 2x yours
        0
    )
    
    # Expansion opportunity (velocity improving)
    expansion_gain = np.where(
        (v90 < -0.05) & growth_validated,
        revenue * 0.08 * (1.0 + np.abs(v90)),
        0
    )
    
    # Total growth
    thirty_day_growth = np.where(
        growth_validated,
        price_lift_gain + conquest_gain + expansion_gain,
        0
    )
    
    # Opportunity type (priority: CONQUEST > PRICE_LIFT > EXPAND)
    opportunity_type = np.where(
        conquest_gain > 0, "CONQUEST",
        np.where(
            price_lift_gain > 0, "PRICE_LIFT",
            np.where(
                expansion_gain > 0, "EXPAND",
                ""
            )
        )
    )
    
    # === COMBINED OPPORTUNITY ALPHA ===
    opportunity_alpha = thirty_day_risk + thirty_day_growth
    
    # === ASSIGN RESULTS TO DATAFRAME (single copy, memory efficient) ===
    result = result.copy()
    
    # === MEMORY OPTIMIZATION ===
    # Convert string columns to category dtype (reduces memory 80%+ for repeated values)
    for col in ['asin', 'brand', 'title']:
        if col in result.columns and result[col].dtype == 'object':
            result[col] = result[col].astype('category')
    
    # Assign computed values with efficient dtypes
    result['thirty_day_risk'] = np.asarray(thirty_day_risk, dtype=np.float32)
    result['thirty_day_growth'] = np.asarray(thirty_day_growth, dtype=np.float32)
    result['opportunity_alpha'] = np.asarray(opportunity_alpha, dtype=np.float32)
    result['predictive_state'] = pd.Categorical(predictive_state, categories=["HOLD", "DEFEND", "EXPLOIT", "REPLENISH"])
    result['opportunity_type'] = pd.Categorical(opportunity_type, categories=["", "PRICE_LIFT", "CONQUEST", "EXPAND"])
    result['growth_validated'] = np.asarray(growth_validated, dtype=bool)
    result['price_erosion_risk'] = np.asarray(price_erosion, dtype=np.float32)
    result['share_erosion_risk'] = np.asarray(share_erosion, dtype=np.float32)
    
    # Model certainty based on data quality
    data_weeks = result.get('data_weeks', pd.Series([4] * len(result))).fillna(4)
    result['model_certainty'] = np.clip(0.40 + (data_weeks / 48) * 0.55, 0.40, 0.95).astype(np.float32)
    
    return result


def calculate_portfolio_predictive_risk(
    portfolio_df: pd.DataFrame,
    total_monthly_revenue: float,
    strategic_bias: str = "Balanced Defense"
) -> Dict[str, Any]:
    """
    Calculate aggregate predictive risk AND growth opportunity for the entire portfolio.
    
    OPTIMIZED: Uses vectorized calculation, then aggregates.
    
    Returns both defensive (risk) and offensive (growth) metrics for unified "Opportunity Alpha".
    
    Args:
        portfolio_df: DataFrame with product data and velocity metrics
        total_monthly_revenue: Total portfolio monthly revenue
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        
    Returns:
        Dict with portfolio-level predictive metrics (risk + growth)
    """
    # === VECTORIZED CALCULATION (100x faster than iterrows) ===
    df_intel = calculate_portfolio_intelligence_vectorized(portfolio_df, strategic_bias)
    
    # === AGGREGATE RESULTS ===
    total_30day_risk = df_intel['thirty_day_risk'].sum()
    total_30day_growth = df_intel['thirty_day_growth'].sum()
    
    # Count by predictive state
    defend_count = (df_intel['predictive_state'] == "DEFEND").sum()
    exploit_count = (df_intel['predictive_state'] == "EXPLOIT").sum()
    replenish_count = (df_intel['predictive_state'] == "REPLENISH").sum()
    
    # Count by opportunity type
    price_lift_count = (df_intel['opportunity_type'] == "PRICE_LIFT").sum()
    conquest_count = (df_intel['opportunity_type'] == "CONQUEST").sum()
    expand_count = (df_intel['opportunity_type'] == "EXPAND").sum()
    
    # Calculate percentages
    risk_pct = (total_30day_risk / total_monthly_revenue * 100) if total_monthly_revenue > 0 else 0
    growth_pct = (total_30day_growth / total_monthly_revenue * 100) if total_monthly_revenue > 0 else 0
    
    # Combined Opportunity Alpha
    opportunity_alpha = total_30day_risk + total_30day_growth
    
    # Determine portfolio health
    if risk_pct > 25:
        portfolio_status = "CRITICAL"
        status_emoji = "🚨"
    elif risk_pct > 15:
        portfolio_status = "ELEVATED"
        status_emoji = "⚠️"
    elif risk_pct > 10:
        portfolio_status = "MODERATE"
        status_emoji = "📊"
    else:
        portfolio_status = "HEALTHY"
        status_emoji = "✅"
    
    return {
        # Defensive metrics (risk)
        "thirty_day_risk": total_30day_risk,
        "risk_pct": risk_pct,
        "portfolio_status": portfolio_status,
        "status_emoji": status_emoji,
        "defend_count": int(defend_count),
        "exploit_count": int(exploit_count),
        "replenish_count": int(replenish_count),
        "action_required_count": int(defend_count + replenish_count),
        
        # Offensive metrics (growth)
        "thirty_day_growth": total_30day_growth,
        "growth_pct": growth_pct,
        "price_lift_count": int(price_lift_count),
        "conquest_count": int(conquest_count),
        "expand_count": int(expand_count),
        "growth_opportunity_count": int(price_lift_count + conquest_count + expand_count),
        
        # Combined (Opportunity Alpha)
        "opportunity_alpha": opportunity_alpha,
        "opportunity_count": int(exploit_count + price_lift_count + conquest_count + expand_count),
        
        # Return enriched DataFrame for downstream use (eliminates redundant recalculation)
        "_enriched_df": df_intel,
    }


# =============================================================================
# SYNCHRONOUS WRAPPER FOR SINGLE PRODUCT
# =============================================================================

class StrategicTriangulator:
    """
    UNIFIED AI ENGINE
    
    Combines Strategic Classification + Predictive Intelligence into a single analysis.
    
    Features:
    1. LLM-powered strategic state classification (FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL)
    2. Predictive 30-day risk forecast with velocity trends
    3. Actionable alerts (Inventory, Pricing, Rank protection)
    4. Model certainty based on data quality
    
    Usage:
        triangulator = StrategicTriangulator()
        brief = triangulator.analyze(product_row, revenue=1000)
        print(brief.thirty_day_risk)     # Predictive: $1,500 at risk
        print(brief.strategic_state)      # Strategic: DISTRESS
        print(brief.ai_recommendation)    # Alert: "📦 Inventory Alert: Stockout by Jan 25..."
    """
    
    def __init__(self, use_llm: bool = True, timeout: float = 10.0, strategic_bias: str = "Balanced Defense"):
        """
        Initialize the unified AI engine.
        
        Args:
            use_llm: Whether to use LLM classification (default True)
            timeout: Timeout for LLM calls in seconds
            strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        """
        self.use_llm = use_llm
        self.timeout = timeout
        self.strategic_bias = strategic_bias
        self._client = None
        self._model = None
    
    def analyze(self, row: Union[pd.Series, Dict], strategic_bias: Optional[str] = None, revenue: Optional[float] = None) -> StrategicBrief:
        """
        UNIFIED ANALYSIS: Strategic Classification + Predictive Intelligence
        
        Performs both strategic state classification AND predictive risk analysis
        in a single call, returning a unified StrategicBrief.
        
        Args:
            row: Product data (Series or dict)
            strategic_bias: Override strategic bias for this analysis (optional)
            revenue: Monthly revenue for predictive calculations (optional, extracted from row if not provided)
            
        Returns:
            StrategicBrief with unified strategic + predictive outputs
        """
        # Use provided bias or fall back to instance bias
        bias = strategic_bias or self.strategic_bias
        
        # Normalize row to dict
        if isinstance(row, pd.Series):
            row_data = row.to_dict()
        else:
            row_data = dict(row)
        
        # Extract revenue if not provided
        if revenue is None:
            revenue = row_data.get('weekly_sales_filled', row_data.get('revenue_proxy', row_data.get('monthly_revenue', 0)))
        
        # === STEP 1: STRATEGIC CLASSIFICATION ===
        if not self.use_llm:
            strategic_brief = _determine_state_fallback(row_data, reason="LLM disabled", strategic_bias=bias)
        else:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                strategic_brief = loop.run_until_complete(
                    analyze_strategy_with_llm(row_data, timeout=self.timeout, strategic_bias=bias)
                )
                loop.close()
            except Exception as e:
                strategic_brief = _determine_state_fallback(row_data, reason=f"Error: {str(e)[:30]}", strategic_bias=bias)
        
        # === STEP 2: PREDICTIVE INTELLIGENCE ===
        # Extract velocity and competitive signals from row data
        v30 = float(row_data.get('velocity_trend_30d', 0.0))
        v90 = float(row_data.get('velocity_trend_90d', 0.0))
        days_to_stockout = row_data.get('days_to_stockout', None)
        if days_to_stockout is not None:
            days_to_stockout = float(days_to_stockout)
        competitor_price_momentum = float(row_data.get('competitor_price_momentum', 0.0))
        price_gap = float(row_data.get('price_gap_vs_competitor', row_data.get('price_delta', 0.0)))
        bsr_trend = float(row_data.get('bsr_trend_30d', row_data.get('rank_delta_30d', 0.0)))
        current_bsr = int(row_data.get('sales_rank_filled', row_data.get('bsr', 100000)))
        
        # Extract data quality from velocity extraction (weeks → months)
        data_weeks = int(row_data.get('data_weeks', 0))
        data_quality_str = str(row_data.get('data_quality', 'LOW'))
        
        # Convert data_quality string or weeks to months_of_data
        if data_quality_str == 'HIGH':
            months_of_data = 12
        elif data_quality_str == 'MEDIUM':
            months_of_data = 6
        elif data_weeks >= 12:
            months_of_data = 12  # 12+ weeks = HIGH
        elif data_weeks >= 8:
            months_of_data = 6   # 8-12 weeks = MEDIUM
        elif data_weeks >= 4:
            months_of_data = 3   # 4-8 weeks = LOW
        else:
            months_of_data = 1   # < 4 weeks = VERY_LOW
        
        # Calculate predictive alpha with actual data quality
        predictive = calculate_predictive_alpha(
            row_data=row_data,
            revenue=revenue,
            velocity_trend_30d=v30,
            velocity_trend_90d=v90,
            days_to_stockout=days_to_stockout,
            competitor_price_momentum=competitor_price_momentum,
            price_gap_vs_competitor=price_gap,
            bsr_trend_30d=bsr_trend,
            current_bsr=current_bsr,
            months_of_data=months_of_data,
            strategic_state=strategic_brief.strategic_state,
            strategic_bias=strategic_bias,
        )
        
        # === STEP 3: GROWTH INTELLIGENCE (Offensive Layer) ===
        competitor_oos = float(row_data.get('competitor_oos_pct', row_data.get('outOfStockPercentage90', 0)) or 0)
        if competitor_oos > 1:  # Normalize if percentage
            competitor_oos = competitor_oos / 100
        
        expansion = calculate_expansion_alpha(
            row_data=row_data,
            revenue=revenue,
            velocity_trend_90d=v90,
            price_gap_vs_competitor=price_gap,
            competitor_oos_pct=competitor_oos,
            strategic_state=strategic_brief.strategic_state,
            strategic_bias=strategic_bias,
        )
        
        # === STEP 4: MERGE INTO UNIFIED OUTPUT ===
        # Enrich strategic brief with predictive intelligence
        strategic_brief.thirty_day_risk = predictive.thirty_day_risk
        strategic_brief.daily_burn_rate = predictive.daily_burn_rate
        strategic_brief.velocity_multiplier = predictive.velocity_multiplier
        strategic_brief.price_erosion_risk = predictive.price_erosion_risk
        strategic_brief.share_erosion_risk = predictive.share_erosion_risk
        strategic_brief.stockout_risk = predictive.stockout_risk
        strategic_brief.predictive_state = predictive.predictive_state
        strategic_brief.predictive_emoji = predictive.state_emoji
        strategic_brief.predictive_description = predictive.state_description
        strategic_brief.cost_of_inaction = predictive.cost_of_inaction
        strategic_brief.ai_recommendation = predictive.ai_recommendation
        strategic_brief.alert_type = predictive.alert_type
        strategic_brief.alert_urgency = predictive.alert_urgency
        strategic_brief.predicted_event_date = predictive.predicted_event_date
        strategic_brief.action_deadline = predictive.action_deadline
        strategic_brief.data_quality = predictive.data_quality
        
        # Enrich with GROWTH INTELLIGENCE
        strategic_brief.thirty_day_growth = expansion.thirty_day_growth
        strategic_brief.price_lift_opportunity = expansion.price_optimization_gain
        strategic_brief.conquest_opportunity = expansion.conquest_revenue
        strategic_brief.expansion_recommendation = expansion.ai_recommendation
        strategic_brief.growth_validated = expansion.velocity_validated
        strategic_brief.opportunity_type = expansion.opportunity_type
        
        # Update confidence to use model certainty from predictive engine
        # (based on data quality and trend consistency)
        strategic_brief.confidence = predictive.model_certainty
        
        return strategic_brief
    
    def analyze_batch(self, rows: List[Union[pd.Series, Dict]]) -> List[StrategicBrief]:
        """
        Analyze multiple products in parallel.
        
        Args:
            rows: List of product data (Series or dicts)
            
        Returns:
            List of StrategicBrief objects
        """
        # Normalize all rows to dicts
        row_dicts = []
        for row in rows:
            if isinstance(row, pd.Series):
                row_dicts.append(row.to_dict())
            else:
                row_dicts.append(dict(row))
        
        if not self.use_llm:
            return [_determine_state_fallback(r, "LLM disabled") for r in row_dicts]
        
        try:
            try:
                asyncio.get_running_loop()
                return [_determine_state_fallback(r, "Sync call in async context") for r in row_dicts]
            except RuntimeError:
                return asyncio.run(analyze_portfolio_async(row_dicts))
        except Exception as e:
            return [_determine_state_fallback(r, f"Error: {str(e)[:30]}") for r in row_dicts]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_float(val, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_portfolio_state_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of strategic states across the portfolio.
    
    Args:
        df: DataFrame with strategic_state column
        
    Returns:
        Dictionary with state distribution and key metrics
    """
    if "strategic_state" not in df.columns:
        return {"error": "DataFrame missing strategic_state column"}
    
    # State distribution by count
    state_counts = df["strategic_state"].value_counts().to_dict()
    
    # State distribution by revenue
    if "weekly_sales_filled" in df.columns:
        state_revenue = df.groupby("strategic_state")["weekly_sales_filled"].sum().to_dict()
        total_revenue = df["weekly_sales_filled"].sum()
        state_revenue_pct = {k: v / total_revenue for k, v in state_revenue.items()} if total_revenue > 0 else {}
    else:
        state_revenue = {}
        state_revenue_pct = {}
    
    # Identify critical issues
    critical_count = state_counts.get("TERMINAL", 0) + state_counts.get("DISTRESS", 0)
    healthy_count = state_counts.get("FORTRESS", 0) + state_counts.get("HARVEST", 0)
    
    # LLM vs fallback breakdown
    if "strategic_source" in df.columns:
        source_counts = df["strategic_source"].value_counts().to_dict()
        llm_pct = source_counts.get("llm", 0) / len(df) if len(df) > 0 else 0
    else:
        source_counts = {}
        llm_pct = 0
    
    return {
        "state_counts": state_counts,
        "state_revenue": state_revenue,
        "state_revenue_pct": state_revenue_pct,
        "total_products": len(df),
        "critical_products": critical_count,
        "healthy_products": healthy_count,
        "fortress_pct": state_counts.get("FORTRESS", 0) / len(df) if len(df) > 0 else 0,
        "source_breakdown": source_counts,
        "llm_classification_pct": llm_pct,
    }


# =============================================================================
# PORTFOLIO-LEVEL STRATEGIC BRIEF
# =============================================================================

async def generate_portfolio_brief(
    portfolio_summary: str,
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None,
    timeout: float = 10.0
) -> Optional[str]:
    """
    Generate an LLM-powered strategic brief for the entire portfolio.
    
    Uses the same AI engine (client/model) as product-level classification
    to ensure consistency across all AI outputs.
    
    Args:
        portfolio_summary: Portfolio metrics summary string
        client: Optional AsyncOpenAI client (will create if not provided)
        model: Optional model name (defaults to gpt-4o-mini)
        timeout: Request timeout in seconds
        
    Returns:
        Strategic brief string, or None if LLM unavailable
    """
    # Get client and model (same as product-level analysis)
    if client is None:
        client = _get_openai_client()
    
    if client is None:
        return None
    
    if model is None:
        model = _get_model_name()
    
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are ShelfGuard's AI strategist, analyzing Amazon portfolio performance. Generate a clear, actionable strategic brief that balances DEFENSIVE (risk avoidance) and OFFENSIVE (growth capture) priorities.

Guidelines:
- Be direct and prescriptive. Focus on what to do, not what happened.
- Start with the most urgent issue — could be Risk OR Growth opportunity.
- If 30-Day Risk is high (>15% of revenue), lead with defensive priorities.
- If Growth Opportunity is significant, call out conquest/price lift opportunities.
- Quantify everything ($ amounts, counts, percentages).
- Reference the Opportunity Alpha (Risk + Growth combined) as the total addressable value.
- End with one clear action for this session.
- Keep it under 100 words.
- Use clear business language - avoid military/tactical jargon.

Example outputs:
- "Risk: $2,400 at risk this month across 5 products. Growth: $1,800 upside via 2 price lifts. Priority: Address ASIN B001 stockout (50% of risk). Total opportunity: $4,200."
- "Growth mode active: $3,100 conquest opportunity — 3 competitors OOS. Risk stable at 8%. Focus: Scale ad spend on vulnerable competitor keywords."""
                    },
                    {
                        "role": "user",
                        "content": f"""Here's the current portfolio status:

{portfolio_summary}

Generate a strategic brief. What should be the focus this session? Consider both Risk (defensive) and Growth (offensive) opportunities."""
                    }
                ],
                max_tokens=200,
                temperature=0.7
            ),
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def generate_portfolio_brief_sync(
    portfolio_summary: str,
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Synchronous wrapper for generate_portfolio_brief.
    
    For use in Streamlit where async/await can be tricky.
    Uses the same AI engine as product-level classification.
    
    Args:
        portfolio_summary: Portfolio metrics summary string
        client: Optional AsyncOpenAI client (will create if not provided)
        model: Optional model name (defaults to gpt-4o-mini)
        
    Returns:
        Strategic brief string, or None if LLM unavailable
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_portfolio_brief(portfolio_summary, client, model)
        )
        loop.close()
        return result
    except Exception:
        return None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import io
    
    # Handle Windows console encoding
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Test with fallback (no API key needed)
    sample_row = {
        "asin": "B01234567X",
        "title": "Sample Coffee K-Cup Pods, 72 Count",
        "filled_price": 45.99,
        "sales_rank_filled": 1500,
        "new_offer_count": 8,
        "amazon_bb_share": 0.75,
        "net_margin": 0.12,
        "velocity_decay": 1.15,
        "review_count": 245,
        "rating": 4.3,
        "weekly_sales_filled": 15000,
    }
    
    print("\n" + "="*60)
    print("STRATEGIC LLM CLASSIFIER TEST")
    print("="*60)
    
    # Test fallback mode
    print("\n[Testing Fallback Mode]")
    triangulator = StrategicTriangulator(use_llm=False)
    brief = triangulator.analyze(sample_row)
    
    print(f"\nState: [{brief.strategic_state}]")
    print(f"Confidence: {brief.confidence:.0%}")
    print(f"Source: {brief.source}")
    print(f"\nReasoning: {brief.reasoning}")
    print(f"Action: {brief.recommended_action}")
    print(f"\nSignals: {', '.join(brief.signals_detected)}")
    
    # Test LLM mode if API key available
    print("\n" + "-"*60)
    print("[Testing LLM Mode]")
    
    triangulator_llm = StrategicTriangulator(use_llm=True)
    brief_llm = triangulator_llm.analyze(sample_row)
    
    print(f"\nState: [{brief_llm.strategic_state}]")
    print(f"Confidence: {brief_llm.confidence:.0%}")
    print(f"Source: {brief_llm.source}")
    print(f"\nReasoning: {brief_llm.reasoning}")
    print(f"Action: {brief_llm.recommended_action}")
