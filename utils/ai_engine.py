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
    The output of the LLM classifier.
    Contains the strategic classification plus human-readable reasoning.
    """
    strategic_state: str
    confidence: float  # 0.0 - 1.0
    reasoning: str  # LLM-generated explanation
    recommended_action: str  # LLM-generated specific action
    
    # Visual properties (filled from STATE_DEFINITIONS)
    state_emoji: str = ""
    state_color: str = ""
    primary_outcome: str = ""
    
    # Source tracking
    source: str = "llm"  # "llm" or "fallback"
    signals_detected: List[str] = field(default_factory=list)
    asin: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dashboard rendering."""
        return {
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
        confidence = float(raw_confidence)
        # If confidence > 1 and <= 100, assume it's a percentage and convert to decimal
        # Most LLMs following the prompt will return 0.0-1.0, so values > 1 are likely percentages
        if 1.0 < confidence <= 100.0:
            confidence = confidence / 100.0
        # Clamp to valid range (0.0 to 1.0)
        confidence = max(0.0, min(1.0, confidence))
        
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
    
    return StrategicBrief(
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
# SYNCHRONOUS WRAPPER FOR SINGLE PRODUCT
# =============================================================================

class StrategicTriangulator:
    """
    Synchronous wrapper for the LLM classifier.
    
    Provides a simple interface for analyzing single products
    from synchronous code (like Streamlit callbacks).
    
    Usage:
        triangulator = StrategicTriangulator()
        brief = triangulator.analyze(product_row)
        result = brief.to_dict()
    """
    
    def __init__(self, use_llm: bool = True, timeout: float = 10.0, strategic_bias: str = "Balanced Defense"):
        """
        Initialize the triangulator.
        
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
    
    def analyze(self, row: Union[pd.Series, Dict], strategic_bias: Optional[str] = None) -> StrategicBrief:
        """
        Analyze a single product and return strategic classification.
        
        Args:
            row: Product data (Series or dict)
            strategic_bias: Override strategic bias for this analysis (optional)
            
        Returns:
            StrategicBrief with classification and reasoning
        """
        # Use provided bias or fall back to instance bias
        bias = strategic_bias or self.strategic_bias
        
        # Normalize row to dict
        if isinstance(row, pd.Series):
            row_data = row.to_dict()
        else:
            row_data = dict(row)
        
        if not self.use_llm:
            return _determine_state_fallback(row_data, reason="LLM disabled", strategic_bias=bias)
        
        # Run async analysis synchronously (same pattern as generate_portfolio_brief_sync)
        try:
            # Create new event loop explicitly to work in Streamlit's async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                analyze_strategy_with_llm(row_data, timeout=self.timeout, strategic_bias=bias)
            )
            loop.close()
            return result
        except Exception as e:
            return _determine_state_fallback(row_data, reason=f"Error: {str(e)[:30]}", strategic_bias=bias)
    
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
                        "content": """You are ShelfGuard's AI strategist, analyzing Amazon portfolio performance. Generate a clear, actionable strategic brief.

Guidelines:
- Be direct and prescriptive. Focus on what to do, not what happened.
- Start with the most important insight or issue.
- Quantify everything ($ amounts, counts, percentages).
- Reference data trends (e.g., "sales declining 15% vs 90-day average").
- End with one clear action for this session.
- Keep it under 100 words.
- Use clear business language - avoid military/tactical jargon."""
                    },
                    {
                        "role": "user",
                        "content": f"""Here's the current portfolio status:

{portfolio_summary}

Generate a strategic brief. What should be the focus this session?"""
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
