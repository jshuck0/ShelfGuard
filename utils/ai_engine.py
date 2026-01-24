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
    thirty_day_risk: float = 0.0      # Projected $ at risk over next 30 days (ACTUAL THREATS ONLY)
    optimization_value: float = 0.0   # $ opportunity for healthy products (NOT risk)
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
            # Predictive outputs (SEMANTIC SPLIT: risk vs optimization)
            "thirty_day_risk": self.thirty_day_risk,          # Actual threats only
            "optimization_value": self.optimization_value,     # Opportunity for healthy products
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

## Strategic States (in order of health)

1. **FORTRESS** - Dominant market position, pricing power
   - Buy Box ownership >80%
   - Healthy margins (>15%) or clear pricing power
   - Strong reviews (500+ with 4.3+ rating)
   - Stable or improving rank
   - Low competitive pressure
   - Action: Optimize, test price increases, harvest profits
   
2. **HARVEST** - Cash cow, maximize extraction
   - Stable rank (not growing, not declining >10%)
   - Good margins (>10%)
   - Mature review base (100+ reviews)
   - Premium price holding despite competition
   - Action: Reduce spend, raise prices, harvest cash

3. **TRENCH_WAR** - Competitive battle, defend share
   - Increasing competitor count (+3 or more in 30d)
   - Price pressure (competitors undercutting 5%+)
   - Rank volatility or declining
   - Buy Box rotation/loss (<60% ownership)
   - Action: Match pricing, defend keywords, protect share

4. **DISTRESS** - Needs intervention, value at risk
   - Margin compression (<10%) or erosion trend
   - Velocity decay (rank worsening 20%+ in 90d)
   - Review velocity stagnant or negative
   - Stock issues or pricing problems
   - Low Buy Box ownership (<40%)
   - Action: Fix root cause, investigate, restructure

5. **TERMINAL** - Exit required, cut losses
   - Severely negative or zero margins (CONFIRMED, not missing data)
   - Sustained rank decline (>50% worse over 90 days)
   - Zero path to profitability
   - CONFIRMED zero Buy Box ownership (<10%) with no recovery path
   - CONFIRMED <50 reviews with 3.5 or lower rating
   - Action: Liquidate, exit, stop all spend
   
   ⚠️ TERMINAL REQUIREMENTS: Use ONLY when metrics CONFIRM failure.
   Products with $5K+/week revenue and top-50 rank are NOT TERMINAL.
   Missing data is NOT confirmation of failure.

## Decision Framework

### Revenue Context (CRITICAL)
- Weekly Revenue >$5K = High-value asset, be cautious about TERMINAL
- Weekly Revenue $1K-$5K = Mid-tier, focus on margin optimization
- Weekly Revenue <$1K = Low volume, assess if worth the effort

### Buy Box Ownership Thresholds
- >80% = Strong (FORTRESS signal)
- 50-80% = Healthy (HARVEST possible)
- 30-50% = Contested (TRENCH_WAR signal)
- <30% = Critical (DISTRESS or TERMINAL)
- Missing/Unknown = Assume 50% (neutral)

### Review Count Context
- 1000+ reviews = Established, hard to displace
- 500-1000 = Strong social proof
- 100-500 = Competitive
- <100 = Vulnerable, needs growth
- 0 or missing = Could be data gap, NOT automatic TERMINAL (check revenue/rank)

### Rank Change Interpretation
- IMPROVING rank (negative % change) = Growth signal
- STABLE rank (±10%) = Steady state
- DECLINING rank (+10-30%) = Warning
- COLLAPSING rank (+30%+) = DISTRESS or TERMINAL

### Data Quality Rules (CRITICAL - READ FIRST)
⚠️ MISSING DATA IS NOT THE SAME AS ZERO DATA ⚠️

- If a metric shows "DATA_UNAVAILABLE" or is missing, do NOT assume the worst
- Missing Buy Box ≠ 0% ownership → Assume 50% (neutral)
- Missing reviews ≠ 0 reviews → Do NOT classify as TERMINAL based on missing review data
- Missing price data = Lower confidence, not automatic DISTRESS
- "buybox_health": "DATA_UNAVAILABLE" means we don't know, NOT that it's zero
- "review_tier": "DATA_UNAVAILABLE" means Keepa didn't return this data

IMPORTANT: Products with strong revenue ($5K+/week) and good rank (#30 or better) 
should NEVER be classified as TERMINAL just because review/BB data is missing.
TERMINAL requires CONFIRMED negative signals, not missing data.

## NEW: Critical Intelligence Signals (High Value)

### Amazon Buy Box Ownership Flags
- `amazon_owns_buybox: YES` = Amazon 1P is the Buy Box winner (HIGH competitive pressure)
- `buybox_is_fba: FBA/FBM` = Fulfillment type of current winner
- `buybox_status: BACKORDER` = Supply chain crisis - URGENT intervention needed

### True Seller Intelligence  
- `true_seller_count: N` = Actual sellers from sellerIds (more accurate than offer count)
- `amazon_is_seller: YES/NO` = Whether Amazon 1P is among the sellers

### Amazon Supply Intelligence (Opportunity Detection)
- `amazon_oos_events_30d: N` = How many times Amazon went OOS in 30 days
  - 3+ events = Amazon supply unstable = CONQUEST OPPORTUNITY
  - 5+ events = Major opportunity to capture Amazon's customers
- `amazon_supply_stability: UNSTABLE` = Frequent OOS signals opportunity

### Velocity Intelligence
- `bsr_velocity_30d: +/-X%` = Pre-calculated rank change from Keepa
  - Positive = rank worsening (DECLINING)
  - Negative = rank improving (GROWTH)
- `momentum: ACCELERATING/DECELERATING` = Growth trajectory

### Pack Size & Unit Economics
- `pack_size: N` = Number of items (e.g., 24 for a 24-pack)
- `price_per_unit: $X.XX` = Per-unit price for fair comparison across pack sizes

### Data Confidence
- `units_data_source: AMAZON ACTUAL` = Using Amazon's monthly sold (HIGH confidence)
- `units_data_source: BSR FORMULA` = Estimated from rank (MODERATE confidence)
- `amazon_monthly_units: N` = Amazon's actual monthly units sold

### Subscription Opportunity
- `sns_eligible: YES` = Subscribe & Save available (loyalty/retention lever)

## Nuanced Pattern Recognition

### FORTRESS Patterns
- High reviews (500+) + High price + Stable rank = Pricing power
- Strong Buy Box (80%+) + Few competitors = Market control
- Premium brand recognition + Loyal customer base
- `amazon_owns_buybox: NO` + High margin = No 1P competition

### HARVEST Patterns  
- Stable mature product + Good margins = Cash cow
- Declining ad efficiency but holding rank = Reduce spend
- Category leader maintaining position organically
- `sns_eligible: YES` = Build subscription base for recurring revenue

### TRENCH_WAR Patterns
- New sellers entering (competitor_count increasing)
- Price compression trend (current price < 90d avg)
- Buy Box rotating among sellers
- `amazon_owns_buybox: YES` = Amazon 1P as competitor (CRITICAL)
- `true_seller_count: 10+` = Crowded listing

### DISTRESS Patterns
- Rank decay + Margin erosion = Spiral risk
- Lost Buy Box + Inventory issues = Revenue at risk
- Review velocity stagnant while competitors grow
- `buybox_status: BACKORDER` = URGENT supply crisis
- `bsr_velocity_30d: +20%+` = Accelerating decline

### TERMINAL Patterns
- Negative margins sustained >90 days
- Rank collapse with no recovery
- Category obsolescence (technology shift)
- `amazon_owns_buybox: YES` + Low margin + Declining rank = Exit

### CONQUEST OPPORTUNITY Patterns (look for these!)
- `amazon_oos_events_30d: 3+` = Amazon supply unstable
- High competitor OOS rate + Your product in stock = Capture demand
- `momentum: ACCELERATING` + Market gaps = Expansion opportunity

## TRIGGER EVENTS (Real-Time Intelligence)

When trigger events are provided, USE THEM to inform your analysis:

### Threat Triggers (adjust state negatively)
- **PRICE_WAR_ACTIVE**: Multiple price drops in 7 days = TRENCH_WAR or worse
- **BUYBOX_COLLAPSE**: Buy Box share dropped >15% = Immediate DISTRESS signal
- **RANK_DETERIORATION**: Sustained rank worsening = Velocity problem
- **NEW_COMPETITOR_SURGE**: 3+ new sellers = TRENCH_WAR pressure
- **RATING_DECLINE**: Rating dropped = Social proof erosion

### Opportunity Triggers (adjust state positively or add to reasoning)
- **COMPETITOR_OOS_IMMINENT**: Competitor inventory <5 units = CONQUEST opportunity
- **AMAZON_SUPPLY_UNSTABLE**: Amazon OOS events = Capture their customers
- **REVIEW_VELOCITY_SPIKE**: Accelerating reviews = Momentum building
- **PRICE_POWER_DETECTED**: Underpriced vs category = Raise price opportunity
- **MOMENTUM_ACCELERATION**: Rank improving = Growth signal
- **SUBSCRIPTION_OPPORTUNITY**: S&S eligible + healthy = Loyalty play

### How to Use Triggers
1. CRITICAL/WARNING triggers should LOWER your confidence if not reflected in state
2. Opportunity triggers should inform your recommended_action
3. Multiple threat triggers = likely DISTRESS or worse
4. Multiple opportunity triggers = likely FORTRESS or HARVEST

## Specific Action Guidelines

ALWAYS make recommendations SPECIFIC and QUANTIFIED:
- BAD: "Optimize pricing" 
- GOOD: "Raise price from $24.99 to $27.99 (+12%)"
- BAD: "Reduce ad spend"
- GOOD: "Cut PPC spend 30%, current ACOS likely unsustainable"
- BAD: "Monitor situation"
- GOOD: "Hold position, review in 2 weeks if rank drops below #500"

## Output Format

Return ONLY valid JSON with this exact structure:
{
    "strategic_state": "STATE_NAME",
    "confidence": 0.85,
    "reasoning": "1-2 clear sentences explaining WHY. Reference specific metrics.",
    "recommended_action": "One specific, quantified action with clear next step"
}

Important:
- Keep reasoning under 150 characters but be SPECIFIC
- ALWAYS explain the SOURCE of risk/opportunity:
  * If HARVEST with risk: Explain if it's "optimization opportunity" (leaving money on table) vs actual competitive threat
  * If TRENCH_WAR: Explain competitive pressure source (new sellers, price undercutting, Buy Box rotation)
  * If DISTRESS: Explain root cause (velocity decline, margin erosion, inventory issues)
  * Reference competitive intelligence if available (competitor count, price gaps, OOS rates)
- Reference actual numbers from the data (rank, price, reviews, Buy Box %, competitor count)
- Make recommended_action specific and measurable
- If data is incomplete, lower confidence and note data gaps
- Return ONLY the JSON object, no other text."""


def _get_strategic_bias_instructions(strategic_bias: str) -> str:
    """
    Generate additional LLM instructions based on the user's strategic focus.
    
    This is the "magic" that makes the entire dashboard shift based on one selector.
    """
    bias_instructions = {
        "Profit Maximization": """
## 🎯 STRATEGIC BIAS: PROFIT MAXIMIZATION

The user prioritizes **Profit over Growth**. Adjust your analysis:

SCORING ADJUSTMENTS:
- Margin <10% → Automatic DISTRESS signal (even with good velocity)
- Margin >20% → Strong FORTRESS/HARVEST signal
- Price increase + stable rank → FORTRESS
- Volume growth + thin margins → DISTRESS ("Unprofitable growth")

RECOMMENDED ACTIONS:
- Always prefer "Raise price" or "Cut ad spend" over "Scale"
- Question any growth investment if margin <15%
- Emphasize cash extraction, reduce reinvestment
- Recommend cutting underperforming products faster

EXAMPLE CLASSIFICATIONS:
- 8% margin + rank improving → DISTRESS ("Growth is unprofitable")
- 25% margin + rank flat → HARVEST ("Extract cash, raise prices")
- Negative margin + any velocity → TERMINAL ("Exit immediately")
""",
        "Balanced Defense": """
## 🎯 STRATEGIC BIAS: BALANCED DEFENSE

The user prioritizes **Balanced approach**. Use standard logic:

SCORING ADJUSTMENTS:
- Weight margin, velocity, competition, and reviews equally
- Consider both short-term profitability and long-term market position
- Don't over-penalize growth investments if they're building share

RECOMMENDED ACTIONS:
- Balance cash extraction with market defense
- Maintain competitive position while optimizing margins
- Invest in products with clear ROI paths

EXAMPLE CLASSIFICATIONS:
- 12% margin + rank stable → HARVEST
- 8% margin + rank improving → TRENCH_WAR
- 20% margin + rank declining → DISTRESS (investigate cause)
""",
        "Aggressive Growth": """
## 🎯 STRATEGIC BIAS: AGGRESSIVE GROWTH

The user prioritizes **Growth over Profit**. Adjust your analysis:

SCORING ADJUSTMENTS:
- Forgive margin compression if rank is improving significantly
- 5% margin + 30% rank improvement → TRENCH_WAR (acceptable growth investment)
- Prioritize velocity gains over margin preservation
- Reward review velocity and market share gains
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
    
    # === TRIGGER EVENT PRE-DETECTION (Feed to LLM for smarter reasoning) ===
    trigger_context = ""
    detected_triggers = []
    if 'historical_df' in row_data:
        try:
            from src.trigger_detection import detect_trigger_events
            asin = row_data.get('asin', '')
            historical_df = row_data['historical_df']
            competitors_df = row_data.get('competitors_df', pd.DataFrame())
            
            if not historical_df.empty:
                detected_triggers = detect_trigger_events(
                    asin=asin,
                    df_historical=historical_df,
                    df_competitors=competitors_df
                )
                
                if detected_triggers:
                    # Build trigger context for LLM
                    trigger_lines = []
                    for t in detected_triggers[:5]:  # Top 5 most severe
                        severity_label = "CRITICAL" if t.severity >= 8 else "WARNING" if t.severity >= 6 else "INFO"
                        trigger_lines.append(f"- [{severity_label}] {t.event_type}: {t.metric_name} changed {t.delta_pct:+.1f}%")
                    
                    trigger_context = "\n\n### DETECTED TRIGGER EVENTS (use these to inform your analysis):\n" + "\n".join(trigger_lines)
                    clean_data["trigger_events_detected"] = len(detected_triggers)
                    clean_data["most_severe_trigger"] = detected_triggers[0].event_type if detected_triggers else None
        except Exception:
            pass  # Silently continue if trigger detection fails
    
    # Build system prompt with strategic bias
    bias_instructions = _get_strategic_bias_instructions(strategic_bias)
    full_system_prompt = f"{STRATEGIST_SYSTEM_PROMPT}\n\n{bias_instructions}"
    
    try:
        # Build user message with trigger context if available
        user_message = f"Analyze this product:\n\n```json\n{json.dumps(clean_data, indent=2)}\n```{trigger_context}"
        
        # Call LLM with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=400,  # Increased for richer reasoning with trigger context
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
        
        # Combine static signal summary with detected trigger events
        signals = _extract_signal_summary(clean_data)
        for trigger in detected_triggers[:5]:
            trigger_signal = f"{trigger.event_type} ({trigger.delta_pct:+.0f}%)"
            if trigger_signal not in signals:
                signals.append(trigger_signal)
        
        return StrategicBrief(
            strategic_state=state_str,
            confidence=confidence,
            reasoning=result.get("reasoning", "LLM analysis complete."),
            recommended_action=result.get("recommended_action", state_def["default_action"]),
            state_emoji=state_def["emoji"],
            state_color=state_def["color"],
            primary_outcome=state_def["primary_outcome"],
            source="llm",
            signals_detected=signals,
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
    
    Comprehensive metric extraction for intelligent AI analysis.
    Includes all available Keepa signals, inventory, competition, and trends.
    """
    clean = {}
    
    # =============================================
    # IDENTITY & CONTEXT
    # =============================================
    if "asin" in row_data:
        clean["asin"] = row_data["asin"]
    if "title" in row_data:
        title = str(row_data["title"])[:80]
        clean["title"] = title + "..." if len(str(row_data.get("title", ""))) > 80 else title
    
    # Brand (critical for premium positioning analysis)
    if "brand" in row_data and row_data["brand"]:
        clean["brand"] = str(row_data["brand"])[:50]
    
    # Category context
    if "category_tree" in row_data:
        clean["category"] = str(row_data["category_tree"])[:100]
    elif "rootCategory" in row_data:
        clean["category"] = str(row_data["rootCategory"])[:100]
    
    # Parent ASIN (for variation analysis)
    if "parent_asin" in row_data and row_data["parent_asin"]:
        clean["parent_asin"] = row_data["parent_asin"]
        clean["is_variation"] = True
    
    # =============================================
    # PRICING METRICS (Core signal)
    # =============================================
    price_fields = ["filled_price", "buy_box_price", "amazon_price", "current_AMAZON", "price"]
    for field in price_fields:
        if field in row_data and row_data[field] is not None:
            val = _safe_float(row_data[field])
            if val and val > 0:
                # Convert cents to dollars if needed
                if val > 500:
                    val = val / 100
                clean["current_price"] = f"${val:.2f}"
                clean["current_price_raw"] = val  # For calculations
                break
    
    # Price 90d average (for trend analysis)
    if "avg90_AMAZON" in row_data or "price_90d_avg" in row_data:
        avg = _safe_float(row_data.get("avg90_AMAZON") or row_data.get("price_90d_avg"))
        if avg and avg > 0:
            if avg > 500:
                avg = avg / 100
            clean["price_90d_avg"] = f"${avg:.2f}"
            # Calculate price trend if we have current price
            if "current_price_raw" in clean:
                price_change_pct = ((clean["current_price_raw"] - avg) / avg) * 100
                if abs(price_change_pct) > 2:  # Only report significant changes
                    clean["price_trend_90d"] = f"{price_change_pct:+.1f}%"
    
    # Min/Max price (volatility indicator)
    if "min90_AMAZON" in row_data:
        min_p = _safe_float(row_data["min90_AMAZON"])
        if min_p and min_p > 0:
            if min_p > 500:
                min_p = min_p / 100
            clean["price_90d_min"] = f"${min_p:.2f}"
    
    if "max90_AMAZON" in row_data:
        max_p = _safe_float(row_data["max90_AMAZON"])
        if max_p and max_p > 0:
            if max_p > 500:
                max_p = max_p / 100
            clean["price_90d_max"] = f"${max_p:.2f}"
    
    # =============================================
    # MARGIN & PROFITABILITY
    # =============================================
    if "net_margin" in row_data:
        margin = _safe_float(row_data["net_margin"])
        if margin is not None:
            clean["net_margin"] = f"{margin*100:.1f}%"
            # Add margin health indicator
            if margin > 0.20:
                clean["margin_health"] = "STRONG"
            elif margin > 0.10:
                clean["margin_health"] = "HEALTHY"
            elif margin > 0:
                clean["margin_health"] = "THIN"
            else:
                clean["margin_health"] = "NEGATIVE"
    
    # =============================================
    # RANK (Sales Velocity Proxy)
    # =============================================
    rank_fields = ["sales_rank_filled", "sales_rank", "current_SALES", "bsr"]
    for field in rank_fields:
        if field in row_data and row_data[field] is not None:
            rank = _safe_float(row_data[field])
            if rank and rank > 0:
                clean["current_sales_rank"] = int(rank)
                # Add rank tier
                if rank < 1000:
                    clean["rank_tier"] = "TOP_1K (High Volume)"
                elif rank < 10000:
                    clean["rank_tier"] = "TOP_10K (Strong)"
                elif rank < 50000:
                    clean["rank_tier"] = "TOP_50K (Moderate)"
                elif rank < 100000:
                    clean["rank_tier"] = "TOP_100K (Low-Moderate)"
                else:
                    clean["rank_tier"] = "LONG_TAIL (Low Volume)"
                break
    
    # Rank trends (critical for velocity analysis)
    if "deltaPercent30_SALES" in row_data or "rank_delta_30d_pct" in row_data:
        delta = _safe_float(row_data.get("deltaPercent30_SALES") or row_data.get("rank_delta_30d_pct"))
        if delta is not None:
            clean["rank_change_30d"] = f"{delta:+.1f}%"
            # Interpret trend
            if delta < -15:
                clean["rank_trend_30d"] = "ACCELERATING"
            elif delta < -5:
                clean["rank_trend_30d"] = "IMPROVING"
            elif delta < 5:
                clean["rank_trend_30d"] = "STABLE"
            elif delta < 15:
                clean["rank_trend_30d"] = "SLOWING"
            else:
                clean["rank_trend_30d"] = "DECLINING"
    
    if "deltaPercent90_SALES" in row_data or "rank_delta_90d_pct" in row_data:
        delta = _safe_float(row_data.get("deltaPercent90_SALES") or row_data.get("rank_delta_90d_pct"))
        if delta is not None:
            clean["rank_change_90d"] = f"{delta:+.1f}%"
            # Interpret long-term trend
            if delta < -20:
                clean["rank_trend_90d"] = "STRONG_GROWTH"
            elif delta < 0:
                clean["rank_trend_90d"] = "GROWING"
            elif delta < 20:
                clean["rank_trend_90d"] = "STABLE"
            elif delta < 50:
                clean["rank_trend_90d"] = "DECLINING"
            else:
                clean["rank_trend_90d"] = "COLLAPSING"
    
    # Velocity decay multiplier
    if "velocity_decay" in row_data:
        decay = _safe_float(row_data["velocity_decay"], default=1.0)
        if decay != 1.0:
            if decay < 0.9:
                clean["velocity_trend"] = f"ACCELERATING ({decay:.2f}x)"
            elif decay > 1.1:
                clean["velocity_trend"] = f"DECAYING ({decay:.2f}x)"
            else:
                clean["velocity_trend"] = "STABLE"
    
    # =============================================
    # BUY BOX (Critical ownership signal)
    # =============================================
    # IMPORTANT: Missing BB data (None) should default to "assumed 50%" not "0%"
    # Zero BB is a CRITICAL signal, but missing data is NOT zero.
    bb_fields = ["amazon_bb_share", "buybox_share", "buyBoxPercentage"]
    bb_found = False
    for field in bb_fields:
        if field in row_data and row_data[field] is not None:
            bb = _safe_float(row_data[field])
            if bb is not None:
                bb_found = True
                # Normalize to 0-1 if percentage
                if bb > 1:
                    bb = bb / 100
                clean["buybox_ownership"] = f"{bb*100:.0f}%"
                # Add Buy Box health indicator
                if bb >= 0.80:
                    clean["buybox_health"] = "DOMINANT"
                elif bb >= 0.50:
                    clean["buybox_health"] = "HEALTHY"
                elif bb >= 0.30:
                    clean["buybox_health"] = "CONTESTED"
                elif bb > 0:
                    clean["buybox_health"] = "AT_RISK"
                else:
                    # Only flag CRITICAL if we have confirmed 0% BB ownership
                    clean["buybox_health"] = "CRITICAL"
                break
    
    # If no BB data found, assume neutral (50%) - don't assume zero!
    if not bb_found:
        clean["buybox_ownership"] = "50% (estimated)"
        clean["buybox_health"] = "DATA_UNAVAILABLE"
        clean["buybox_data_note"] = "Buy Box data not available - assuming 50% (neutral)"
    
    # Buy Box 30d stats
    if "buyBoxStatsAmazon30" in row_data:
        bb = _safe_float(row_data["buyBoxStatsAmazon30"])
        if bb is not None:
            clean["amazon_buybox_30d"] = f"{bb:.0f}%"
    
    # Buy Box switches (volatility indicator)
    if "buy_box_switches" in row_data:
        switches = _safe_float(row_data["buy_box_switches"])
        if switches is not None and switches > 0:
            clean["buybox_switches"] = int(switches)
            if switches > 10:
                clean["buybox_volatility"] = "HIGH"
            elif switches > 3:
                clean["buybox_volatility"] = "MODERATE"
            else:
                clean["buybox_volatility"] = "LOW"
    
    # =============================================
    # COMPETITION
    # =============================================
    offer_fields = ["new_offer_count", "current_COUNT_NEW", "offerCountNew"]
    for field in offer_fields:
        if field in row_data and row_data[field] is not None:
            count = _safe_float(row_data[field])
            if count is not None:
                clean["competitor_count"] = int(count)
                # Add competition level
                if count <= 3:
                    clean["competition_level"] = "LOW"
                elif count <= 10:
                    clean["competition_level"] = "MODERATE"
                elif count <= 25:
                    clean["competition_level"] = "HIGH"
                else:
                    clean["competition_level"] = "SATURATED"
                break
    
    # Competitor change (market dynamics)
    if "delta30_COUNT_NEW" in row_data:
        delta = _safe_float(row_data["delta30_COUNT_NEW"])
        if delta is not None and delta != 0:
            clean["competitor_change_30d"] = f"{delta:+.0f} sellers"
            if delta > 5:
                clean["competition_trend"] = "INCREASING_RAPIDLY"
            elif delta > 0:
                clean["competition_trend"] = "INCREASING"
            elif delta < -5:
                clean["competition_trend"] = "DECREASING_RAPIDLY"
            else:
                clean["competition_trend"] = "DECREASING"
    
    # Price gap vs competitors
    if "price_gap" in row_data:
        gap = _safe_float(row_data["price_gap"])
        if gap is not None and abs(gap) > 0.03:
            clean["price_vs_competitor"] = f"{gap*100:+.0f}%"
    
    # Competitor out of stock (opportunity signal)
    if "competitor_oos_pct" in row_data or "outOfStockPercentage90" in row_data:
        oos = _safe_float(row_data.get("competitor_oos_pct") or row_data.get("outOfStockPercentage90"))
        if oos is not None and oos > 0:
            if oos > 1:  # Normalize percentage
                oos = oos / 100
            clean["competitor_oos_rate"] = f"{oos*100:.0f}%"
            if oos > 0.20:
                clean["oos_opportunity"] = "HIGH (competitors struggling)"
    
    # =============================================
    # REVIEWS & SOCIAL PROOF
    # =============================================
    # IMPORTANT: Missing review data (None/0) does NOT mean "zero reviews"
    # It often means Keepa didn't return this data for the product.
    # AI should NOT assume TERMINAL state based on missing review data alone.
    review_fields = ["review_count", "current_COUNT_REVIEWS", "reviewCount"]
    review_found = False
    for field in review_fields:
        if field in row_data and row_data[field] is not None:
            count = _safe_float(row_data[field])
            if count is not None and count > 0:
                clean["review_count"] = int(count)
                review_found = True
                # Add review tier
                if count >= 1000:
                    clean["review_tier"] = "ESTABLISHED (1K+)"
                elif count >= 500:
                    clean["review_tier"] = "STRONG (500+)"
                elif count >= 100:
                    clean["review_tier"] = "COMPETITIVE (100+)"
                elif count >= 25:
                    clean["review_tier"] = "DEVELOPING (25+)"
                else:
                    clean["review_tier"] = "NEW (<25)"
                break
    
    # If no review data found, mark as UNKNOWN (not zero!)
    if not review_found:
        clean["review_tier"] = "DATA_UNAVAILABLE"
        clean["review_data_note"] = "Review count not available from Keepa - do not assume zero reviews"
    
    # Review velocity (growth signal)
    if "delta30_COUNT_REVIEWS" in row_data:
        delta = _safe_float(row_data["delta30_COUNT_REVIEWS"])
        if delta is not None and delta != 0:
            clean["reviews_added_30d"] = int(delta)
            if delta > 50:
                clean["review_velocity"] = "EXPLOSIVE"
            elif delta > 20:
                clean["review_velocity"] = "STRONG"
            elif delta > 5:
                clean["review_velocity"] = "HEALTHY"
            elif delta > 0:
                clean["review_velocity"] = "SLOW"
            else:
                clean["review_velocity"] = "NEGATIVE"
    
    # Rating
    rating_fields = ["rating", "current_RATING", "avgRating"]
    for field in rating_fields:
        if field in row_data and row_data[field] is not None:
            rating = _safe_float(row_data[field])
            if rating is not None and rating > 0:
                # Keepa stores as rating*10
                if rating > 10:
                    rating = rating / 10
                clean["rating"] = f"{rating:.1f}★"
                # Add rating health
                if rating >= 4.5:
                    clean["rating_health"] = "EXCELLENT"
                elif rating >= 4.0:
                    clean["rating_health"] = "GOOD"
                elif rating >= 3.5:
                    clean["rating_health"] = "CONCERNING"
                else:
                    clean["rating_health"] = "CRITICAL"
                break
    
    # =============================================
    # REVENUE & SALES VOLUME
    # =============================================
    rev_fields = ["weekly_sales_filled", "revenue_proxy", "estimated_weekly_revenue"]
    for field in rev_fields:
        if field in row_data:
            rev = _safe_float(row_data[field])
            if rev is not None and rev > 0:
                clean["weekly_revenue"] = f"${rev:,.0f}"
                clean["monthly_revenue_est"] = f"${rev * 4.33:,.0f}"
                # Add revenue tier
                if rev >= 10000:
                    clean["revenue_tier"] = "HIGH_VALUE ($10K+/wk)"
                elif rev >= 5000:
                    clean["revenue_tier"] = "STRONG ($5K+/wk)"
                elif rev >= 1000:
                    clean["revenue_tier"] = "MODERATE ($1K+/wk)"
                else:
                    clean["revenue_tier"] = "LOW (<$1K/wk)"
                break
    
    # Monthly units sold
    if "monthlySold" in row_data:
        sold = _safe_float(row_data["monthlySold"])
        if sold is not None and sold > 0:
            clean["monthly_units_sold"] = f"{int(sold):,}+"
    
    # Estimated units (from various sources)
    if "estimated_units" in row_data:
        units = _safe_float(row_data["estimated_units"])
        if units is not None and units > 0:
            clean["estimated_weekly_units"] = int(units)
    
    # =============================================
    # INVENTORY & STOCK (If available)
    # =============================================
    if "days_to_stockout" in row_data:
        days = _safe_float(row_data["days_to_stockout"])
        if days is not None and days < 90:
            clean["days_to_stockout"] = int(days)
            if days < 7:
                clean["stockout_risk"] = "CRITICAL"
            elif days < 14:
                clean["stockout_risk"] = "HIGH"
            elif days < 30:
                clean["stockout_risk"] = "MODERATE"
            else:
                clean["stockout_risk"] = "LOW"
    
    if "fbaFees" in row_data or "fba_fees" in row_data:
        fees = _safe_float(row_data.get("fbaFees") or row_data.get("fba_fees"))
        if fees is not None and fees > 0:
            if fees > 100:  # Convert cents
                fees = fees / 100
            clean["fba_fees"] = f"${fees:.2f}"
    
    # =============================================
    # COMPETITIVE CONTEXT (Enriched from market analysis)
    # =============================================
    
    # Price gap vs median (market positioning)
    if "price_gap_vs_median" in row_data:
        gap = _safe_float(row_data["price_gap_vs_median"])
        if gap is not None:
            pct = gap * 100
            if abs(pct) > 3:
                clean["price_vs_market_median"] = f"{pct:+.0f}%"
                if pct < -10:
                    clean["price_position"] = "UNDERPRICED (opportunity to raise)"
                elif pct < -5:
                    clean["price_position"] = "Below market"
                elif pct > 10:
                    clean["price_position"] = "PREMIUM (justify or reduce)"
                elif pct > 5:
                    clean["price_position"] = "Above market"
                else:
                    clean["price_position"] = "At market"
    
    # Median competitor price (benchmark)
    if "median_competitor_price" in row_data:
        median_price = _safe_float(row_data["median_competitor_price"])
        if median_price and median_price > 0:
            clean["market_median_price"] = f"${median_price:.2f}"
    
    # Review advantage
    if "review_advantage_pct" in row_data:
        adv = _safe_float(row_data["review_advantage_pct"])
        if adv is not None:
            pct = adv * 100
            if abs(pct) > 20:
                clean["review_vs_market"] = f"{pct:+.0f}%"
                if pct > 50:
                    clean["review_position"] = "DOMINANT (strong moat)"
                elif pct > 20:
                    clean["review_position"] = "Advantage"
                elif pct < -50:
                    clean["review_position"] = "VULNERABLE (needs growth)"
                elif pct < -20:
                    clean["review_position"] = "Disadvantage"
    
    # Competitor count (explicit)
    if "competitor_count" in row_data and "competitor_count" not in clean:
        count = _safe_float(row_data["competitor_count"])
        if count is not None and count > 0:
            clean["total_market_competitors"] = int(count)
    
    # Best/worst competitor rank (market intensity)
    if "best_competitor_rank" in row_data:
        best_rank = _safe_float(row_data["best_competitor_rank"])
        if best_rank and best_rank > 0:
            clean["best_competitor_rank"] = int(best_rank)
    
    # Competitor OOS (opportunity)
    if "competitor_oos_pct" in row_data:
        oos = _safe_float(row_data["competitor_oos_pct"])
        if oos is not None and oos > 0.05:  # >5% OOS
            oos_pct = oos * 100 if oos <= 1 else oos
            clean["competitor_oos_rate"] = f"{oos_pct:.0f}%"
            if oos_pct > 30:
                clean["oos_opportunity"] = "HIGH (competitors struggling - conquest opportunity)"
            elif oos_pct > 15:
                clean["oos_opportunity"] = "MODERATE (some supply issues)"
    
    # =============================================
    # NEW CRITICAL METRICS (2026-01-21)
    # =============================================
    
    # Buy Box ownership flags (more precise than percentage)
    if "buybox_is_amazon" in row_data and row_data["buybox_is_amazon"] is not None:
        is_amazon = row_data["buybox_is_amazon"]
        clean["amazon_owns_buybox"] = "YES" if is_amazon else "NO"
        if is_amazon:
            clean["buybox_status"] = "AMAZON 1P (competitive pressure high)"
    
    if "buybox_is_fba" in row_data and row_data["buybox_is_fba"] is not None:
        is_fba = row_data["buybox_is_fba"]
        clean["buybox_is_fba"] = "FBA" if is_fba else "FBM/Other"
    
    if "buybox_is_backorder" in row_data and row_data["buybox_is_backorder"]:
        clean["supply_status"] = "BACKORDER (supply chain issue)"
    
    # True seller count from sellerIds (more accurate than offer count)
    if "seller_count" in row_data:
        seller_ct = _safe_float(row_data["seller_count"])
        if seller_ct is not None and seller_ct > 0:
            clean["true_seller_count"] = int(seller_ct)
            # Override competitor count with more accurate data
            if "competitor_count" not in clean:
                clean["competitor_count"] = int(seller_ct)
    
    # Amazon as seller indicator
    if "has_amazon_seller" in row_data and row_data["has_amazon_seller"] is not None:
        has_amz = row_data["has_amazon_seller"]
        clean["amazon_is_seller"] = "YES" if has_amz else "NO"
    
    # OOS event counts (more actionable than percentage)
    if "oos_count_amazon_30" in row_data:
        oos_30 = _safe_float(row_data["oos_count_amazon_30"])
        if oos_30 is not None and oos_30 > 0:
            clean["amazon_oos_events_30d"] = int(oos_30)
            if oos_30 >= 5:
                clean["amazon_supply_stability"] = "UNSTABLE (frequent OOS)"
            elif oos_30 >= 2:
                clean["amazon_supply_stability"] = "MODERATE (occasional OOS)"
    
    if "oos_count_amazon_90" in row_data:
        oos_90 = _safe_float(row_data["oos_count_amazon_90"])
        if oos_90 is not None and oos_90 > 0:
            clean["amazon_oos_events_90d"] = int(oos_90)
    
    # Pre-calculated velocity from Keepa
    if "velocity_30d" in row_data:
        vel_30 = _safe_float(row_data["velocity_30d"])
        if vel_30 is not None:
            clean["bsr_velocity_30d"] = f"{vel_30:+.1f}%"
            if vel_30 > 20:
                clean["momentum"] = "ACCELERATING (strong growth)"
            elif vel_30 < -20:
                clean["momentum"] = "DECELERATING (declining)"
    
    if "velocity_90d" in row_data:
        vel_90 = _safe_float(row_data["velocity_90d"])
        if vel_90 is not None:
            clean["bsr_velocity_90d"] = f"{vel_90:+.1f}%"
    
    # Pack size and per-unit price (for fair comparison)
    if "number_of_items" in row_data:
        pack_size = _safe_float(row_data["number_of_items"])
        if pack_size is not None and pack_size > 1:
            clean["pack_size"] = int(pack_size)
    
    if "price_per_unit" in row_data:
        ppu = _safe_float(row_data["price_per_unit"])
        if ppu is not None and ppu > 0:
            clean["price_per_unit"] = f"${ppu:.2f}"
    
    # Units source tracking
    if "units_source" in row_data:
        src = row_data["units_source"]
        if src == "amazon_monthly_sold":
            clean["units_data_source"] = "AMAZON ACTUAL (high confidence)"
        else:
            clean["units_data_source"] = "BSR FORMULA (estimated)"
    
    # Amazon's monthly sold (if available)
    if "monthly_sold" in row_data:
        sold = _safe_float(row_data["monthly_sold"])
        if sold is not None and sold > 0:
            clean["amazon_monthly_units"] = f"{int(sold):,}"
    
    # Subscribe & Save eligibility
    if "is_sns" in row_data and row_data["is_sns"]:
        clean["sns_eligible"] = "YES (Subscribe & Save)"
    
    # =============================================
    # DATA QUALITY INDICATOR
    # =============================================
    # Count how many key metrics we have
    key_metrics = ["current_price", "current_sales_rank", "buybox_ownership", "review_count", "rating", "weekly_revenue"]
    metrics_present = sum(1 for m in key_metrics if m in clean)
    
    # Also count competitive context
    competitive_metrics = ["price_vs_market_median", "review_vs_market", "total_market_competitors"]
    competitive_present = sum(1 for m in competitive_metrics if m in clean)
    
    # Count new critical metrics (2026-01-21)
    new_critical_metrics = ["amazon_owns_buybox", "true_seller_count", "bsr_velocity_30d", "amazon_monthly_units", "amazon_oos_events_30d"]
    new_metrics_present = sum(1 for m in new_critical_metrics if m in clean)
    
    if metrics_present >= 5:
        clean["data_quality"] = "HIGH"
        if competitive_present >= 2:
            clean["competitive_context"] = "ENRICHED"
        if new_metrics_present >= 3:
            clean["data_richness"] = "PREMIUM (Amazon direct data available)"
    elif metrics_present >= 3:
        clean["data_quality"] = "MEDIUM"
    else:
        clean["data_quality"] = "LOW"
        clean["data_quality_note"] = f"Only {metrics_present}/6 key metrics available"
    
    # Remove raw values used for calculations
    clean.pop("current_price_raw", None)
    
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
    
    # NEW: Use new critical metrics when available (2026-01-21)
    # Use seller_count if available (more accurate than offer_count)
    seller_count = _safe_float(row_data.get("seller_count"))
    if seller_count is not None and seller_count > 0:
        offer_count = seller_count
    
    # Use pre-calculated velocity if available
    velocity_30d = _safe_float(row_data.get("velocity_30d"))
    if velocity_30d is not None:
        # Convert velocity percentage to decay factor
        # Positive velocity = growth, negative = decay
        velocity_decay = 1.0 - (velocity_30d / 100.0) if velocity_30d < 0 else 1.0
    
    # Check if Amazon owns Buy Box (competitive pressure indicator)
    amazon_owns_bb = row_data.get("buybox_is_amazon")
    is_backorder = row_data.get("buybox_is_backorder", False)
    
    # OOS events indicate supply instability
    oos_count_30 = _safe_float(row_data.get("oos_count_amazon_30"), default=0)
    
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
    opportunities = []
    
    # =========================================================================
    # NEW: Extract intelligence from new critical metrics (2026-01-21)
    # =========================================================================
    
    # Amazon as Buy Box winner = major competitive pressure
    amazon_1p_pressure = amazon_owns_bb is True
    if amazon_1p_pressure:
        signals.append("Amazon 1P owns Buy Box (HIGH competitive pressure)")
    
    # Backorder = supply chain crisis
    supply_crisis = is_backorder is True
    if supply_crisis:
        signals.append("BACKORDER STATUS (supply chain issue)")
    
    # Amazon OOS events = conquest opportunity  
    amazon_unstable = oos_count_30 >= 3
    if amazon_unstable:
        opportunities.append(f"Amazon OOS {int(oos_count_30)}x in 30d (conquest opportunity)")
    
    # Get additional new metrics
    is_sns = row_data.get("is_sns", False)
    monthly_sold = _safe_float(row_data.get("monthly_sold"), default=0)
    units_source = row_data.get("units_source", "bsr_formula")
    number_of_items = _safe_float(row_data.get("number_of_items"), default=1)
    has_amazon_seller = row_data.get("has_amazon_seller")
    
    # High-confidence data available
    premium_data = units_source == "amazon_monthly_sold" and monthly_sold > 0
    if premium_data:
        signals.append(f"Amazon data: {int(monthly_sold):,} units/mo (HIGH confidence)")
    
    # S&S opportunity
    if is_sns:
        opportunities.append("Subscribe & Save eligible (loyalty opportunity)")
    
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
    
    # DISTRESS: Margin compression or velocity decay or supply crisis
    elif margin < margin_distress or velocity_decay > 1.3 or supply_crisis:
        state = StrategicState.DISTRESS
        confidence = 0.80 if supply_crisis else 0.75
        
        if supply_crisis:
            reasoning = f"SUPPLY CRISIS: Product is backordered. This damages rank, reviews, and long-term velocity."
            action = "URGENT: Expedite inventory. Consider air freight. Communicate with supplier daily."
            signals.append("BACKORDER (supply chain crisis)")
        else:
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
    
    # TRENCH_WAR: High competition or BB pressure or Amazon 1P
    elif offer_count > 10 or bb_share < 0.60 or price_gap > 0.10 or amazon_1p_pressure:
        state = StrategicState.TRENCH_WAR
        confidence = 0.75 if amazon_1p_pressure else 0.70
        
        if amazon_1p_pressure:
            reasoning = f"Amazon 1P competition detected. {int(offer_count)} sellers, BB share {bb_share*100:.0f}%."
            if strategic_bias == "Profit Maximization":
                action = "Differentiate on value-adds Amazon can't match. Focus on bundle/subscription."
            elif strategic_bias == "Aggressive Growth":
                action = "Compete on keywords Amazon doesn't bid on. Build brand moat."
            else:
                action = "Defend with brand differentiation. Avoid direct price war with Amazon."
        else:
            reasoning = f"Competitive pressure detected. {int(offer_count)} sellers, BB share {bb_share*100:.0f}%."
            if strategic_bias == "Profit Maximization":
                action = f"Avoid price war. Focus on differentiation. Consider raising price (gap {price_gap*100:+.0f}%)."
            elif strategic_bias == "Aggressive Growth":
                action = f"Defend aggressively. Match pricing (gap {price_gap*100:+.0f}%). Scale defensive ads."
            else:
                action = f"Defend position. Match competitor pricing (gap {price_gap*100:+.0f}%). Increase visibility spend."
        
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
            action = f"Maximize extraction. Raise price +10% (current margin {margin*100:.0f}%). Cut ad spend 30%."
        elif strategic_bias == "Aggressive Growth":
            action = "Invest for scale. Test ad expansion to adjacent keywords."
        else:
            action = f"Test price increase +5% (margin {margin*100:.0f}%). Reduce ad spend. Maximize profit."
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
    
    # Merge opportunities into signals for display
    all_signals = signals + opportunities
    
    # Boost confidence if we have premium Amazon data
    if premium_data:
        confidence = min(confidence + 0.10, 0.95)
    
    # Add opportunity context to reasoning
    if opportunities:
        reasoning += f" Opportunities: {'; '.join(opportunities)}."
    
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
        signals_detected=all_signals,
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
        
        # === FIX: Explicitly Map Optimization & Risk for Dashboard ===
        # Map growth to optimization value
        brief.optimization_value = expansion.thirty_day_growth
        
        # Calculate Risk based on State & Revenue
        # This was previously missing, causing 0s in the dashboard
        risk_factor = 0.0
        if state == StrategicState.TERMINAL:
            risk_factor = 0.90 # 90% of revenue at risk
        elif state == StrategicState.DISTRESS:
            risk_factor = 0.50 # 50% at risk
        elif state == StrategicState.TRENCH_WAR:
            risk_factor = 0.20 # 20% at risk due to competition
        
        # Set the calculated risk
        brief.thirty_day_risk = revenue * risk_factor
        
    except Exception:
        # Fallback values if expansion calc fails
        brief.optimization_value = 0.0
        brief.thirty_day_risk = revenue * 0.1 if state in [StrategicState.DISTRESS, StrategicState.TERMINAL] else 0.0
    
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
    
    SEMANTIC CLARITY:
    - thirty_day_risk = ACTUAL THREATS (will lose money if you don't act)
    - optimization_value = OPPORTUNITY (could gain money with optimization)
    """
    # Core prediction (required fields - no defaults)
    thirty_day_risk: float              # Predicted $ at risk over next 30 days (ACTUAL THREATS ONLY)
    daily_burn_rate: float              # Current daily loss rate
    velocity_multiplier: float          # Trend-based adjustment (>1 = accelerating, <1 = decelerating)
    
    # Risk components (required fields - no defaults)
    price_erosion_risk: float           # Risk from competitor pricing pressure
    share_erosion_risk: float           # Risk from market share loss
    stockout_risk: float                # Risk from inventory stockout
    
    # Predictive state (required fields - no defaults)
    predictive_state: str               # DEFEND, EXPLOIT, REPLENISH, HOLD
    state_emoji: str                    # Visual indicator
    state_description: str              # Human-readable explanation
    
    # Model confidence (required fields - no defaults)
    model_certainty: float              # R-squared / confidence (0-1)
    data_quality: str                   # "HIGH" (12+ months), "MEDIUM" (3-12), "LOW" (<3)
    
    # Actionable insight (required field - no default)
    cost_of_inaction: str               # Human-readable consequence
    
    # === OPTIONAL FIELDS WITH DEFAULTS (must come after required fields) ===
    # SEMANTIC SPLIT: Separate risk from opportunity
    optimization_value: float = 0.0     # $ optimization opportunity (NOT risk) for healthy products
    
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
    # SEMANTIC FIX: Separate ACTUAL THREATS from OPTIMIZATION OPPORTUNITY
    # 
    # ACTUAL RISK = weighted_stockout_risk + weighted_price_risk + weighted_share_risk
    #   - Only include base_30day_risk if there are actual threat signals
    #
    # OPTIMIZATION VALUE = base_30day_risk (for healthy products with no threats)
    #   - This represents "money left on the table" not "money you'll lose"
    
    # Apply strategic bias weights to risk components
    weighted_stockout_risk = stockout_risk * inventory_weight
    weighted_price_risk = max(0, price_erosion_risk) * price_weight
    weighted_share_risk = share_erosion_risk * rank_weight
    
    # Actual threat total (without baseline)
    actual_threat_total = weighted_stockout_risk + weighted_price_risk + weighted_share_risk
    
    # Determine if there are ACTUAL threats (not just baseline opportunity)
    has_actual_threats = (
        stockout_risk > 0 or                    # Inventory threat
        price_erosion_risk > 0 or               # Price defense threat  
        share_erosion_risk > revenue * 0.02 or  # Meaningful velocity decline
        bsr_trend_30d > 0.15                    # Significant rank decline
    )
    
    # THIRTY_DAY_RISK = only actual threats
    # For healthy products without threats, this should be $0 (matching vectorized calculation)
    if has_actual_threats:
        # Include base risk only when there are actual threat signals
        thirty_day_risk = max(0, base_30day_risk + actual_threat_total)
    else:
        # No actual threats = $0 risk (optimization goes elsewhere)
        thirty_day_risk = 0.0
    
    # OPTIMIZATION VALUE = baseline opportunity for healthy products
    # This is what shows as "could gain" not "will lose"
    optimization_value = base_30day_risk if not has_actual_threats else 0.0
    
    # Daily burn rate for trending (use actual threat rate, not optimization)
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
        
        # For HOLD/HARVEST products, explain the optimization opportunity source
        if strategic_state == "HARVEST" and thirty_day_risk > 0:
            # Harvest products: optimization opportunity, not actual threat
            if stockout_risk > thirty_day_risk * 0.5:
                cost_of_inaction = f"${thirty_day_risk:,.0f} optimization opportunity from inventory management"
            elif abs(price_erosion_risk) > thirty_day_risk * 0.5 and price_erosion_risk < 0:
                # Negative price_erosion_risk = pricing opportunity
                cost_of_inaction = f"${thirty_day_risk:,.0f} optimization opportunity from pricing power (rank #{current_bsr} supports price test)"
            elif price_erosion_risk > thirty_day_risk * 0.5:
                cost_of_inaction = f"${thirty_day_risk:,.0f} optimization opportunity from pricing defense"
            elif share_erosion_risk > thirty_day_risk * 0.5:
                cost_of_inaction = f"${thirty_day_risk:,.0f} optimization opportunity from market share protection"
            else:
                cost_of_inaction = f"${thirty_day_risk:,.0f} optimization opportunity (stable position, test pricing/spend efficiency)"
        elif thirty_day_risk > 0:
            # Other HOLD products with risk - explain source
            if price_erosion_risk > thirty_day_risk * 0.5:
                cost_of_inaction = f"${thirty_day_risk:,.0f} at risk from pricing pressure"
            elif share_erosion_risk > thirty_day_risk * 0.5:
                cost_of_inaction = f"${thirty_day_risk:,.0f} at risk from velocity decline"
            else:
                cost_of_inaction = f"${thirty_day_risk:,.0f} projected optimization opportunity"
        else:
            cost_of_inaction = f"${thirty_day_risk:,.0f} projected optimization opportunity"
    
    # Default AI recommendation if no alert triggered
    if not ai_recommendation:
        if strategic_state == "HARVEST" and thirty_day_risk > 0:
            # Harvest products: explain optimization opportunity
            if abs(price_erosion_risk) > thirty_day_risk * 0.5 and price_erosion_risk < 0:
                # Negative price_erosion_risk = pricing opportunity
                ai_recommendation = f"Stable position. ${thirty_day_risk:,.0f} optimization opportunity from pricing power (rank #{current_bsr} supports price test)."
            elif price_erosion_risk > thirty_day_risk * 0.5:
                ai_recommendation = f"Stable position. ${thirty_day_risk:,.0f} optimization opportunity from pricing defense."
            elif share_erosion_risk > thirty_day_risk * 0.5:
                ai_recommendation = f"Stable position. ${thirty_day_risk:,.0f} optimization opportunity from market share protection."
            else:
                ai_recommendation = f"Stable position. ${thirty_day_risk:,.0f} optimization opportunity available (test pricing/spend efficiency)."
        elif thirty_day_risk > revenue * 0.2:
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
        optimization_value=optimization_value,  # NEW: Separate from risk
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
    # Use NEW Amazon OOS intelligence for better conquest detection
    amazon_oos_30 = row_data.get('oos_count_amazon_30', 0) or 0
    amazon_oos_90 = row_data.get('oos_count_amazon_90', 0) or 0
    amazon_owns_bb = row_data.get('buybox_is_amazon', False)
    amazon_pct_30 = row_data.get('bb_stats_amazon_30', 0) or 0
    
    # Amazon instability detection - when Amazon has OOS events, that's conquest gold
    amazon_unstable = amazon_oos_30 >= 3 or amazon_oos_90 >= 5
    amazon_is_competitor = amazon_owns_bb or amazon_pct_30 > 0.20
    
    # Calculate effective competitor vulnerability
    # Priority: Amazon OOS data > generic competitor OOS
    if amazon_unstable and amazon_is_competitor:
        # Amazon is having supply issues - major conquest opportunity!
        amazon_oos_rate = min(1.0, amazon_oos_30 / 10)  # Normalize to 0-1
        effective_oos_pct = max(competitor_oos_pct, amazon_oos_rate)
        target_is_amazon = True
    else:
        effective_oos_pct = competitor_oos_pct
        target_is_amazon = False
    
    if effective_oos_pct > 0.30 or amazon_unstable:
        target_competitor_asin = row_data.get('top_competitor_asin', '')
        effective_competitor_rev = competitor_monthly_rev if competitor_monthly_rev > 0 else revenue * 2
        
        # Amazon conquest is worth more (larger customer base)
        if target_is_amazon:
            conquest_revenue = effective_competitor_rev * effective_oos_pct * conquest_capture_rate * 1.5
        else:
            conquest_revenue = effective_competitor_rev * effective_oos_pct * conquest_capture_rate
        
        if not opportunity_type or conquest_revenue > price_optimization_gain:
            opportunity_type = "CONQUEST"
            opportunity_urgency = "HIGH" if effective_oos_pct > 0.50 or amazon_unstable else "MEDIUM"
            
            if target_is_amazon:
                ai_recommendation = f"🎯 AMAZON CONQUEST: Amazon went OOS {int(amazon_oos_30)}x in 30 days! Attack with aggressive ads on their keywords to capture ${conquest_revenue:,.0f} in 30-day revenue."
            elif target_competitor_asin:
                ai_recommendation = f"🎯 Conquest Opportunity: Competitor {target_competitor_asin} is {int(effective_oos_pct*100)}% OOS. Redirect ad budget to their keywords to capture ${conquest_revenue:,.0f} in 30-day revenue."
            else:
                ai_recommendation = f"🎯 Conquest Opportunity: Competitor {int(effective_oos_pct*100)}% OOS. Increase ad spend on category keywords to capture ${conquest_revenue:,.0f}."
    
    # ========== KEYWORD EXPANSION ANALYSIS ==========
    # Use NEW pre-calculated velocity from Keepa when available
    velocity_30d = row_data.get('velocity_30d')
    velocity_90d = row_data.get('velocity_90d')
    
    # Prefer Keepa's velocity calculation if available
    if velocity_30d is not None and velocity_30d < 0:
        # Keepa velocity: negative = improving (opposite of our convention)
        effective_velocity = velocity_30d / 100.0  # Already negative
    elif velocity_trend_90d < -0.05:
        effective_velocity = velocity_trend_90d
    else:
        effective_velocity = 0
    
    if effective_velocity < -0.05:  # Improving by more than 5%
        velocity_multiplier = 1.0 + abs(effective_velocity)
        keyword_expansion_gain = revenue * expansion_rate * velocity_multiplier
        
        if not opportunity_type:
            opportunity_type = "EXPAND"
            opportunity_urgency = "MEDIUM" if effective_velocity < -0.10 else "LOW"
            ai_recommendation = f"🚀 Expansion Opportunity: Momentum detected (velocity +{abs(effective_velocity)*100:.0f}%). Scale keyword coverage to capture ${keyword_expansion_gain:,.0f} in new revenue."
    
    # ========== S&S RETENTION OPPORTUNITY ==========
    # NEW: Detect Subscribe & Save upsell opportunities
    is_sns = row_data.get('is_sns', False)
    sns_opportunity = 0.0
    
    if is_sns and strategic_state in ["FORTRESS", "HARVEST"]:
        # Products with good fundamentals + S&S = subscription upsell opportunity
        # Estimate: 5-15% of revenue can be converted to recurring subscriptions
        sns_conversion_rate = 0.10 if "Growth" in strategic_bias else 0.05
        sns_opportunity = revenue * sns_conversion_rate * 12 / 30  # Monthly value / 30 days
        
        if not opportunity_type and sns_opportunity > 1000:
            opportunity_type = "SUBSCRIBE"
            opportunity_urgency = "MEDIUM"
            ai_recommendation = f"🔄 S&S Opportunity: Product eligible for Subscribe & Save. Push subscription messaging to capture ${sns_opportunity:,.0f}/mo in recurring revenue."
    
    # ========== TOTAL 30-DAY GROWTH ==========
    thirty_day_growth = price_optimization_gain + conquest_revenue + keyword_expansion_gain + sns_opportunity
    
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
    
    # Get actual DataFrame length for consistent Series creation
    n_rows = len(df)
    
    # Work with a copy to avoid modifying original (we'll return enriched version)
    result = df.copy()
    
    # === EXTRACT BASE METRICS (vectorized) - Ensure all Series have same length ===
    # Use proper DataFrame column access with fallbacks
    if 'weekly_sales_filled' in result.columns:
        revenue = result['weekly_sales_filled'].fillna(1000).values
    elif 'revenue_proxy' in result.columns:
        revenue = result['revenue_proxy'].fillna(1000).values
    else:
        revenue = np.full(n_rows, 1000.0, dtype=np.float32)
    
    if 'velocity_trend_30d' in result.columns:
        v30 = result['velocity_trend_30d'].fillna(0.0).values
    else:
        v30 = np.zeros(n_rows, dtype=np.float32)
    
    if 'velocity_trend_90d' in result.columns:
        v90 = result['velocity_trend_90d'].fillna(0.0).values
    else:
        v90 = np.zeros(n_rows, dtype=np.float32)
    
    if 'competitor_oos_pct' in result.columns:
        competitor_oos = result['competitor_oos_pct'].fillna(0.0).values
    elif 'outOfStockPercentage90' in result.columns:
        competitor_oos = result['outOfStockPercentage90'].fillna(0.0).values
    else:
        competitor_oos = np.zeros(n_rows, dtype=np.float32)
    
    if 'price_gap_vs_competitor' in result.columns:
        price_gap = result['price_gap_vs_competitor'].fillna(0.0).values
    elif 'price_delta' in result.columns:
        price_gap = result['price_delta'].fillna(0.0).values
    else:
        price_gap = np.zeros(n_rows, dtype=np.float32)
    
    # Normalize competitor OOS to 0-1 range (vectorized)
    competitor_oos = np.where(competitor_oos > 1, competitor_oos / 100, competitor_oos)
    
    # === NEW CRITICAL METRICS (2026-01-21) ===
    # Amazon OOS intelligence - for conquest detection
    amazon_oos_30 = result['oos_count_amazon_30'].fillna(0).values if 'oos_count_amazon_30' in result.columns else np.zeros(n_rows)
    amazon_oos_90 = result['oos_count_amazon_90'].fillna(0).values if 'oos_count_amazon_90' in result.columns else np.zeros(n_rows)
    
    # Amazon Buy Box ownership - competitive pressure indicator
    amazon_owns_bb = result['buybox_is_amazon'].fillna(False).values if 'buybox_is_amazon' in result.columns else np.zeros(n_rows, dtype=bool)
    bb_amazon_share_30 = result['bb_stats_amazon_30'].fillna(0).values if 'bb_stats_amazon_30' in result.columns else np.zeros(n_rows)
    
    # True seller count - more accurate than offer count
    seller_count = result['seller_count'].fillna(5).values if 'seller_count' in result.columns else np.full(n_rows, 5)
    
    # Supply chain status
    is_backorder = result['buybox_is_backorder'].fillna(False).values if 'buybox_is_backorder' in result.columns else np.zeros(n_rows, dtype=bool)
    
    # Subscribe & Save eligibility
    is_sns = result['is_sns'].fillna(False).values if 'is_sns' in result.columns else np.zeros(n_rows, dtype=bool)
    
    # Pre-calculated velocity from Keepa
    velocity_30d_keepa = result['velocity_30d'].fillna(0).values if 'velocity_30d' in result.columns else np.zeros(n_rows)
    velocity_90d_keepa = result['velocity_90d'].fillna(0).values if 'velocity_90d' in result.columns else np.zeros(n_rows)
    
    # Units source for confidence weighting
    has_amazon_units = result['units_source'].fillna('').values == 'amazon_monthly_sold' if 'units_source' in result.columns else np.zeros(n_rows, dtype=bool)
    
    # Amazon instability detection (conquest opportunity)
    amazon_unstable = (amazon_oos_30 >= 3) | (amazon_oos_90 >= 5)
    amazon_is_competitor = amazon_owns_bb | (bb_amazon_share_30 > 0.20)
    
    # Enhance competitor OOS with Amazon-specific intelligence
    amazon_oos_pct = np.minimum(1.0, amazon_oos_30 / 10.0)  # Normalize to 0-1
    enhanced_oos = np.maximum(competitor_oos, np.where(amazon_unstable & amazon_is_competitor, amazon_oos_pct, 0))
    
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
    
    # === RISK vs OPPORTUNITY CALCULATION (vectorized) ===
    # SEMANTIC FIX: Separate ACTUAL RISK (will lose) from OPTIMIZATION (could gain)
    # 
    # ACTUAL RISK = Things that will hurt you if you don't act:
    #   - Stockout (inventory runs out)
    #   - Price war (competitors undercutting, losing Buy Box)
    #   - Velocity crash (rank dropping fast)
    #
    # OPTIMIZATION = Things you could improve (not urgent):
    #   - Pricing power (you're underpriced, could raise)
    #   - Ad efficiency (reduce wasteful spend)
    #   - Position stable (no action needed)
    
    # 1. ACTUAL PRICING RISK - only if competitors are UNDERCUTTING you
    # price_gap > 0 means YOU are more expensive (risk of losing Buy Box)
    is_being_undercut = price_gap > 0.05  # Competitors 5%+ cheaper = real risk
    price_erosion = np.where(
        is_being_undercut,
        revenue * np.minimum(price_gap, 0.30) * 0.20 * price_weight,  # 20% of gap as risk
        0
    )
    
    # 2. ACTUAL VELOCITY RISK - only if rank is DECLINING meaningfully  
    # v90 > 0.10 means rank dropped 10%+ over 90 days = real concern
    is_declining_fast = v90 > 0.10  # More than 10% velocity decline
    share_erosion = np.where(
        is_declining_fast,
        revenue * 0.15 * np.minimum(v90, 0.50) * rank_weight,
        0
    )
    
    # 3. ACTUAL STOCKOUT RISK - only with real inventory data
    if 'days_until_stockout' in result.columns:
        days_stockout = result['days_until_stockout'].fillna(999).values
        stockout_risk_val = np.where(
            days_stockout < 30,
            revenue * 0.10 * (30 - days_stockout) / 30 * inventory_weight,
            0
        )
    else:
        stockout_risk_val = np.zeros_like(revenue)  # Zero when no actual data
    
    # ACTUAL 30-DAY RISK = Sum of real threats (NOT fabricated baseline)
    thirty_day_risk = np.maximum(0, price_erosion + share_erosion + stockout_risk_val)
    
    # For Root Cause Analysis display (components should sum to total)
    component_total = price_erosion + share_erosion + stockout_risk_val
    # Only normalize if we have actual risk components
    if np.any(component_total > 0):
        scale_factor = np.where(component_total > 0, np.where(thirty_day_risk > 0, thirty_day_risk / component_total, 1), 1)
        price_erosion = price_erosion * scale_factor
        share_erosion = share_erosion * scale_factor
        stockout_risk_val = stockout_risk_val * scale_factor
    
    # === EXTRACT RANK (needed for predictive state) ===
    if 'sales_rank_filled' in result.columns:
        rank = result['sales_rank_filled'].fillna(5000).values
    elif 'sales_rank' in result.columns:
        rank = result['sales_rank'].fillna(5000).values
    else:
        rank = np.full(n_rows, 1000, dtype=np.float32)
    
    # === PREDICTIVE STATE (SEMANTIC FIX) ===
    # States now have clear meanings:
    # - DEFEND: Real threat - you'll LOSE revenue without action
    # - REPLENISH: Stockout imminent - act now or lose sales
    # - EXPLOIT: Competitor weakness - opportunity to gain
    # - GROW: Position strong - pricing power opportunity
    # - STABLE: No action needed - healthy
    has_actual_stockout_data = 'days_until_stockout' in result.columns
    
    # Determine if there's ACTUAL risk (not just optimization headroom)
    has_actual_risk = thirty_day_risk > revenue * 0.02  # More than 2% of revenue at actual risk
    
    # NEW: Backorder status is URGENT supply crisis
    has_supply_crisis = is_backorder
    
    predictive_state = np.where(
        has_supply_crisis, "REPLENISH",  # BACKORDER = URGENT supply action
        np.where(
            has_actual_stockout_data & (stockout_risk_val > revenue * 0.05), "REPLENISH",  # Stockout imminent
            np.where(
                amazon_owns_bb & is_declining_fast, "DEFEND",  # Amazon 1P + declining = serious threat
                np.where(
                    is_declining_fast & has_actual_risk, "DEFEND",  # Velocity crash with real impact
                    np.where(
                        is_being_undercut & has_actual_risk, "DEFEND",  # Price war with real impact
                        np.where(
                            amazon_unstable & amazon_is_competitor, "EXPLOIT",  # Amazon OOS = premium conquest
                            np.where(
                                enhanced_oos > 0.20, "EXPLOIT",  # Competitor OOS - conquest opportunity
                                np.where(
                                    (rank <= 100) & (v90 <= 0.05), "GROW",  # Market leader, stable - pricing power
                                    "STABLE"  # Healthy position, no urgent action
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    
    # NOTE: optimization_value will be calculated AFTER growth components are computed
    # See below after sns_gain is calculated
    
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
    
    # Conquest opportunity (competitor OOS - enhanced with Amazon intelligence)
    # Amazon conquest is worth 1.5x more (larger customer base, higher intent)
    amazon_conquest_bonus = np.where(amazon_unstable & amazon_is_competitor, 1.5, 1.0)
    conquest_gain = np.where(
        (enhanced_oos > 0.30) & growth_validated,
        revenue * 2 * enhanced_oos * conquest_rate * amazon_conquest_bonus,
        0
    )
    
    # Expansion opportunity (velocity improving)
    expansion_gain = np.where(
        (v90 < -0.05) & growth_validated,
        revenue * 0.08 * (1.0 + np.abs(v90)),
        0
    )
    
    # === NEW: PRICE POWER OPPORTUNITY ===
    # For market leaders (top 100 rank, high revenue), assume 3-5% price increase potential
    # This is a realistic growth signal that doesn't require rare competitor data
    is_market_leader = (rank <= 100) & (revenue >= 5000)  # Top 100 rank, $5K+/week
    is_stable_position = (v90 <= 0.10)  # Not declining significantly
    
    # Price power: stable leaders can test 3-5% price increases
    # Conservative estimate: 4% price increase captures 70% of the lift (30% volume loss)
    price_power_rate = 0.04 * 0.70  # 4% price * 70% retention = 2.8% net gain
    price_power_gain = np.where(
        is_market_leader & is_stable_position & growth_validated,
        revenue * price_power_rate,
        0
    )
    
    # === NEW: REVIEW MOAT OPPORTUNITY ===
    # Products with strong review counts can command premium pricing
    if 'review_count' in result.columns:
        review_count = result['review_count'].fillna(0).values
    else:
        review_count = np.zeros(n_rows, dtype=np.float32)
    
    has_strong_reviews = review_count >= 500  # 500+ reviews = pricing power
    review_moat_gain = np.where(
        has_strong_reviews & is_stable_position & growth_validated,
        revenue * 0.02,  # 2% potential from review moat
        0
    )
    
    # === NEW: S&S SUBSCRIPTION OPPORTUNITY ===
    # Products with S&S eligibility and strong fundamentals = subscription upsell
    sns_conversion_rate = np.where(np.char.find(np.full(n_rows, strategic_bias), 'Growth') >= 0, 0.10, 0.05)
    sns_gain = np.where(
        is_sns & growth_validated & is_stable_position,
        revenue * sns_conversion_rate * 12 / 30,  # Monthly subscription value / 30 days
        0
    )
    
    # Total growth (include new opportunity types)
    thirty_day_growth = np.where(
        growth_validated,
        price_lift_gain + conquest_gain + expansion_gain + price_power_gain + review_moat_gain + sns_gain,
        0
    )
    
    # Opportunity type (priority: CONQUEST > PRICE_LIFT > PRICE_POWER > SUBSCRIBE > REVIEW_MOAT > EXPAND)
    opportunity_type = np.where(
        conquest_gain > 0, "CONQUEST",
        np.where(
            price_lift_gain > 0, "PRICE_LIFT",
            np.where(
                price_power_gain > 0, "PRICE_POWER",
                np.where(
                    sns_gain > 1000, "SUBSCRIBE",  # Only flag if $1K+ opportunity
                    np.where(
                        review_moat_gain > 0, "REVIEW_MOAT",
                        np.where(
                            expansion_gain > 0, "EXPAND",
                            ""
                        )
                    )
                )
            )
        )
    )
    
    # === OPTIMIZATION VALUE (for STABLE/GROW products) ===
    # This is UPSIDE POTENTIAL for healthy products = sum of actual opportunity types
    # NOT a flat percentage - uses real calculated opportunities
    optimization_value = np.where(
        thirty_day_risk < revenue * 0.02,  # No significant actual risk = healthy product
        # Sum all opportunity types (the actual $ values, not just growth flag)
        price_lift_gain + price_power_gain + review_moat_gain + sns_gain,  # Exclude conquest (that's thirty_day_growth)
        0  # Products with real risk don't get optimization value
    )
    
    # === COMBINED OPPORTUNITY ALPHA ===
    opportunity_alpha = thirty_day_risk + thirty_day_growth
    
    # === STRATEGIC STATE (vectorized rule-based classification) ===
    # Uses same logic as LLM classifier but vectorized for speed
    # Extract additional metrics if available
    bb_share = result['amazon_bb_share'].fillna(0.5).values if 'amazon_bb_share' in result.columns else np.full(n_rows, 0.5)
    new_offer_count = result['new_offer_count'].fillna(5).values if 'new_offer_count' in result.columns else np.full(n_rows, 5)
    
    # Normalize bb_share if > 1 (assume percentage)
    bb_share = np.where(bb_share > 1, bb_share / 100, bb_share)
    
    # Use true seller count if available (more accurate than offer count)
    effective_seller_count = np.where(seller_count > 0, seller_count, new_offer_count)
    
    # Strategic state classification (priority order)
    # TERMINAL: Very low BB (<10%), significant risk, negative velocity
    is_terminal = (bb_share < 0.10) & (v90 > 0.30)
    
    # DISTRESS: Supply crisis (backorder) OR Low BB (<40%) + declining velocity
    is_distress = has_supply_crisis | ((bb_share < 0.40) & (v90 > 0.20) & (~is_terminal))
    
    # TRENCH_WAR: Amazon 1P owns BB OR Contested BB (30-60%) with many competitors
    is_amazon_trench = amazon_owns_bb & (bb_share < 0.80)  # Amazon 1P = competitive pressure
    is_trench = (is_amazon_trench | ((bb_share >= 0.30) & (bb_share < 0.60) & (effective_seller_count >= 8))) & (~is_distress) & (~is_terminal)
    
    # FORTRESS: High BB (>80%), stable or improving velocity, no Amazon 1P
    is_fortress = (bb_share >= 0.80) & (v90 <= 0.10) & (~amazon_owns_bb)
    
    # HARVEST: Everything else (stable, good BB)
    strategic_state = np.where(
        is_terminal, "TERMINAL",
        np.where(
            is_distress, "DISTRESS",
            np.where(
                is_trench, "TRENCH_WAR",
                np.where(
                    is_fortress, "FORTRESS",
                    "HARVEST"
                )
            )
        )
    )
    
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
    result['optimization_value'] = np.asarray(optimization_value, dtype=np.float32)  # NEW: Upside potential (not risk)
    result['opportunity_alpha'] = np.asarray(opportunity_alpha, dtype=np.float32)
    result['predictive_state'] = pd.Categorical(predictive_state, categories=["STABLE", "GROW", "DEFEND", "EXPLOIT", "REPLENISH"])  # UPDATED categories
    result['opportunity_type'] = pd.Categorical(opportunity_type, categories=["", "PRICE_LIFT", "CONQUEST", "EXPAND", "PRICE_POWER", "REVIEW_MOAT", "SUBSCRIBE"])
    result['growth_validated'] = np.asarray(growth_validated, dtype=bool)
    result['price_erosion_risk'] = np.asarray(price_erosion, dtype=np.float32)
    result['share_erosion_risk'] = np.asarray(share_erosion, dtype=np.float32)
    result['stockout_risk'] = np.asarray(stockout_risk_val, dtype=np.float32)
    result['strategic_state'] = pd.Categorical(strategic_state, categories=["FORTRESS", "HARVEST", "TRENCH_WAR", "DISTRESS", "TERMINAL"])
    
    # NEW: Store Amazon 1P and S&S flags for dashboard display
    result['amazon_1p_competitor'] = amazon_owns_bb
    result['sns_eligible'] = is_sns
    result['amazon_unstable'] = amazon_unstable
    
    # Model certainty based on data quality
    # BOOSTED when we have Amazon's actual monthly sold data
    if 'data_weeks' in result.columns:
        data_weeks = result['data_weeks'].fillna(4).values
    else:
        data_weeks = np.full(n_rows, 4.0, dtype=np.float32)
    
    # Base certainty from data weeks
    base_certainty = 0.40 + (data_weeks / 48) * 0.50
    
    # Boost certainty if we have Amazon's actual unit data
    amazon_data_boost = np.where(has_amazon_units, 0.10, 0.0)
    
    result['model_certainty'] = np.clip(base_certainty + amazon_data_boost, 0.40, 0.95).astype(np.float32)
    
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
    
    def __init__(self, use_llm: bool = True, timeout: float = 10.0, strategic_bias: str = "Balanced Defense",
                 enable_triggers: bool = False, enable_network: bool = False):
        """
        Initialize the unified AI engine.

        Args:
            use_llm: Whether to use LLM classification (default True)
            timeout: Timeout for LLM calls in seconds
            strategic_bias: User's strategic focus (Profit/Balanced/Growth)
            enable_triggers: Enable trigger event detection (requires historical data)
            enable_network: Enable network intelligence (requires Supabase connection)
        """
        self.use_llm = use_llm
        self.timeout = timeout
        self.strategic_bias = strategic_bias
        self.enable_triggers = enable_triggers
        self.enable_network = enable_network
        self._client = None
        self._model = None

        # Initialize network intelligence if enabled
        if self.enable_network:
            try:
                from src.network_intelligence import NetworkIntelligence
                from supabase import create_client
                import os

                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

                if supabase_url and supabase_key:
                    supabase = create_client(supabase_url, supabase_key)
                    self.network_intel = NetworkIntelligence(supabase)
                else:
                    self.network_intel = None
                    if STREAMLIT_AVAILABLE:
                        import streamlit as st
                        # Try Streamlit secrets
                        try:
                            supabase_url = st.secrets.get("supabase", {}).get("url") or st.secrets.get("url")
                            supabase_key = st.secrets.get("supabase", {}).get("service_key") or st.secrets.get("key")
                            if supabase_url and supabase_key:
                                supabase = create_client(supabase_url, supabase_key)
                                self.network_intel = NetworkIntelligence(supabase)
                        except:
                            pass
            except Exception as e:
                self.network_intel = None
        else:
            self.network_intel = None
    
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
        # SEMANTIC SPLIT: thirty_day_risk = actual threats, optimization_value = opportunity
        strategic_brief.thirty_day_risk = predictive.thirty_day_risk
        strategic_brief.optimization_value = predictive.optimization_value  # NEW: separate from risk
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

        # === STEP 5: TRIGGER DETECTION (Optional Enhancement) ===
        if self.enable_triggers and 'historical_df' in row_data:
            try:
                from src.trigger_detection import detect_trigger_events

                asin = row_data.get('asin', '')
                historical_df = row_data['historical_df']
                competitors_df = row_data.get('competitors_df', pd.DataFrame())

                if not historical_df.empty:
                    triggers = detect_trigger_events(
                        asin=asin,
                        df_historical=historical_df,
                        df_competitors=competitors_df
                    )

                    # Add trigger events to reasoning if detected
                    if triggers:
                        trigger_summary = "\n\n🎯 Trigger Events Detected:\n"
                        for t in triggers[:3]:  # Top 3 most severe
                            severity_emoji = "🔴" if t.severity >= 8 else "🟡" if t.severity >= 6 else "🟢"
                            trigger_summary += f"{severity_emoji} {t.event_type}: {t.metric_name} changed {t.delta_pct:+.1f}% (severity {t.severity}/10)\n"

                        strategic_brief.reasoning += trigger_summary

                        # Track trigger types in signals_detected
                        for t in triggers[:5]:
                            if t.event_type not in strategic_brief.signals_detected:
                                strategic_brief.signals_detected.append(t.event_type)
            except Exception as e:
                # Silently skip trigger detection if it fails
                pass

        # === STEP 6: NETWORK INTELLIGENCE (Optional Enhancement) ===
        if self.enable_network and self.network_intel:
            try:
                category_id = row_data.get('category_id')
                asin = row_data.get('asin', '')
                price = float(row_data.get('price', row_data.get('filled_price', row_data.get('buy_box_price', 0))))

                if category_id and price > 0:
                    # Get category benchmarks
                    benchmarks = self.network_intel.get_category_benchmarks(category_id)

                    if benchmarks and benchmarks.get('median_price'):
                        # Calculate competitive position
                        position = self.network_intel.get_competitive_position(asin, category_id)

                        # Build network intelligence summary
                        network_summary = "\n\n📊 Network Intelligence:\n"

                        # Price vs category median
                        median_price = benchmarks['median_price']
                        price_vs_median = ((price / median_price) - 1) * 100
                        network_summary += f"• Your price: ${price:.2f} ({price_vs_median:+.1f}% vs category median of ${median_price:.2f})\n"

                        # Review count vs category
                        review_count = int(row_data.get('review_count', 0))
                        median_reviews = benchmarks.get('median_review_count', 0)
                        if median_reviews > 0:
                            review_vs_median = ((review_count / median_reviews) - 1) * 100
                            network_summary += f"• Reviews: {review_count:,} ({review_vs_median:+.1f}% vs median of {int(median_reviews):,})\n"

                        # Competitive advantages
                        if position and position.get('competitive_advantages'):
                            advantages = position['competitive_advantages'][:2]  # Top 2
                            if advantages:
                                network_summary += f"• Advantages: {', '.join(advantages)}\n"

                        # Add network summary to reasoning
                        strategic_brief.reasoning += network_summary
            except Exception as e:
                # Silently skip network intelligence if it fails
                pass

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
    timeout: float = 10.0,
    strategic_bias: str = "Balanced Defense"
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
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        
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
    
    # Get strategic bias instructions
    bias_instructions = _get_strategic_bias_instructions(strategic_bias)
    
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are ShelfGuard's AI strategist analyzing Amazon portfolio health. Generate a clear, actionable executive brief.

{bias_instructions}

## ANALYSIS FRAMEWORK

1. **REVENUE HEALTH** (Most important)
   - Total Monthly Revenue → Portfolio scale
   - At-Risk Revenue (30-day) → Defensive priority (spread across ALL products, not just 1-2)
   - Growth Opportunity → Offensive priority

2. **PRODUCT STATUS DISTRIBUTION**
   - FORTRESS/HARVEST = Healthy (extract value)
   - TRENCH_WAR = Defend (maintain position, especially if Amazon 1P competitor detected)
   - DISTRESS = Fix (check for backorder status, supply issues)
   - TERMINAL = Exit

3. **PREDICTIVE STATES** (Action urgency)
   - EXPLOIT = Amazon/competitor supply unstable = conquest opportunity NOW
   - DEFEND = Velocity declining or price war = protect revenue
   - REPLENISH = Backorder or stockout imminent = supply chain URGENT
   - GROW = Market leader position stable = test price increases
   - STABLE = No urgent action needed

3. **RISK INTERPRETATION (CRITICAL)**
   - "At-Risk Revenue" is the SUM of risk across the ENTIRE portfolio
   - Do NOT attribute total portfolio risk to individual products unless data explicitly shows individual risk
   - If "Top Risk Products" are listed, use THOSE specific amounts per product
   - If no individual risk data, describe portfolio-wide actions (pricing strategy, inventory review, etc.)

## BRIEF REQUIREMENTS

FORMAT YOUR RESPONSE AS:
1. **Status Line**: One sentence portfolio health summary with $ amounts
2. **Priority Action**: The single most important action with specific $ impact and ASINs (if provided)
3. **Secondary Focus**: A DIFFERENT strategic insight - NOT just repeating the priority action. Examples:
   - Market structure analysis (e.g., "72 competitors = fragmented market, focus on top 3 SKUs")
   - Competitive dynamics (e.g., "Competitor pricing pressure on mid-tier products suggests defensive pricing needed")
   - Portfolio composition (e.g., "70% of revenue from top 5 SKUs = concentration risk, diversify")
   - Data quality insights (e.g., "Limited historical data on new launches = prioritize data collection")
   - Strategic positioning (e.g., "Market leader position enables premium tier testing")
4. **Opportunity Alpha**: Strategic interpretation of the total addressable value - NOT just repeating the number. Examples:
   - "Total addressable value of $520.9K represents 15% of portfolio revenue at stake — defensive actions can preserve $445.3K while growth initiatives capture $75.7K"
   - "Combined opportunity of $520.9K: $445.3K defensive (velocity decline across 15 products) + $75.7K offensive (price power on top-30 rank SKUs)"

RULES:
- ONLY reference ASINs that appear in "Top Risk Products" or "Top Growth Products" sections
- If no specific products are listed, give portfolio-wide recommendations (not fake ASINs)
- Do NOT invent or fabricate ASINs - if you don't have specific ASIN data, say "review portfolio" instead
- QUANTIFY: Always include $ impact of recommended actions
- Secondary Focus MUST be different from Priority Action - provide strategic context, not tactical repetition
- Opportunity Alpha MUST interpret the number strategically, not just restate it
- MAX 150 WORDS (increased to allow for more sophisticated insights)

## GROWTH OPPORTUNITY TYPES (Use these in recommendations)
- PRICE_POWER: Top-100 rank products can test 3-5% price increases
- REVIEW_MOAT: 500+ reviews = pricing power, test premium positioning
- CONQUEST: Competitor out of stock = capture their customers
- AMAZON_CONQUEST: Amazon 1P supply unstable (3+ OOS events) = attack their keywords
- SUBSCRIBE: Products with S&S eligibility = push subscription messaging for recurring revenue
- EXPAND: Improving velocity = increase ad spend to accelerate

## NEW INTELLIGENCE SIGNALS (High-value insights)
- **Amazon 1P Competition**: If Amazon owns Buy Box, brand faces pricing pressure. Recommend differentiation.
- **Amazon Supply Instability**: Amazon going OOS frequently = prime conquest opportunity.
- **Backorder Status**: Products in backorder need URGENT supply chain action.
- **True Seller Count**: More accurate than offer count for competition assessment.
- **S&S Eligibility**: Subscription opportunity for customer retention.
- **Amazon Actual Units**: When "units_source: AMAZON ACTUAL" is shown, data confidence is HIGH.
- **Pack Size Arbitrage**: Compare price-per-unit across pack sizes for pricing optimization.

## SOPHISTICATED RECOMMENDATIONS (Not kindergarten advice)
GOOD - Specific and quantified:
- "Test 5% price increase on top 3 SKUs (rank #29-30) — $X/mo upside if volume holds"
- "Review variety pack pricing — variety packs at $28 vs singles at $17 may have elasticity opportunity"
- "Optimize ad spend allocation — shift budget from #216 rank products to top-30 performers"
- "A/B test Prime-exclusive pricing on [ASIN] — category leaders often capture 8% premium"

BAD - Generic and vague:
- "Maintain brand visibility" (meaningless)
- "Monitor competitor activity" (not actionable)
- "Ensure marketing efforts maintain market share" (kindergarten level)

## MARKET LEADER ADVICE (For 50%+ market share)
When brand is market leader, focus on:
1. Price optimization testing (you have pricing power)
2. Variety pack vs singles pricing arbitrage
3. Premium tier product launches
4. Ad efficiency (reduce waste on already-dominant SKUs)

GOOD PATTERNS:
- (For market leaders): "Price power opportunity: Top-30 rank with 50%+ share supports 4-6% price tests on [ASIN] — $X/mo upside"
- "Optimize ad spend: Reduce defensive spend on dominant SKUs, reallocate to variety packs with lower share"
- (Only if specific ASINs provided): "Priority: [ACTUAL_ASIN] has $X/mo growth potential from [SPECIFIC_REASON]"

BAD PATTERNS:
- "Address stockout risk for ASIN B0XYZ" (invented ASIN)
- "Maintain brand visibility" (useless advice)
- Attributing entire portfolio risk to 1-2 products when no individual data provided"""
                    },
                    {
                        "role": "user",
                        "content": f"""Here's the current portfolio status:

{portfolio_summary}

Generate an executive strategic brief. Focus on the highest-impact actions. If Risk is significant, prioritize defense. If Risk is low, prioritize growth. Always be specific and quantify recommendations."""
                    }
                ],
                max_tokens=250,
                temperature=0.5  # Lower temperature for more consistent output
            ),
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def generate_portfolio_brief_sync(
    portfolio_summary: str,
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None,
    strategic_bias: str = "Balanced Defense"
) -> Optional[str]:
    """
    Synchronous wrapper for generate_portfolio_brief.
    
    For use in Streamlit where async/await can be tricky.
    Uses the same AI engine as product-level classification.
    
    Args:
        portfolio_summary: Portfolio metrics summary string
        client: Optional AsyncOpenAI client (will create if not provided)
        model: Optional model name (defaults to gpt-4o-mini)
        strategic_bias: User's strategic focus (Profit/Balanced/Growth)
        
    Returns:
        Strategic brief string, or None if LLM unavailable
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_portfolio_brief(portfolio_summary, client, model, strategic_bias=strategic_bias)
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
