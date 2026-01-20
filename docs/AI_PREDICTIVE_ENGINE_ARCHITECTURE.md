# ShelfGuard AI & Predictive Intelligence Engine Architecture
## Unified Strategic Intelligence System v2.0

**Status:** Architecture Complete - Ready for Implementation
**Author:** Technical Architecture Team
**Date:** 2026-01-19
**Related:** INSIGHT_ENGINE_REFACTOR_PLAN.md

---

## Executive Summary

This document defines the **unified architecture** that integrates:
1. **Strategic Classification Engine** (LLM-based state analysis)
2. **Predictive Intelligence Layer** (30-day risk forecasting)
3. **Trigger Event Detection** (causal event system)
4. **Insight Generation Pipeline** (high-fidelity recommendations)

**Key Architectural Principles:**
- **Single Data Flow**: Trigger Events → Strategic Analysis → Predictive Forecast → Actionable Insights
- **Causal Reasoning**: Every prediction cites specific trigger events
- **Quantified Outputs**: All forecasts include specific dollar amounts
- **Quality Gates**: Validate outputs before storage
- **Feedback Loops**: Track accuracy to improve models

---

## Part 1: Unified System Architecture

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAW KEEPA DATA                                  │
│  (90 days historical: price, BSR, reviews, inventory, buybox, etc.)    │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   TRIGGER EVENT DETECTION                                │
│  - Inventory drops (<5 units)                                           │
│  - Price wars (3+ drops in 7d)                                          │
│  - Review velocity spikes                                               │
│  - BuyBox share collapse                                                │
│  - Rank degradation (30%+ worse)                                        │
│  - Competitor stockouts                                                 │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STRATEGIC CLASSIFICATION (LLM)                              │
│  Input: Keepa metrics + Trigger events                                 │
│  Output: Product Status (CRITICAL/OPPORTUNITY/WATCH/STABLE)            │
│  Model: GPT-4o-mini with structured prompt                             │
│  Validation: Quality gate (must cite triggers + quantify)              │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            PREDICTIVE INTELLIGENCE LAYER                                 │
│  - 30-day risk forecast (velocity-adjusted)                             │
│  - Stockout risk calculation                                            │
│  - Price erosion modeling                                               │
│  - Market share erosion tracking                                        │
│  - Growth opportunity detection                                         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              INSIGHT GENERATION (LLM)                                    │
│  Input: Status + Triggers + Predictions + Competitive context          │
│  Output: Specific recommendation with $ amounts                         │
│  Validation: Must have $ amount + causal reason + time horizon         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   STRATEGIC INSIGHTS DATABASE                            │
│  - Store validated insights                                             │
│  - Track trigger events                                                 │
│  - Monitor user actions                                                 │
│  - Measure prediction accuracy                                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ACTION QUEUE UI                                       │
│  - Filtered by priority (hide STABLE)                                  │
│  - Visual distinction (profit vs risk)                                 │
│  - User feedback collection                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Integration Map

```python
# CURRENT ARCHITECTURE (utils/ai_engine.py)
class StrategicBrief:
    strategic_state: str          # FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL
    confidence: float
    reasoning: str
    recommended_action: str
    thirty_day_risk: float        # Predictive intelligence
    thirty_day_growth: float      # Growth intelligence

# NEW ARCHITECTURE (integrated with insights)
class UnifiedIntelligence:
    """
    Unified output combining strategic classification, predictive forecasting,
    and actionable insights.
    """
    # === IDENTITY ===
    asin: str
    timestamp: datetime

    # === TRIGGER EVENTS (NEW) ===
    trigger_events: List[TriggerEvent]       # Detected market changes
    primary_trigger: str                      # Most important event

    # === STRATEGIC CLASSIFICATION (EXISTING - ENHANCED) ===
    product_status: ProductStatus             # NEW unified status enum
    strategic_state: StrategicState           # Keep for backward compatibility
    confidence: float
    reasoning: str                            # Now must cite trigger events

    # === PREDICTIVE INTELLIGENCE (EXISTING - ENHANCED) ===
    thirty_day_risk: float                    # Risk forecast
    thirty_day_growth: float                  # Growth forecast
    net_expected_value: float                 # NEW: risk-adjusted upside

    # Risk components
    price_erosion_risk: float
    share_erosion_risk: float
    stockout_risk: float

    # === ACTIONABLE INSIGHT (NEW) ===
    recommendation: str                       # Specific action with $ amount
    projected_upside_monthly: float           # NEW: quantified opportunity
    downside_risk_monthly: float              # NEW: quantified risk
    action_type: str                          # PROFIT_CAPTURE or RISK_MITIGATION
    time_horizon_days: int                    # When to act

    # === VALIDATION & QUALITY ===
    validation_passed: bool
    validation_errors: List[str]
    data_quality: str                         # HIGH, MEDIUM, LOW

    # === USER INTERACTION ===
    user_dismissed: bool
    user_feedback: Optional[str]
```

---

## Part 2: Enhanced Strategic Classification Engine

### 2.1 Integration with Trigger Events

**OLD PROMPT (Current ai_engine.py):**
```python
# ❌ No context about WHY things are happening
"""
Analyze this product:
- Revenue: $5,000
- BSR: 15,000
- Price: $24.99
What strategic state is it in?
"""
```

**NEW PROMPT (Trigger Event Aware):**
```python
def build_enhanced_strategic_prompt(
    row_data: Dict[str, Any],
    trigger_events: List[TriggerEvent],
    competitor_context: Dict[str, Any]
) -> str:
    """
    Build LLM prompt enriched with trigger events and competitive context.

    This replaces the current STRATEGIST_SYSTEM_PROMPT with causal intelligence.
    """

    # Format trigger events with severity ranking
    trigger_context = "\n".join([
        f"{i+1}. [{e.severity}/10] {e.event_type.upper()}: "
        f"{e.metric_name} changed {e.delta_pct:+.1f}% "
        f"(from {e.baseline_value} → {e.current_value})"
        for i, e in enumerate(trigger_events[:5])  # Top 5 events
    ]) if trigger_events else "No significant trigger events detected in last 30 days."

    prompt = f"""You are ShelfGuard's Senior CPG Strategist with 20 years of Amazon experience.

PRODUCT CURRENT STATE (ASIN: {row_data['asin']}):
- Monthly Revenue: ${row_data['monthly_revenue']:,.0f}
- Current Price: ${row_data['price']:.2f}
- Sales Rank (BSR): {row_data['bsr']:,}
- Review Count: {row_data['review_count']} ({row_data.get('avg_rating', 0):.1f}★)
- BuyBox Share: {row_data.get('buybox_share', 0):.0%}
- Estimated Margin: {row_data.get('margin_pct', 0):.1f}%
- Inventory Level: {row_data.get('inventory_count', 'Unknown')} units

VELOCITY TRENDS:
- 30-Day Revenue Change: {row_data.get('velocity_30d', 0):+.1%}
- 90-Day Revenue Change: {row_data.get('velocity_90d', 0):+.1%}
- BSR Trend (30d): {row_data.get('bsr_velocity_30d', 0):+.1f}% (negative = improving rank)

COMPETITIVE LANDSCAPE:
- Category Avg Price: ${competitor_context.get('avg_price', 0):.2f}
- Your Price Positioning: {((row_data['price'] / competitor_context.get('avg_price', row_data['price'])) - 1) * 100:+.1f}%
- Category Avg Reviews: {competitor_context.get('avg_reviews', 0):.0f}
- Your Review Advantage: {((row_data['review_count'] / max(1, competitor_context.get('avg_reviews', 1))) - 1) * 100:+.1f}%
- Top 3 Competitor Avg Inventory: {competitor_context.get('top3_inventory_avg', 0):.0f} units
- Competitor Out-of-Stock Rate: {competitor_context.get('oos_rate', 0):.1%}

DETECTED TRIGGER EVENTS (Last 30 Days - by Severity):
{trigger_context}

TASK: Classify this product into ONE of 5 Strategic States.

## Strategic States

1. **FORTRESS** - Dominant market position, pricing power
   - Buy Box >80%, Margin >15%, Rank stable/improving
   - Example: Market leader with strong moat

2. **HARVEST** - Cash cow, maximize extraction
   - Stable rank, good margins, premium price holding
   - Example: Mature product with loyal customer base

3. **TRENCH_WAR** - Competitive battle, defend share
   - Price pressure, rank volatility, BuyBox rotation
   - Example: Category under attack from new entrants

4. **DISTRESS** - Needs intervention, value at risk
   - Margin compression (<10%), velocity decay, review stagnation
   - Example: Product losing ground to competition

5. **TERMINAL** - Exit required, liquidation mode
   - Negative margins, severe velocity decay, market collapsed
   - Example: Dying category or fundamentally broken product

## Output Requirements

You MUST provide a JSON response with these fields:

{{
    "product_status": "<one of: stable_fortress, stable_cash_cow, opportunity_price_power, watch_price_war, critical_margin_collapse, etc.>",
    "strategic_state": "<one of: FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL>",
    "confidence": <0.0-1.0>,
    "reasoning": "<2-3 sentences explaining WHY, citing specific trigger events>",
    "recommended_action": "<Specific action with timeframe>",
    "trigger_events_cited": ["<event_type>", "<event_type>"],
    "risk_assessment": "<Assessment of downside risk>",
    "opportunity_assessment": "<Assessment of upside opportunity>"
}}

## Critical Rules

1. **Cite Trigger Events**: Your reasoning MUST reference specific trigger events from the list above
2. **Be Specific**: Use actual numbers from the data (e.g., "inventory dropped from 47 to 3 units")
3. **No Generic Advice**: Don't say "monitor closely" - say "if BSR crosses 20,000, reduce ad spend by 20%"
4. **Map to New Status**: Use the granular product_status taxonomy (13 options), not just the 5 strategic states
5. **Confidence Calibration**:
   - >0.85 = Strong signal, multiple confirming triggers
   - 0.70-0.85 = Moderate signal, some ambiguity
   - <0.70 = Weak signal, conflicting indicators

## Product Status Mapping

Map your strategic assessment to ONE of these granular statuses:

**CRITICAL (Priority 100):**
- critical_margin_collapse: Margin <5% and declining
- critical_inventory_risk: OOS in <7 days
- critical_buybox_loss: BuyBox <30%
- critical_velocity_crash: Revenue -50%+ in 30d

**OPPORTUNITY (Priority 75):**
- opportunity_price_power: Can raise price (competitor weakness + review advantage)
- opportunity_ad_waste: Can cut ad spend 15%+ without rank impact
- opportunity_review_gap: Launch Vine (reviews <50% of category avg)
- opportunity_competitor_weakness: Rival OOS or pricing high

**WATCH (Priority 50):**
- watch_new_competitor: New ASIN with BSR <10k entered
- watch_price_war: 3+ price drops in 7d detected
- watch_seasonal_anomaly: Unusual pattern vs historical
- watch_rank_volatility: BSR variance >50%

**STABLE (Priority 0 - filter from default view):**
- stable_fortress: Market leader, defended position
- stable_cash_cow: Consistent revenue, low volatility
- stable_niche: Small but profitable

Generate the classification now.
"""

    return prompt
```

### 2.2 Refactored Classification Function

**File:** `utils/ai_engine_v2.py`

```python
async def analyze_product_with_intelligence(
    asin: str,
    row_data: Dict[str, Any],
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame,
    client: Optional[AsyncOpenAI] = None,
    strategic_bias: str = "Balanced Defense"
) -> UnifiedIntelligence:
    """
    UNIFIED INTELLIGENCE PIPELINE

    Combines trigger detection, strategic classification, predictive forecasting,
    and insight generation into a single output.

    This is the NEW entry point that replaces the old separate calls to:
    - analyze_strategy_with_llm()
    - calculate_predictive_alpha()
    - generate_resolution_cards()

    Args:
        asin: Product ASIN
        row_data: Current product metrics
        df_historical: 90-day historical data for this ASIN
        df_competitors: Current competitor data in same category
        client: OpenAI client (created if not provided)
        strategic_bias: User preference (Profit/Balanced/Growth)

    Returns:
        UnifiedIntelligence object with complete strategic + predictive + actionable data
    """

    # ========== STEP 1: TRIGGER EVENT DETECTION ==========
    from src.trigger_detection import detect_trigger_events

    trigger_events = detect_trigger_events(
        asin=asin,
        df_historical=df_historical,
        df_competitors=df_competitors
    )

    # Store top 10 events sorted by severity
    trigger_events = sorted(trigger_events, key=lambda e: e.severity, reverse=True)[:10]
    primary_trigger = trigger_events[0].event_type if trigger_events else None

    # ========== STEP 2: COMPETITIVE CONTEXT ==========
    competitor_context = {
        'avg_price': df_competitors['price'].mean(),
        'avg_reviews': df_competitors['review_count'].mean(),
        'top3_inventory_avg': df_competitors.nlargest(3, 'monthly_revenue')['inventory_count'].mean(),
        'oos_rate': (df_competitors['inventory_count'] < 5).mean(),
        'avg_margin': df_competitors['margin_pct'].mean() if 'margin_pct' in df_competitors else 0
    }

    # ========== STEP 3: STRATEGIC CLASSIFICATION (LLM) ==========
    prompt = build_enhanced_strategic_prompt(
        row_data=row_data,
        trigger_events=trigger_events,
        competitor_context=competitor_context
    )

    # Call LLM with enhanced prompt
    if client is None:
        client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Parse unified status
        product_status = ProductStatus(result.get('product_status', 'stable_cash_cow'))
        strategic_state = StrategicState(result.get('strategic_state', 'HARVEST'))
        confidence = float(result.get('confidence', 0.7))
        reasoning = result.get('reasoning', '')
        recommended_action = result.get('recommended_action', '')

    except Exception as e:
        # Fallback to deterministic classification
        product_status, strategic_state, confidence, reasoning = _classify_fallback(
            row_data, trigger_events, strategic_bias
        )
        recommended_action = STATE_DEFINITIONS[strategic_state]['default_action']

    # ========== STEP 4: PREDICTIVE INTELLIGENCE ==========
    # Calculate 30-day risk forecast with trigger event context
    predictions = calculate_enhanced_predictive_alpha(
        row_data=row_data,
        df_historical=df_historical,
        trigger_events=trigger_events,
        competitor_context=competitor_context,
        strategic_state=strategic_state.value,
        strategic_bias=strategic_bias
    )

    # ========== STEP 5: INSIGHT GENERATION (LLM) ==========
    # Generate specific, quantified recommendation
    insight = await generate_actionable_insight(
        asin=asin,
        product_status=product_status,
        trigger_events=trigger_events,
        predictions=predictions,
        competitor_context=competitor_context,
        client=client
    )

    # ========== STEP 6: QUALITY VALIDATION ==========
    validation_passed, validation_errors = validate_insight_quality(insight)

    # ========== STEP 7: BUILD UNIFIED OUTPUT ==========
    intelligence = UnifiedIntelligence(
        asin=asin,
        timestamp=datetime.now(),

        # Trigger events
        trigger_events=trigger_events,
        primary_trigger=primary_trigger,

        # Strategic classification
        product_status=product_status,
        strategic_state=strategic_state,
        confidence=confidence,
        reasoning=reasoning,

        # Predictive intelligence
        thirty_day_risk=predictions.thirty_day_risk,
        thirty_day_growth=predictions.thirty_day_growth,
        net_expected_value=(insight['projected_upside_monthly'] * confidence) - insight['downside_risk_monthly'],
        price_erosion_risk=predictions.price_erosion_risk,
        share_erosion_risk=predictions.share_erosion_risk,
        stockout_risk=predictions.stockout_risk,

        # Actionable insight
        recommendation=insight['recommendation'],
        projected_upside_monthly=insight['projected_upside_monthly'],
        downside_risk_monthly=insight['downside_risk_monthly'],
        action_type=insight['action_type'],
        time_horizon_days=insight['time_horizon_days'],

        # Validation
        validation_passed=validation_passed,
        validation_errors=validation_errors,
        data_quality=_assess_data_quality(df_historical),

        # User interaction
        user_dismissed=False,
        user_feedback=None
    )

    return intelligence
```

---

## Part 3: Enhanced Predictive Intelligence Layer

### 3.1 Trigger Event Integration

**Current:** Predictive alpha calculation uses basic velocity trends.
**New:** Enhanced with trigger event severity weighting.

```python
def calculate_enhanced_predictive_alpha(
    row_data: Dict[str, Any],
    df_historical: pd.DataFrame,
    trigger_events: List[TriggerEvent],
    competitor_context: Dict[str, Any],
    strategic_state: str = "HARVEST",
    strategic_bias: str = "Balanced Defense"
) -> PredictiveAlpha:
    """
    ENHANCED PREDICTIVE ALPHA with Trigger Event Intelligence.

    Key Enhancement: Risk forecast is now trigger-event-aware.
    Instead of generic velocity decay, we model specific threats:
    - Competitor OOS → price lift opportunity
    - Price war → margin compression risk
    - Review spike → defensive action needed

    This replaces the existing calculate_predictive_alpha() function
    in utils/ai_engine.py with trigger-aware forecasting.
    """

    revenue = row_data.get('monthly_revenue', 0)

    # ========== BASE CALCULATION (EXISTING) ==========
    # Keep the core formula: (Daily_Leakage * 30) * (1 + velocity_trend_90d)
    base_opportunity_rate = 0.15
    daily_leakage = (revenue * base_opportunity_rate) / 30.0

    velocity_trend_90d = row_data.get('velocity_90d', 0)
    velocity_adjustment = 1.0 + velocity_trend_90d
    base_30day_risk = daily_leakage * 30 * velocity_adjustment

    # ========== TRIGGER EVENT RISK AMPLIFICATION (NEW) ==========
    # Each high-severity trigger amplifies risk forecast
    trigger_risk_multiplier = 1.0
    stockout_risk = 0.0
    price_erosion_risk = 0.0
    share_erosion_risk = 0.0

    for event in trigger_events:
        if event.event_type == "competitor_oos_imminent" and event.severity >= 8:
            # Competitor OOS = OPPORTUNITY (negative risk = growth potential)
            # Revenue opportunity = (daily revenue * days competitor will be OOS * capture rate)
            days_oos_estimated = 14  # Historical average restock time
            capture_rate = 0.25  # Conservative: capture 25% of their sales
            daily_revenue = revenue / 30
            stockout_opportunity = -(daily_revenue * days_oos_estimated * capture_rate)
            price_erosion_risk += stockout_opportunity  # Negative = opportunity

        elif event.event_type == "buybox_share_collapse" and event.severity >= 9:
            # BuyBox loss = CRITICAL RISK
            # Lost revenue = (current revenue * BuyBox loss % * 30 days)
            buybox_loss_pct = abs(event.delta_pct) / 100
            lost_revenue = revenue * buybox_loss_pct
            share_erosion_risk += lost_revenue
            trigger_risk_multiplier *= 1.5  # Amplify urgency

        elif event.event_type == "price_war_active" and event.severity >= 7:
            # Price war = MARGIN COMPRESSION
            # Estimate: will need to drop price 10% to maintain position
            margin_pct = row_data.get('margin_pct', 20) / 100
            price_drop_impact = revenue * 0.10  # 10% revenue reduction
            margin_loss = price_drop_impact * margin_pct
            price_erosion_risk += margin_loss
            trigger_risk_multiplier *= 1.3

        elif event.event_type == "rank_degradation" and event.severity >= 8:
            # Rank degrading = SHARE EROSION
            # Model: for every 10% BSR increase, lose ~5% revenue
            rank_change_pct = abs(event.delta_pct) / 100
            revenue_loss_rate = rank_change_pct * 0.5
            share_loss = revenue * revenue_loss_rate
            share_erosion_risk += share_loss
            trigger_risk_multiplier *= 1.2

        elif event.event_type == "review_velocity_spike" and event.severity >= 6:
            # Review spike on YOUR product = POSITIVE signal (reduce risk)
            if event.affected_asin == row_data['asin']:
                trigger_risk_multiplier *= 0.85  # Reduce risk 15%
            else:
                # Review spike on competitor = WARNING
                trigger_risk_multiplier *= 1.1

    # ========== APPLY TRIGGER AMPLIFICATION ==========
    thirty_day_risk = (base_30day_risk * trigger_risk_multiplier) + price_erosion_risk + share_erosion_risk
    thirty_day_risk = max(0, thirty_day_risk)  # Floor at 0

    # ========== INVENTORY STOCKOUT RISK (EXISTING LOGIC) ==========
    days_to_stockout = row_data.get('days_to_stockout')
    supplier_lead_time = 7

    if days_to_stockout is not None and days_to_stockout < 14:
        stockout_window = max(0, 30 - days_to_stockout - supplier_lead_time)
        daily_profit = (revenue / 30.0) * 0.25
        stockout_risk = stockout_window * daily_profit

        # Stockout risk is ADDITIVE to trigger risks
        thirty_day_risk += stockout_risk

    # ========== GENERATE AI RECOMMENDATION (TRIGGER-AWARE) ==========
    ai_recommendation = _generate_trigger_aware_recommendation(
        trigger_events=trigger_events,
        thirty_day_risk=thirty_day_risk,
        stockout_risk=stockout_risk,
        price_erosion_risk=price_erosion_risk,
        share_erosion_risk=share_erosion_risk
    )

    # ========== GROWTH INTELLIGENCE (EXISTING LOGIC) ==========
    thirty_day_growth = _calculate_growth_opportunities(
        row_data=row_data,
        competitor_context=competitor_context,
        trigger_events=trigger_events
    )

    return PredictiveAlpha(
        thirty_day_risk=thirty_day_risk,
        daily_burn_rate=thirty_day_risk / 30.0,
        velocity_multiplier=trigger_risk_multiplier,
        price_erosion_risk=price_erosion_risk,
        share_erosion_risk=share_erosion_risk,
        stockout_risk=stockout_risk,
        thirty_day_growth=thirty_day_growth,
        ai_recommendation=ai_recommendation,
        alert_type=_determine_alert_type(trigger_events),
        alert_urgency=_determine_alert_urgency(trigger_events, thirty_day_risk, revenue),
        predictive_state=_determine_predictive_state(trigger_events),
        data_quality=_assess_data_quality(df_historical)
    )


def _generate_trigger_aware_recommendation(
    trigger_events: List[TriggerEvent],
    thirty_day_risk: float,
    stockout_risk: float,
    price_erosion_risk: float,
    share_erosion_risk: float
) -> str:
    """
    Generate AI recommendation that CITES specific trigger events.

    OLD: "Monitor closely: $1,500 at risk"
    NEW: "🚨 Competitor B (B08XYZ) has <5 units remaining. Raise price by $2 to capture $1,850 opportunity."
    """
    if not trigger_events:
        return f"${thirty_day_risk:,.0f} optimization opportunity available."

    # Priority 1: Stockout risk (most urgent)
    for event in trigger_events:
        if event.event_type == "competitor_oos_imminent" and event.severity >= 8:
            return (
                f"💰 OPPORTUNITY: Competitor {event.related_asin} has {event.current_value:.0f} units "
                f"remaining (down from {event.baseline_value:.0f}). Raise price to capture "
                f"${abs(price_erosion_risk):,.0f} opportunity before they restock (~14 days)."
            )

    # Priority 2: BuyBox collapse (critical defense)
    for event in trigger_events:
        if event.event_type == "buybox_share_collapse" and event.severity >= 9:
            return (
                f"🚨 CRITICAL: BuyBox share dropped {abs(event.delta_pct):.0f}% "
                f"(from {event.baseline_value:.0%} to {event.current_value:.0%}). "
                f"${share_erosion_risk:,.0f} at risk. Investigate pricing/stock immediately."
            )

    # Priority 3: Price war (defensive action)
    for event in trigger_events:
        if event.event_type == "price_war_active" and event.severity >= 7:
            return (
                f"⚔️ PRICE WAR: {event.current_value:.0f} price drops detected in 7 days. "
                f"${price_erosion_risk:,.0f} margin at risk. Consider matching lowest competitor or exit category."
            )

    # Priority 4: Rank degradation (share loss)
    for event in trigger_events:
        if event.event_type == "rank_degradation" and event.severity >= 8:
            return (
                f"📉 RANK DECLINING: BSR worsened {abs(event.delta_pct):.0f}% "
                f"(from {event.baseline_value:,.0f} to {event.current_value:,.0f}). "
                f"${share_erosion_risk:,.0f} revenue at risk. Increase ad spend or improve listing."
            )

    # Default: Generic risk alert
    return f"${thirty_day_risk:,.0f} at risk over next 30 days based on current trajectory."
```

---

## Part 4: Actionable Insight Generation (LLM Layer 2)

### 4.1 Second LLM Call for Specific Recommendations

**Architecture Decision:** Use TWO sequential LLM calls:
1. **Strategic Classification** (which state is it in?)
2. **Insight Generation** (what should we DO about it?)

**Rationale:** Separating classification from recommendation allows:
- Better prompt engineering for each task
- Easier validation (validate classification, then validate recommendation)
- Fallback flexibility (deterministic classification + LLM recommendation)

```python
async def generate_actionable_insight(
    asin: str,
    product_status: ProductStatus,
    trigger_events: List[TriggerEvent],
    predictions: PredictiveAlpha,
    competitor_context: Dict[str, Any],
    client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    SECOND LLM CALL: Generate specific, quantified recommendation.

    This is called AFTER strategic classification and predictive forecasting.
    The LLM now has full context to make a specific recommendation.

    Args:
        asin: Product ASIN
        product_status: Classified status from first LLM call
        trigger_events: Detected market changes
        predictions: 30-day risk/growth forecast
        competitor_context: Competitive landscape data
        client: OpenAI client

    Returns:
        Dict with recommendation, upside, downside, action_type, time_horizon
    """

    # Build context-rich prompt
    trigger_summary = "\n".join([
        f"- {e.event_type}: {e.metric_name} changed {e.delta_pct:+.1f}% (severity {e.severity}/10)"
        for e in trigger_events[:5]
    ])

    prompt = f"""You are ShelfGuard's AI Strategist generating a specific, actionable recommendation.

PRODUCT STATUS: {product_status.value}
(This product has been classified as: {product_status.name})

PREDICTIVE FORECAST:
- 30-Day Risk: ${predictions.thirty_day_risk:,.0f}
- 30-Day Growth Potential: ${predictions.thirty_day_growth:,.0f}
- Stockout Risk: ${predictions.stockout_risk:,.0f}
- Price Erosion Risk: ${predictions.price_erosion_risk:,.0f}
- Share Erosion Risk: ${predictions.share_erosion_risk:,.0f}

TRIGGER EVENTS DETECTED:
{trigger_summary if trigger_events else "No significant events detected."}

COMPETITIVE CONTEXT:
- Category Avg Price: ${competitor_context['avg_price']:.2f}
- Category Avg Reviews: {competitor_context['avg_reviews']:.0f}
- Competitor OOS Rate: {competitor_context['oos_rate']:.1%}

TASK: Generate a SPECIFIC, QUANTIFIED recommendation.

## Output Requirements (JSON)

You MUST provide these fields:

{{
    "recommendation": "<SPECIFIC action with dollar amount or percentage>",
    "reasoning": "<WHY this action, citing trigger events>",
    "projected_upside_monthly": <dollar amount if action succeeds>,
    "downside_risk_monthly": <dollar amount if action fails>,
    "action_type": "<PROFIT_CAPTURE or RISK_MITIGATION>",
    "time_horizon_days": <days to implement>,
    "confidence_factors": ["<reason 1>", "<reason 2>", "<reason 3>"]
}}

## Critical Rules

1. **Be Specific**:
   - ✅ "Raise price from $24.99 to $26.99 (+$2.00)"
   - ❌ "Consider raising price"

2. **Cite Triggers**:
   - ✅ "...because Competitor B inventory <5 units (down from 47)"
   - ❌ "...because of competitor weakness"

3. **Quantify Impact**:
   - Upside: Revenue gained if action works
   - Downside: Revenue lost if action backfires
   - Be realistic (not overly optimistic)

4. **Time Horizon**:
   - CRITICAL status: 1-3 days
   - OPPORTUNITY status: 7-14 days
   - WATCH status: 14-30 days
   - STABLE status: No action needed

5. **Action Type**:
   - PROFIT_CAPTURE: Opportunity to increase revenue/margin
   - RISK_MITIGATION: Defensive action to prevent loss

## Example Output

{{
    "recommendation": "Raise price from $24.99 to $26.99 (+$2.00) and test for 7 days",
    "reasoning": "Competitor B (ASIN B08XYZ789) inventory dropped to 3 units (down from 47 units last week). Your review count (450) is 2x category average, providing pricing power. Historical price elasticity of -0.4 suggests minimal volume impact at this price point.",
    "projected_upside_monthly": 1850,
    "downside_risk_monthly": 280,
    "action_type": "PROFIT_CAPTURE",
    "time_horizon_days": 7,
    "confidence_factors": [
        "Competitor supply constraint verified via Keepa inventory tracking",
        "Strong review advantage (450 vs 220 category avg) supports premium positioning",
        "Historical price elasticity (-0.4) indicates low demand sensitivity",
        "Category average price is $27.15 (you're currently 8% below)"
    ]
}}

Generate the recommendation now.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
            response_format={"type": "json_object"}
        )

        insight = json.loads(response.choices[0].message.content)

        # Validate required fields
        required_fields = ['recommendation', 'reasoning', 'projected_upside_monthly',
                          'downside_risk_monthly', 'action_type', 'time_horizon_days']

        for field in required_fields:
            if field not in insight:
                raise ValueError(f"Missing required field: {field}")

        return insight

    except Exception as e:
        # Fallback to deterministic insight
        return _generate_deterministic_insight(
            product_status, trigger_events, predictions
        )
```

---

## Part 5: Unified Pipeline Orchestrator

### 5.1 Main Entry Point

**File:** `src/intelligence_pipeline.py`

```python
"""
Unified Intelligence Pipeline

This is the MAIN ENTRY POINT for the entire AI + Predictive Intelligence system.
It orchestrates:
1. Trigger detection
2. Strategic classification
3. Predictive forecasting
4. Insight generation
5. Quality validation
6. Database storage

USAGE:
    from src.intelligence_pipeline import generate_portfolio_intelligence

    insights = await generate_portfolio_intelligence(
        asins=['B08ABC123', 'B08XYZ789'],
        lookback_days=90
    )
"""

from typing import List, Dict
import pandas as pd
import asyncio
from datetime import datetime

from src.models.unified_intelligence import UnifiedIntelligence
from utils.ai_engine_v2 import analyze_product_with_intelligence
from src.trigger_detection import detect_trigger_events
from database.insights_db import store_insight, expire_old_insights


async def generate_portfolio_intelligence(
    asins: List[str],
    lookback_days: int = 90,
    strategic_bias: str = "Balanced Defense",
    batch_size: int = 10
) -> List[UnifiedIntelligence]:
    """
    Generate unified intelligence for a portfolio of ASINs.

    This is the MAIN orchestrator that processes multiple ASINs in parallel.

    Args:
        asins: List of ASINs to analyze
        lookback_days: Days of historical data to analyze
        strategic_bias: User's strategic preference
        batch_size: Number of ASINs to process in parallel

    Returns:
        List of UnifiedIntelligence objects (one per ASIN)
    """

    # ========== DATA LOADING ==========
    # Load historical data for all ASINs
    df_historical = load_historical_data(asins, lookback_days)
    df_competitors = load_competitor_data(asins)

    # ========== PARALLEL PROCESSING ==========
    # Process ASINs in batches to avoid rate limits
    all_intelligence = []

    for i in range(0, len(asins), batch_size):
        batch_asins = asins[i:i+batch_size]

        # Create tasks for parallel execution
        tasks = []
        for asin in batch_asins:
            row_data = get_current_metrics(asin, df_historical)
            asin_historical = df_historical[df_historical['asin'] == asin]

            task = analyze_product_with_intelligence(
                asin=asin,
                row_data=row_data,
                df_historical=asin_historical,
                df_competitors=df_competitors,
                strategic_bias=strategic_bias
            )
            tasks.append(task)

        # Execute batch in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        for result in batch_results:
            if isinstance(result, UnifiedIntelligence):
                all_intelligence.append(result)
            else:
                # Log error but continue
                print(f"Error processing ASIN: {result}")

    # ========== POST-PROCESSING ==========
    # Filter out low-priority stable products
    actionable_intelligence = [
        intel for intel in all_intelligence
        if intel.product_status.value not in ['stable_fortress', 'stable_cash_cow', 'stable_niche']
    ]

    # ========== DATABASE STORAGE ==========
    # Store insights in database
    for intel in all_intelligence:
        if intel.validation_passed:
            await store_insight(intel)

    # Expire old insights
    await expire_old_insights()

    return all_intelligence


async def generate_executive_brief(
    portfolio_intelligence: List[UnifiedIntelligence]
) -> str:
    """
    Generate executive brief from unified intelligence.

    This replaces the existing executive brief logic in shelfguard_app.py
    with causal reasoning based on trigger events.
    """

    # Aggregate metrics
    total_risk = sum(i.thirty_day_risk for i in portfolio_intelligence)
    total_growth = sum(i.thirty_day_growth for i in portfolio_intelligence)
    critical_count = sum(1 for i in portfolio_intelligence if 'critical_' in i.product_status.value)
    opportunity_count = sum(1 for i in portfolio_intelligence if 'opportunity_' in i.product_status.value)

    # Find most severe trigger events across portfolio
    all_triggers = []
    for intel in portfolio_intelligence:
        all_triggers.extend(intel.trigger_events)

    top_triggers = sorted(all_triggers, key=lambda t: t.severity, reverse=True)[:3]

    # Build brief context
    brief_context = f"""
PORTFOLIO INTELLIGENCE SUMMARY:
- Total ASINs Analyzed: {len(portfolio_intelligence)}
- Critical Issues: {critical_count}
- Optimization Opportunities: {opportunity_count}
- 30-Day Risk: ${total_risk:,.0f}
- 30-Day Growth Potential: ${total_growth:,.0f}

TOP TRIGGER EVENTS:
{chr(10).join([f"- {t.event_type}: {t.affected_asin} - {t.metric_name} changed {t.delta_pct:+.1f}%" for t in top_triggers])}

CRITICAL PRODUCTS REQUIRING IMMEDIATE ACTION:
{chr(10).join([f"- {i.asin}: {i.recommendation}" for i in portfolio_intelligence if 'critical_' in i.product_status.value][:3])}

OPTIMIZATION OPPORTUNITIES:
{chr(10).join([f"- {i.asin}: ${i.projected_upside_monthly:,.0f} upside - {i.recommendation}" for i in portfolio_intelligence if 'opportunity_' in i.product_status.value][:3])}
"""

    # Call LLM for executive brief
    client = _get_openai_client()

    prompt = f"""You are ShelfGuard's executive strategist. Generate a concise strategic brief.

{brief_context}

Generate a 75-100 word brief that:
1. Leads with the most urgent threat or opportunity
2. Cites specific trigger events and ASINs
3. Quantifies financial impact
4. Ends with one executable command

Use action language: "Protocol Activated", "Execute immediately", "Deploy"
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content
```

---

## Part 6: Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

**Day 1-2: Data Models**
- [ ] Create `src/models/unified_intelligence.py`
- [ ] Create `src/models/trigger_event.py`
- [ ] Create `src/models/product_status.py` (enum)

**Day 3-4: Trigger Detection**
- [ ] Create `src/trigger_detection.py`
- [ ] Implement 6 core detectors (inventory, price, reviews, buybox, rank, competitor)
- [ ] Unit tests

**Day 5: Database Layer**
- [ ] Create `database/insights_db.py`
- [ ] Implement `store_insight()`, `fetch_actionable_insights()`
- [ ] Create indexes

### Phase 2: AI Engine Refactor (Week 2)

**Day 1-2: Enhanced Classification**
- [ ] Create `utils/ai_engine_v2.py`
- [ ] Implement `build_enhanced_strategic_prompt()`
- [ ] Migrate from `analyze_strategy_with_llm()` to new version

**Day 3: Enhanced Predictive Layer**
- [ ] Refactor `calculate_enhanced_predictive_alpha()`
- [ ] Add trigger event risk amplification
- [ ] Implement `_generate_trigger_aware_recommendation()`

**Day 4: Insight Generation**
- [ ] Implement `generate_actionable_insight()` (2nd LLM call)
- [ ] Implement quality validation gate

**Day 5: Integration**
- [ ] Create `analyze_product_with_intelligence()` (unified entry point)
- [ ] Test end-to-end flow

### Phase 3: Pipeline & UI (Week 3)

**Day 1-2: Pipeline Orchestrator**
- [ ] Create `src/intelligence_pipeline.py`
- [ ] Implement `generate_portfolio_intelligence()`
- [ ] Implement parallel batch processing

**Day 3-4: UI Integration**
- [ ] Update `apps/shelfguard_app.py` to use new pipeline
- [ ] Update Action Queue rendering
- [ ] Add trigger event display

**Day 5: Executive Brief**
- [ ] Implement `generate_executive_brief()` with triggers
- [ ] Update dashboard header

### Phase 4: Testing & Validation (Week 4)

**Day 1-2: Unit Tests**
- [ ] Test trigger detection accuracy
- [ ] Test LLM prompt quality
- [ ] Test validation gates

**Day 3-4: Integration Tests**
- [ ] End-to-end pipeline test
- [ ] Load test (100+ ASINs)
- [ ] LLM failure scenarios

**Day 5: UAT**
- [ ] Verify zero UI contradictions
- [ ] Verify all insights have $ amounts
- [ ] Verify trigger events cited
- [ ] Performance benchmarking

---

## Part 7: Success Metrics

### Quantitative KPIs

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| **Insight Specificity** | ~20% have $ amounts | 100% | % of insights with `$` or `%` in recommendation |
| **Trigger Citation Rate** | 0% | 100% | % of insights that cite ≥1 trigger event |
| **Prediction Accuracy** | Unknown | >70% | Track actual outcomes vs predicted upside/risk |
| **False Positive Rate** | ~40% | <10% | % of "action needed" on stable products |
| **LLM Latency** | N/A | <3s per ASIN | p95 latency for full intelligence pipeline |
| **Quality Gate Pass Rate** | N/A | >90% | % of LLM outputs passing validation |

### Qualitative Success Criteria

✅ **Causal Transparency**: Every recommendation cites specific trigger events
✅ **Financial Quantification**: 100% include projected upside + downside risk
✅ **No Contradictions**: Product status matches across all UI elements
✅ **Signal Filtering**: Stable products hidden from default Action Queue view
✅ **Executive Confidence**: Briefs include causal reasoning, not observations

---

## Part 8: Risk Mitigation

### Technical Risks

**Risk: LLM generates invalid insights**
*Mitigation:* Quality validation gate rejects outputs missing $ amounts or trigger citations. Fallback to deterministic recommendations.

**Risk: Trigger detection has false positives**
*Mitigation:* Set conservative severity thresholds. Log all detections for manual review in first 2 weeks.

**Risk: Two LLM calls doubles latency**
*Mitigation:* Run calls in parallel where possible. Cache classification results. Use gpt-4o-mini for speed.

**Risk: Integration breaks existing features**
*Mitigation:* Keep old `ai_engine.py` for 2 weeks. Dual-write insights. A/B test in production.

### Product Risks

**Risk: Predictions are wildly inaccurate**
*Mitigation:* Track actual outcomes in `insight_outcomes` table. Display confidence scores prominently. Start with conservative forecasts.

**Risk: Too many CRITICAL alerts create fatigue**
*Mitigation:* Cap CRITICAL alerts at 5 per view. Strict severity thresholds (9/10 minimum).

**Risk: Users don't trust AI recommendations**
*Mitigation:* Always show trigger events + reasoning. Allow user feedback. Track "helpful" vs "not helpful" ratings.

---

## Appendix A: Architecture Comparison

### OLD Architecture (Current)
```
Keepa Data → Strategic Classification (LLM) → UI Display
                     ↓
           Predictive Alpha (deterministic) → Alerts
```

**Problems:**
- Strategic classification disconnected from triggers
- Predictive alpha uses generic velocity, not events
- No causal reasoning
- No specific $ amounts
- Multiple conflicting status fields

### NEW Architecture (Unified)
```
Keepa Data → Trigger Detection → Strategic Classification (LLM + triggers)
                                           ↓
                                  Predictive Alpha (trigger-aware)
                                           ↓
                                  Insight Generation (LLM + predictions)
                                           ↓
                                  Quality Validation
                                           ↓
                                  Database Storage
                                           ↓
                                  Action Queue (filtered)
```

**Benefits:**
- Single unified flow
- Causal reasoning at every layer
- Quantified predictions
- Quality gates prevent bad outputs
- Feedback loops for improvement

---

## Appendix B: File Structure

```
ShelfGuard/
├── src/
│   ├── models/
│   │   ├── unified_intelligence.py      (NEW - UnifiedIntelligence dataclass)
│   │   ├── trigger_event.py             (NEW - TriggerEvent dataclass)
│   │   └── product_status.py            (NEW - ProductStatus enum)
│   ├── trigger_detection.py             (NEW - detect_trigger_events())
│   ├── intelligence_pipeline.py         (NEW - main orchestrator)
│   └── recommendations.py               (ARCHIVE - migrate to new system)
├── utils/
│   ├── ai_engine.py                     (KEEP - existing strategic classification)
│   └── ai_engine_v2.py                  (NEW - enhanced with triggers)
├── database/
│   └── insights_db.py                   (NEW - store/fetch insights)
├── schemas/
│   ├── strategic_insights.sql           (NEW - unified insights table)
│   ├── trigger_events.sql               (NEW - trigger tracking)
│   └── insight_outcomes.sql             (NEW - accuracy tracking)
├── apps/
│   └── shelfguard_app.py                (REFACTOR - use new pipeline)
└── docs/
    ├── INSIGHT_ENGINE_REFACTOR_PLAN.md  (Architecture - insights layer)
    └── AI_PREDICTIVE_ENGINE_ARCHITECTURE.md  (THIS FILE - unified system)
```

---

**End of Architecture Document. Ready for Implementation.**
