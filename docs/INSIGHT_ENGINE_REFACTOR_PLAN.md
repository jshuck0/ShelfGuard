# ShelfGuard Insight Engine Refactor: Strategic Intelligence System
## Staff Engineering Plan v1.0

**Status:** Design Complete - Ready for Implementation
**Author:** Technical Architecture Team
**Date:** 2026-01-19

---

## Executive Summary

**The Problem:**
Current insights are weak, contradictory, and generate noise instead of signal:
- Generic advice ("Reduce Ad Spend") without quantified specifics
- Contradictory UI states (Red "Raise Price" card + "HOLD - Monitor" table row)
- False positives (flagging stable "Cash Cows" as needing action)
- Fluffy executive briefs lacking causal reasoning

**The Solution:**
Transform from **Monitoring Tool** → **Strategic Intelligence System** with:
1. **Unified Status Taxonomy** (One Source of Truth)
2. **Causal Reasoning Engine** (Trigger Events → Insights)
3. **Quantified Recommendations** (Specific $ amounts and reasoning)
4. **Signal Filtering** (Hide stable products, surface real opportunities)

**Expected Outcomes:**
- 80% reduction in false positive alerts
- 100% of recommendations include specific dollar amounts
- Zero contradictions between UI elements
- Executive trust through transparent causal reasoning

---

## Part 1: Unified Status Taxonomy - "The One Truth"

### 1.1 Current State Analysis (Problems Identified)

**Existing Fields Creating Confusion:**
```python
# From ai_engine.py
strategic_state: str  # FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL

# From recommendations.py
alert["type"]: str    # volume_stealer, efficiency_gap, new_entrant
alert["severity"]: str # high, medium, low

# From UI rendering
problem_category: str  # "✅ Your Brand - Healthy", "Cash Cow"
capital_zone: str      # "🏰 FORTRESS (Cash Flow)", "📉 DRAG (Waste)"
```

**The Contradiction Example:**
```
Action Card (Red):     "🚨 RAISE PRICE - High Priority"
Table Row State:       "HARVEST - Cash Cow"
Table Recommendation:  "HOLD - Monitor position, stable"
Executive Brief:       "Focus on maintaining market share"
```

### 1.2 New Unified Status Taxonomy

**Replace all conflicting fields with a single `product_status` enum:**

```python
class ProductStatus(Enum):
    """
    UNIFIED STATUS CLASSIFICATION

    This is THE SINGLE SOURCE OF TRUTH for product state.
    All UI elements, alerts, and recommendations derive from this field.
    """

    # CRITICAL - Immediate action required (losing money NOW)
    CRITICAL_MARGIN_COLLAPSE = "critical_margin_collapse"      # Margin < 10% and declining
    CRITICAL_INVENTORY_RISK = "critical_inventory_risk"        # OOS imminent (< 7 days)
    CRITICAL_BUYBOX_LOSS = "critical_buybox_loss"              # BuyBox < 30%
    CRITICAL_VELOCITY_CRASH = "critical_velocity_crash"        # Revenue -50%+ in 30d

    # OPPORTUNITY - Optimization available (capture upside)
    OPPORTUNITY_PRICE_POWER = "opportunity_price_power"        # Can raise price $X
    OPPORTUNITY_AD_WASTE = "opportunity_ad_waste"              # Cut spend by $X
    OPPORTUNITY_REVIEW_GAP = "opportunity_review_gap"          # Launch Vine (↑ conv X%)
    OPPORTUNITY_COMPETITOR_WEAKNESS = "opportunity_competitor_weakness"  # Rival OOS

    # WATCH - Volatile/Changing (monitor closely)
    WATCH_NEW_COMPETITOR = "watch_new_competitor"              # New ASIN w/ BSR < 10k
    WATCH_PRICE_WAR = "watch_price_war"                        # 3+ price drops in 7d
    WATCH_SEASONAL_ANOMALY = "watch_seasonal_anomaly"          # Unusual pattern detected
    WATCH_RANK_VOLATILITY = "watch_rank_volatility"            # BSR variance > 50%

    # STABLE - Healthy/Cash Cow (filter from default view)
    STABLE_FORTRESS = "stable_fortress"                        # Market leader, defended
    STABLE_CASH_COW = "stable_cash_cow"                        # Consistent revenue, low risk
    STABLE_NICHE = "stable_niche"                              # Small but profitable
```

**Status Hierarchy (for filtering):**
```python
STATUS_PRIORITY = {
    "CRITICAL": 100,      # Always show, red alerts
    "OPPORTUNITY": 75,    # Show by default, green/yellow
    "WATCH": 50,          # Show if user toggles "monitoring"
    "STABLE": 0           # Hidden by default (toggle to view)
}
```

### 1.3 Mapping Old → New Schema

**Migration Logic:**

| Old `strategic_state` | Old `severity` | New `product_status` |
|----------------------|----------------|---------------------|
| TERMINAL             | high           | CRITICAL_MARGIN_COLLAPSE |
| DISTRESS             | high           | CRITICAL_VELOCITY_CRASH |
| TRENCH_WAR           | medium         | WATCH_PRICE_WAR |
| HARVEST              | (any)          | OPPORTUNITY_PRICE_POWER |
| FORTRESS             | low            | STABLE_FORTRESS |

**Database Migration:**
```sql
-- Add new unified status column
ALTER TABLE products ADD COLUMN product_status TEXT;
ALTER TABLE products ADD COLUMN status_priority INT DEFAULT 0;
ALTER TABLE products ADD COLUMN status_changed_at TIMESTAMPTZ DEFAULT NOW();

-- Create index for fast filtering
CREATE INDEX idx_products_status_priority ON products(status_priority DESC, status_changed_at DESC);

-- Backfill data (transition period)
UPDATE products
SET product_status = CASE
    WHEN strategic_state = 'TERMINAL' THEN 'critical_margin_collapse'
    WHEN strategic_state = 'DISTRESS' THEN 'critical_velocity_crash'
    WHEN strategic_state = 'HARVEST' THEN 'opportunity_price_power'
    WHEN strategic_state = 'FORTRESS' THEN 'stable_fortress'
    ELSE 'stable_cash_cow'
END,
status_priority = CASE
    WHEN strategic_state IN ('TERMINAL', 'DISTRESS') THEN 100
    WHEN strategic_state = 'HARVEST' THEN 75
    WHEN strategic_state = 'TRENCH_WAR' THEN 50
    ELSE 0
END;
```

---

## Part 2: Insight Reasoning Framework - "The Why Engine"

### 2.1 Trigger Event System

**Problem:** Current insights say WHAT ("Raise price") but not WHY ("because competitor X is OOS").

**Solution:** Structured Trigger Events that feed into LLM context.

```python
@dataclass
class TriggerEvent:
    """
    A Trigger Event is a discrete, measurable market change that justifies an insight.

    Examples:
    - Competitor inventory dropped below 5 units
    - Review count increased by 50 in 7 days
    - BuyBox share fell from 85% to 45%
    """
    event_type: str                    # "competitor_oos", "review_spike", "price_war"
    severity: int                      # 1-10 (10 = most urgent)
    detected_at: datetime
    metric_name: str                   # "inventory_count", "review_count", "buybox_share"
    baseline_value: float              # Value before change
    current_value: float               # Value after change
    delta_pct: float                   # % change
    affected_asin: str
    related_asin: Optional[str] = None # Competitor ASIN if relevant

    def to_llm_context(self) -> str:
        """Format for LLM prompt injection."""
        return (
            f"EVENT DETECTED: {self.event_type.upper()}\n"
            f"- Metric: {self.metric_name} changed from {self.baseline_value} → {self.current_value} "
            f"({self.delta_pct:+.1f}%)\n"
            f"- Severity: {self.severity}/10\n"
            f"- Detected: {self.detected_at.strftime('%Y-%m-%d %H:%M')}\n"
        )
```

**Trigger Event Detectors (expanded from recommendations.py):**

```python
def detect_trigger_events(
    asin: str,
    df_historical: pd.DataFrame,
    df_competitors: pd.DataFrame
) -> List[TriggerEvent]:
    """
    Scan Keepa data for discrete trigger events.

    Returns all detected events for this ASIN in last 30 days.
    """
    events = []

    # 1. COMPETITOR INVENTORY EVENTS
    for _, comp in df_competitors.iterrows():
        if comp['fba_inventory'] < 5 and comp['fba_inventory_7d_ago'] >= 10:
            events.append(TriggerEvent(
                event_type="competitor_oos_imminent",
                severity=9,
                detected_at=datetime.now(),
                metric_name="fba_inventory",
                baseline_value=comp['fba_inventory_7d_ago'],
                current_value=comp['fba_inventory'],
                delta_pct=-70.0,
                affected_asin=asin,
                related_asin=comp['asin']
            ))

    # 2. REVIEW VELOCITY EVENTS
    current_reviews = df_historical['review_count'].iloc[-1]
    reviews_30d_ago = df_historical['review_count'].iloc[-30] if len(df_historical) >= 30 else current_reviews
    review_delta = current_reviews - reviews_30d_ago

    if review_delta > 50:  # Significant spike
        events.append(TriggerEvent(
            event_type="review_velocity_spike",
            severity=6,
            detected_at=datetime.now(),
            metric_name="review_count",
            baseline_value=reviews_30d_ago,
            current_value=current_reviews,
            delta_pct=(review_delta / reviews_30d_ago * 100) if reviews_30d_ago > 0 else 0,
            affected_asin=asin
        ))

    # 3. BUYBOX OWNERSHIP EVENTS
    current_bb = df_historical['buybox_share'].iloc[-1]
    bb_7d_ago = df_historical['buybox_share'].iloc[-7] if len(df_historical) >= 7 else current_bb

    if current_bb < 0.5 and bb_7d_ago >= 0.8:
        events.append(TriggerEvent(
            event_type="buybox_share_collapse",
            severity=10,
            detected_at=datetime.now(),
            metric_name="buybox_share",
            baseline_value=bb_7d_ago,
            current_value=current_bb,
            delta_pct=((current_bb - bb_7d_ago) / bb_7d_ago * 100),
            affected_asin=asin
        ))

    # 4. PRICE WAR EVENTS
    price_changes = df_historical['price'].diff().tail(7)
    price_drops = (price_changes < 0).sum()

    if price_drops >= 3:
        events.append(TriggerEvent(
            event_type="price_war_active",
            severity=7,
            detected_at=datetime.now(),
            metric_name="price_change_frequency",
            baseline_value=0,
            current_value=price_drops,
            delta_pct=0,
            affected_asin=asin
        ))

    # 5. BSR VELOCITY EVENTS (both directions)
    current_bsr = df_historical['sales_rank'].iloc[-1]
    bsr_7d_ago = df_historical['sales_rank'].iloc[-7] if len(df_historical) >= 7 else current_bsr
    bsr_velocity = ((bsr_7d_ago - current_bsr) / bsr_7d_ago * 100) if bsr_7d_ago > 0 else 0

    if bsr_velocity < -30:  # Rank worsened 30%+
        events.append(TriggerEvent(
            event_type="rank_degradation",
            severity=8,
            detected_at=datetime.now(),
            metric_name="sales_rank",
            baseline_value=bsr_7d_ago,
            current_value=current_bsr,
            delta_pct=bsr_velocity,
            affected_asin=asin
        ))

    return sorted(events, key=lambda e: e.severity, reverse=True)
```

### 2.2 LLM Prompt Engineering - Strategy Mode

**OLD PROMPT (Observation Mode):**
```python
# ❌ BAD - Vague, no context, no specificity
"""
You are analyzing product performance. Here's the data:
- Revenue: $5,000
- BSR: 15,000
What should we do?
"""
```

**NEW PROMPT (Strategy Mode with Trigger Events):**

```python
def build_strategic_prompt(
    asin: str,
    product_metrics: Dict,
    trigger_events: List[TriggerEvent],
    competitor_context: Dict
) -> str:
    """
    Build high-fidelity LLM prompt with causal context.

    CRITICAL RULES:
    1. Every recommendation MUST include specific dollar amount or percentage
    2. Every recommendation MUST cite the trigger event (causal reason)
    3. Distinguish "Profit Capture" (opportunity) from "Risk Mitigation" (defense)
    """

    # Format trigger events
    trigger_context = "\n\n".join([e.to_llm_context() for e in trigger_events])

    prompt = f"""
You are ShelfGuard's Strategic Intelligence System analyzing ASIN {asin}.

PRODUCT CURRENT STATE:
- Current Price: ${product_metrics['price']:.2f}
- Monthly Revenue: ${product_metrics['monthly_revenue']:,.0f}
- Sales Rank (BSR): {product_metrics['bsr']:,}
- Review Count: {product_metrics['review_count']} ({product_metrics['avg_rating']:.1f}★)
- BuyBox Share: {product_metrics['buybox_share']:.0%}
- Estimated Margin: {product_metrics['est_margin']:.1%}

COMPETITIVE CONTEXT:
- Category Avg Price: ${competitor_context['avg_price']:.2f}
- Your Price vs Avg: {((product_metrics['price'] / competitor_context['avg_price'] - 1) * 100):+.1f}%
- Top 3 Competitor Inventory: {competitor_context['top3_inventory_avg']:.0f} units
- Category Avg Reviews: {competitor_context['avg_reviews']:.0f}

DETECTED TRIGGER EVENTS (Last 30 Days):
{trigger_context if trigger_events else "No significant trigger events detected."}

TASK: Generate a Strategic Recommendation.

REQUIREMENTS:
1. **Status Classification**: Choose ONE status from:
   - CRITICAL: Immediate fix needed (losing money NOW)
   - OPPORTUNITY: Optimization available (capture upside)
   - WATCH: Volatile/changing (monitor closely)
   - STABLE: Healthy/cash cow (no action needed)

2. **Specific Recommendation**: You MUST provide:
   - Exact dollar amount OR percentage (e.g., "Raise price by $1.50" or "Cut ad spend by 15%")
   - Time horizon (e.g., "Test for 7 days," "Implement within 48 hours")

3. **Causal Reasoning**: Cite the SPECIFIC trigger event that justifies this action.
   - ✅ GOOD: "Raise price by $2.00 because Competitor B (ASIN B08XYZ) is out of stock (inventory: 3 units)"
   - ❌ BAD: "Consider raising price to improve margins"

4. **Risk Quantification**: Estimate the financial impact:
   - **Projected Upside**: Revenue opportunity if action succeeds (e.g., "+$1,200/month")
   - **Downside Risk**: Potential loss if action fails (e.g., "-$300/month if demand drops")

5. **Action Type**: Classify as:
   - PROFIT_CAPTURE (opportunity - green)
   - RISK_MITIGATION (defense - red)

OUTPUT FORMAT (JSON):
{{
    "status": "OPPORTUNITY_PRICE_POWER",
    "confidence": 0.85,
    "recommendation": "Raise price from $24.99 to $26.99 (+$2.00) over next 7 days",
    "reasoning": "Competitor B (ASIN B08XYZ789) has <5 units inventory remaining and historically takes 14 days to restock. Your review count (450) is 2x category average, giving pricing power. Current price is 8% below category average ($27.15).",
    "trigger_events_cited": ["competitor_oos_imminent", "review_advantage"],
    "projected_upside_monthly": 1850,
    "downside_risk_monthly": 280,
    "action_type": "PROFIT_CAPTURE",
    "time_horizon_days": 7,
    "confidence_factors": [
        "Strong review advantage (450 vs 220 avg)",
        "Competitor supply constraint verified",
        "Historical price elasticity: -0.4 (low sensitivity)"
    ]
}}
"""
    return prompt
```

### 2.3 Insight Quality Validation

**Post-LLM Quality Gate:**

```python
def validate_insight_quality(insight: Dict) -> Tuple[bool, List[str]]:
    """
    Reject low-quality LLM outputs that don't meet Strategic Intelligence standards.

    Returns: (is_valid, list_of_violations)
    """
    violations = []

    # Rule 1: Must have specific dollar amount or percentage
    recommendation = insight.get('recommendation', '')
    if not any(char in recommendation for char in ['$', '%', '+']):
        violations.append("Missing specific dollar amount or percentage")

    # Rule 2: Must cite at least one trigger event
    if not insight.get('trigger_events_cited') or len(insight['trigger_events_cited']) == 0:
        violations.append("No trigger events cited in reasoning")

    # Rule 3: Must quantify financial impact
    if insight.get('projected_upside_monthly', 0) == 0 and insight.get('downside_risk_monthly', 0) == 0:
        violations.append("Missing financial impact quantification")

    # Rule 4: Reasoning must be >50 characters (no generic fluff)
    reasoning = insight.get('reasoning', '')
    if len(reasoning) < 50:
        violations.append("Reasoning too short/generic")

    # Rule 5: Confidence must be reasonable (0.3 - 1.0)
    confidence = insight.get('confidence', 0)
    if confidence < 0.3 or confidence > 1.0:
        violations.append(f"Unreasonable confidence score: {confidence}")

    is_valid = len(violations) == 0
    return is_valid, violations
```

---

## Part 3: Database Schema - High-Fidelity Insights

### 3.1 New `strategic_insights` Table

```sql
-- Drop old conflicting tables
-- DROP TABLE IF EXISTS recommendations CASCADE;
-- DROP TABLE IF EXISTS alerts CASCADE;

-- Create unified insights table
CREATE TABLE strategic_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core Identity
    asin TEXT NOT NULL REFERENCES products(asin),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    is_active BOOLEAN DEFAULT TRUE,

    -- Unified Status (THE ONE TRUTH)
    product_status TEXT NOT NULL,  -- ProductStatus enum value
    status_priority INT NOT NULL,  -- 100=CRITICAL, 75=OPPORTUNITY, 50=WATCH, 0=STABLE
    status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- LLM-Generated Insight
    recommendation TEXT NOT NULL,           -- Specific action (e.g., "Raise price by $2.00")
    reasoning TEXT NOT NULL,                -- Causal explanation (>50 chars)
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    llm_model TEXT DEFAULT 'gpt-4o-mini',
    llm_prompt_hash TEXT,                   -- Cache key for identical prompts

    -- Trigger Events (What caused this insight?)
    trigger_events JSONB NOT NULL,          -- Array of TriggerEvent objects
    primary_trigger TEXT,                   -- Most important trigger event type

    -- Financial Impact (Quantified Upside/Risk)
    projected_upside_monthly DECIMAL(10,2), -- Revenue opportunity (e.g., +$1,850)
    downside_risk_monthly DECIMAL(10,2),    -- Potential loss (e.g., -$280)
    net_expected_value DECIMAL(10,2),       -- Upside * confidence - downside risk

    -- Action Metadata
    action_type TEXT NOT NULL,              -- PROFIT_CAPTURE or RISK_MITIGATION
    time_horizon_days INT,                  -- Recommended implementation window
    confidence_factors TEXT[],              -- List of reasons for confidence score

    -- Validation & Quality
    validation_passed BOOLEAN DEFAULT TRUE,
    validation_errors TEXT[],               -- Quality gate violations

    -- User Interaction
    user_dismissed BOOLEAN DEFAULT FALSE,
    dismissed_at TIMESTAMPTZ,
    user_feedback TEXT,                     -- "helpful", "not_helpful", "incorrect"

    -- Indexes
    CONSTRAINT valid_status CHECK (product_status IN (
        'critical_margin_collapse', 'critical_inventory_risk', 'critical_buybox_loss', 'critical_velocity_crash',
        'opportunity_price_power', 'opportunity_ad_waste', 'opportunity_review_gap', 'opportunity_competitor_weakness',
        'watch_new_competitor', 'watch_price_war', 'watch_seasonal_anomaly', 'watch_rank_volatility',
        'stable_fortress', 'stable_cash_cow', 'stable_niche'
    )),
    CONSTRAINT valid_action_type CHECK (action_type IN ('PROFIT_CAPTURE', 'RISK_MITIGATION'))
);

-- Performance indexes
CREATE INDEX idx_insights_status_priority ON strategic_insights(status_priority DESC, created_at DESC);
CREATE INDEX idx_insights_asin_active ON strategic_insights(asin, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_insights_action_type ON strategic_insights(action_type, status_priority DESC);
CREATE INDEX idx_insights_expires ON strategic_insights(expires_at) WHERE is_active = TRUE;

-- Auto-expire old insights
CREATE OR REPLACE FUNCTION expire_old_insights()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE strategic_insights
    SET is_active = FALSE
    WHERE expires_at < NOW() AND is_active = TRUE;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_expire_insights
AFTER INSERT ON strategic_insights
EXECUTE FUNCTION expire_old_insights();
```

### 3.2 Supporting Tables

**Trigger Events Tracking:**

```sql
CREATE TABLE trigger_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asin TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity INT CHECK (severity >= 1 AND severity <= 10),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name TEXT NOT NULL,
    baseline_value DECIMAL(12,2),
    current_value DECIMAL(12,2),
    delta_pct DECIMAL(6,2),
    related_asin TEXT,  -- Competitor ASIN if relevant

    -- Track if this event triggered an insight
    generated_insight_id UUID REFERENCES strategic_insights(id),

    -- Deduplication
    UNIQUE(asin, event_type, detected_at)
);

CREATE INDEX idx_trigger_events_asin ON trigger_events(asin, detected_at DESC);
CREATE INDEX idx_trigger_events_severity ON trigger_events(severity DESC, detected_at DESC);
```

**Insight Performance Tracking:**

```sql
CREATE TABLE insight_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID NOT NULL REFERENCES strategic_insights(id),

    -- Did the user take action?
    action_taken BOOLEAN,
    action_taken_at TIMESTAMPTZ,

    -- Did the insight prediction come true?
    actual_outcome TEXT,  -- 'upside_realized', 'downside_occurred', 'no_change', 'unknown'
    actual_revenue_impact DECIMAL(10,2),  -- Measured result
    predicted_revenue_impact DECIMAL(10,2),  -- What we predicted
    prediction_error_pct DECIMAL(6,2),  -- How accurate was our LLM?

    -- Feedback loop for LLM improvement
    measured_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_outcomes_insight ON insight_outcomes(insight_id);
```

---

## Part 4: Action Queue Refactoring

### 4.1 Filtering Logic - Signal vs Noise

**Current Problem:** Action Queue shows everything, including stable "Cash Cows".

**New Filtering System:**

```python
class ActionQueueFilter:
    """
    Smart filtering to show ONLY actionable items by default.

    Philosophy: "If it's stable and healthy, don't distract the user."
    """

    DEFAULT_FILTERS = {
        'min_priority': 50,           # Hide STABLE (priority 0)
        'show_stable': False,          # User must explicitly toggle
        'max_items': 25,               # Top 25 by priority
        'action_types': ['PROFIT_CAPTURE', 'RISK_MITIGATION']
    }

    @staticmethod
    def apply_filters(
        df_insights: pd.DataFrame,
        user_filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Filter insights to show only actionable signal.
        """
        filters = {**ActionQueueFilter.DEFAULT_FILTERS, **(user_filters or {})}

        # Start with active insights only
        df = df_insights[df_insights['is_active'] == True].copy()

        # Filter by priority threshold
        df = df[df['status_priority'] >= filters['min_priority']]

        # Filter out STABLE unless user opts in
        if not filters['show_stable']:
            df = df[~df['product_status'].str.startswith('stable_')]

        # Filter by action type if specified
        if filters['action_types']:
            df = df[df['action_type'].isin(filters['action_types'])]

        # Sort by priority, then by expected value
        df = df.sort_values(
            ['status_priority', 'net_expected_value'],
            ascending=[False, False]
        )

        # Limit to top N
        df = df.head(filters['max_items'])

        return df
```

### 4.2 Visual Distinction - Profit vs Risk

**UI Color Coding:**

```python
ACTION_TYPE_STYLING = {
    'PROFIT_CAPTURE': {
        'color': '#28a745',          # Green
        'icon': '💰',
        'badge': 'OPPORTUNITY',
        'description': 'Revenue Growth Opportunity'
    },
    'RISK_MITIGATION': {
        'color': '#dc3545',          # Red
        'icon': '🛡️',
        'badge': 'DEFENSE',
        'description': 'Risk Mitigation Required'
    }
}

def render_action_card(insight: Dict):
    """
    Render action card with clear visual distinction between opportunity and risk.
    """
    styling = ACTION_TYPE_STYLING[insight['action_type']]

    st.markdown(f"""
    <div style="
        border-left: 5px solid {styling['color']};
        background: white;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 24px;">{styling['icon']}</span>
                <span style="
                    background: {styling['color']};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 700;
                    margin-left: 8px;
                ">{styling['badge']}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 20px; font-weight: 700; color: {styling['color']};">
                    ${insight['projected_upside_monthly']:,.0f}/mo
                </div>
                <div style="font-size: 11px; color: #999;">
                    Risk: ${insight['downside_risk_monthly']:,.0f}/mo
                </div>
            </div>
        </div>

        <div style="margin-top: 12px;">
            <div style="font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                {insight['recommendation']}
            </div>
            <div style="font-size: 13px; color: #666; line-height: 1.4;">
                {insight['reasoning']}
            </div>
        </div>

        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e0e0e0;">
            <div style="font-size: 11px; color: #999;">
                ⏱️ {insight['time_horizon_days']} day window
                | 🎯 {insight['confidence']:.0%} confidence
                | 🔥 {', '.join(insight['trigger_events_cited'][:2])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

### 4.3 Action Queue SQL Query

```sql
-- Fetch Action Queue (only actionable items)
SELECT
    si.asin,
    p.title,
    p.brand,
    si.product_status,
    si.status_priority,
    si.recommendation,
    si.reasoning,
    si.action_type,
    si.projected_upside_monthly,
    si.downside_risk_monthly,
    si.net_expected_value,
    si.confidence,
    si.time_horizon_days,
    si.trigger_events,
    si.created_at,

    -- Current metrics for context
    p.price,
    p.monthly_revenue,
    p.sales_rank,
    p.review_count

FROM strategic_insights si
JOIN products p ON si.asin = p.asin

WHERE
    si.is_active = TRUE
    AND si.status_priority >= 50  -- Filter out STABLE by default
    AND si.user_dismissed = FALSE
    AND si.validation_passed = TRUE

ORDER BY
    si.status_priority DESC,
    si.net_expected_value DESC

LIMIT 25;
```

---

## Part 5: Implementation Plan

### Phase 1: Schema & Data Layer (Week 1)

**Day 1-2: Database Migration**
```bash
# Files to create/modify:
- schemas/strategic_insights.sql         (NEW)
- schemas/trigger_events.sql             (NEW)
- schemas/insight_outcomes.sql           (NEW)
- scripts/migrate_old_insights.py        (NEW - backfill data)
```

**Day 3-4: Core Data Models**
```bash
- src/models/product_status.py           (NEW - ProductStatus enum)
- src/models/trigger_event.py            (NEW - TriggerEvent dataclass)
- src/models/strategic_insight.py        (NEW - StrategicInsight dataclass)
```

**Day 5: Trigger Event Detectors**
```bash
- src/trigger_detection.py               (NEW - detect_trigger_events())
- tests/test_trigger_detection.py        (NEW)
```

### Phase 2: Insight Engine Refactor (Week 2)

**Day 1-3: LLM Prompt Engineering**
```bash
- utils/ai_engine_v2.py                  (REFACTOR)
  - build_strategic_prompt()
  - validate_insight_quality()
  - generate_strategic_insight()
```

**Day 4-5: Insight Pipeline**
```bash
- src/insight_pipeline.py                (NEW)
  - Run trigger detection
  - Generate LLM insights
  - Validate quality
  - Store to database
```

### Phase 3: Action Queue & UI (Week 3)

**Day 1-2: Action Queue Backend**
```bash
- src/action_queue.py                    (NEW)
  - ActionQueueFilter class
  - fetch_actionable_insights()
  - apply_user_filters()
```

**Day 3-4: UI Refactoring**
```bash
- apps/shelfguard_app.py                 (REFACTOR)
  - Remove old problem_category logic
  - Replace with product_status
  - Render with action_type styling

- components/action_card.py              (NEW)
  - render_action_card()
  - render_action_queue()
```

**Day 5: Executive Brief LLM Update**
```bash
- utils/executive_brief.py               (REFACTOR)
  - Inject trigger events into brief
  - Use new insight data structure
```

### Phase 4: Testing & Validation (Week 4)

**Day 1-2: Unit Tests**
```bash
- tests/test_product_status.py
- tests/test_trigger_events.py
- tests/test_insight_quality.py
- tests/test_action_queue_filters.py
```

**Day 3-4: Integration Tests**
```bash
- tests/integration/test_insight_pipeline.py
- tests/integration/test_ui_consistency.py
```

**Day 5: User Acceptance Testing**
```bash
- Load production-like data
- Verify zero UI contradictions
- Verify all insights have $ amounts
- Verify stable products hidden by default
```

---

## Part 6: Success Metrics

### Quantitative KPIs

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| False Positive Rate | ~40% | <10% | % of "action needed" alerts on stable products |
| Insight Specificity | ~20% | 100% | % of recommendations with $ amounts |
| UI Contradictions | ~15 per view | 0 | Manual audit of Action Queue vs Table |
| Action Queue Length | ~80 items | <25 items | Default view item count |
| Insight Confidence | N/A | >70% avg | LLM confidence scores |

### Qualitative Success Criteria

✅ **Zero Contradictions:**
Every action card status matches the table row status exactly.

✅ **Causal Transparency:**
Every insight cites at least one specific trigger event.

✅ **Financial Quantification:**
100% of recommendations include specific dollar amounts or percentages.

✅ **Signal Filtering:**
Default view shows ONLY items requiring action (CRITICAL + OPPORTUNITY).

✅ **Executive Trust:**
Strategic briefs include causal reasoning, not generic observations.

---

## Part 7: Risk Mitigation

### Technical Risks

**Risk: LLM fails to generate valid insights**
*Mitigation:* Fallback to deterministic rules if validation fails. Track failure rate.

**Risk: Schema migration breaks existing features**
*Mitigation:* Keep old columns during transition period. Dual-write for 2 weeks.

**Risk: Trigger event detection misses edge cases**
*Mitigation:* Start with conservative thresholds. Add logging for manual review.

### Product Risks

**Risk: Users want to see stable products**
*Mitigation:* Add "Show All Products" toggle (opt-in for STABLE items).

**Risk: Too many CRITICAL alerts create alert fatigue**
*Mitigation:* Set strict thresholds for CRITICAL status. Cap at 5 items max.

**Risk: Financial predictions are wildly inaccurate**
*Mitigation:* Track actual outcomes in `insight_outcomes` table. Calibrate LLM over time.

---

## Appendix A: Example Output

**Before (Current System):**
```
Action Card: 🚨 HIGH PRIORITY - Fix Ad Spend
Table Status: HARVEST - Cash Cow
Table Recommendation: Monitor - position stable
Executive Brief: Focus on maintaining market share
```

**After (New System):**
```
Action Card:
💰 OPPORTUNITY - PROFIT CAPTURE
Raise price from $24.99 to $26.99 (+$2.00)

Reasoning: Competitor B (B08XYZ789) has <5 units remaining (down from 47 units 7 days ago).
Your review count (450) is 2x category average, providing pricing power. Historical price
elasticity of -0.4 suggests minimal volume impact.

Projected Upside: +$1,850/month
Downside Risk: -$280/month (if demand drops 10%)
Confidence: 85%
Time Window: 7 days
Trigger Events: competitor_oos_imminent, review_advantage

[Execute] [Dismiss] [Details]
```

---

## Appendix B: File Checklist

**New Files to Create:**
- [ ] `schemas/strategic_insights.sql`
- [ ] `schemas/trigger_events.sql`
- [ ] `schemas/insight_outcomes.sql`
- [ ] `src/models/product_status.py`
- [ ] `src/models/trigger_event.py`
- [ ] `src/models/strategic_insight.py`
- [ ] `src/trigger_detection.py`
- [ ] `src/insight_pipeline.py`
- [ ] `src/action_queue.py`
- [ ] `utils/ai_engine_v2.py`
- [ ] `components/action_card.py`
- [ ] `scripts/migrate_old_insights.py`
- [ ] `tests/test_trigger_detection.py`
- [ ] `tests/test_insight_quality.py`
- [ ] `tests/test_action_queue_filters.py`

**Files to Refactor:**
- [ ] `apps/shelfguard_app.py` (Action Queue rendering)
- [ ] `src/recommendations.py` (Merge into new system)
- [ ] `utils/ai_engine.py` (Migrate to ai_engine_v2.py)
- [ ] `utils/executive_brief.py` (Use new insight data)

**Files to Archive (Deprecated):**
- [ ] Old `alerts` table logic
- [ ] Old `problem_category` rendering
- [ ] Conflicting status fields

---

**End of Plan. Ready for Implementation.**
