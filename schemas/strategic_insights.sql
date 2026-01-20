-- ========================================
-- ShelfGuard Strategic Insights Schema
-- ========================================
-- Insight Engine: High-fidelity recommendation generation
-- Part of unified intelligence system

-- ========================================
-- 1. STRATEGIC INSIGHTS TABLE
-- ========================================
-- Stores validated, actionable insights for products
CREATE TABLE IF NOT EXISTS strategic_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core Identity
    asin TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    is_active BOOLEAN DEFAULT TRUE,

    -- Unified Status (THE ONE TRUTH)
    product_status TEXT NOT NULL,  -- ProductStatus enum value
    status_priority INT NOT NULL,  -- 100=CRITICAL, 75=OPPORTUNITY, 50=WATCH, 0=STABLE
    status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Strategic Classification (backward compat)
    strategic_state TEXT,  -- FORTRESS, HARVEST, TRENCH_WAR, DISTRESS, TERMINAL

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

    -- Predictive Intelligence
    thirty_day_risk DECIMAL(10,2),          -- 30-day risk forecast
    thirty_day_growth DECIMAL(10,2),        -- 30-day growth potential
    price_erosion_risk DECIMAL(10,2),
    share_erosion_risk DECIMAL(10,2),
    stockout_risk DECIMAL(10,2),

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
CREATE INDEX IF NOT EXISTS idx_insights_status_priority ON strategic_insights(status_priority DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_insights_asin_active ON strategic_insights(asin, is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_insights_action_type ON strategic_insights(action_type, status_priority DESC);
CREATE INDEX IF NOT EXISTS idx_insights_expires ON strategic_insights(expires_at) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_insights_asin ON strategic_insights(asin);

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


-- ========================================
-- 2. TRIGGER EVENTS TABLE
-- ========================================
-- Tracks discrete market changes that justify insights
CREATE TABLE IF NOT EXISTS trigger_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    asin TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'competitor_oos_imminent', 'price_war_active', etc.
    severity INT CHECK (severity >= 1 AND severity <= 10),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event Details
    metric_name TEXT NOT NULL,              -- 'fba_inventory', 'buy_box_price', etc.
    baseline_value DECIMAL(12,2),           -- Value before change
    current_value DECIMAL(12,2),            -- Value after change
    delta_pct DECIMAL(6,2),                 -- Percentage change

    -- Related Products
    related_asin TEXT,                      -- Competitor ASIN if relevant

    -- Insight Linkage
    generated_insight_id UUID REFERENCES strategic_insights(id) ON DELETE SET NULL,

    -- Deduplication
    UNIQUE(asin, event_type, detected_at)
);

CREATE INDEX IF NOT EXISTS idx_trigger_events_asin ON trigger_events(asin, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_trigger_events_severity ON trigger_events(severity DESC, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_trigger_events_type ON trigger_events(event_type, detected_at DESC);


-- ========================================
-- 3. INSIGHT OUTCOMES TABLE
-- ========================================
-- Tracks prediction accuracy for feedback loop
CREATE TABLE IF NOT EXISTS insight_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID NOT NULL REFERENCES strategic_insights(id) ON DELETE CASCADE,

    -- Did the user take action?
    action_taken BOOLEAN,
    action_taken_at TIMESTAMPTZ,

    -- Did the insight prediction come true?
    actual_outcome TEXT,  -- 'upside_realized', 'downside_occurred', 'no_change', 'unknown'
    actual_revenue_impact DECIMAL(10,2),        -- Measured result
    predicted_revenue_impact DECIMAL(10,2),     -- What we predicted
    prediction_error_pct DECIMAL(6,2),          -- How accurate was our LLM?

    -- Feedback loop for LLM improvement
    measured_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_outcome CHECK (actual_outcome IN ('upside_realized', 'downside_occurred', 'no_change', 'unknown'))
);

CREATE INDEX IF NOT EXISTS idx_outcomes_insight ON insight_outcomes(insight_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_measured ON insight_outcomes(measured_at DESC);


-- ========================================
-- RLS POLICIES
-- ========================================
-- Strategic insights are private to users

ALTER TABLE strategic_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE trigger_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE insight_outcomes ENABLE ROW LEVEL SECURITY;

-- For now, allow service role full access
-- TODO: Add user-based policies when we have user_id tracking

CREATE POLICY "Service role can manage strategic_insights"
    ON strategic_insights
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role can manage trigger_events"
    ON trigger_events
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role can manage insight_outcomes"
    ON insight_outcomes
    USING (true)
    WITH CHECK (true);


-- ========================================
-- HELPER VIEWS
-- ========================================

-- Active insights by priority
CREATE OR REPLACE VIEW active_insights AS
SELECT *
FROM strategic_insights
WHERE is_active = TRUE
  AND validation_passed = TRUE
  AND user_dismissed = FALSE
ORDER BY status_priority DESC, net_expected_value DESC;

-- Critical alerts only
CREATE OR REPLACE VIEW critical_alerts AS
SELECT *
FROM strategic_insights
WHERE is_active = TRUE
  AND validation_passed = TRUE
  AND user_dismissed = FALSE
  AND status_priority = 100
ORDER BY created_at DESC;

-- Opportunities only
CREATE OR REPLACE VIEW opportunities AS
SELECT *
FROM strategic_insights
WHERE is_active = TRUE
  AND validation_passed = TRUE
  AND user_dismissed = FALSE
  AND status_priority = 75
  AND action_type = 'PROFIT_CAPTURE'
ORDER BY projected_upside_monthly DESC;
