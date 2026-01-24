-- ========================================
-- P3: Historical Attribution Tracking Migration
-- ========================================
-- Run this to update revenue_attributions table for ASIN-level tracking
-- Created: 2026-01-24
-- ========================================

-- Drop existing revenue_attributions table if it exists (will recreate with new schema)
DROP TABLE IF EXISTS attribution_drivers CASCADE;
DROP TABLE IF EXISTS revenue_attributions CASCADE;

-- 12. REVENUE_ATTRIBUTIONS TABLE (P3: Enhanced)
-- Stores top-level causal breakdown per ASIN per date
CREATE TABLE revenue_attributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    attribution_date DATE NOT NULL,

    -- Revenue Deltas
    total_revenue_delta NUMERIC(12, 2),
    previous_revenue NUMERIC(12, 2),
    current_revenue NUMERIC(12, 2),

    -- Causal Categories (Absolute $)
    internal_contribution NUMERIC(12, 2),
    competitive_contribution NUMERIC(12, 2),
    macro_contribution NUMERIC(12, 2),
    platform_contribution NUMERIC(12, 2),

    -- Confidence & Metadata
    explained_variance NUMERIC(4, 3), -- 0.000 to 1.000
    confidence NUMERIC(4, 3),
    residual NUMERIC(12, 2),

    -- Detailed Drivers (JSONB for flexibility)
    internal_drivers JSONB DEFAULT '[]'::jsonb,
    competitive_drivers JSONB DEFAULT '[]'::jsonb,
    macro_drivers JSONB DEFAULT '[]'::jsonb,
    platform_drivers JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, asin, attribution_date)
);

CREATE INDEX idx_attribution_project_asin ON revenue_attributions(project_id, asin, attribution_date DESC);

ALTER TABLE revenue_attributions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view revenue_attributions" ON revenue_attributions;
DROP POLICY IF EXISTS "Anyone can insert revenue_attributions" ON revenue_attributions;
DROP POLICY IF EXISTS "Anyone can update revenue_attributions" ON revenue_attributions;

CREATE POLICY "Anyone can view revenue_attributions"
    ON revenue_attributions FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert revenue_attributions"
    ON revenue_attributions FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Anyone can update revenue_attributions"
    ON revenue_attributions FOR UPDATE
    USING (true);


-- 13. KEYWORD_RANKS TABLE (P3: Share of Voice Tracking)
-- Stores keyword ranking history for Share of Voice analysis
CREATE TABLE IF NOT EXISTS keyword_ranks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    keyword TEXT NOT NULL,
    rank INTEGER,
    search_volume INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(asin, keyword, timestamp)
);

CREATE INDEX idx_keyword_ranks_asin ON keyword_ranks(asin, timestamp DESC);
CREATE INDEX idx_keyword_ranks_project ON keyword_ranks(project_id, timestamp DESC);

ALTER TABLE keyword_ranks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view keyword_ranks" ON keyword_ranks;
DROP POLICY IF EXISTS "Anyone can insert keyword_ranks" ON keyword_ranks;

CREATE POLICY "Anyone can view keyword_ranks"
    ON keyword_ranks FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert keyword_ranks"
    ON keyword_ranks FOR INSERT
    WITH CHECK (true);


-- 14. FORECAST_ACCURACY TABLE (P3: Track Forecast Performance)
-- Stores predicted vs actual outcomes for forecast validation
CREATE TABLE IF NOT EXISTS forecast_accuracy (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_horizon_days INTEGER NOT NULL, -- 30, 60, or 90

    -- Predictions
    predicted_revenue NUMERIC(12, 2),
    predicted_lower_bound NUMERIC(12, 2),
    predicted_upper_bound NUMERIC(12, 2),
    confidence_interval NUMERIC(3, 2), -- 0.80 = 80% CI

    -- Actuals (filled in after horizon passes)
    actual_revenue NUMERIC(12, 2),
    actual_date DATE,

    -- Accuracy Metrics
    absolute_error NUMERIC(12, 2),
    percentage_error NUMERIC(6, 2),
    within_confidence_interval BOOLEAN,

    -- Metadata
    forecast_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, asin, forecast_date, forecast_horizon_days)
);

CREATE INDEX idx_forecast_accuracy_project ON forecast_accuracy(project_id, asin, forecast_date DESC);

ALTER TABLE forecast_accuracy ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view forecast_accuracy" ON forecast_accuracy;
DROP POLICY IF EXISTS "Anyone can insert forecast_accuracy" ON forecast_accuracy;
DROP POLICY IF EXISTS "Anyone can update forecast_accuracy" ON forecast_accuracy;

CREATE POLICY "Anyone can view forecast_accuracy"
    ON forecast_accuracy FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert forecast_accuracy"
    ON forecast_accuracy FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Anyone can update forecast_accuracy"
    ON forecast_accuracy FOR UPDATE
    USING (true);
