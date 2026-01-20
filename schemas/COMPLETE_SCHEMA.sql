-- ========================================
-- ShelfGuard COMPLETE Schema
-- ========================================
-- Run this entire file in Supabase SQL Editor
-- Created: 2026-01-19
--
-- This combines:
-- 1. search_to_state.sql (core tables)
-- 2. network_intelligence.sql (AI/network tables)
-- ========================================


-- ========================================
-- PART 1: CORE TABLES
-- ========================================

-- 1. PROJECTS TABLE
-- Stores user-created market monitoring projects
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID,  -- NULL for anonymous users
    project_name TEXT NOT NULL,
    mission_type TEXT NOT NULL CHECK (mission_type IN ('bodyguard', 'scout', 'surgeon')),
    asin_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true
);

-- Index for fast user lookups
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC);

-- Row Level Security (RLS)
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (prevents errors on re-run)
DROP POLICY IF EXISTS "Users can view their own projects" ON projects;
DROP POLICY IF EXISTS "Users can insert their own projects" ON projects;
DROP POLICY IF EXISTS "Users can update their own projects" ON projects;
DROP POLICY IF EXISTS "Users can delete their own projects" ON projects;
DROP POLICY IF EXISTS "Anyone can view projects" ON projects;
DROP POLICY IF EXISTS "Anyone can insert projects" ON projects;
DROP POLICY IF EXISTS "Anyone can update projects" ON projects;
DROP POLICY IF EXISTS "Anyone can delete projects" ON projects;

-- Permissive policies for anonymous access
CREATE POLICY "Anyone can view projects"
    ON projects FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert projects"
    ON projects FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Anyone can update projects"
    ON projects FOR UPDATE
    USING (true);

CREATE POLICY "Anyone can delete projects"
    ON projects FOR DELETE
    USING (true);


-- 2. TRACKED_ASINS TABLE
-- Links ASINs to projects (many-to-many)
CREATE TABLE IF NOT EXISTS tracked_asins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    UNIQUE(project_id, asin)
);

CREATE INDEX IF NOT EXISTS idx_tracked_asins_project_id ON tracked_asins(project_id);
CREATE INDEX IF NOT EXISTS idx_tracked_asins_asin ON tracked_asins(asin);

ALTER TABLE tracked_asins ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view tracked_asins" ON tracked_asins;
DROP POLICY IF EXISTS "Anyone can insert tracked_asins" ON tracked_asins;

CREATE POLICY "Anyone can view tracked_asins"
    ON tracked_asins FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert tracked_asins"
    ON tracked_asins FOR INSERT
    WITH CHECK (true);


-- 3. PRODUCT_SNAPSHOTS TABLE
-- Time-series storage for harvested product data
CREATE TABLE IF NOT EXISTS product_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asin TEXT NOT NULL,
    snapshot_date DATE NOT NULL,

    -- Core signals
    buy_box_price NUMERIC(10, 2),
    amazon_price NUMERIC(10, 2),
    new_fba_price NUMERIC(10, 2),
    sales_rank INTEGER,
    amazon_bb_share NUMERIC(5, 4),
    buy_box_switches INTEGER DEFAULT 0,
    new_offer_count INTEGER,
    review_count INTEGER,
    rating NUMERIC(3, 2),

    -- Calculated metrics
    estimated_units INTEGER,
    estimated_weekly_revenue NUMERIC(12, 2),
    filled_price NUMERIC(10, 2),

    -- Product metadata
    title TEXT,
    brand TEXT,
    parent_asin TEXT,
    main_image TEXT,

    -- Category metadata (for network intelligence)
    category_id INTEGER,
    category_name TEXT,
    category_tree TEXT[],
    category_root TEXT,

    -- Metadata
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source TEXT DEFAULT 'keepa',

    UNIQUE(asin, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_asin_date ON product_snapshots(asin, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON product_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_parent_asin ON product_snapshots(parent_asin);
CREATE INDEX IF NOT EXISTS idx_snapshots_category ON product_snapshots(category_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_category_root ON product_snapshots(category_root, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_brand_category ON product_snapshots(brand, category_root, snapshot_date DESC);

ALTER TABLE product_snapshots ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Anyone can insert product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Anyone can update product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Service role can insert product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Service role can update product snapshots" ON product_snapshots;

CREATE POLICY "Anyone can view product snapshots"
    ON product_snapshots FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert product snapshots"
    ON product_snapshots FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Anyone can update product snapshots"
    ON product_snapshots FOR UPDATE
    USING (true);


-- 4. HISTORICAL_METRICS TABLE
-- Stores 90-day historical Price & BSR data from Keepa backfill
CREATE TABLE IF NOT EXISTS historical_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    sales_rank INTEGER,
    buy_box_price NUMERIC(10, 2),
    amazon_price NUMERIC(10, 2),
    new_fba_price NUMERIC(10, 2),
    -- Additional fields for velocity extraction
    price NUMERIC(10, 2),
    bsr INTEGER,
    revenue NUMERIC(12, 2),
    units INTEGER,
    review_count INTEGER,
    rating NUMERIC(3, 2),
    amazon_bb_share NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(project_id, asin, datetime)
);

CREATE INDEX IF NOT EXISTS idx_historical_metrics_project_asin
    ON historical_metrics(project_id, asin, datetime DESC);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_datetime
    ON historical_metrics(datetime DESC);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_asin
    ON historical_metrics(asin);

ALTER TABLE historical_metrics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view historical_metrics" ON historical_metrics;
DROP POLICY IF EXISTS "Anyone can insert historical_metrics" ON historical_metrics;

CREATE POLICY "Anyone can view historical_metrics"
    ON historical_metrics FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert historical_metrics"
    ON historical_metrics FOR INSERT
    WITH CHECK (true);


-- 5. RESOLUTION_CARDS TABLE
-- Stores generated alerts/recommendations
CREATE TABLE IF NOT EXISTS resolution_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('high', 'medium', 'low')),
    asin TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    action TEXT,
    priority_score NUMERIC(5, 2),
    is_dismissed BOOLEAN DEFAULT false,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_resolution_cards_project_id
    ON resolution_cards(project_id, created_at DESC);

ALTER TABLE resolution_cards ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view resolution_cards" ON resolution_cards;
DROP POLICY IF EXISTS "Anyone can insert resolution_cards" ON resolution_cards;

CREATE POLICY "Anyone can view resolution_cards"
    ON resolution_cards FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert resolution_cards"
    ON resolution_cards FOR INSERT
    WITH CHECK (true);


-- ========================================
-- PART 2: NETWORK INTELLIGENCE TABLES
-- ========================================

-- 6. CATEGORY_INTELLIGENCE TABLE
-- Stores category-level benchmarks
CREATE TABLE IF NOT EXISTS category_intelligence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id INTEGER NOT NULL,
    category_name TEXT NOT NULL,
    category_root TEXT,
    snapshot_date DATE NOT NULL,

    -- Price Benchmarks
    median_price NUMERIC(10, 2),
    p75_price NUMERIC(10, 2),
    p25_price NUMERIC(10, 2),
    avg_price NUMERIC(10, 2),
    price_volatility_score NUMERIC(5, 2),

    -- Quality Benchmarks
    median_rating NUMERIC(3, 2),
    avg_rating NUMERIC(3, 2),
    median_review_count INTEGER,
    avg_review_count INTEGER,

    -- Rank Benchmarks
    median_bsr INTEGER,
    p25_bsr INTEGER,
    p75_bsr INTEGER,

    -- Market Structure
    total_asins_tracked INTEGER,
    avg_offer_count NUMERIC(5, 2),
    avg_bb_share NUMERIC(5, 4),

    -- Revenue Benchmarks
    total_weekly_revenue NUMERIC(12, 2),
    median_weekly_revenue NUMERIC(10, 2),

    -- Intelligence Quality
    data_quality TEXT CHECK (data_quality IN ('HIGH', 'MEDIUM', 'LOW')),
    snapshot_count INTEGER,
    last_updated TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(category_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_category_intelligence_date ON category_intelligence(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_category_intelligence_root ON category_intelligence(category_root, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_category_intelligence_id ON category_intelligence(category_id);

ALTER TABLE category_intelligence ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view category_intelligence" ON category_intelligence;
DROP POLICY IF EXISTS "Anyone can insert category_intelligence" ON category_intelligence;

CREATE POLICY "Anyone can view category_intelligence"
    ON category_intelligence FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert category_intelligence"
    ON category_intelligence FOR INSERT
    WITH CHECK (true);


-- 7. BRAND_INTELLIGENCE TABLE
-- Stores brand-level aggregates
CREATE TABLE IF NOT EXISTS brand_intelligence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand TEXT NOT NULL,
    category_root TEXT NOT NULL,

    -- Aggregate Metrics
    total_asins_tracked INTEGER DEFAULT 0,
    avg_price NUMERIC(10, 2),
    median_price NUMERIC(10, 2),
    avg_rating NUMERIC(3, 2),
    avg_review_count INTEGER,
    total_weekly_revenue NUMERIC(12, 2),
    avg_weekly_revenue NUMERIC(10, 2),
    market_share_pct NUMERIC(5, 2),

    -- Time Series
    first_seen DATE,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    snapshot_count INTEGER DEFAULT 0,

    UNIQUE(brand, category_root)
);

CREATE INDEX IF NOT EXISTS idx_brand_intelligence_category ON brand_intelligence(category_root);
CREATE INDEX IF NOT EXISTS idx_brand_intelligence_brand ON brand_intelligence(brand);
CREATE INDEX IF NOT EXISTS idx_brand_intelligence_revenue ON brand_intelligence(total_weekly_revenue DESC);

ALTER TABLE brand_intelligence ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view brand_intelligence" ON brand_intelligence;
DROP POLICY IF EXISTS "Anyone can insert brand_intelligence" ON brand_intelligence;

CREATE POLICY "Anyone can view brand_intelligence"
    ON brand_intelligence FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert brand_intelligence"
    ON brand_intelligence FOR INSERT
    WITH CHECK (true);


-- 8. MARKET_PATTERNS TABLE
-- Historical pattern library
CREATE TABLE IF NOT EXISTS market_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type TEXT NOT NULL,
    category_root TEXT,
    trigger_conditions JSONB NOT NULL,
    typical_outcome TEXT,
    success_rate NUMERIC(5, 2),
    avg_revenue_impact NUMERIC(10, 2),
    avg_duration_days INTEGER,
    observed_count INTEGER DEFAULT 1,
    first_observed DATE,
    last_observed DATE,
    example_asins TEXT[],
    confidence_score NUMERIC(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patterns_type_category ON market_patterns(pattern_type, category_root);
CREATE INDEX IF NOT EXISTS idx_patterns_observed_count ON market_patterns(observed_count DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON market_patterns(confidence_score DESC);

ALTER TABLE market_patterns ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can view market_patterns" ON market_patterns;
DROP POLICY IF EXISTS "Anyone can insert market_patterns" ON market_patterns;

CREATE POLICY "Anyone can view market_patterns"
    ON market_patterns FOR SELECT
    USING (true);

CREATE POLICY "Anyone can insert market_patterns"
    ON market_patterns FOR INSERT
    WITH CHECK (true);


-- ========================================
-- PART 3: HELPER FUNCTIONS & VIEWS
-- ========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for projects table
DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Latest snapshot per ASIN (for dashboard)
CREATE OR REPLACE VIEW latest_snapshots AS
SELECT DISTINCT ON (asin)
    *
FROM product_snapshots
ORDER BY asin, snapshot_date DESC;

-- Trend comparison view (today vs yesterday)
CREATE OR REPLACE VIEW snapshot_trends AS
SELECT
    t.asin,
    t.snapshot_date as current_date,
    t.buy_box_price as current_price,
    t.sales_rank as current_rank,
    t.amazon_bb_share as current_bb_share,
    t.estimated_weekly_revenue as current_revenue,
    y.buy_box_price as prev_price,
    y.sales_rank as prev_rank,
    y.amazon_bb_share as prev_bb_share,
    y.estimated_weekly_revenue as prev_revenue,
    CASE WHEN y.buy_box_price > 0 THEN
        (t.buy_box_price - y.buy_box_price) / y.buy_box_price * 100
    END as price_change_pct,
    CASE WHEN y.sales_rank > 0 THEN
        (y.sales_rank - t.sales_rank)::float / y.sales_rank * 100
    END as rank_improvement_pct,
    (t.amazon_bb_share - y.amazon_bb_share) * 100 as bb_share_change_pct
FROM product_snapshots t
LEFT JOIN product_snapshots y
    ON t.asin = y.asin
    AND y.snapshot_date = t.snapshot_date - INTERVAL '1 day'
WHERE t.snapshot_date = CURRENT_DATE;

-- Latest category intelligence
CREATE OR REPLACE VIEW latest_category_intelligence AS
SELECT DISTINCT ON (category_id)
    *
FROM category_intelligence
ORDER BY category_id, snapshot_date DESC;

-- Top brands by revenue
CREATE OR REPLACE VIEW top_brands AS
SELECT
    brand,
    category_root,
    total_weekly_revenue,
    total_asins_tracked,
    market_share_pct,
    avg_price,
    avg_rating
FROM brand_intelligence
ORDER BY total_weekly_revenue DESC
LIMIT 100;

-- High confidence patterns
CREATE OR REPLACE VIEW reliable_patterns AS
SELECT *
FROM market_patterns
WHERE confidence_score > 0.7
  AND observed_count >= 5
ORDER BY confidence_score DESC, observed_count DESC;


-- ========================================
-- DONE!
-- ========================================
-- Tables created:
--   1. projects
--   2. tracked_asins
--   3. product_snapshots (with category columns)
--   4. historical_metrics
--   5. resolution_cards
--   6. category_intelligence
--   7. brand_intelligence
--   8. market_patterns
--
-- Views created:
--   - latest_snapshots
--   - snapshot_trends
--   - latest_category_intelligence
--   - top_brands
--   - reliable_patterns
-- ========================================
