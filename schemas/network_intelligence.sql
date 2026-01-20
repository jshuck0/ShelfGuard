-- ========================================
-- ShelfGuard Network Intelligence Schema
-- ========================================
-- Self-learning AI system via data accumulation
-- Part of unified intelligence system

-- ========================================
-- 1. EXTEND product_snapshots TABLE
-- ========================================
-- Add category metadata for network intelligence

-- Check if columns exist before adding
DO $$
BEGIN
    -- Add category_id if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='product_snapshots' AND column_name='category_id') THEN
        ALTER TABLE product_snapshots ADD COLUMN category_id INTEGER;
    END IF;

    -- Add category_name if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='product_snapshots' AND column_name='category_name') THEN
        ALTER TABLE product_snapshots ADD COLUMN category_name TEXT;
    END IF;

    -- Add category_tree if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='product_snapshots' AND column_name='category_tree') THEN
        ALTER TABLE product_snapshots ADD COLUMN category_tree TEXT[];
    END IF;

    -- Add category_root if not exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='product_snapshots' AND column_name='category_root') THEN
        ALTER TABLE product_snapshots ADD COLUMN category_root TEXT;
    END IF;
END $$;

-- Create indexes for category queries
CREATE INDEX IF NOT EXISTS idx_snapshots_category ON product_snapshots(category_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_category_root ON product_snapshots(category_root, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_brand_category ON product_snapshots(brand, category_root, snapshot_date DESC);


-- ========================================
-- 2. CATEGORY INTELLIGENCE TABLE
-- ========================================
-- Stores category-level benchmarks calculated from accumulated data
CREATE TABLE IF NOT EXISTS category_intelligence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    category_id INTEGER NOT NULL,
    category_name TEXT NOT NULL,
    category_root TEXT,
    snapshot_date DATE NOT NULL,

    -- Price Benchmarks
    median_price NUMERIC(10, 2),
    p75_price NUMERIC(10, 2),               -- 75th percentile (premium)
    p25_price NUMERIC(10, 2),               -- 25th percentile (budget)
    avg_price NUMERIC(10, 2),
    price_volatility_score NUMERIC(5, 2),  -- Std dev of prices

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
    snapshot_count INTEGER,                 -- How many ASINs we have data for
    last_updated TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(category_id, snapshot_date)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_category_intelligence_date ON category_intelligence(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_category_intelligence_root ON category_intelligence(category_root, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_category_intelligence_id ON category_intelligence(category_id);


-- ========================================
-- 3. BRAND INTELLIGENCE TABLE
-- ========================================
-- Stores brand-level aggregates calculated from accumulated data
CREATE TABLE IF NOT EXISTS brand_intelligence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity
    brand TEXT NOT NULL,
    category_root TEXT NOT NULL,

    -- Aggregate Metrics (updated daily)
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

    -- Metadata
    snapshot_count INTEGER DEFAULT 0,       -- How many data points we have

    UNIQUE(brand, category_root)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_brand_intelligence_category ON brand_intelligence(category_root);
CREATE INDEX IF NOT EXISTS idx_brand_intelligence_brand ON brand_intelligence(brand);
CREATE INDEX IF NOT EXISTS idx_brand_intelligence_revenue ON brand_intelligence(total_weekly_revenue DESC);


-- ========================================
-- 4. MARKET PATTERNS TABLE
-- ========================================
-- Historical pattern library: store patterns we've observed
CREATE TABLE IF NOT EXISTS market_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pattern Identity
    pattern_type TEXT NOT NULL,             -- 'competitor_oos_price_spike', 'review_advantage_premium', etc.
    category_root TEXT,

    -- Pattern Signature (what triggers it)
    trigger_conditions JSONB NOT NULL,      -- {"competitor_inventory": "<5", "review_advantage": ">2x"}

    -- Historical Outcomes (what usually happens)
    typical_outcome TEXT,
    success_rate NUMERIC(5, 2),             -- % of times pattern leads to predicted outcome
    avg_revenue_impact NUMERIC(10, 2),
    avg_duration_days INTEGER,

    -- Sample Data
    observed_count INTEGER DEFAULT 1,
    first_observed DATE,
    last_observed DATE,
    example_asins TEXT[],                   -- Sample ASINs where we saw this

    -- Metadata
    confidence_score NUMERIC(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_patterns_type_category ON market_patterns(pattern_type, category_root);
CREATE INDEX IF NOT EXISTS idx_patterns_observed_count ON market_patterns(observed_count DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON market_patterns(confidence_score DESC);


-- ========================================
-- HELPER FUNCTIONS
-- ========================================

-- Function to update brand intelligence aggregates
CREATE OR REPLACE FUNCTION update_brand_intelligence()
RETURNS TRIGGER AS $$
BEGIN
    -- Recalculate brand aggregates when new snapshot added
    INSERT INTO brand_intelligence (
        brand,
        category_root,
        total_asins_tracked,
        avg_price,
        median_price,
        avg_rating,
        avg_review_count,
        total_weekly_revenue,
        snapshot_count,
        first_seen,
        last_updated
    )
    SELECT
        NEW.brand,
        NEW.category_root,
        COUNT(DISTINCT asin),
        AVG(buy_box_price),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY buy_box_price),
        AVG(rating),
        AVG(review_count),
        SUM(estimated_weekly_revenue),
        COUNT(*),
        MIN(snapshot_date),
        NOW()
    FROM product_snapshots
    WHERE brand = NEW.brand
      AND category_root = NEW.category_root
      AND snapshot_date >= CURRENT_DATE - INTERVAL '30 days'
    ON CONFLICT (brand, category_root)
    DO UPDATE SET
        total_asins_tracked = EXCLUDED.total_asins_tracked,
        avg_price = EXCLUDED.avg_price,
        median_price = EXCLUDED.median_price,
        avg_rating = EXCLUDED.avg_rating,
        avg_review_count = EXCLUDED.avg_review_count,
        total_weekly_revenue = EXCLUDED.total_weekly_revenue,
        snapshot_count = EXCLUDED.snapshot_count,
        last_updated = NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update brand intelligence
-- Note: Disabled by default for performance, run as batch job instead
-- CREATE TRIGGER trigger_update_brand_intelligence
--     AFTER INSERT ON product_snapshots
--     FOR EACH ROW
--     EXECUTE FUNCTION update_brand_intelligence();


-- Function to update category intelligence aggregates
CREATE OR REPLACE FUNCTION calculate_category_intelligence(
    p_category_id INTEGER,
    p_snapshot_date DATE DEFAULT CURRENT_DATE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO category_intelligence (
        category_id,
        category_name,
        category_root,
        snapshot_date,
        median_price,
        p75_price,
        p25_price,
        avg_price,
        price_volatility_score,
        median_rating,
        avg_rating,
        median_review_count,
        avg_review_count,
        median_bsr,
        p25_bsr,
        p75_bsr,
        total_asins_tracked,
        total_weekly_revenue,
        median_weekly_revenue,
        data_quality,
        snapshot_count,
        last_updated
    )
    SELECT
        p_category_id,
        MAX(category_name),
        MAX(category_root),
        p_snapshot_date,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY buy_box_price),
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY buy_box_price),
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY buy_box_price),
        AVG(buy_box_price),
        STDDEV(buy_box_price),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rating),
        AVG(rating),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY review_count),
        AVG(review_count),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sales_rank),
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY sales_rank),
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY sales_rank),
        COUNT(DISTINCT asin),
        SUM(estimated_weekly_revenue),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY estimated_weekly_revenue),
        CASE
            WHEN COUNT(*) > 100 THEN 'HIGH'
            WHEN COUNT(*) > 20 THEN 'MEDIUM'
            ELSE 'LOW'
        END,
        COUNT(*),
        NOW()
    FROM product_snapshots
    WHERE category_id = p_category_id
      AND snapshot_date = p_snapshot_date
    ON CONFLICT (category_id, snapshot_date)
    DO UPDATE SET
        median_price = EXCLUDED.median_price,
        p75_price = EXCLUDED.p75_price,
        p25_price = EXCLUDED.p25_price,
        avg_price = EXCLUDED.avg_price,
        price_volatility_score = EXCLUDED.price_volatility_score,
        median_rating = EXCLUDED.median_rating,
        avg_rating = EXCLUDED.avg_rating,
        median_review_count = EXCLUDED.median_review_count,
        avg_review_count = EXCLUDED.avg_review_count,
        median_bsr = EXCLUDED.median_bsr,
        p25_bsr = EXCLUDED.p25_bsr,
        p75_bsr = EXCLUDED.p75_bsr,
        total_asins_tracked = EXCLUDED.total_asins_tracked,
        total_weekly_revenue = EXCLUDED.total_weekly_revenue,
        median_weekly_revenue = EXCLUDED.median_weekly_revenue,
        data_quality = EXCLUDED.data_quality,
        snapshot_count = EXCLUDED.snapshot_count,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;


-- ========================================
-- HELPER VIEWS
-- ========================================

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
-- RLS POLICIES
-- ========================================

ALTER TABLE category_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE brand_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_patterns ENABLE ROW LEVEL SECURITY;

-- Network intelligence is readable by all users (aggregate public data)
CREATE POLICY "Anyone can view category_intelligence"
    ON category_intelligence FOR SELECT
    USING (true);

CREATE POLICY "Anyone can view brand_intelligence"
    ON brand_intelligence FOR SELECT
    USING (true);

CREATE POLICY "Anyone can view market_patterns"
    ON market_patterns FOR SELECT
    USING (true);

-- Only service role can write
CREATE POLICY "Service role can manage category_intelligence"
    ON category_intelligence
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role can manage brand_intelligence"
    ON brand_intelligence
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role can manage market_patterns"
    ON market_patterns
    USING (true)
    WITH CHECK (true);
