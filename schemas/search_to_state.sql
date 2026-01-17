-- ========================================
-- ShelfGuard Search-to-State Schema
-- ========================================
-- Run this in Supabase SQL Editor to create required tables

-- 1. PROJECTS TABLE
-- Stores user-created market monitoring projects
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,  -- NULL for anonymous users
    project_name TEXT NOT NULL,
    mission_type TEXT NOT NULL CHECK (mission_type IN ('bodyguard', 'scout', 'surgeon')),
    asin_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true
);

-- Index for fast user lookups
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC);

-- Row Level Security (RLS) - Users can only see their own projects
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own projects"
    ON projects FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert their own projects"
    ON projects FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own projects"
    ON projects FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects"
    ON projects FOR DELETE
    USING (auth.uid() = user_id);


-- 2. TRACKED_ASINS TABLE
-- Links ASINs to projects (many-to-many)
CREATE TABLE IF NOT EXISTS tracked_asins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    UNIQUE(project_id, asin)  -- Prevent duplicate ASINs in same project
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_tracked_asins_project_id ON tracked_asins(project_id);
CREATE INDEX IF NOT EXISTS idx_tracked_asins_asin ON tracked_asins(asin);

-- RLS: Inherit from projects table
ALTER TABLE tracked_asins ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view tracked ASINs for their projects"
    ON tracked_asins FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = tracked_asins.project_id
            AND (projects.user_id = auth.uid() OR projects.user_id IS NULL)
        )
    );

CREATE POLICY "Users can insert tracked ASINs for their projects"
    ON tracked_asins FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = tracked_asins.project_id
            AND (projects.user_id = auth.uid() OR projects.user_id IS NULL)
        )
    );


-- 3. HISTORICAL_METRICS TABLE
-- Stores 90-day historical Price & BSR data from Keepa backfill
CREATE TABLE IF NOT EXISTS historical_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    sales_rank INTEGER,
    buy_box_price NUMERIC(10, 2),
    amazon_price NUMERIC(10, 2),
    new_fba_price NUMERIC(10, 2),
    UNIQUE(project_id, asin, datetime)  -- Prevent duplicates
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_historical_metrics_project_asin
    ON historical_metrics(project_id, asin, datetime DESC);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_datetime
    ON historical_metrics(datetime DESC);

-- RLS: Inherit from projects
ALTER TABLE historical_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view historical metrics for their projects"
    ON historical_metrics FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = historical_metrics.project_id
            AND (projects.user_id = auth.uid() OR projects.user_id IS NULL)
        )
    );

CREATE POLICY "Users can insert historical metrics for their projects"
    ON historical_metrics FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = historical_metrics.project_id
            AND (projects.user_id = auth.uid() OR projects.user_id IS NULL)
        )
    );


-- 4. RESOLUTION_CARDS TABLE (Optional - for persistence)
-- Stores generated alerts/recommendations
CREATE TABLE IF NOT EXISTS resolution_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    alert_type TEXT NOT NULL,  -- 'volume_stealer', 'efficiency_gap', etc.
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

-- Index for fast project lookups
CREATE INDEX IF NOT EXISTS idx_resolution_cards_project_id
    ON resolution_cards(project_id, created_at DESC);

-- RLS: Inherit from projects
ALTER TABLE resolution_cards ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view resolution cards for their projects"
    ON resolution_cards FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = resolution_cards.project_id
            AND (projects.user_id = auth.uid() OR projects.user_id IS NULL)
        )
    );


-- ========================================
-- HELPER FUNCTIONS
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
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ========================================
-- MIGRATION NOTES
-- ========================================
/*
Run this script in Supabase SQL Editor.

After running:
1. Verify RLS policies are active
2. Test anonymous access (user_id = NULL works)
3. Grant appropriate permissions for service role if using server-side functions

For production:
- Add indexes for specific query patterns
- Consider partitioning historical_metrics by project_id if data grows large
- Add retention policy for old resolution_cards (auto-delete after 30 days)
*/
