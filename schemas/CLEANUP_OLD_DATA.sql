-- CLEANUP_OLD_DATA.sql
-- =====================
-- Run this AFTER deploying the rating/review index fix to clear corrupted data
-- This truncates metric data tables while preserving project metadata
--
-- HOW TO RUN:
-- 1. Go to Supabase Dashboard > SQL Editor
-- 2. Paste this entire script
-- 3. Click "Run"
--
-- SAFE TABLES (NOT truncated):
-- - projects (project names, settings)
-- - project_asins / tracked_asins (ASIN lists per project)

-- Disable triggers temporarily for faster truncation
SET session_replication_role = replica;

-- Truncate core data tables (these should always exist)
TRUNCATE TABLE IF EXISTS product_snapshots CASCADE;
TRUNCATE TABLE IF EXISTS historical_metrics CASCADE;
TRUNCATE TABLE IF EXISTS category_intelligence CASCADE;

-- Truncate optional tables (may not exist in all deployments)
TRUNCATE TABLE IF EXISTS llm_cache CASCADE;
TRUNCATE TABLE IF EXISTS strategic_insights CASCADE;
TRUNCATE TABLE IF EXISTS trigger_events CASCADE;
TRUNCATE TABLE IF EXISTS brand_intelligence CASCADE;
TRUNCATE TABLE IF EXISTS market_patterns CASCADE;
TRUNCATE TABLE IF EXISTS resolution_cards CASCADE;

-- Re-enable triggers
SET session_replication_role = DEFAULT;

-- Verify core tables are empty
SELECT 'product_snapshots' as table_name, COUNT(*) as row_count FROM product_snapshots
UNION ALL SELECT 'historical_metrics', COUNT(*) FROM historical_metrics
UNION ALL SELECT 'category_intelligence', COUNT(*) FROM category_intelligence;

-- Expected output: All tables should show row_count = 0
