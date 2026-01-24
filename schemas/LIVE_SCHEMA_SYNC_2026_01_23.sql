-- ========================================
-- LIVE SCHEMA SYNC (2026-01-23)
-- Synced from live Supabase instance
-- ========================================

-- 1. STRATEGIC_INSIGHTS TABLE
-- Stores AI-generated intelligence, attribution, and forecasting
CREATE TABLE IF NOT EXISTS strategic_insights (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    asin text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    expires_at timestamp with time zone NOT NULL DEFAULT (now() + '30 days'::interval),
    is_active boolean DEFAULT true,
    product_status text NOT NULL,
    status_priority integer NOT NULL,
    status_changed_at timestamp with time zone NOT NULL DEFAULT now(),
    strategic_state text,
    recommendation text NOT NULL,
    reasoning text NOT NULL,
    confidence double precision,
    llm_model text DEFAULT 'gpt-4o-mini'::text,
    llm_prompt_hash text,
    trigger_events jsonb NOT NULL,
    primary_trigger text,
    projected_upside_monthly numeric,
    downside_risk_monthly numeric,
    net_expected_value numeric,
    thirty_day_risk numeric,
    thirty_day_growth numeric,
    price_erosion_risk numeric,
    share_erosion_risk numeric,
    stockout_risk numeric,
    action_type text NOT NULL,
    time_horizon_days integer,
    confidence_factors ARRAY,
    validation_passed boolean DEFAULT true,
    validation_errors ARRAY,
    user_dismissed boolean DEFAULT false,
    dismissed_at timestamp with time zone,
    user_feedback text,
    
    -- Phase 2/2.5 Columns
    revenue_attribution jsonb,
    anticipated_events jsonb DEFAULT '[]'::jsonb,
    scenarios jsonb DEFAULT '[]'::jsonb,
    sustainable_run_rate numeric,
    category_benchmarks_summary text,
    competitive_position_summary text
);

-- 2. TRIGGER_EVENTS TABLE
-- Stores individual detected market events
CREATE TABLE IF NOT EXISTS trigger_events (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    asin text NOT NULL,
    event_type text NOT NULL,
    severity integer,
    detected_at timestamp with time zone NOT NULL DEFAULT now(),
    metric_name text NOT NULL,
    baseline_value numeric,
    current_value numeric,
    delta_pct numeric,
    related_asin text,
    generated_insight_id uuid
);
