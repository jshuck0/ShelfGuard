# ShelfGuard Tree Shake Audit Report
**Date:** 2026-01-20
**Auditor:** Claude Code Agent
**Scope:** Complete codebase, database schema, and data flow audit

---

## EXECUTIVE SUMMARY

| Category | Status | Issues Found |
|----------|--------|--------------|
| **Codebase Structure** | ✅ Good | 21 active modules, 10 orphaned |
| **Database Schema** | ❌ CRITICAL | 3 missing tables, 2 missing columns |
| **Data Flow (Write→Read)** | ⚠️ BROKEN | Column name mismatches |
| **Column Naming** | ⚠️ INCONSISTENT | Discovery vs Keepa naming clash |

---

## PART 1: CRITICAL DATABASE ISSUES

### 1.1 Missing Tables in COMPLETE_SCHEMA.sql

The following tables are referenced in code but **NOT in the schema**:

| Table | Used By | Purpose |
|-------|---------|---------|
| `llm_cache` | two_phase_discovery.py:219, 257 | Cache LLM responses to reduce API costs |
| `strategic_insights` | intelligence_pipeline.py:457, 505 | Store AI-generated insights |
| `trigger_events` | intelligence_pipeline.py:479 | Store market trigger events |

**FIX REQUIRED:** Add these tables to COMPLETE_SCHEMA.sql

### 1.2 Missing Columns in product_snapshots

The code writes these columns but they're **NOT in the schema**:

| Column | Written By | Purpose |
|--------|-----------|---------|
| `seller_count` | data_accumulation.py:118 | Number of sellers |
| `competitor_oos_pct` | data_accumulation.py:119 | Competitor out-of-stock % |

**FIX REQUIRED:** Add columns to product_snapshots table

### 1.3 Critical Column Name Mismatch: historical_metrics

**THE PROBLEM:**

`backfill.py` WRITES these columns:
```
- sales_rank
- buy_box_price
- amazon_price
- new_fba_price
```

`supabase_reader.py` READS expecting these columns:
```
- price → filled_price
- bsr → sales_rank_filled
- revenue → weekly_sales_filled
- units → estimated_units
```

**RESULT:** The reader maps columns that don't exist! Velocity extraction fails silently.

**FIX OPTIONS:**
1. Update backfill.py to write the expected column names
2. Update supabase_reader.py to map the actual column names
3. Add both sets of columns to the table (redundant but safe)

---

## PART 2: COLUMN NAMING INCONSISTENCIES

### 2.1 Discovery Phase vs Keepa Phase Naming

The codebase has TWO naming conventions that clash:

**Discovery Phase (two_phase_discovery, data_accumulation):**
```python
- price          # Buy box price
- bsr            # Best Seller Rank
- revenue_proxy  # Estimated revenue
- monthly_units  # Monthly sales units
```

**Keepa/Dashboard Phase (keepa_client, supabase_reader):**
```python
- filled_price        # Buy box price (filled gaps)
- sales_rank_filled   # BSR (interpolated)
- weekly_sales_filled # Weekly revenue
- estimated_units     # Weekly units
```

**IMPACT:**
- data_accumulation.py expects `price`, `bsr`, `revenue_proxy`
- supabase_reader.py normalizes to `filled_price`, `sales_rank_filled`
- Dashboard expects `revenue_proxy` (fixed with alias)
- AI engine gets confused by inconsistent naming

### 2.2 Complete Column Mapping Table

| Discovery Name | Keepa Name | DB Column | Dashboard Expects |
|---------------|------------|-----------|-------------------|
| `price` | `filled_price` | `buy_box_price` | `buy_box_price` |
| `bsr` | `sales_rank_filled` | `sales_rank` | `sales_rank_filled` |
| `revenue_proxy` | `weekly_sales_filled` | `estimated_weekly_revenue` | `revenue_proxy` ✅ (aliased) |
| `monthly_units` | `estimated_units` | `estimated_units` | `estimated_units` |

---

## PART 3: ORPHANED FILES (Safe to Delete)

### 3.1 Debug/Test Files (10 files)
| File | Reason |
|------|--------|
| `apps/debug_llm_engine.py` | Debug-only UI, not imported |
| `scrapers/aislegopher_scraper.py` | No active imports |
| `scrapers/flipp_scraper.py` | No active imports |
| `scrapers/popgot_scraper.py` | No active imports |
| `pipelines/weekly_refresh_allscrapers.py` | Only imports orphaned scrapers |
| `pipelines/smoke_test_pipeline.py` | Test utility |
| `pipelines/run_migration.py` | One-time utility |
| `tests/audit_finance.py` | Test file |
| `tests/tests_keepa_processing.py` | Test file |
| `examples/use_intelligence_pipeline.py` | Documentation example |

**RECOMMENDATION:** Keep for now but exclude from production deployment.

---

## PART 4: DATA FLOW AUDIT

### 4.1 product_snapshots Table

**WRITES (3 sources):**
1. `supabase_reader.cache_market_snapshot()` - After discovery
2. `data_accumulation._store_product_snapshots()` - Network intelligence
3. `pipelines/harvest_tracked_asins.py` - Weekly refresh

**READS (4 locations):**
1. `supabase_reader.load_latest_snapshots()` - Dashboard load
2. `supabase_reader.load_snapshot_history()` - Trend detection
3. `network_intelligence.get_category_position()` - AI context
4. `two_phase_discovery._check_search_cache()` - Cache check

**STATUS:** ✅ Working - normalized in `_normalize_snapshot_to_dashboard()`

### 4.2 historical_metrics Table

**WRITES (1 source):**
1. `backfill.upsert_historical_metrics()` - After project creation

**READS (2 locations):**
1. `supabase_reader.load_historical_metrics_from_db()` - Velocity extraction
2. `search_to_state_ui.py:826` - Project dashboard

**STATUS:** ❌ BROKEN - Column name mismatch (see 1.3)

### 4.3 category_intelligence Table

**WRITES (1 source):**
1. `data_accumulation._update_category_intelligence()` - After discovery

**READS (2 locations):**
1. `supabase_reader.get_market_snapshot_with_network_intelligence()` - Dashboard
2. `network_intelligence.get_category_benchmarks()` - AI context

**STATUS:** ✅ Working

### 4.4 brand_intelligence Table

**WRITES (1 source):**
1. `data_accumulation._update_brand_intelligence()` - After discovery

**READS (1 location):**
1. `network_intelligence.get_brand_performance()` - AI context

**STATUS:** ✅ Working

### 4.5 market_patterns Table

**WRITES (1 source):**
1. `data_accumulation._store_market_pattern()` - Pattern detection

**READS (1 location):**
1. `network_intelligence.get_historical_pattern()` - AI context

**STATUS:** ✅ Working

### 4.6 Missing Table Data Flows

| Table | Writes | Reads | Status |
|-------|--------|-------|--------|
| `llm_cache` | two_phase_discovery.py | two_phase_discovery.py | ❌ Table missing |
| `strategic_insights` | intelligence_pipeline.py | intelligence_pipeline.py | ❌ Table missing |
| `trigger_events` | intelligence_pipeline.py | (none found) | ❌ Table missing |

---

## PART 5: FIXES REQUIRED

### 5.1 Schema Fixes (COMPLETE_SCHEMA.sql)

**Add missing tables:**

```sql
-- LLM CACHE TABLE
CREATE TABLE IF NOT EXISTS llm_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key TEXT NOT NULL UNIQUE,
    prompt_hash TEXT NOT NULL,
    response JSONB NOT NULL,
    model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_llm_cache_key ON llm_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_llm_cache_expires ON llm_cache(expires_at);

-- STRATEGIC INSIGHTS TABLE
CREATE TABLE IF NOT EXISTS strategic_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asin TEXT NOT NULL,
    user_id UUID,
    generated_at TIMESTAMPTZ NOT NULL,
    product_status TEXT NOT NULL,
    strategic_state TEXT NOT NULL,
    confidence NUMERIC(3, 2),
    reasoning TEXT,
    recommendation TEXT,
    action_type TEXT,
    projected_upside_monthly NUMERIC(12, 2),
    downside_risk_monthly NUMERIC(12, 2),
    net_expected_value NUMERIC(12, 2),
    thirty_day_risk NUMERIC(5, 2),
    thirty_day_growth NUMERIC(5, 2),
    time_horizon_days INTEGER,
    primary_trigger_type TEXT,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_insights_asin ON strategic_insights(asin);
CREATE INDEX IF NOT EXISTS idx_insights_status ON strategic_insights(status);

-- TRIGGER EVENTS TABLE
CREATE TABLE IF NOT EXISTS trigger_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id UUID REFERENCES strategic_insights(id) ON DELETE CASCADE,
    asin TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT,
    detected_at TIMESTAMPTZ NOT NULL,
    metric_name TEXT,
    baseline_value NUMERIC,
    current_value NUMERIC,
    delta_pct NUMERIC(5, 2),
    affected_asin TEXT,
    related_asin TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triggers_insight ON trigger_events(insight_id);
CREATE INDEX IF NOT EXISTS idx_triggers_asin ON trigger_events(asin);

-- Add missing columns to product_snapshots
ALTER TABLE product_snapshots ADD COLUMN IF NOT EXISTS seller_count INTEGER;
ALTER TABLE product_snapshots ADD COLUMN IF NOT EXISTS competitor_oos_pct NUMERIC(5, 4);
```

### 5.2 Code Fixes

**Fix historical_metrics column mapping in supabase_reader.py:**

```python
# CURRENT (BROKEN):
column_map = {
    "datetime": "week_start",
    "price": "filled_price",      # ❌ Column doesn't exist
    "bsr": "sales_rank_filled",   # ❌ Column doesn't exist
    "revenue": "weekly_sales_filled",
    "units": "estimated_units"
}

# FIX: Map actual columns written by backfill.py
column_map = {
    "datetime": "week_start",
    "buy_box_price": "filled_price",    # ✅ Actual column
    "sales_rank": "sales_rank_filled",  # ✅ Actual column
    # Note: revenue and units not written by backfill
}
```

---

## PART 6: ACTIVE MODULE SUMMARY

### Production Modules (21 files)
```
apps/
├── shelfguard_app.py      # Main dashboard
├── search_to_state_ui.py  # Discovery UI
├── finance.py             # Financial metrics
└── synthetic_intel.py     # Data enrichment

src/
├── persistence.py         # Project/ASIN storage
├── supabase_reader.py     # Data reads
├── backfill.py            # Historical backfill
├── two_phase_discovery.py # Discovery pipeline
├── intelligence_pipeline.py # AI orchestration
├── data_accumulation.py   # Network intelligence
├── trigger_detection.py   # Event detection
├── network_intelligence.py # Category benchmarks
├── trend_detection.py     # Trend analysis
├── recommendations.py     # Action generation
├── family_harvester.py    # SKU grouping
└── models/ (3 files)      # Data models

utils/
├── ai_engine.py           # LLM classifier
└── data_healer.py         # Gap filling

scrapers/
└── keepa_client.py        # Keepa API wrapper

pipelines/
├── harvest_tracked_asins.py
└── run_keepa_weekly_kcup_top1000.py
```

---

## PART 7: SINGLETON PATTERN VERIFICATION

**Current Implementation (supabase_reader.py):**
```python
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(...)
    return _supabase_client
```

**STATUS:** ✅ Implemented correctly - avoids connection spam.

**Files still using old pattern:**
- `backfill.py` - Uses `create_supabase_client()` (redirects to singleton ✅)
- `persistence.py` - Creates own client ⚠️ (should use singleton)
- `intelligence_pipeline.py` - Accepts client as parameter ✅
- `data_accumulation.py` - Accepts client as parameter ✅

---

## RECOMMENDED ACTION PLAN

### Priority 1: Fix Database Schema
1. Add `llm_cache`, `strategic_insights`, `trigger_events` tables
2. Add `seller_count`, `competitor_oos_pct` columns to product_snapshots
3. Run updated COMPLETE_SCHEMA.sql in Supabase

### Priority 2: Fix Column Mapping
1. Update `supabase_reader.load_historical_metrics_from_db()` to map actual column names
2. OR update `backfill.py` to write expected column names

### Priority 3: Standardize Naming
1. Document the naming convention in a central location
2. Consider normalizing all discovery phase code to use Keepa naming

### Priority 4: Cleanup (Optional)
1. Move orphaned files to `_deprecated/` folder
2. Add `.gitignore` entries for test/debug files

---

## AUDIT COMPLETE

**Files Analyzed:** 41 Python files
**Tables Audited:** 9 (6 existing + 3 missing)
**Critical Issues:** 5
**Warnings:** 3
**Orphaned Files:** 10

**Next Step:** Update COMPLETE_SCHEMA.sql with the fixes above, then fix the column mapping in supabase_reader.py.
