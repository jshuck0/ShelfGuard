"""
Run the product_snapshots migration against Supabase.
This creates the new table and views needed for the stateful architecture.
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Only the NEW parts of the schema (product_snapshots table and views)
MIGRATION_SQL = """
-- 5. PRODUCT_SNAPSHOTS TABLE
-- Time-series storage for harvested product data (The Oracle's Data Warehouse)
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
    
    -- Metadata
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source TEXT DEFAULT 'keepa',
    
    UNIQUE(asin, snapshot_date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_snapshots_asin_date ON product_snapshots(asin, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON product_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_parent_asin ON product_snapshots(parent_asin);

-- RLS
ALTER TABLE product_snapshots ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for re-run safety)
DROP POLICY IF EXISTS "Anyone can view product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Service role can insert product snapshots" ON product_snapshots;
DROP POLICY IF EXISTS "Service role can update product snapshots" ON product_snapshots;

CREATE POLICY "Anyone can view product snapshots"
    ON product_snapshots FOR SELECT
    USING (true);

CREATE POLICY "Service role can insert product snapshots"
    ON product_snapshots FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Service role can update product snapshots"
    ON product_snapshots FOR UPDATE
    USING (true);

-- Views
CREATE OR REPLACE VIEW latest_snapshots AS
SELECT DISTINCT ON (asin)
    *
FROM product_snapshots
ORDER BY asin, snapshot_date DESC;

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
"""

def run_migration():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[ERROR] SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        return False
    
    print(f"[INFO] Connecting to Supabase: {SUPABASE_URL[:50]}...")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Test connection by querying the table
        try:
            result = supabase.table("product_snapshots").select("id").limit(1).execute()
            print("[SUCCESS] product_snapshots table already exists!")
            print(f"[INFO] Found {len(result.data)} existing records")
            return True
        except Exception as e:
            error_str = str(e)
            if "does not exist" in error_str or "relation" in error_str.lower():
                print("[WARNING] Table doesn't exist yet.")
                print("[INFO] Please run the SQL from schemas/search_to_state.sql in Supabase SQL Editor")
                print("[INFO] Specifically lines 164-279 for the product_snapshots table")
            else:
                print(f"[WARNING] Connection test result: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        return False

if __name__ == "__main__":
    run_migration()
