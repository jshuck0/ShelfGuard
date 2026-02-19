-- ========================================
-- ShelfGuard Phase A — Add New Columns to product_snapshots
-- ========================================
-- Safe to re-run: all statements use IF NOT EXISTS checks.
-- Adds Phase A signal columns and missing extended product columns.

DO $$
BEGIN
    -- Extended product signals (may already exist from prior work)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='monthly_sold') THEN
        ALTER TABLE product_snapshots ADD COLUMN monthly_sold INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='number_of_items') THEN
        ALTER TABLE product_snapshots ADD COLUMN number_of_items INTEGER DEFAULT 1;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='price_per_unit') THEN
        ALTER TABLE product_snapshots ADD COLUMN price_per_unit NUMERIC(10, 2);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='buybox_is_amazon') THEN
        ALTER TABLE product_snapshots ADD COLUMN buybox_is_amazon BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='buybox_is_fba') THEN
        ALTER TABLE product_snapshots ADD COLUMN buybox_is_fba BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='buybox_is_backorder') THEN
        ALTER TABLE product_snapshots ADD COLUMN buybox_is_backorder BOOLEAN DEFAULT false;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='has_amazon_seller') THEN
        ALTER TABLE product_snapshots ADD COLUMN has_amazon_seller BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='seller_count') THEN
        ALTER TABLE product_snapshots ADD COLUMN seller_count INTEGER DEFAULT 1;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='oos_count_amazon_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN oos_count_amazon_30 INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='oos_count_amazon_90') THEN
        ALTER TABLE product_snapshots ADD COLUMN oos_count_amazon_90 INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='oos_pct_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN oos_pct_30 NUMERIC(5, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='oos_pct_90') THEN
        ALTER TABLE product_snapshots ADD COLUMN oos_pct_90 NUMERIC(5, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='velocity_30d') THEN
        ALTER TABLE product_snapshots ADD COLUMN velocity_30d NUMERIC(10, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='velocity_90d') THEN
        ALTER TABLE product_snapshots ADD COLUMN velocity_90d NUMERIC(10, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='bb_seller_count_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN bb_seller_count_30 INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='bb_top_seller_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN bb_top_seller_30 NUMERIC(5, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='units_source') THEN
        ALTER TABLE product_snapshots ADD COLUMN units_source TEXT DEFAULT 'bsr_formula';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='is_sns') THEN
        ALTER TABLE product_snapshots ADD COLUMN is_sns BOOLEAN DEFAULT false;
    END IF;

    -- Phase A columns (new)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='return_rate') THEN
        ALTER TABLE product_snapshots ADD COLUMN return_rate INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='sales_rank_drops_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN sales_rank_drops_30 INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='sales_rank_drops_90') THEN
        ALTER TABLE product_snapshots ADD COLUMN sales_rank_drops_90 INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='monthly_sold_delta') THEN
        ALTER TABLE product_snapshots ADD COLUMN monthly_sold_delta INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='top_comp_bb_share_30') THEN
        ALTER TABLE product_snapshots ADD COLUMN top_comp_bb_share_30 NUMERIC(5, 4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='active_ingredients_raw') THEN
        ALTER TABLE product_snapshots ADD COLUMN active_ingredients_raw TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='item_type_keyword') THEN
        ALTER TABLE product_snapshots ADD COLUMN item_type_keyword TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='has_buybox_stats') THEN
        ALTER TABLE product_snapshots ADD COLUMN has_buybox_stats BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='has_monthly_sold_history') THEN
        ALTER TABLE product_snapshots ADD COLUMN has_monthly_sold_history BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='has_active_ingredients') THEN
        ALTER TABLE product_snapshots ADD COLUMN has_active_ingredients BOOLEAN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='product_snapshots' AND column_name='has_sales_rank_drops') THEN
        ALTER TABLE product_snapshots ADD COLUMN has_sales_rank_drops BOOLEAN;
    END IF;
END $$;
