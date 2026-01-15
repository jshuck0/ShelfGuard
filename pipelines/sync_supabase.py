import math
import os
import httpx
import pandas as pd
import numpy as np
import time
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# 1. STABLE CONNECTION SETUP
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

stable_session = httpx.Client(
    http2=False, 
    timeout=httpx.Timeout(60.0, connect=10.0)
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
supabase.postgrest.session = stable_session

# --- Configuration ---
INPUT_ROWS = "outputs/keepa_weekly_kcup_top1000_rows.csv"
INPUT_TOTALS = "outputs/keepa_weekly_kcup_top1000_totals.csv"
T_ROWS = "keepa_weekly_rows"
T_TOTALS = "weekly_totals"
T_ASIN = "asin_master"

ROWS_COLS = [
    "week_start", "asin", "parent_asin", "title", "variation_attributes",
    "weekly_sales_filled", "estimated_units", "filled_price", "sales_rank_filled",
    "amazon_bb_share", "buy_box_switches", 
    "new_fba_price", "new_fbm_price", "new_offer_count", "rating", "review_count",
    "weeks_of_cover", "package_weight_g", "package_vol_cf", 
    "fba_fees"
]

def to_iso_z(dt_series):
    return pd.to_datetime(dt_series, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def _clean_value(x):
    if x is None or pd.isna(x): return None
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x): return None
        return float(x)
    return x

def upsert_in_chunks(table, df, chunk_size=100):
    if df.empty: return
    on_conflict = "asin,week_start" if table == T_ROWS else "week_start" if table == T_TOTALS else "asin"
    records = df.to_dict(orient="records")
    
    print(f"🔼 Upserting {len(records):,} records to '{table}'...")
    for i in range(0, len(records), chunk_size):
        chunk = [{k: _clean_value(v) for k, v in r.items()} for r in records[i : i + chunk_size]]
        try:
            supabase.table(table).upsert(chunk, on_conflict=on_conflict).execute()
            time.sleep(0.05) 
        except Exception as e:
            print(f"⚠️ Batch {i} failed, waiting 5s before retry...")
            time.sleep(5)
            try:
                supabase.table(table).upsert(chunk, on_conflict=on_conflict).execute()
            except Exception as e2:
                print(f"❌ Batch {i} failed permanently: {e2}")

def sync_to_cloud():
    if not os.path.exists(INPUT_ROWS):
        return print(f"❌ Error: {INPUT_ROWS} not found.")

    df_full = pd.read_csv(INPUT_ROWS)
    df_full["asin"] = df_full["asin"].astype(str).str.strip().str.upper()
    
    # --- UPDATED SANITY GATE ---
    if "filled_price" in df_full.columns:
        df_full.loc[df_full["filled_price"] > 500, "filled_price"] /= 100.0
    
    # TRUST THE CSV: Do not divide by 4.33 again here.
    if "weekly_sales_filled" not in df_full.columns and "estimated_units" in df_full.columns:
        df_full["weekly_sales_filled"] = df_full["estimated_units"] * df_full["filled_price"].fillna(0)

    # 1. Sync Unique Products
    master_cols = ["asin", "parent_asin", "title", "brand", "manufacturer", "main_image", "is_starbucks"]
    avail_master = [c for c in master_cols if c in df_full.columns]
    upsert_in_chunks(T_ASIN, df_full[avail_master].drop_duplicates("asin"))

    # 2. Sync Market Totals (Scale is preserved from run_keepa_weekly.py)
    if os.path.exists(INPUT_TOTALS):
        df_t = pd.read_csv(INPUT_TOTALS)
        df_t["week_start"] = to_iso_z(df_t["week_start"])
        upsert_in_chunks(T_TOTALS, df_t)

    # 3. Sync Granular Rows
    avail_rows = [c for c in ROWS_COLS if c in df_full.columns]
    df_r = df_full[avail_rows].copy()
    
    if "variation_attributes" in df_r.columns:
        df_r["variation_attributes"] = df_r["variation_attributes"].fillna("").astype(str)

    df_r["week_start"] = to_iso_z(df_r["week_start"])
    upsert_in_chunks(T_ROWS, df_r)

    print("\n🚀 SUCCESS: Financial OS is clean and live in Supabase.")

if __name__ == "__main__":
    sync_to_cloud()