import sys
import time
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import keepa
from dotenv import load_dotenv

# 1. SETUP PATHS
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# 2. LOAD ENVIRONMENT
load_dotenv(ROOT / ".env", override=True)

# Import our cleaned scraper logic
from scrapers.keepa_client import build_keepa_weekly_table

# 3. CONFIGURATION
KCUP_ASIN_CSV = "data/raw/kcup_asins_top1000.csv"
STARBUCKS_ASIN_CSV = "data/raw/starbucks_kcup_asins.csv"
OUT_WEEKLY_ROWS_CSV = "outputs/keepa_weekly_kcup_top1000_rows.csv"
OUT_WEEKLY_TOTALS_CSV = "outputs/keepa_weekly_kcup_top1000_totals.csv"

def _load_asins(path: str) -> list[str]:
    """Helper to load ASINs from CSV safely."""
    full_path = ROOT / path
    if not full_path.exists():
        print(f"⚠️ Warning: File not found {path}")
        return []
    df = pd.read_csv(full_path)
    col = next((c for c in df.columns if c.lower() in ["asin", "asins"]), df.columns[0])
    return df[col].dropna().astype(str).str.strip().str.upper().unique().tolist()

def npy_serializer(obj):
    """Handles NumPy types that standard json.dump cannot."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return str(obj)

def main():
    # 4. KEY AUTHENTICATION
    key = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")
    if not key:
        print(f"❌ ERROR: KEEPA_KEY not found in .env at {ROOT / '.env'}")
        return
    
    print(f"✅ Environment Secured. Key starts with: {key[:5]}...")

    # 5. ASIN LOADING
    all_asins = _load_asins(KCUP_ASIN_CSV) 
    sbux_asin_list = _load_asins(STARBUCKS_ASIN_CSV)
    sbux_asin_set = set(sbux_asin_list)
    
    if not all_asins:
        print("❌ ERROR: No ASINs loaded. Check your CSV paths.")
        return

    # 6. IMPROVED RESUME LOGIC (Batch Folder)
    batch_dir = ROOT / "data/raw/keepa_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    existing_batches = list(batch_dir.glob("batch_*.json"))
    fetched_asins = set()
    products = []
    
    print(f"📂 Scanning {len(existing_batches)} existing batch files...")
    for b in existing_batches:
        try:
            with open(b, 'r') as f:
                batch_data = json.load(f)
                if batch_data:
                    products.extend(batch_data)
                    fetched_asins.update({p['asin'] for p in batch_data if 'asin' in p})
        except Exception:
            continue

    remaining_asins = [a for a in all_asins if a not in fetched_asins]
    print(f"🔄 Checkpoint: {len(fetched_asins)} fetched. {len(remaining_asins)} remaining.")

    # 7. HARDENED FETCHING (RESTORED WITH PIKE PLACE FIX)
    if remaining_asins:
        api = keepa.Keepa(key)
        # Process in batches of 10 to respect Keepa credit limits
        for i in range(0, len(remaining_asins), 10):
            batch = remaining_asins[i : i + 10]
            current_idx = len(fetched_asins) + i
            
            success = False
            attempts = 0
            while not success and attempts < 3:
                try:
                    attempts += 1
                    print(f"Fetching {current_idx}/{len(all_asins)} (Attempt {attempts})...")
                    
                    # --- THE FIX: update=24 forces Keepa to send Titles/Variations ---
                    res = api.query(batch, domain="US", buybox=True, update=24, stats=90)
                    
                    final_path = batch_dir / f"batch_{current_idx}.json"
                    temp_path = batch_dir / f"batch_{current_idx}.tmp"
                    
                    with open(temp_path, 'w') as f:
                        json.dump(res, f, default=npy_serializer)
                    
                    if final_path.exists():
                        final_path.unlink()
                    temp_path.rename(final_path)
                    
                    products.extend(res)
                    print(f"✅ Batch {current_idx} secured.")
                    success = True
                except Exception as e:
                    print(f"⚠️ Attempt {attempts} failed: {e}. Retrying in 30s...")
                    time.sleep(30)
            
            if not success:
                print(f"❌ Batch {current_idx} failed after 3 attempts. Moving to processing.")
                break
            
            time.sleep(2) # Politeness delay

    # 8. DATA PROCESSING
    if not products:
        print("❌ ERROR: No product data in memory.")
        return

    print("📊 All data fetched. Cleaning hierarchy and calculating sales...")
    weekly = build_keepa_weekly_table(products)
    
    if weekly.empty or "asin" not in weekly.columns:
        print("❌ CRITICAL: The weekly table is empty. Check keepa_client.py extraction.")
        return

    # Tag Starbucks products
    weekly["is_starbucks"] = weekly["asin"].isin(sbux_asin_set).astype(int)
    
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    weekly.to_csv(ROOT / OUT_WEEKLY_ROWS_CSV, index=False)

    # 9. AGGREGATION & INSIGHTS
    group_cols = ["week_start", "parent_asin"]
    if all(col in weekly.columns for col in group_cols + ["weekly_sales_filled"]):
        parent_weekly = weekly.groupby(group_cols, as_index=False).agg(
            p_sales=("weekly_sales_filled", "sum"),  # CAPTURE ALL VARIATIONS
            is_sbux=("is_starbucks", "max")
        )
        
        totals = parent_weekly.groupby("week_start", as_index=False).agg(kcup_sales=("p_sales", "sum"))
        sbux_totals = parent_weekly[parent_weekly["is_sbux"] == 1].groupby("week_start", as_index=False).agg(sbux_sales=("p_sales", "sum"))
        
        final_totals = totals.merge(sbux_totals, on="week_start", how="left").fillna(0)
        final_totals.to_csv(ROOT / OUT_WEEKLY_TOTALS_CSV, index=False)

        print("-" * 30)
        print(f"✅ PIPELINE SUCCESS.")
        if not final_totals.empty:
            latest_row = final_totals.sort_values("week_start").iloc[-1]
            print(f"Latest Week:        {latest_row['week_start']}")
            print(f"Total Market Size: ${latest_row['kcup_sales']:,.2f}")
            print(f"Starbucks Rev:      ${latest_row['sbux_sales']:,.2f}")
            share = (latest_row['sbux_sales'] / latest_row['kcup_sales'] * 100) if latest_row['kcup_sales'] > 0 else 0
            print(f"Starbucks Share:   {share:.1f}%")
        print("-" * 30)

if __name__ == "__main__":
    main()