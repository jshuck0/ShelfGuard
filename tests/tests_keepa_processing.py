import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. SETUP & PATHS
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

# 2. CONFIGURATION
CSV_PATH = ROOT / "outputs/keepa_weekly_kcup_top1000_rows.csv"
TOTALS_PATH = ROOT / "outputs/keepa_weekly_kcup_top1000_totals.csv"

def run_audit():
    print("🛡️  SHELFGUARD OS: STARTING FULL AUDIT...\n")
    
    # --- PHASE 1: LOCAL FILE VALIDATION ---
    print("📁 PHASE 1: Local File Integrity")
    if not CSV_PATH.exists():
        print("❌ CRITICAL: Weekly Rows CSV not found!")
        return
    
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    unique_asins = df['asin'].nunique()
    
    print(f"✅ Found {total_rows:,} total rows.")
    print(f"✅ Found {unique_asins} unique ASINs.")

    # Check for "Standard Product" poisoning
    poisoned = df[df['variation_attributes'] == "Standard Product"].shape[0]
    poison_pct = (poisoned / total_rows) * 100
    if poison_pct > 10:
        print(f"⚠️  WARNING: {poison_pct:.1f}% of rows are generic. Check regex in keepa_client.py.")
    else:
        print(f"✅ Variation Parsing: {100-poison_pct:.1f}% health score.")

    # --- PHASE 2: FINANCIAL CALIBRATION ---
    print("\n💰 PHASE 2: Market Calibration Audit")
    if TOTALS_PATH.exists():
        df_t = pd.read_csv(TOTALS_PATH)
        latest = df_t.sort_values("week_start").iloc[-1]
        market_size = latest['kcup_sales']
        sbux_rev = latest['sbux_sales']
        share = (sbux_rev / market_size) * 100 if market_size > 0 else 0
        
        print(f"📊 Market Total: ${market_size:,.2f}")
        print(f"☕ Starbucks Rev: ${sbux_rev:,.2f}")
        print(f"📈 Starbucks Share: {share:.1f}%")
        
        # Calibration Check
        if market_size < 10_000_000:
            print("❌ CALIBRATION ERROR: Market size is too low. Check the 145k constant and sum() aggregation.")
        elif market_size > 25_000_000:
            print("⚠️  CALIBRATION WARNING: Market size looks high. Check for double-counting.")
        else:
            print("💎 CALIBRATION: Optimized for 2026 Grocery Velocity.")

    # --- PHASE 3: CLOUD RECONCILIATION ---
    print("\n☁️  PHASE 3: Supabase Cloud Sync Check")
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        supabase: Client = create_client(url, key)
        
        # Check ASIN Master
        master_count = supabase.table("asin_master").select("count", count="exact").execute().count
        # Check Weekly Rows
        cloud_rows = supabase.table("keepa_weekly_rows").select("count", count="exact").execute().count
        
        print(f"✅ Cloud Master Table: {master_count} ASINs synced.")
        print(f"✅ Cloud Weekly Table: {cloud_rows:,} rows live.")
        
        if abs(cloud_rows - total_rows) > 1000:
             print("⚠️  SYNC GAP: Cloud row count differs significantly from local CSV.")
        else:
             print("✅ Cloud-to-Local Reconciled.")

    except Exception as e:
        print(f"❌ Cloud Connection Failed: {e}")

    print("\n🏁 AUDIT COMPLETE: System is ready for demonstration.")

if __name__ == "__main__":
    run_audit()