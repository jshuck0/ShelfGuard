import sys, os, json, pandas as pd, numpy as np
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from scrapers.keepa_client import build_keepa_weekly_table, rolling_36m_start

# Config paths
STARBUCKS_ASIN_CSV = "data/raw/starbucks_kcup_asins.csv"
OUT_WEEKLY_ROWS_CSV = "outputs/keepa_weekly_kcup_top1000_rows.csv"
OUT_WEEKLY_TOTALS_CSV = "outputs/keepa_weekly_kcup_top1000_totals.csv"

def main():
    # 1. Load Starbucks set for identifier tagging
    sbux_path = ROOT / STARBUCKS_ASIN_CSV
    sbux_set = set()
    if sbux_path.exists():
        sbux_set = set(pd.read_csv(sbux_path).iloc[:,0].dropna().astype(str).str.upper().tolist())

    # 2. Determine 36-month lookback window
    window_start = rolling_36m_start()
    batch_dir = ROOT / "data/raw/keepa_batches"
    
    # 3. Load local JSON data without extra API calls
    products = []
    print("📂 Loading local JSON data...")
    for b in batch_dir.glob("batch_*.json"):
        with open(b, 'r') as f:
            products.extend(json.load(f))

    if not products:
        return print("❌ Error: No local batches found.")

    # 4. Run the high-fidelity extraction and sales model
    print(f"📊 Analyzing {len(products)} products since {window_start}...")
    weekly = build_keepa_weekly_table(products, window_start=window_start)
    
    if weekly.empty:
        return print("❌ Error: No data processed.")

    # 5. Tag Starbucks products and save the granular "Master" CSV
    weekly["is_starbucks"] = weekly["asin"].str.upper().isin(sbux_set).astype(int)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    weekly.to_csv(ROOT / OUT_WEEKLY_ROWS_CSV, index=False)

    # 6. Professional Aggregation (The "Director's Fix")
    # This logic groups by Parent ASIN and calculates the best/max metrics for the brand
    agg_map = {
        "weekly_sales_filled": "max",     # Revenue potential of the top variation
        "is_starbucks": "max",            # Maintain brand tag
        "amazon_bb_share": "mean",        # Average brand presence in Buy Box
        "buy_box_switches": "sum",        # Total volatility across variations
        "new_fba_price": "min",           # Most competitive Prime price
        "new_fbm_price": "min",           # Most competitive merchant price
        "rating": "max",                  # Brand's highest customer rating
        "review_count": "max"             # Highest social proof count
    }
    
    # Filter only available columns to prevent aggregation crashes
    actual_agg = {k: v for k, v in agg_map.items() if k in weekly.columns}
    
    # Group by week and parent to prevent variation double-counting
    parent_weekly = weekly.groupby(["week_start", "parent_asin"], as_index=False).agg(actual_agg)
    
    # 7. Final Market Insight Calculations
    totals = parent_weekly.groupby("week_start", as_index=False)["weekly_sales_filled"].sum().rename(columns={"weekly_sales_filled": "total_mkt"})
    sbux = parent_weekly[parent_weekly["is_starbucks"] == 1].groupby("week_start", as_index=False)["weekly_sales_filled"].sum().rename(columns={"weekly_sales_filled": "sbux_rev"})
    
    # Merge and final file export for dashboard totals
    final = totals.merge(sbux, on="week_start", how="left").fillna(0)
    final.to_csv(ROOT / OUT_WEEKLY_TOTALS_CSV, index=False)

    # 8. Success Output
    if not final.empty:
        latest = final.sort_values("week_start").iloc[-1]
        print("-" * 40)
        print(f"🚀 SUCCESS: {len(weekly):,} rows processed.")
        print(f"Latest Week:  {latest['week_start']}")
        print(f"Total Market: ${latest['total_mkt']:,.2f}")
        print(f"Starbucks:    ${latest['sbux_rev']:,.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main()