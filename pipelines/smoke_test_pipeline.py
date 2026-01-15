import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Pathing fix
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

import keepa
import pandas as pd
from scrapers.keepa_client import build_keepa_weekly_table

def run_bulletproof_check():
    key = os.getenv("KEEPA_API_KEY")
    api = keepa.Keepa(key)
    # Test a Starbucks ASIN and a different brand to ensure regex isn't too narrow
    test_asins = ["B00P0ZGH6O", "B007TGDXNO"] 

    print("🚀 STARTING FINAL VALIDATION...")
    
    # 1. API CONNECTION & IMAGE CHECK
    print("\n--- 1. API & IMAGE VERIFICATION ---")
    products = api.query(test_asins, domain="US", buybox=True, update=24)
    
    for p in products:
        raw_images = p.get("imagesCSV")
        if raw_images:
            print(f"✅ ASIN {p['asin']}: Found {len(raw_images.split(','))} images.")
        else:
            print(f"⚠️ ASIN {p['asin']}: No images found in raw data.")

    # 2. PARSER & VARIATION CHECK
    print("\n--- 2. PARSER & ATTRIBUTE VERIFICATION ---")
    df = build_keepa_weekly_table(products)
    
    # Get the most recent row for each ASIN
    latest = df.sort_values("week_start").groupby("asin").tail(1)
    
    for _, row in latest.iterrows():
        print(f"ASIN: {row['asin']}")
        print(f"  - Title: {row['title'][:50]}...")
        print(f"  - Variation: {row['variation_attributes']}")
        print(f"  - Image URL: {row['main_image']}")
        
        # 3. DATA INTEGRITY CHECKS
        if pd.isna(row['main_image']) or "None" in str(row['main_image']):
            print("  ❌ ERROR: Image URL failed to generate.")
        elif not str(row['main_image']).startswith("https://"):
            print("  ❌ ERROR: Invalid Image URL format.")
        else:
            print("  ✅ Image URL looks solid.")

        if "Standard Product" in str(row['variation_attributes']):
            print("  ⚠️ WARNING: Variation parsing fell back to default.")
        else:
            print("  ✅ Variation string parsed successfully.")

    print("\n--- 🏁 VALIDATION COMPLETE ---")

if __name__ == "__main__":
    run_bulletproof_check()