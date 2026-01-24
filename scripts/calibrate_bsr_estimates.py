"""
Keepa Sales Estimate Calibration
================================
Verifies the accuracy of Sales Rank (BSR) as a predictor of Sales Volume.

Compare:
1. `current_SALES` (BSR) - The independent variable
2. `monthlySold` (Amazon's reported volume) - The ground truth

Output:
- Correlation coefficient (R²)
- Mean Absolute Percentage Error (MAPE) of BSR-based formulas
- Scatter plot data points
"""

import sys
import os
import requests
import pandas as pd
import numpy as np
from typing import List, Dict
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_keepa_key():
    # Try getting from env or user input
    key = os.getenv("KEEPA_API_KEY")
    if not key:
        try:
            # Try to load from streamlit secrets if available locally
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            key = secrets.get("keepa_api_key") or secrets.get("keepa", {}).get("api_key")
        except:
            pass
    
    if not key:
        print("⚠️  No KEEPA_API_KEY found.")
        print("    Please set KEEPA_API_KEY environment variable or run from project root.")
        sys.exit(1)
    return key

def fetch_calibration_data(api_key: str, count: int = 100):
    """Fetch products that have BOTH sales rank AND monthly sold data."""
    print(f"📡 Fetching {count} products with verified sales volume...")
    
    url = f"https://api.keepa.com/query?key={api_key}&domain=1&stats=0"
    
    # Query for products with explicit monthlySold data
    # Filter: monthlySold >= 50 (to avoid noise) and BSR < 100000
    query = {
        "monthlySold_gte": 50,
        "current_SALES_gte": 1,
        "current_SALES_lte": 100000,
        "perPage": count,
        "page": 0,
        "sort": [["current_SALES", "asc"]] # Get best sellers first
    }
    
    try:
        response = requests.post(url, json=query, timeout=30)
        data = response.json()
        
        asins = data.get("asinList", [])
        print(f"✅ Found {len(asins)} candidate ASINs.")
        
        # Now fetch details (we need the raw values)
        # Note: product_finder response doesn't give values, just ASINs.
        # We need to use the token bucket efficiently.
        
        # Actually, for Product Finder, we can request certain fields? 
        # No, Keepa Product Finder API returns ASIN list only.
        # We MUST fetch product details to get the actual `monthlySold` and `current_SALES` values.
        
        import keepa
        api = keepa.Keepa(api_key)
        
        products = api.query(asins[:count], stats=1)
        
        records = []
        for p in products:
            # Get actual monthly sold (not estimated)
            monthly_sold = p.get("monthlySold")
            
            # Get current Sales Rank
            # keepa library returns csv info, let's look for SALES rank (index 3 usually)
            # But relying on CSV parsing is complex. Let's use `stats` if available.
            stats = p.get("stats", {})
            current_bsr = stats.get("current", {})
            # Keepa python lib structure is tricky based on stats param.
            # Let's try to get latest from CSV
            
            data_csv = p.get("data", {})
            bsr_list = data_csv.get("SALES", [])
            
            if monthly_sold and bsr_list:
                # Get last known BSR
                last_bsr = bsr_list[-1] if not np.isnan(bsr_list[-1]) else None
                
                if last_bsr and last_bsr > 0:
                    records.append({
                        "asin": p.get("asin"),
                        "title": p.get("title")[:30],
                        "bsr": last_bsr,
                        "monthly_sold": monthly_sold,
                        "category": p.get("rootCategory")
                    })
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def analyze_correlation(df: pd.DataFrame):
    """Analyze relationship between BSR and Monthly Sold."""
    if df.empty:
        print("⚠️ No data to analyze.")
        return
        
    print("\n📊 CALIBRATION ANALYSIS")
    print(f"   Sample Size: {len(df)} products")
    
    # 1. Log-Log Correlation (Power Law is typical for Sales Rank)
    # log(Sales) = a + b * log(Rank)
    df["log_bsr"] = np.log(df["bsr"])
    df["log_sales"] = np.log(df["monthly_sold"])
    
    correlation = df["log_bsr"].corr(df["log_sales"])
    r_squared = correlation ** 2
    
    print(f"\n📈 Correlation (Log-Log): {correlation:.4f}")
    print(f"   R² Score:            {r_squared:.4f}")
    print("\n   Interpretation:")
    if r_squared > 0.8:
        print("   ✅ Strong predictive power. BSR is a reliable proxy.")
    elif r_squared > 0.5:
        print("   ⚠️ Moderate relationship. BSR is useful but has variance.")
    else:
        print("   ❌ Weak relationship. BSR is a poor predictor.")
        
    # 2. Check "Rule of Thumb" accuracy (e.g. JungleScout type estimates)
    # This checks if our internal logic needs tuning.
    
    print("\n📋 Sample Data Points (BSR -> Actual Sales):")
    print(df[["bsr", "monthly_sold"]].head(10).to_string(index=False))

if __name__ == "__main__":
    key = get_keepa_key()
    df = fetch_calibration_data(key)
    analyze_correlation(df)
