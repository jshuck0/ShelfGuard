import streamlit as st
import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_SERVICE_KEY"]

# Initialize Supabase Client
supabase: Client = create_client(url, key)

@st.cache_data(ttl=3600)
def get_all_data():
    """
    Fetches full historical rows and merges with master metadata.
    Optimized for predictive 36-month velocity analysis and sparkline rendering.
    """
    all_rows, chunk_size, start = [], 1000, 0
    
    # 1. FETCH WEEKLY PERFORMANCE ROWS
    while True:
        # Added 'sales_rank_filled' for predictive logic
        resp = supabase.table("keepa_weekly_rows").select(
            "week_start, asin, parent_asin, title, weekly_sales_filled, estimated_units, "
            "amazon_bb_share, new_fba_price, filled_price, weeks_of_cover, "
            "variation_attributes, fba_fees, package_vol_cf, sales_rank_filled"
        ).range(start, start + chunk_size - 1).execute()
        
        all_rows.extend(resp.data)
        if len(resp.data) < chunk_size: 
            break
        start += chunk_size

    df_rows = pd.DataFrame(all_rows)
    
    # 2. FETCH MASTER METADATA (Starbucks Tags & Product Images)
    df_master = pd.DataFrame(
        supabase.table("asin_master")
        .select("asin, is_starbucks, main_image")
        .execute().data
    )
    
    if df_rows.empty:
        return pd.DataFrame()

    # 3. DATA CLEANING & OPTIMIZATION
    # Standardize ASINs for perfect merging
    df_rows["asin"] = df_rows["asin"].astype(str).str.strip().str.upper()
    
    # Convert dates to datetime objects for time-series math
    df_rows["week_start"] = pd.to_datetime(df_rows["week_start"])
    
    # --- BULLETPROOF NUMERIC HANDLING ---
    # We force these to numeric immediately to prevent string-pollution in sparklines
    numeric_cols = ["weekly_sales_filled", "sales_rank_filled", "filled_price", "weeks_of_cover"]
    for col in numeric_cols:
        if col in df_rows.columns:
            # Errors='coerce' turns junk into NaN, then we fill with 0 to keep the math alive
            df_rows[col] = pd.to_numeric(df_rows[col], errors='coerce').fillna(0).astype("float32")

    if not df_master.empty:
        df_master["asin"] = df_master["asin"].astype(str).str.strip().str.upper()
        
        # Merge ensures is_starbucks flows into the historical rows
        df_rows = df_rows.merge(df_master, on="asin", how="left")
        
        # Final cleanup for missing categorical values
        df_rows["is_starbucks"] = df_rows["is_starbucks"].fillna(0).astype(int)
        df_rows["title"] = df_rows["title"].fillna("Unknown Product")
        df_rows["main_image"] = df_rows["main_image"].fillna("")
    
    return df_rows