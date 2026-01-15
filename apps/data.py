import streamlit as st  # <--- ADD THIS LINE
import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

@st.cache_data(ttl=3600)
def get_all_data():
    all_rows, chunk_size, start = [], 1000, 0
    while True:
        resp = supabase.table("keepa_weekly_rows").select(
            "week_start, asin, parent_asin, title, weekly_sales_filled, estimated_units, "
            "amazon_bb_share, new_fba_price, filled_price, weeks_of_cover, "
            "variation_attributes, fba_fees, package_vol_cf"
        ).range(start, start + chunk_size - 1).execute()
        
        all_rows.extend(resp.data)
        if len(resp.data) < chunk_size: break
        start += chunk_size

    df_rows = pd.DataFrame(all_rows)
    
    # Fetch Master Metadata (Starbucks Tags & Images)
    df_master = pd.DataFrame(
        supabase.table("asin_master")
        .select("asin, is_starbucks, main_image")
        .execute().data
    )
    
    if not df_rows.empty and not df_master.empty:
        # --- THE KEY FIX: STANDARDIZE ASINS ---
        df_rows["asin"] = df_rows["asin"].astype(str).str.strip().str.upper()
        df_master["asin"] = df_master["asin"].astype(str).str.strip().str.upper()
        
        # Merge ensures is_starbucks flows into the weekly rows
        df_rows = df_rows.merge(df_master, on="asin", how="left")
        
        # Final cleanup
        df_rows["is_starbucks"] = df_rows["is_starbucks"].fillna(0).astype(int)
        df_rows["title"] = df_rows["title"].fillna("Unknown Product")
        df_rows["main_image"] = df_rows["main_image"].fillna("")
    
    return df_rows