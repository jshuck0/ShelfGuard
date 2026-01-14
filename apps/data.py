import streamlit as st
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
            "week_start, asin, parent_asin, title, weekly_sales_filled, amazon_bb_share, new_fba_price, filled_price"
        ).range(start, start + chunk_size - 1).execute()
        all_rows.extend(resp.data)
        if len(resp.data) < chunk_size: break
        start += chunk_size

    df_rows = pd.DataFrame(all_rows)
    df_master = pd.DataFrame(supabase.table("asin_master").select("asin, is_starbucks").execute().data)
    
    if not df_rows.empty:
        df_rows = df_rows.merge(df_master, on="asin", how="left")
        df_rows["is_starbucks"] = df_rows["is_starbucks"].fillna(0).astype(int)
        # Normalize date to midnight to ensure exact matches with the sidebar
        df_rows["week_start"] = pd.to_datetime(df_rows["week_start"]).dt.normalize().dt.tz_localize(None)
        df_rows["amazon_bb_share"] = pd.to_numeric(df_rows["amazon_bb_share"], errors='coerce').fillna(1.0)
    
    return df_rows