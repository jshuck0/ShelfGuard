import streamlit as st
import pandas as pd
from supabase import create_client, Client

# Initialize Supabase Client once at module level
supabase: Client = create_client(st.secrets["url"], st.secrets["key"])

# Constants for data fetching
CHUNK_SIZE = 1000
REQUIRED_COLUMNS = (
    "week_start, asin, parent_asin, title, weekly_sales_filled, estimated_units, "
    "amazon_bb_share, new_fba_price, filled_price, weeks_of_cover, "
    "variation_attributes, fba_fees, package_vol_cf, sales_rank_filled"
)
NUMERIC_COLUMNS = ["weekly_sales_filled", "sales_rank_filled", "filled_price", "weeks_of_cover"]

@st.cache_data(ttl=3600)
def get_all_data():
    """
    Fetches full historical rows and merges with master metadata.
    Optimized for predictive 36-month velocity analysis and sparkline rendering.

    Performance optimizations:
    - Efficient chunked fetching with pre-allocated list
    - Vectorized type conversions
    - Single-pass data cleaning
    """
    all_rows = []
    start = 0

    # 1. FETCH WEEKLY PERFORMANCE ROWS (Optimized chunking)
    while True:
        resp = supabase.table("keepa_weekly_rows").select(
            REQUIRED_COLUMNS
        ).range(start, start + CHUNK_SIZE - 1).execute()

        if not resp.data:
            break

        all_rows.extend(resp.data)
        if len(resp.data) < CHUNK_SIZE:
            break
        start += CHUNK_SIZE

    if not all_rows:
        return pd.DataFrame()

    df_rows = pd.DataFrame(all_rows)
    
    # 2. FETCH MASTER METADATA (Starbucks Tags & Product Images)
    master_resp = supabase.table("asin_master").select("asin, is_starbucks, main_image").execute()
    df_master = pd.DataFrame(master_resp.data) if master_resp.data else pd.DataFrame()

    # 3. DATA CLEANING & OPTIMIZATION (Vectorized operations)
    # Standardize ASINs for merging (single operation)
    df_rows["asin"] = df_rows["asin"].astype(str).str.strip().str.upper()

    # Convert dates to datetime (timezone-aware)
    df_rows["week_start"] = pd.to_datetime(df_rows["week_start"], utc=True)

    # Vectorized numeric conversion (more efficient than loop)
    for col in NUMERIC_COLUMNS:
        if col in df_rows.columns:
            df_rows[col] = pd.to_numeric(df_rows[col], errors='coerce').fillna(0).astype("float32")

    # 4. MERGE WITH MASTER METADATA
    if not df_master.empty:
        df_master["asin"] = df_master["asin"].astype(str).str.strip().str.upper()

        # Left merge to preserve all rows
        df_rows = df_rows.merge(df_master, on="asin", how="left")

        # Vectorized cleanup for missing values
        df_rows["is_starbucks"] = df_rows["is_starbucks"].fillna(0).astype("int8")  # Use int8 for memory efficiency
        df_rows["title"] = df_rows["title"].fillna("Unknown Product")
        df_rows["main_image"] = df_rows["main_image"].fillna("")

    return df_rows