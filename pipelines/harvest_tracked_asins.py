"""
ShelfGuard Product Harvesting Pipeline
=======================================
The Oracle's Data Warehouse Ingestion Worker

This pipeline:
1. Queries all active ASINs from tracked_asins table
2. Batch-fetches from Keepa API (respecting rate limits)
3. Transforms using keepa_client.py logic
4. Upserts into product_snapshots table

Run Schedule: Every 6-24 hours via cron/GitHub Actions
Usage: python pipelines/harvest_tracked_asins.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Set

import pandas as pd
import numpy as np
import httpx
import keepa
from dotenv import load_dotenv
from supabase import create_client, Client

# Setup path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scrapers.keepa_client import build_keepa_weekly_table

# Load environment
load_dotenv(ROOT / ".env", override=True)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
KEEPA_API_KEY = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")

# Harvesting settings
BATCH_SIZE = 20  # ASINs per Keepa request (conservative for rate limits)
CHUNK_SIZE = 100  # Records per Supabase upsert
RETRY_DELAY = 30  # Seconds between retries
MAX_RETRIES = 3
POLITENESS_DELAY = 2  # Seconds between Keepa batches

# Table name
T_SNAPSHOTS = "product_snapshots"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_supabase_client() -> Client:
    """Create Supabase client with stable connection."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    
    stable_session = httpx.Client(
        http2=False,
        timeout=httpx.Timeout(60.0, connect=10.0)
    )
    
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    client.postgrest.session = stable_session
    return client


def get_all_tracked_asins(supabase: Client) -> Set[str]:
    """
    Query all unique active ASINs from tracked_asins table.
    
    Returns:
        Set of unique ASIN strings
    """
    try:
        result = supabase.table("tracked_asins").select("asin").eq("is_active", True).execute()
        
        if not result.data:
            logger.warning("No tracked ASINs found in database")
            return set()
        
        asins = {row["asin"].strip().upper() for row in result.data if row.get("asin")}
        logger.info(f"📋 Found {len(asins)} unique tracked ASINs")
        return asins
        
    except Exception as e:
        logger.error(f"❌ Failed to query tracked_asins: {e}")
        return set()


def fetch_keepa_products(asins: List[str], api_key: str) -> List[dict]:
    """
    Fetch product data from Keepa API in batches.
    
    Args:
        asins: List of ASINs to fetch
        api_key: Keepa API key
        
    Returns:
        List of Keepa product dictionaries
    """
    if not asins:
        return []
    
    api = keepa.Keepa(api_key)
    all_products = []
    
    for i in range(0, len(asins), BATCH_SIZE):
        batch = asins[i:i + BATCH_SIZE]
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"📡 Fetching batch {i//BATCH_SIZE + 1}/{(len(asins)-1)//BATCH_SIZE + 1} ({len(batch)} ASINs)...")
                
                # Fetch with stats for 90 days + buybox data
                products = api.query(batch, domain="US", buybox=True, stats=90)
                all_products.extend(products)
                
                logger.info(f"✅ Batch {i//BATCH_SIZE + 1} complete ({len(products)} products)")
                break
                
            except Exception as e:
                logger.warning(f"⚠️ Batch failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"❌ Batch permanently failed after {MAX_RETRIES} attempts")
        
        # Politeness delay between batches
        if i + BATCH_SIZE < len(asins):
            time.sleep(POLITENESS_DELAY)
    
    logger.info(f"📊 Total products fetched: {len(all_products)}")
    return all_products


def transform_to_snapshots(products: List[dict], snapshot_date: date) -> pd.DataFrame:
    """
    Transform Keepa products into product_snapshots format.
    
    Uses the existing keepa_client.py logic for consistency.
    
    Args:
        products: List of Keepa product dictionaries
        snapshot_date: Date for the snapshot
        
    Returns:
        DataFrame ready for upsert to product_snapshots
    """
    if not products:
        return pd.DataFrame()
    
    # Use existing weekly table builder (extracts all metrics)
    weekly_df = build_keepa_weekly_table(products)
    
    if weekly_df.empty:
        logger.warning("⚠️ Weekly table is empty after transformation")
        return pd.DataFrame()
    
    # Get the latest week's data for each ASIN
    if "week_start" in weekly_df.columns:
        # Sort by week_start descending and take first occurrence per ASIN
        weekly_df = weekly_df.sort_values("week_start", ascending=False)
        latest_df = weekly_df.drop_duplicates(subset=["asin"], keep="first").copy()
    else:
        latest_df = weekly_df.drop_duplicates(subset=["asin"]).copy()
    
    # Map columns to snapshot schema
    snapshot_df = pd.DataFrame({
        "asin": latest_df["asin"].str.strip().str.upper(),
        "snapshot_date": snapshot_date.isoformat(),
        "buy_box_price": latest_df.get("buy_box_price"),
        "amazon_price": latest_df.get("amazon_price"),
        "new_fba_price": latest_df.get("new_fba_price"),
        "sales_rank": latest_df.get("sales_rank_filled", latest_df.get("sales_rank")),
        "amazon_bb_share": latest_df.get("amazon_bb_share"),
        "buy_box_switches": latest_df.get("buy_box_switches"),
        "new_offer_count": latest_df.get("new_offer_count"),
        "review_count": latest_df.get("review_count"),
        "rating": latest_df.get("rating"),
        "estimated_units": latest_df.get("estimated_units"),
        "estimated_weekly_revenue": latest_df.get("weekly_sales_filled"),
        "filled_price": latest_df.get("filled_price"),
        "title": latest_df.get("title"),
        "brand": latest_df.get("brand"),
        "parent_asin": latest_df.get("parent_asin"),
        "main_image": latest_df.get("main_image"),
        "source": "keepa"
    })
    
    logger.info(f"🔄 Transformed {len(snapshot_df)} products for snapshot")
    return snapshot_df


def _clean_value(x):
    """Clean value for Supabase insertion."""
    if x is None or pd.isna(x):
        return None
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def upsert_snapshots(supabase: Client, df: pd.DataFrame) -> int:
    """
    Upsert snapshot DataFrame to Supabase.
    
    Args:
        supabase: Supabase client
        df: DataFrame of snapshots
        
    Returns:
        Number of records upserted
    """
    if df.empty:
        return 0
    
    records = df.to_dict(orient="records")
    total_upserted = 0
    
    logger.info(f"🔼 Upserting {len(records)} snapshots to '{T_SNAPSHOTS}'...")
    
    for i in range(0, len(records), CHUNK_SIZE):
        chunk = [{k: _clean_value(v) for k, v in r.items()} for r in records[i:i + CHUNK_SIZE]]
        
        try:
            supabase.table(T_SNAPSHOTS).upsert(
                chunk, 
                on_conflict="asin,snapshot_date"
            ).execute()
            total_upserted += len(chunk)
            
        except Exception as e:
            logger.error(f"⚠️ Batch {i//CHUNK_SIZE + 1} failed: {e}")
            # Retry once
            time.sleep(RETRY_DELAY)
            try:
                supabase.table(T_SNAPSHOTS).upsert(
                    chunk,
                    on_conflict="asin,snapshot_date"
                ).execute()
                total_upserted += len(chunk)
            except Exception as e2:
                logger.error(f"❌ Batch permanently failed: {e2}")
    
    return total_upserted


def harvest_all():
    """
    Main harvesting orchestrator.
    
    Workflow:
    1. Connect to Supabase
    2. Get all tracked ASINs
    3. Fetch from Keepa
    4. Transform to snapshot format
    5. Upsert to product_snapshots
    """
    logger.info("=" * 60)
    logger.info("🌾 STARTING HARVEST PIPELINE")
    logger.info(f"📅 Snapshot Date: {date.today().isoformat()}")
    logger.info("=" * 60)
    
    # Validate configuration
    if not KEEPA_API_KEY:
        logger.error("❌ KEEPA_API_KEY not found in environment")
        return False
    
    try:
        # 1. Connect to Supabase
        supabase = create_supabase_client()
        logger.info("✅ Connected to Supabase")
        
        # 2. Get tracked ASINs
        tracked_asins = get_all_tracked_asins(supabase)
        
        if not tracked_asins:
            logger.warning("⚠️ No ASINs to harvest. Add products via Market Discovery first.")
            return True
        
        # 3. Fetch from Keepa
        asin_list = list(tracked_asins)
        products = fetch_keepa_products(asin_list, KEEPA_API_KEY)
        
        if not products:
            logger.error("❌ No products returned from Keepa")
            return False
        
        # 4. Transform to snapshots
        snapshot_date = date.today()
        snapshot_df = transform_to_snapshots(products, snapshot_date)
        
        if snapshot_df.empty:
            logger.error("❌ Transformation produced empty DataFrame")
            return False
        
        # 5. Upsert to Supabase
        upserted_count = upsert_snapshots(supabase, snapshot_df)
        
        logger.info("=" * 60)
        logger.info("🚀 HARVEST COMPLETE")
        logger.info(f"   ASINs requested: {len(tracked_asins)}")
        logger.info(f"   Products fetched: {len(products)}")
        logger.info(f"   Snapshots saved: {upserted_count}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Harvest failed with error: {e}")
        return False


def harvest_specific_asins(asins: List[str]):
    """
    Harvest specific ASINs (useful for new project creation).
    
    Args:
        asins: List of ASIN strings to harvest
    """
    logger.info(f"🎯 Harvesting {len(asins)} specific ASINs...")
    
    if not KEEPA_API_KEY:
        logger.error("❌ KEEPA_API_KEY not found")
        return False
    
    try:
        supabase = create_supabase_client()
        
        # Normalize ASINs
        asins = [a.strip().upper() for a in asins if a]
        
        # Fetch and transform
        products = fetch_keepa_products(asins, KEEPA_API_KEY)
        snapshot_df = transform_to_snapshots(products, date.today())
        
        # Upsert
        upserted = upsert_snapshots(supabase, snapshot_df)
        logger.info(f"✅ Harvested and saved {upserted} snapshots")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Specific harvest failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Harvest tracked ASINs from Keepa to Supabase")
    parser.add_argument("--asins", nargs="+", help="Specific ASINs to harvest (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Test without writing to database")
    
    args = parser.parse_args()
    
    if args.asins:
        success = harvest_specific_asins(args.asins)
    else:
        success = harvest_all()
    
    sys.exit(0 if success else 1)
