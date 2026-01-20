"""
ShelfGuard Data Pipeline Tests
==============================
Tests for the stateful architecture (product_snapshots + historical_metrics)

Replaces legacy K-Cup specific tests.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")


def run_audit():
    """Run a health check on the ShelfGuard data pipeline."""
    print("🛡️  SHELFGUARD OS: DATA PIPELINE AUDIT\n")
    
    # --- PHASE 1: SUPABASE CONNECTION ---
    print("☁️  PHASE 1: Supabase Connection")
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            print("❌ SUPABASE_URL or SUPABASE_SERVICE_KEY not set in .env")
            return
        
        supabase = create_client(url, key)
        print(f"✅ Connected to Supabase")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # --- PHASE 2: CORE TABLES CHECK ---
    print("\n📊 PHASE 2: Core Tables Status")
    
    core_tables = [
        "product_snapshots",
        "projects", 
        "tracked_asins",
        "historical_metrics",
        "resolution_cards"
    ]
    
    for table in core_tables:
        try:
            result = supabase.table(table).select("*", count="exact").limit(1).execute()
            count = result.count if hasattr(result, 'count') else len(result.data)
            print(f"  ✅ {table}: {count:,} rows")
        except Exception as e:
            print(f"  ❌ {table}: Error - {e}")
    
    # --- PHASE 3: INTELLIGENCE TABLES CHECK ---
    print("\n🧠 PHASE 3: Intelligence Tables Status")
    
    intel_tables = [
        "category_intelligence",
        "brand_intelligence",
        "strategic_insights",
        "trigger_events",
        "market_patterns"
    ]
    
    for table in intel_tables:
        try:
            result = supabase.table(table).select("*", count="exact").limit(1).execute()
            count = result.count if hasattr(result, 'count') else len(result.data)
            status = "✅" if count > 0 else "⚪"
            print(f"  {status} {table}: {count:,} rows")
        except Exception as e:
            print(f"  ⚠️  {table}: Not found or error - {e}")
    
    # --- PHASE 4: DATA QUALITY CHECK ---
    print("\n🔍 PHASE 4: Product Snapshots Quality")
    
    try:
        result = supabase.table("product_snapshots").select(
            "asin, estimated_weekly_revenue, sales_rank, filled_price"
        ).limit(10).execute()
        
        if not result.data:
            print("  ⚪ No snapshots cached yet (run discovery first)")
        else:
            has_revenue = sum(1 for r in result.data if r.get("estimated_weekly_revenue"))
            has_rank = sum(1 for r in result.data if r.get("sales_rank"))
            has_price = sum(1 for r in result.data if r.get("filled_price"))
            
            total = len(result.data)
            print(f"  Revenue populated: {has_revenue}/{total}")
            print(f"  Rank populated: {has_rank}/{total}")
            print(f"  Price populated: {has_price}/{total}")
            
            if has_revenue == 0 and has_rank == 0:
                print("  ⚠️  WARNING: Snapshots have NULL metrics - check cache_market_snapshot()")
            else:
                print("  ✅ Data quality looks good")
                
    except Exception as e:
        print(f"  ❌ Quality check failed: {e}")
    
    print("\n🏁 AUDIT COMPLETE")


if __name__ == "__main__":
    run_audit()
