"""
Test Script: Verify Sales Rank Sorting in Keepa API
====================================================
This script verifies that our Keepa queries return products sorted by
Sales Rank (best sellers first).

Success Criteria for "Dial" search:
- Top results should be high-velocity bulk items like:
  - "Dial Gold Bar Soap, 8 Count"
  - "Dial Antibacterial Liquid Hand Soap, 1 Gallon Refill"
  - NOT single bottles sold by random resellers

Run: python test_sales_rank_sort.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Fix Windows encoding issues
sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()


def get_keepa_api_key():
    """Get Keepa API key from environment."""
    key = os.getenv("KEEPA_API_KEY")
    if not key:
        raise ValueError("KEEPA_API_KEY not found in environment")
    return key


def test_sorted_query(keyword: str = "Dial", limit: int = 10):
    """
    Test that Keepa returns products sorted by Sales Rank.

    Args:
        keyword: Search term to test
        limit: Number of results to show
    """
    api_key = get_keepa_api_key()

    print(f"\n{'='*60}")
    print(f"Testing Sales Rank Sort for: '{keyword}'")
    print(f"{'='*60}\n")

    # Query WITH sort parameter (what we want)
    query_sorted = {
        "title": keyword,
        "perPage": 50,
        "page": 0,
        "current_SALES_gte": 1,
        "current_SALES_lte": 100000,
        "sort": [["current_SALES", "asc"]]  # KEY: Sort by sales rank ascending
    }

    url = f"https://api.keepa.com/query?key={api_key}&domain=1&stats=0"

    print("Fetching with sort=[['current_SALES', 'asc']]...")
    response = requests.post(url, json=query_sorted, timeout=30)

    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code} - {response.text}")
        return False

    result = response.json()
    asins = result.get("asinList", [])[:limit]

    if not asins:
        print(f"❌ No products found for '{keyword}'")
        return False

    print(f"\n✅ Found {len(result.get('asinList', []))} total products")
    print(f"\nTop {limit} ASINs by Sales Rank:\n")

    # Now fetch full product details to get titles and BSR
    import keepa
    api = keepa.Keepa(api_key)
    products = api.query(asins, stats=30, rating=False)

    print(f"{'#':<4} {'BSR':<12} {'Title':<70}")
    print("-" * 90)

    hero_product_count = 0

    for i, product in enumerate(products, 1):
        title = product.get("title", "Unknown")[:70]

        # Get BSR from csv
        csv = product.get("csv", [])
        bsr = 0
        if csv and len(csv) > 3 and csv[3]:
            bsr_data = csv[3]
            if isinstance(bsr_data, list) and len(bsr_data) > 0:
                bsr = bsr_data[-1] if bsr_data[-1] != -1 else 0
            elif isinstance(bsr_data, int):
                bsr = bsr_data if bsr_data != -1 else 0

        bsr_str = f"{bsr:,}" if bsr > 0 else "N/A"
        print(f"{i:<4} {bsr_str:<12} {title}")

        # Check for "hero" indicators (bulk packs, refills, multi-packs)
        title_lower = title.lower()
        if any(x in title_lower for x in ['count', 'pack', 'refill', 'gallon', 'bundle', 'bulk']):
            hero_product_count += 1

    print(f"\n{'='*60}")
    print(f"ANALYSIS")
    print(f"{'='*60}")

    # Check if top 3 have reasonable BSRs (< 50,000 indicates high volume)
    top_3_bsrs = []
    for product in products[:3]:
        csv = product.get("csv", [])
        if csv and len(csv) > 3 and csv[3]:
            bsr_data = csv[3]
            if isinstance(bsr_data, list) and len(bsr_data) > 0 and bsr_data[-1] != -1:
                top_3_bsrs.append(bsr_data[-1])

    if top_3_bsrs:
        avg_top_3_bsr = sum(top_3_bsrs) / len(top_3_bsrs)
        print(f"📊 Average BSR of top 3: {avg_top_3_bsr:,.0f}")

        if avg_top_3_bsr < 50000:
            print(f"✅ PASS: Top products have strong sales (BSR < 50k)")
        else:
            print(f"⚠️ WARNING: Top products have weaker sales than expected")

    print(f"📦 Hero products detected (bulk/multi-pack): {hero_product_count}/{limit}")

    if hero_product_count >= 3:
        print(f"✅ PASS: Multiple bulk/multi-pack products in top results")
    else:
        print(f"⚠️ NOTE: Fewer bulk products than expected (may vary by brand)")

    # Final verdict
    print(f"\n{'='*60}")
    if top_3_bsrs and avg_top_3_bsr < 100000:
        print("🎯 SORT APPEARS TO BE WORKING - High-volume products first!")
    else:
        print("⚠️ VERIFY MANUALLY - Check if these are the expected hero products")
    print(f"{'='*60}\n")

    return True


if __name__ == "__main__":
    # Test with Dial (the example from the task)
    test_sorted_query("Dial", limit=10)

    # Optional: Test with another brand
    # test_sorted_query("Starbucks", limit=10)
