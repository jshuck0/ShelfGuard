import os
import httpx
from dotenv import load_dotenv

load_dotenv()
KEEPA_KEY = os.getenv("KEEPA_API_KEY")
# Using a different Starbucks ASIN just to verify if the first one is "dead"
TEST_ASIN = "B007TGDXNO" 

def force_update_test():
    url = "https://api.keepa.com/product"
    params = {
        "key": KEEPA_KEY,
        "domain": 1,
        "asin": TEST_ASIN,
        "update": 24, # Force a fresh scrape if data is > 24hrs old
        "stats": 1
    }
    
    with httpx.Client() as client:
        print(f"📡 Requesting fresh data for {TEST_ASIN}...")
        r = client.get(url, params=params, timeout=30.0)
        data = r.json()
        
        products = data.get("products", [])
        if not products or products[0] is None:
            print("❌ Still no product found.")
            return

        p = products[0]
        title = p.get('title')
        print(f"✅ Title Found: {title if title else 'STILL NONE'}")
        print(f"✅ Variation Key Check: {'variationAttributes' in p}")
        
        if title is None:
            print("💡 Observation: The API returns the keys but the values are still empty.")
            print("💡 This often means your API key has enough credits to connect, but not enough to 'trigger' a new scrape.")

if __name__ == "__main__":
    force_update_test()