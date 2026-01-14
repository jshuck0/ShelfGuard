from scrapers.keepa_client import fetch_products, extract_weekly_series
from scrapers.popgot_scraper import scrape_category
import pandas as pd

ASINS = [
    # Starbucks ASINs here
]

def run():
    products = fetch_products(ASINS)

    keepa_rows = []
    for p in products:
        df = extract_weekly_series(p)
        df["asin"] = p["asin"]
        keepa_rows.append(df)

    keepa_df = pd.concat(keepa_rows)
    keepa_df.to_csv("data/processed/keepa_weekly.csv", index=False)

    popgot_df = scrape_category("https://popgot.com/k-cup-coffee-pods")
    popgot_df.to_csv("data/raw/popgot_snapshot.csv", index=False)

if __name__ == "__main__":
    run()