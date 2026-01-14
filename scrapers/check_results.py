import pandas as pd
from pathlib import Path

# 1. Point to your output file
file_path = Path("outputs/keepa_weekly_kcup_top1000_rows.csv")

if not file_path.exists():
    print(f"❌ Could not find {file_path}. Please check your folder.")
else:
    df = pd.read_csv(file_path)
    df['week_start'] = pd.to_datetime(df['week_start'])

    # 2. Get the most recent week of data
    latest_week = df['week_start'].max()
    latest_df = df[df['week_start'] == latest_week].copy()

    # 3. Apply the "Director's Fix" (De-duplication by Parent ASIN)
    # This prevents double-counting variations (e.g., 24-pack vs 72-pack)
    parent_summary = latest_df.groupby('parent_asin').agg({
        'weekly_sales_filled': 'max',
        'is_starbucks': 'max',
        'title': 'first'
    }).reset_index()

    # 4. Calculate Market Metrics
    total_market_rev = parent_summary['weekly_sales_filled'].sum()
    sbux_rev = parent_summary[parent_summary['is_starbucks'] == 1]['weekly_sales_filled'].sum()
    sbux_share = (sbux_rev / total_market_rev * 100) if total_market_rev > 0 else 0

    # 5. Competitor Analysis (Guessing Brand from Title)
    parent_summary['brand'] = parent_summary['title'].str.split().str[0].str.upper()
    top_brands = parent_summary.groupby('brand')['weekly_sales_filled'].sum().sort_values(ascending=False).head(6)

    print("-" * 50)
    print(f"📊 ADSIGHTFUL EXECUTIVE SUMMARY | {latest_week.date()}")
    print("-" * 50)
    print(f"Total Market Revenue (Est):  ${total_market_rev:,.2f}")
    print(f"Starbucks Revenue (Est):     ${sbux_rev:,.2f}")
    print(f"Starbucks Market Share:      {sbux_share:.2f}%")
    print("-" * 50)
    print("🏆 MARKET LEADERS (Weekly Revenue)")
    for brand, rev in top_brands.items():
        prefix = "⭐️ " if "STARBUCKS" in brand else "   "
        print(f"{prefix}{brand:15}: ${rev:,.2f}")
    print("-" * 50)