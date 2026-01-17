"""
Demo Data Module: Fetch and process ASINs for prospect demos.
Allows anyone to paste their ASINs and see ShelfGuard in action.
"""
import streamlit as st
import pandas as pd
import numpy as np
import keepa
import time
from pathlib import Path
import sys

# Add parent path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scrapers.keepa_client import build_keepa_weekly_table


def has_keepa_key() -> bool:
    """Check if a Keepa API key is configured (without calling the API)."""
    # Try secrets first
    try:
        key = st.secrets.get("keepa", {}).get("api_key")
        if key and key.strip():
            return True
    except:
        pass
    
    # Try environment variable
    import os
    key = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")
    if key and key.strip():
        return True
    
    return False


def get_keepa_api():
    """
    Initialize Keepa API with key from secrets or environment.
    Only call this when you're actually about to make API requests.
    """
    # Try secrets first (Streamlit Cloud)
    try:
        key = st.secrets.get("keepa", {}).get("api_key")
        if key and key.strip():
            api = keepa.Keepa(key)
            return api
    except RuntimeError as e:
        # Catches PAYMENT_REQUIRED, INVALID_KEY, etc.
        st.error(f"❌ Keepa API error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Keepa connection failed: {str(e)}")
        return None
    
    # Try environment variable
    import os
    key = os.getenv("KEEPA_API_KEY") or os.getenv("KEEPA_KEY")
    if key and key.strip():
        try:
            api = keepa.Keepa(key)
            return api
        except RuntimeError as e:
            st.error(f"❌ Keepa API error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Keepa connection failed: {str(e)}")
            return None
    
    return None


def generate_sample_data(asins: list) -> pd.DataFrame:
    """
    Generate realistic sample data for demos when Keepa API is unavailable.
    This allows prospects to see the dashboard with their ASINs even without API access.
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    weeks = pd.date_range(end=datetime.now(), periods=52, freq='W-MON')
    
    rows = []
    for asin in asins:
        # Generate semi-realistic patterns
        base_price = np.random.uniform(15, 75)
        base_units = np.random.uniform(50, 500)
        trend = np.random.choice([-0.02, 0, 0.01, 0.03])  # Weekly trend
        volatility = np.random.uniform(0.1, 0.3)
        
        for i, week in enumerate(weeks):
            # Add trend and random noise
            price = base_price * (1 + np.random.uniform(-0.05, 0.05))
            units = base_units * (1 + trend * i) * (1 + np.random.uniform(-volatility, volatility))
            units = max(0, units)
            
            # Seasonal boost (Q4)
            if week.month in [10, 11, 12]:
                units *= 1.3
            
            rows.append({
                'asin': asin,
                'parent_asin': asin,
                'title': f"Product {asin}",
                'week_start': week,
                'weekly_sales_filled': units * price,
                'estimated_units': units,
                'filled_price': price,
                'amazon_bb_share': np.random.uniform(0.6, 1.0),
                'new_fba_price': price * np.random.uniform(0.95, 1.05),
                'sales_rank_filled': int(np.random.uniform(1000, 50000) * (1 - trend * i * 0.5)),
                'velocity_decay': 1.0 + trend * 20,
                'weeks_of_cover': np.random.uniform(2, 12),
                'variation_attributes': f"Variety | {np.random.choice(['12', '24', '48', '72', '96'])} Count",
                'fba_fees': price * 0.15,
                'package_vol_cf': np.random.uniform(0.02, 0.15),
                'is_starbucks': 1,
                'main_image': ''
            })
    
    df = pd.DataFrame(rows)
    return df


def parse_asin_input(raw_input: str) -> list:
    """
    Parse ASINs from various input formats.
    Handles: comma-separated, newline-separated, space-separated, URLs
    """
    if not raw_input:
        return []
    
    # Clean up the input
    text = raw_input.strip()
    
    # Extract ASINs from Amazon URLs if present
    import re
    url_pattern = r'/dp/([A-Z0-9]{10})'
    url_asins = re.findall(url_pattern, text, re.IGNORECASE)
    
    # Split by common delimiters
    parts = re.split(r'[,\n\s]+', text)
    
    # Filter to valid ASIN format (10 alphanumeric characters)
    asin_pattern = re.compile(r'^[A-Z0-9]{10}$', re.IGNORECASE)
    direct_asins = [p.strip().upper() for p in parts if asin_pattern.match(p.strip())]
    
    # Combine and dedupe
    all_asins = list(set(url_asins + direct_asins))
    
    return all_asins[:50]  # Limit to 50 ASINs for demo


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_keepa_data(asins: tuple) -> pd.DataFrame:
    """
    Fetch data from Keepa API for given ASINs.
    Cached for 1 hour to avoid redundant API calls.
    """
    api = get_keepa_api()
    if not api:
        st.error("❌ Keepa API key not configured. Add `keepa.api_key` to `.streamlit/secrets.toml`")
        return pd.DataFrame()
    
    asin_list = list(asins)
    all_products = []
    
    # Fetch in batches of 10 (Keepa limit)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(asin_list), 10):
        batch = asin_list[i:i+10]
        batch_num = i // 10 + 1
        total_batches = (len(asin_list) + 9) // 10
        
        status_text.text(f"Fetching batch {batch_num}/{total_batches}...")
        progress_bar.progress((i + 10) / len(asin_list))
        
        try:
            # buybox=True for Buy Box history, update=24 for fresh data, stats=90 for 90-day stats
            res = api.query(batch, domain="US", buybox=True, update=24, stats=90)
            all_products.extend(res)
            time.sleep(1)  # Respect rate limits
        except Exception as e:
            st.warning(f"⚠️ Batch {batch_num} failed: {str(e)[:100]}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_products:
        st.error("❌ No data returned from Keepa. Check your ASINs.")
        return pd.DataFrame()
    
    # Process through the existing Keepa client
    df = build_keepa_weekly_table(all_products)
    
    if df.empty:
        st.error("❌ Failed to process Keepa data.")
        return pd.DataFrame()
    
    # Mark all as "target brand" for demo purposes
    df["is_starbucks"] = 1
    df["main_image"] = df.get("main_image", "")
    
    return df


def render_asin_upload_ui():
    """
    Render the ASIN upload interface.
    Returns the processed DataFrame if ASINs were uploaded and fetched.
    """
    st.markdown("### 🚀 Try With Your Own ASINs")
    st.caption("Paste up to 50 ASINs to see ShelfGuard analyze YOUR portfolio.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        asin_input = st.text_area(
            "Paste ASINs (one per line, comma-separated, or Amazon URLs)",
            height=120,
            placeholder="B08XYZ1234\nB09ABC5678\nhttps://amazon.com/dp/B07DEF9012\n...",
            key="asin_input"
        )
    
    with col2:
        st.markdown("**Accepted formats:**")
        st.caption("• One ASIN per line")
        st.caption("• Comma-separated")
        st.caption("• Amazon product URLs")
        st.caption("• Mix of all above")
    
    # Parse and validate
    asins = parse_asin_input(asin_input)
    
    if asins:
        st.success(f"✅ {len(asins)} valid ASINs detected")
        
        # Show preview
        with st.expander("Preview ASINs", expanded=False):
            st.code("\n".join(asins[:10]) + (f"\n... and {len(asins)-10} more" if len(asins) > 10 else ""))
    
    # Check for API key (doesn't call the API, just checks if key exists)
    has_api = has_keepa_key()
    
    # Fetch button
    fetch_col1, fetch_col2 = st.columns([1, 3])
    with fetch_col1:
        if has_api:
            fetch_button = st.button("🔍 Analyze Portfolio", type="primary", disabled=len(asins) == 0)
        else:
            fetch_button = st.button("🎲 Demo with Sample Data", type="secondary", disabled=len(asins) == 0)
    
    with fetch_col2:
        if not asins:
            st.caption("Paste ASINs above to enable analysis")
        elif not has_api:
            st.caption("⚠️ No Keepa key — will generate sample data for demo")
    
    # Process if button clicked (only NOW do we call the Keepa API)
    if fetch_button and asins:
        if has_api:
            with st.spinner("Fetching data from Keepa..."):
                df = fetch_keepa_data(tuple(asins))
                # If Keepa failed, fall back to sample data
                if df.empty:
                    st.warning("Keepa fetch failed. Generating sample data instead...")
                    df = generate_sample_data(asins)
        else:
            with st.spinner("Generating sample data for demo..."):
                df = generate_sample_data(asins)
                st.info("📊 Using sample data (no Keepa API key). Add key to `.streamlit/secrets.toml` for real data.")
        
        if not df.empty:
            # Store in session state for use in main dashboard
            st.session_state['demo_data'] = df
            st.session_state['demo_mode'] = True
            st.success(f"✅ Loaded {len(df['asin'].unique())} products with {len(df)} weekly records")
            st.rerun()
    
    return None


def get_demo_data():
    """
    Get the demo data from session state if available.
    Returns None if no demo data loaded.
    """
    if st.session_state.get('demo_mode') and 'demo_data' in st.session_state:
        return st.session_state['demo_data']
    return None


def clear_demo_data():
    """Clear demo data and return to default mode."""
    st.session_state['demo_mode'] = False
    if 'demo_data' in st.session_state:
        del st.session_state['demo_data']
