
import streamlit as st
import pandas as pd
import hashlib

# Import dependencies
try:
    from src.supabase_reader import (
        check_data_freshness,
        get_market_snapshot_from_cache,
        load_historical_metrics_from_db,
        load_historical_metrics_by_asins,
        get_market_snapshot_with_network_intelligence,
        SUPABASE_CACHE_ENABLED
    )
except ImportError:
    SUPABASE_CACHE_ENABLED = False
    get_market_snapshot_with_network_intelligence = None

try:
    from utils.data_healer import heal_market_snapshot, clean_and_interpolate_metrics
except ImportError:
    heal_market_snapshot = None

try:
    from utils.ai_engine import extract_portfolio_velocity
except ImportError:
    # Will fallback individually
    pass

# Helper for money formatting
def f_money(val):
    if val >= 1000000:
        return f"${val/1000000:.1f}M"
    if val >= 1000:
        return f"${val/1000:.0f}K"
    return f"${val:.0f}"

def ensure_data_loaded():
    """
    Loads project data, performs healing, calculates intelligence, and returns dashboard objects.
    Returns:
        tuple: (res, fin, enriched_portfolio_df, portfolio_context)
    """
    
    # 2. DATA INGESTION - CACHE-FIRST ARCHITECTURE (The Oracle)
    project_name = st.session_state.get('active_project_name', 'Unknown Project')
    seed_asin = st.session_state.get('active_project_asin', None)
    project_asins = st.session_state.get('active_project_all_asins', [])
    seed_brand = st.session_state.get('active_project_seed_brand', '')
    
    if not seed_asin:
        return None, None, pd.DataFrame(), ""

    # Initialize data containers
    df_weekly = pd.DataFrame()
    market_snapshot = pd.DataFrame()
    data_source = "none"
    network_stats = {}

    # === SUPABASE CACHE (PRIMARY) ===
    category_id = st.session_state.get('active_project_category_id')

    if SUPABASE_CACHE_ENABLED and project_asins:
        try:
            if get_market_snapshot_with_network_intelligence:
                market_snapshot, cache_stats = get_market_snapshot_with_network_intelligence(
                    project_asins, seed_brand, category_id
                )
                if cache_stats.get('has_network_context'):
                    network_stats = cache_stats.get('network_intelligence', {})
            elif get_market_snapshot_from_cache:
                market_snapshot, cache_stats = get_market_snapshot_from_cache(project_asins, seed_brand)

            if not market_snapshot.empty:
                data_source = "supabase"
                freshness = check_data_freshness(project_asins) if check_data_freshness else {}
        except Exception as e:
            pass
    
    # === GET REAL WEEKLY DATA FROM SESSION STATE ===
    df_weekly = st.session_state.get('active_project_data', pd.DataFrame())
    
    if df_weekly.empty:
        df_weekly = market_snapshot.copy() if not market_snapshot.empty else pd.DataFrame()
    
    if market_snapshot.empty:
        market_snapshot = st.session_state.get('active_project_market_snapshot', pd.DataFrame())
        if not market_snapshot.empty:
            data_source = "session"

    if df_weekly.empty or market_snapshot.empty:
        return None, None, pd.DataFrame(), ""
    
    # === DATA HEALER ===
    if heal_market_snapshot:
        market_snapshot = heal_market_snapshot(market_snapshot, verbose=False)
        if clean_and_interpolate_metrics:
            market_snapshot = clean_and_interpolate_metrics(market_snapshot, group_by_column="asin", verbose=False)
        if not df_weekly.equals(market_snapshot):
            df_weekly = heal_market_snapshot(df_weekly, verbose=False)
    
    # Store df_weekly
    st.session_state['df_weekly'] = df_weekly
    
    # === VELOCITY EXTRACTION ===
    project_id = st.session_state.get('active_project_id')
    velocity_df = pd.DataFrame()
    historical_df_for_velocity = pd.DataFrame()
    
    if SUPABASE_CACHE_ENABLED and load_historical_metrics_from_db and project_id:
        try:
            historical_df_for_velocity = load_historical_metrics_from_db(project_id)
        except: pass
        
    if historical_df_for_velocity.empty and not df_weekly.empty:
        historical_df_for_velocity = df_weekly

    try:
        from utils.ai_engine import extract_portfolio_velocity
        if not historical_df_for_velocity.empty and 'asin' in historical_df_for_velocity.columns:
            velocity_df = extract_portfolio_velocity(historical_df_for_velocity)
            if not velocity_df.empty:
                market_snapshot = market_snapshot.merge(
                    velocity_df[['asin', 'velocity_trend_30d', 'velocity_trend_90d', 'data_quality', 'data_weeks']],
                    on='asin', how='left'
                )
    except: pass

    # === BRAND IDENTIFICATION ===
    if 'brand' not in market_snapshot.columns:
        market_snapshot['brand'] = market_snapshot['title'].apply(lambda x: x.split()[0] if pd.notna(x) and x else "Unknown")

    seed_product = market_snapshot[market_snapshot['asin'] == seed_asin]
    if seed_product.empty:
         # Simplified fallback
         target_brand = seed_brand if seed_brand else "Your Brand"
    else:
         target_brand = seed_product['brand'].iloc[0]

    target_brand_lower = target_brand.lower().strip() if target_brand else ""
    market_snapshot['brand_lower'] = market_snapshot['brand'].str.lower().str.strip().fillna("")
    
    market_snapshot['is_your_brand'] = market_snapshot['brand_lower'].str.contains(
        target_brand_lower, case=False, na=False, regex=False
    ) if target_brand_lower else False
    
    title_match = market_snapshot['title'].str.lower().str.contains(
        target_brand_lower, case=False, na=False, regex=False
    ) if target_brand_lower else False
    
    market_snapshot['is_your_brand'] = market_snapshot['is_your_brand'] | title_match
    
    # Revenue Proxy
    if 'revenue_proxy' not in market_snapshot.columns or market_snapshot['revenue_proxy'].isna().all():
        if 'avg_weekly_revenue' in market_snapshot.columns:
            market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['avg_weekly_revenue'], errors='coerce').fillna(0) * 4.33
        else:
            market_snapshot['revenue_proxy'] = 0.0
    else:
        market_snapshot['revenue_proxy'] = pd.to_numeric(market_snapshot['revenue_proxy'], errors='coerce').fillna(0)

    portfolio_df = market_snapshot[market_snapshot['is_your_brand']].copy()
    market_df = market_snapshot

    # Competitive Intel
    price_col = 'current_price' if 'current_price' in market_snapshot.columns else 'avg_price'
    if price_col and price_col in market_snapshot.columns:
        competitor_prices = market_snapshot.loc[~market_snapshot['is_your_brand'], price_col]
        market_avg_price = competitor_prices.mean() if len(competitor_prices) > 0 else 0
        if market_avg_price > 0:
            market_snapshot['price_gap_vs_competitor'] = (
                (market_snapshot[price_col] - market_avg_price) / market_avg_price
            ).fillna(0)
    
    # Metrics
    portfolio_rev_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in portfolio_df.columns else 'revenue_proxy'
    portfolio_revenue = portfolio_df[portfolio_rev_col].sum() if portfolio_rev_col in portfolio_df.columns else 0
    portfolio_product_count = len(portfolio_df)
    
    market_rev_col = 'revenue_proxy_adjusted' if 'revenue_proxy_adjusted' in market_df.columns else 'revenue_proxy'
    total_market_revenue = market_df[market_rev_col].sum() if market_rev_col in market_df.columns else 0
    total_market_products = len(market_df)
    
    competitor_revenue = total_market_revenue - portfolio_revenue
    competitor_product_count = total_market_products - portfolio_product_count
    
    your_market_share = (portfolio_revenue / total_market_revenue * 100) if total_market_revenue > 0 else 0

    # Snapshot DF for Dashboard
    portfolio_snapshot_df = portfolio_df.copy()
    if 'revenue_proxy' in portfolio_snapshot_df.columns:
        portfolio_snapshot_df['revenue_proxy'] = pd.to_numeric(portfolio_snapshot_df['revenue_proxy'], errors='coerce').fillna(0)
    else:
        portfolio_snapshot_df['revenue_proxy'] = 0.0
    
    portfolio_snapshot_df['weekly_sales_filled'] = portfolio_snapshot_df['revenue_proxy'].copy()
    portfolio_snapshot_df['problem_category'] = '✅ Your Brand - Healthy'
    portfolio_snapshot_df['predictive_zone'] = '✅ HOLD'
    portfolio_snapshot_df['is_healthy'] = True

    # RES Object
    res = {
        'data': portfolio_snapshot_df,
        'total_rev': portfolio_revenue,
        'yoy_delta': 0.0,
        'share_delta': 0.0,
        'predictive_zones': {
            '✅ HOLD': portfolio_revenue,
            '🛡️ DEFEND': 0,
            '⚡ EXPLOIT': 0,
            '🔄 REPLENISH': 0
        },
        'demand_forecast': {},
        'hierarchy': {}
    }
    
    # FIN Object
    fin = {
        'efficiency_score': int(your_market_share),
        'portfolio_status': 'Active',
    }

    # Context
    portfolio_context = f"""
    BRAND PERFORMANCE SNAPSHOT:
    - Brand: {target_brand}
    - Market Share: {your_market_share:.1f}%
    - Portfolio Revenue: ${portfolio_revenue:,.0f}/month
    - Total Market Size: ${total_market_revenue:,.0f}/month
    """

    # === INTELLIGENCE CALCULATION ===
    # Use the cached intelligence function (assuming it's imported in main app, we can redo it here or rely on app)
    # But this function is in a separate module. We should assume the app handles the caching decoration 
    # OR we implement non-decorated version here.
    
    # Actually, the app has `_cached_portfolio_intelligence` which is decorated.
    # We can't call it here easily.
    # We will return the PREPARED data (res, portfolio_snapshot_df) 
    # and let the App call the Intelligence function.
    
    return res, fin, portfolio_snapshot_df, portfolio_context
