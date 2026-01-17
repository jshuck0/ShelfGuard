import pandas as pd
import numpy as np
from synthetic_intel import (
    calculate_synthetic_cogs, 
    calculate_synthetic_volume,
    calculate_landed_logistics,
    enrich_synthetic_financials,
    interpolate_keepa_gaps
)

def _safe_float(val, default=0.0):
    """Bulletproof float conversion for financial calculations."""
    if pd.notna(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    return default


def get_historical_context(history_df: pd.DataFrame, asin: str, current_value: float, metric_name='weekly_sales_filled') -> dict:
    """
    Analyze 36-month history to provide context for current metric.
    
    Returns:
    - percentile: Where current value ranks (0-100)
    - all_time_high/low: Historical extremes
    - median: Typical value
    - volatility: How much it varies
    - is_anomaly: Is current value unusual?
    - trend: Overall direction over 36M
    """
    asin_history = history_df[history_df['asin'] == asin].sort_values('week_start')
    
    if len(asin_history) < 12:
        return {'insufficient_data': True}
    
    # Check if metric exists in history
    if metric_name not in asin_history.columns:
        return {'insufficient_data': True, 'missing_column': True}
    
    historical_values = asin_history[metric_name].fillna(0).values
    
    # Calculate percentile rank
    percentile = (historical_values < current_value).mean() * 100
    
    # Calculate volatility (coefficient of variation)
    mean_val = historical_values.mean()
    volatility = (historical_values.std() / mean_val) if mean_val > 0 else 0
    
    # Detect anomalies (> 2 standard deviations from mean)
    is_anomaly = abs(current_value - mean_val) > 2 * historical_values.std()
    
    # Calculate long-term trend
    if len(historical_values) >= 24:
        first_half = historical_values[:len(historical_values)//2].mean()
        second_half = historical_values[len(historical_values)//2:].mean()
        trend_direction = (second_half - first_half) / first_half if first_half > 0 else 0
    else:
        trend_direction = 0
    
    return {
        'percentile': round(percentile, 1),
        'all_time_high': float(historical_values.max()),
        'all_time_low': float(historical_values.min()),
        'median': float(np.median(historical_values)),
        'mean': float(mean_val),
        'volatility': round(volatility, 2),
        'is_anomaly': bool(is_anomaly),
        'trend_direction': round(trend_direction, 3),
        'data_points': len(historical_values)
    }


def calculate_demand_forecast(history_df: pd.DataFrame, target_date) -> dict:
    """
    Calculate 8-week forward demand projections with actionable intelligence.

    Uses:
    - Recent velocity trends (8-week momentum)
    - YoY seasonal patterns (same period last year)
    - Growth rate extrapolation
    - Spike/trough detection for timing

    Returns dict of ASIN -> comprehensive forecast data

    Performance optimizations:
    - Vectorized timezone handling
    - Pre-computed date ranges
    - Reduced redundant calculations
    """
    if history_df.empty:
        return {}

    forecast = {}

    # Ensure timezone-aware datetime (vectorized)
    target_dt = pd.to_datetime(target_date)
    if hasattr(target_dt, 'tzinfo') and target_dt.tzinfo is None:
        target_dt = target_dt.tz_localize('UTC')

    # Ensure week_start is timezone-aware (one-time operation)
    if 'week_start' in history_df.columns and history_df['week_start'].dt.tz is None:
        history_df = history_df.copy()
        history_df['week_start'] = history_df['week_start'].dt.tz_localize('UTC')

    # Pre-compute date ranges for seasonal analysis (avoid recalculating in loop)
    ly_dates = [(target_dt - pd.Timedelta(days=365*i)) for i in range(1, 4)]

    # Process each ASIN
    for asin in history_df['asin'].unique():
        asin_history = history_df[history_df['asin'] == asin].sort_values('week_start')
        
        if len(asin_history) < 4:
            continue
        
        # Recent 8 weeks revenue
        recent_8w = asin_history.tail(8)['weekly_sales_filled']
        recent_avg = recent_8w.mean() if len(recent_8w) > 0 else 0
        
        # Week-over-week trend (last 4 weeks)
        recent_4w = asin_history.tail(4)['weekly_sales_filled'].tolist()
        if len(recent_4w) >= 3:
            wow_trend = (recent_4w[-1] - recent_4w[0]) / recent_4w[0] if recent_4w[0] > 0 else 0
        else:
            wow_trend = 0
        
        # Calculate growth rate from trend
        if len(asin_history) >= 12:
            older_8w = asin_history.iloc[-16:-8]['weekly_sales_filled'] if len(asin_history) >= 16 else asin_history.head(8)['weekly_sales_filled']
            older_avg = older_8w.mean() if len(older_8w) > 0 else recent_avg
            growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # === ENHANCED MULTI-YEAR SEASONAL ANALYSIS ===
        # Analyze all 3 years (not just 1) for more accurate seasonal patterns
        lookahead_samples = []

        # Use pre-computed ly_dates to avoid redundant date calculations
        for ly_date in ly_dates:
            # Vectorized date range filtering
            ly_lookahead_start = ly_date + pd.Timedelta(days=7)
            ly_lookahead_end = ly_date + pd.Timedelta(days=63)

            # Single filtered operation
            ly_lookahead = asin_history[
                (asin_history['week_start'] >= ly_lookahead_start) &
                (asin_history['week_start'] <= ly_lookahead_end)
            ]

            if len(ly_lookahead) > 0:
                lookahead_samples.append(ly_lookahead['weekly_sales_filled'].mean())
        
        # Calculate expected value from all available years (vectorized)
        if lookahead_samples:
            expected_8w_avg = np.mean(lookahead_samples)
            seasonal_std = np.std(lookahead_samples) if len(lookahead_samples) > 1 else expected_8w_avg * 0.2

            # Confidence based on consistency across years (optimized branching)
            if len(lookahead_samples) >= 2 and expected_8w_avg > 0:
                cv = seasonal_std / expected_8w_avg
                seasonal_confidence = 0.95 if cv < 0.15 else (0.75 if cv < 0.30 else 0.50)
            else:
                seasonal_confidence = 0.60
            
            # Detect seasonal patterns
            if recent_avg > 0:
                seasonal_change = (expected_8w_avg - recent_avg) / recent_avg
                seasonal_peak_coming = seasonal_change > 0.25
                seasonal_trough_coming = seasonal_change < -0.20
            else:
                seasonal_change = 0
                seasonal_peak_coming = False
                seasonal_trough_coming = False
            
            seasonal_factor = expected_8w_avg / recent_avg if recent_avg > 0 else 1.0
        else:
            # Fallback to simple method if insufficient history
            expected_8w_avg = recent_avg
            seasonal_factor = 1.0
            seasonal_confidence = 0.30
            seasonal_peak_coming = False
            seasonal_trough_coming = False
        
        # === ENHANCED PROJECTION WITH CONFIDENCE ===
        # Blend recent trends with historical seasonal patterns (optimized calculation)
        trend_projection = recent_avg * (1 + (growth_rate * 0.5))
        seasonal_projection = expected_8w_avg if lookahead_samples else recent_avg

        # Weight by confidence (simplified logic)
        weight_seasonal = seasonal_confidence if lookahead_samples else 0.5
        weight_trend = 1 - weight_seasonal

        projected_weekly = (trend_projection * weight_trend) + (seasonal_projection * weight_seasonal)
        projected_8w_total = projected_weekly * 8
        current_8w_total = recent_avg * 8

        forecast_change = (projected_8w_total - current_8w_total) / current_8w_total if current_8w_total > 0 else 0

        # Calculate confidence level (optimized conditional)
        if lookahead_samples:
            confidence_label = "HIGH" if seasonal_confidence >= 0.90 else ("MEDIUM" if seasonal_confidence >= 0.70 else "LOW")
        else:
            confidence_label = "LOW"
        
        # Determine demand trajectory (trending to zero detection)
        trending_to_zero = growth_rate < -0.25 and wow_trend < -0.15
        accelerating_growth = growth_rate > 0.15 and wow_trend > 0.10
        
        # Determine forecast signal
        if forecast_change > 0.20:
            signal = "📈 SURGE"
            action_modifier = "pre_scale"
        elif forecast_change > 0.08:
            signal = "↗️ GROWTH"
            action_modifier = "invest"
        elif forecast_change < -0.15:
            signal = "📉 DECLINE"
            action_modifier = "reduce"
        elif forecast_change < -0.05:
            signal = "↘️ SOFTENING"
            action_modifier = "caution"
        else:
            signal = "→ STABLE"
            action_modifier = "maintain"
        
        # === ACTIONABLE DIRECTIVES ===
        
        # 1. INVENTORY INTELLIGENCE
        if forecast_change > 0.20:
            inventory_action = "🚨 REORDER NOW"
            inventory_reason = f"Demand ↑{forecast_change*100:.0f}% next 8 weeks"
        elif forecast_change < -0.15:
            inventory_action = "⚠️ REDUCE PO"
            inventory_reason = f"Demand ↓{abs(forecast_change)*100:.0f}% projected"
        elif trending_to_zero:
            inventory_action = "🛑 STOP REPLENISHMENT"
            inventory_reason = "Trajectory approaching zero"
        else:
            inventory_action = "✅ MAINTAIN LEVELS"
            inventory_reason = "Stable demand pattern"
        
        # 2. MEDIA SPEND TIMING
        if seasonal_peak_coming:
            media_timing = "🎯 SCALE 2 WEEKS EARLY"
            media_reason = "Seasonal peak detected (YoY pattern)"
        elif forecast_change > 0.15:
            media_timing = "📈 PRE-POSITION SPEND"
            media_reason = "Capture demand wave, don't chase"
        elif seasonal_trough_coming:
            media_timing = "⏸️ CONSERVE BUDGET"
            media_reason = "Seasonal trough approaching"
        elif forecast_change < -0.10:
            media_timing = "⏸️ REDUCE & WAIT"
            media_reason = "Save spend for better ROI period"
        else:
            media_timing = "⚖️ STEADY PACE"
            media_reason = "Maintain current allocation"
        
        # 3. PRICING OPTIMIZATION (stored for use with inventory context)
        if forecast_change > 0.20:
            pricing_action = "📈 RAISE PRICE (+10%)"
            pricing_reason = "High demand supports premium"
        elif forecast_change < -0.15:
            pricing_action = "🎫 PROMO TO CLEAR"
            pricing_reason = "Convert inventory before decay"
        elif accelerating_growth:
            pricing_action = "💰 HOLD PREMIUM"
            pricing_reason = "Momentum supports price"
        else:
            pricing_action = "✅ MAINTAIN PRICE"
            pricing_reason = "No forecast-driven change"
        
        # 4. PORTFOLIO STRATEGY
        if trending_to_zero:
            portfolio_action = "💀 ACCELERATE EXIT"
            portfolio_reason = "Pod trending to zero"
        elif accelerating_growth and forecast_change > 0.15:
            portfolio_action = "🚀 DOUBLE DOWN"
            portfolio_reason = "Validate expansion investment"
        elif seasonal_peak_coming:
            portfolio_action = "📅 PLAN SEASONAL ALLOCATION"
            portfolio_reason = "Pre-position for peak"
        elif seasonal_trough_coming and forecast_change < -0.10:
            portfolio_action = "⏳ HIBERNATE POSITION"
            portfolio_reason = "Ride out seasonal low"
        else:
            portfolio_action = "📊 MONITOR"
            portfolio_reason = "Standard tracking"
        
        forecast[asin] = {
            'current_weekly_avg': round(recent_avg, 2),
            'projected_weekly_avg': round(projected_weekly, 2),
            'forecast_change_pct': round(forecast_change, 3),
            'growth_rate': round(growth_rate, 3),
            'wow_trend': round(wow_trend, 3),
            'seasonal_factor': round(seasonal_factor, 2),
            'signal': signal,
            'action_modifier': action_modifier,
            'projected_8w_total': round(projected_8w_total, 2),
            # === 36M INTELLIGENCE ===
            'confidence': confidence_label,
            'years_analyzed': len(lookahead_samples) if 'lookahead_samples' in locals() else 1,
            'historical_pattern': f"Based on {len(lookahead_samples)} year(s) of data" if 'lookahead_samples' in locals() else "Limited history",
            # Actionable Intelligence
            'inventory_action': inventory_action,
            'inventory_reason': inventory_reason,
            'media_timing': media_timing,
            'media_reason': media_reason,
            'pricing_action': pricing_action,
            'pricing_reason': pricing_reason,
            'portfolio_action': portfolio_action,
            'portfolio_reason': portfolio_reason,
            # Flags
            'seasonal_peak_coming': seasonal_peak_coming,
            'seasonal_trough_coming': seasonal_trough_coming,
            'trending_to_zero': trending_to_zero,
            'accelerating_growth': accelerating_growth
        }
    
    return forecast

def calculate_net_realization(row, use_synthetic=True):
    """
    Calibrated for 2026 Amazon Fee structures (Referral 15% + FBA Hikes).
    
    Now uses AI Synthetic Intelligence for COGS, Volume, and Logistics
    instead of hardcoded defaults.
    """
    price = _safe_float(row.get('filled_price'))
    if price <= 0: return 0.0
    
    # Updated Fee Logic
    fba_fee = _safe_float(row.get('fba_fees'), default=4.50) 
    referral_fee = price * 0.15 
    
    if use_synthetic:
        # AI Synthetic: Use intelligent estimates
        storage_vol = _safe_float(row.get('synthetic_vol_cf'), default=_safe_float(row.get('package_vol_cf'), default=0.15))
        cogs = _safe_float(row.get('synthetic_cogs'), default=price * 0.25)
        logistics = _safe_float(row.get('landed_logistics'), default=0.0)
    else:
        # Fallback: Original hardcoded defaults
        storage_vol = _safe_float(row.get('package_vol_cf'), default=0.05)
        cogs = price * 0.25
        logistics = 0.0
    
    storage_cost = storage_vol * 0.87  # FBA storage rate per CF
    
    net_profit = price - (fba_fee + referral_fee + storage_cost + cogs + logistics)
    return net_profit / price

def calculate_row_efficiency(bb, gap, weeks_cover, net_margin, velocity_factor):
    """Portfolio efficiency scoring (0-100) with Predictive Multi-Year Weighting."""
    base_score = bb * 40
    price_score = max(0, 30 - (gap * 300))
    margin_score = min(30, max(0, (net_margin - 0.10) * 100))
    
    # Momentum Penalty: If velocity is decaying (factor > 1.1), penalize efficiency
    momentum_penalty = 1 / velocity_factor if velocity_factor > 1.1 else 1.0
    stock_factor = 1.0 if weeks_cover >= 2.0 else 0.2
    
    return min(100, int((base_score + price_score + margin_score) * stock_factor * momentum_penalty))

def analyze_strategic_matrix(row):
    """Assigns products to Strategic Zones using 36-month Predictive Signals."""
    bb = _safe_float(row.get('amazon_bb_share'), default=1.0)
    price = _safe_float(row.get('filled_price'), default=0.0)
    comp = _safe_float(row.get('new_fba_price'), default=price) 
    gap = (price - comp) / comp if comp > 0 else 0
    weeks_cover = _safe_float(row.get('weeks_of_cover'), default=4.0)
    velocity_decay = _safe_float(row.get('velocity_decay'), default=1.0)
    
    net_margin = calculate_net_realization(row)
    efficiency_score = calculate_row_efficiency(bb, gap, weeks_cover, net_margin, velocity_decay)

    # 1. PROBLEM-BASED SEGMENTATION (Clear, actionable categories)
    # Primary problem determines category
    if net_margin < 0.05:
        problem_category = "🔥 Losing Money"
        problem_reason = "Negative margin"
    elif velocity_decay > 1.5:
        problem_category = "📉 Losing Share" 
        problem_reason = "Velocity decay >1.5x"
    elif gap > 0.08 or bb < 0.70:
        problem_category = "💰 Price Problem"
        problem_reason = "Price gap or low Buy Box"
    elif velocity_decay < 0.9 and net_margin > 0.15 and bb > 0.85:
        problem_category = "🚀 Scale Winner"
        problem_reason = "Growing + high margin + strong BB"
    elif net_margin > 0.12 and bb > 0.75:
        problem_category = "✅ Healthy"
        problem_reason = "On track"
    else:
        problem_category = "📊 Monitor"
        problem_reason = "Watch closely"
    
    # Keep legacy capital_zone for backwards compatibility
    if "Losing Money" in problem_category:
        capital_zone = "🩸 BLEED (Negative Margin)"
    elif "Losing Share" in problem_category:
        capital_zone = "📉 DRAG (Terminal Decay)"
    elif "Scale Winner" in problem_category:
        capital_zone = "🏰 FORTRESS (Cash Flow)"
    elif "Healthy" in problem_category:
        capital_zone = "🚀 FRONTIER (Growth)"
    else:
        capital_zone = "📉 DRAG (Waste)"

    # 2. ACTIONABLE DIRECTIVES (tied to problem)
    if "Losing Money" in problem_category:
        ecom_action = "💀 EXIT — Stop bleeding"
        ad_action = "🛑 PAUSE ADS"
    elif "Losing Share" in problem_category:
        ecom_action = "🚨 FIX VISIBILITY"
        ad_action = "📢 INCREASE BIDS"
    elif "Price Problem" in problem_category:
        ecom_action = "🎫 CLIP COUPON / REPRICE"
        ad_action = "⚖️ HOLD SPEND"
    elif "Scale Winner" in problem_category:
        ecom_action = "📈 RAISE PRICE (+5%)"
        ad_action = "🚀 SCALE +25%"
    else:
        ecom_action = "✅ MAINTAIN"
        ad_action = "⚖️ OPTIMIZE ROAS"

    return pd.Series([ad_action, ecom_action, capital_zone, gap, efficiency_score, net_margin, problem_category, problem_reason])

def run_weekly_analysis(all_rows, selected_week):
    """
    Executes the analysis, benchmarks Category Growth, and fixes taxonomy.

    Performance optimizations:
    - Reduced redundant sorting operations
    - Vectorized operations where possible
    - Optimized groupby aggregations
    """
    # Use timezone-aware datetime for consistent comparisons
    target_dt = pd.to_datetime(selected_week)
    if target_dt.tzinfo is None:
        target_dt = target_dt.tz_localize('UTC')

    target_date = target_dt.date()
    ly_dt = target_dt - pd.Timedelta(days=364)  # Seasonal Benchmark
    ly_date = ly_dt.date()

    # 1. PREDICTIVE ENGINE (Optimized filtering)
    if all_rows["week_start"].dt.tz is not None:
        history = all_rows[all_rows["week_start"] <= target_dt].copy()
    else:
        history = all_rows[all_rows["week_start"].dt.date <= target_date].copy()

    # Ensure numeric type (vectorized)
    history['sales_rank_filled'] = pd.to_numeric(history['sales_rank_filled'], errors='coerce').fillna(0)

    # Optimized aggregations - single groupby operation
    lt_avg = history.groupby('asin')['sales_rank_filled'].mean()

    # Sort once and reuse
    history_sorted = history.sort_values(['asin', 'week_start'], ascending=[True, False])
    rt_avg = history_sorted.groupby('asin').head(8).groupby('asin')['sales_rank_filled'].mean()

    # Trend arrays (optimized lambda)
    trend_arrays = history.sort_values('week_start').groupby('asin')['sales_rank_filled'].apply(
        lambda x: x.tolist()  # More efficient than list comprehension
    )
    
    velocity_intel = pd.DataFrame({
        'velocity_decay': (rt_avg / lt_avg).fillna(1.0).round(2),
        'Trend (36M)': trend_arrays
    }).reset_index()
    
    # 1b. DEMAND FORECASTING ENGINE
    # Calculate 8-week forward projections based on velocity trends and seasonality
    demand_forecast = calculate_demand_forecast(history, target_dt)

    # 2. SNAPSHOTS
    if all_rows["week_start"].dt.tz is not None:
        df_snapshot = all_rows[all_rows["week_start"].dt.date == target_date].copy()
        ly_snapshot = all_rows[all_rows["week_start"].dt.date == ly_date].copy()
    else:
        df_snapshot = all_rows[all_rows["week_start"].dt.date == target_date].copy()
        ly_snapshot = all_rows[all_rows["week_start"].dt.date == ly_date].copy()
    
    sbux = df_snapshot[df_snapshot["is_starbucks"] == 1].copy()
    total_rev_ly = ly_snapshot[ly_snapshot["is_starbucks"] == 1]['weekly_sales_filled'].sum()

    if sbux.empty: 
        return {"data": pd.DataFrame(), "capital_flow": {}, "total_rev": 0, "total_rev_ly": 0, "share_delta": 0, "yoy_delta": 0}

    # 3. AI SYNTHETIC INTELLIGENCE: Keepa Gap Interpolation
    # Fill missing BSR, Buy Box prices, and Competitor MAP using historical patterns
    sbux = interpolate_keepa_gaps(sbux, history)

    # 4. COMPETITIVE INTELLIGENCE (CI)
    category_growth_rate = 0.06 # Fixed benchmark for 2026 Starbucks Portfolio
    total_rev_curr = sbux['weekly_sales_filled'].sum()
    yoy_delta = (total_rev_curr - total_rev_ly) / total_rev_ly if total_rev_ly > 0 else 0
    share_delta = yoy_delta - category_growth_rate

    # 5. TAXONOMY FIX
    sbux = sbux.merge(velocity_intel, on='asin', how='left')
    
    # 5b. ADD FORECAST DATA TO EACH ASIN (Optimized mapping - create dict once per ASIN)
    def get_forecast_data(asin):
        """Extract forecast data for ASIN with defaults."""
        f = demand_forecast.get(asin, {})
        return {
            'forecast_change': f.get('forecast_change_pct', 0),
            'forecast_signal': f.get('signal', '→ STABLE'),
            'forecast_action': f.get('action_modifier', 'maintain'),
            'inventory_action': f.get('inventory_action', '✅ MAINTAIN LEVELS'),
            'inventory_reason': f.get('inventory_reason', 'Stable demand'),
            'media_timing': f.get('media_timing', '⚖️ STEADY PACE'),
            'media_reason': f.get('media_reason', 'Standard allocation'),
            'pricing_action': f.get('pricing_action', '✅ MAINTAIN PRICE'),
            'pricing_reason': f.get('pricing_reason', 'No change'),
            'portfolio_action': f.get('portfolio_action', '📊 MONITOR'),
            'portfolio_reason': f.get('portfolio_reason', 'Standard tracking'),
            'seasonal_peak': f.get('seasonal_peak_coming', False),
            'seasonal_trough': f.get('seasonal_trough_coming', False),
            'trending_to_zero': f.get('trending_to_zero', False),
            'accelerating_growth': f.get('accelerating_growth', False)
        }

    # Apply all forecast columns at once
    forecast_data = pd.DataFrame([get_forecast_data(asin) for asin in sbux['asin']], index=sbux.index)
    sbux = pd.concat([sbux, forecast_data], axis=1)
    
    # Parse variation_attributes with intelligent flavor/count detection
    # Pre-compile regex pattern outside function for performance
    import re
    count_pattern = re.compile(r'(\d+)\s*(ct|count|pods?)\b', re.IGNORECASE)

    # Known flavor keywords (prioritized)
    KNOWN_FLAVORS = ['Pike Place', 'Breakfast Blend', 'French Roast', 'Sumatra',
                     'Caramel', 'Caffe Verona', 'Veranda', 'House Blend', 'Espresso']

    def parse_variation(row):
        attr = str(row.get('variation_attributes', '')).strip()
        title = str(row.get('title', '')).strip()

        flavor = "Standard"
        count = "Standard"

        if '|' in attr:
            parts = attr.split('|', 1)  # Split only once for efficiency
            part0, part1 = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""

            # Check for count pattern in parts
            count_match_0 = count_pattern.search(part0)
            count_match_1 = count_pattern.search(part1)

            if count_match_0:
                count = f"{count_match_0.group(1)} Count"
                flavor = part1 if part1 and not count_match_1 else "Standard"
            elif count_match_1:
                count = f"{count_match_1.group(1)} Count"
                flavor = part0 if part0 != "Standard" else "Standard"
            else:
                flavor = part0
                count = part1 if part1 else "Standard"
        else:
            # No separator - check if it's a count or flavor
            count_match = count_pattern.search(attr)
            if count_match:
                count = f"{count_match.group(1)} Count"
            elif attr and attr != "Standard":
                flavor = attr

        # If flavor is still "Standard", extract from title (optimized search)
        if flavor == "Standard":
            title_lower = title.lower()
            for f in KNOWN_FLAVORS:
                if f.lower() in title_lower:
                    flavor = f
                    break

        return pd.Series([flavor, count])

    sbux[['Flavor', 'Count']] = sbux.apply(parse_variation, axis=1)
    
    # 6. AI SYNTHETIC INTELLIGENCE: Financial Enrichment
    # Add synthetic COGS, Volumetric CF, and Landed Logistics
    sbux = enrich_synthetic_financials(sbux)
    
    sbux[[
        'ad_action', 'ecom_action', 'capital_zone', 'price_gap', 
        'efficiency_score', 'net_margin', 'problem_category', 'problem_reason'
    ]] = sbux.apply(analyze_strategic_matrix, axis=1)
    
    capital_flow = sbux.groupby("capital_zone")['weekly_sales_filled'].sum().to_dict()
    
    # 7. COMPETITIVE INTELLIGENCE CONTEXT + FORECAST
    # Portfolio-level forecast summary
    portfolio_forecast_change = sbux['forecast_change'].mean() if 'forecast_change' in sbux.columns else 0
    surge_count = len(sbux[sbux['forecast_signal'].str.contains('SURGE', na=False)])
    decline_count = len(sbux[sbux['forecast_signal'].str.contains('DECLINE', na=False)])
    
    ci_context = {
        'share_delta': share_delta,        # vs 6% category benchmark
        'yoy_delta': yoy_delta,             # absolute YoY change
        'category_growth': category_growth_rate,
        'is_losing_share': share_delta < 0,
        'is_growing': yoy_delta > 0,
        'share_severity': 'critical' if share_delta < -0.10 else 'warning' if share_delta < 0 else 'healthy',
        # Forecast signals
        'portfolio_forecast_change': portfolio_forecast_change,
        'forecast_trending_up': portfolio_forecast_change > 0.05,
        'forecast_trending_down': portfolio_forecast_change < -0.05,
        'surge_sku_count': surge_count,
        'decline_sku_count': decline_count
    }
    
    # 8. HIERARCHICAL POD AGGREGATION with CI + Forecast + 36M History
    # Create L1 (Flavor), L2 (Count), L3 (ASIN) hierarchy
    hierarchy = build_pod_hierarchy(sbux, history, ci_context)
    
    return {
        "data": sbux, 
        "hierarchy": hierarchy,
        "capital_flow": capital_flow, 
        "total_rev": total_rev_curr,
        "total_rev_ly": total_rev_ly,
        "demand_forecast": demand_forecast,
        "share_delta": share_delta,
        "yoy_delta": yoy_delta,
        "ci_context": ci_context
    }


def build_pod_hierarchy(df: pd.DataFrame, history_df: pd.DataFrame = None, ci_context: dict = None) -> dict:
    """
    Build hierarchical Pod structure for Director-level analysis.
    
    Hierarchy:
    - L1: Flavor (Brand Health) - Portfolio Strategy
    - L2: Count (Logistics/P&L) - Ops Strategy  
    - L3: ASIN (Individual Listing) - Media Directive
    
    Metrics are rolled up from ASIN → Count → Flavor.
    CI Context and 36-month Historical Intelligence are incorporated into directives.
    """
    hierarchy = {}
    ci_context = ci_context or {}
    history_df = history_df if history_df is not None else pd.DataFrame()
    
    for flavor in df['Flavor'].unique():
        flavor_df = df[df['Flavor'] == flavor]
        
        # CI Metrics at Flavor Level
        avg_bb_share = flavor_df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in flavor_df else 1.0
        avg_price_gap = flavor_df['price_gap'].fillna(0).mean() if 'price_gap' in flavor_df else 0
        
        # Forecast Metrics at Flavor Level
        avg_forecast_change = flavor_df['forecast_change'].mean() if 'forecast_change' in flavor_df.columns else 0
        surge_count = len(flavor_df[flavor_df['forecast_signal'].str.contains('SURGE', na=False)]) if 'forecast_signal' in flavor_df.columns else 0
        decline_count = len(flavor_df[flavor_df['forecast_signal'].str.contains('DECLINE', na=False)]) if 'forecast_signal' in flavor_df.columns else 0
        forecast_signal = _get_aggregate_forecast_signal(avg_forecast_change, surge_count, decline_count, len(flavor_df))
        
        # Aggregate Actionable Intelligence at Flavor Level
        flavor_actionable = _aggregate_actionable_intel(flavor_df)
        
        # L1: Flavor-level aggregation with CI + Forecast + Actionable Intel
        flavor_metrics = {
            'level': 'L1_Flavor',
            'name': flavor,
            'total_revenue': flavor_df['weekly_sales_filled'].sum(),
            'avg_margin': flavor_df['net_margin'].mean(),
            'avg_efficiency': flavor_df['efficiency_score'].mean(),
            'avg_velocity_decay': flavor_df['velocity_decay'].mean(),
            'sku_count': len(flavor_df),
            'synthetic_cogs_total': flavor_df['synthetic_cogs'].sum() if 'synthetic_cogs' in flavor_df else 0,
            'dominant_zone': flavor_df.groupby('capital_zone')['weekly_sales_filled'].sum().idxmax(),
            'health_status': _calculate_pod_health(flavor_df),
            # CI Metrics
            'avg_bb_share': avg_bb_share,
            'avg_price_gap': avg_price_gap,
            'ci_share_delta': ci_context.get('share_delta', 0),
            'ci_alert': _get_ci_alert(flavor_df, ci_context),
            # Forecast Metrics
            'forecast_change': avg_forecast_change,
            'forecast_signal': forecast_signal,
            'strategic_action': _get_flavor_directive(flavor_df, history_df, ci_context),
            # Actionable Intelligence (aggregated)
            **flavor_actionable,
            'counts': {}
        }
        
        # L2: Count-level aggregation within each Flavor
        for count in flavor_df['Count'].unique():
            count_df = flavor_df[flavor_df['Count'] == count]
            
            # CI Metrics at Count Level
            count_avg_bb = count_df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in count_df else 1.0
            count_avg_gap = count_df['price_gap'].fillna(0).mean() if 'price_gap' in count_df else 0
            
            # Forecast Metrics at Count Level
            count_forecast_change = count_df['forecast_change'].mean() if 'forecast_change' in count_df.columns else 0
            count_surge = len(count_df[count_df['forecast_signal'].str.contains('SURGE', na=False)]) if 'forecast_signal' in count_df.columns else 0
            count_decline = len(count_df[count_df['forecast_signal'].str.contains('DECLINE', na=False)]) if 'forecast_signal' in count_df.columns else 0
            count_forecast_signal = _get_aggregate_forecast_signal(count_forecast_change, count_surge, count_decline, len(count_df))
            
            # Aggregate Actionable Intelligence at Count Level
            count_actionable = _aggregate_actionable_intel(count_df)
            
            count_metrics = {
                'level': 'L2_Count',
                'name': count,
                'pod_id': f"{flavor}|{count}",
                'total_revenue': count_df['weekly_sales_filled'].sum(),
                'avg_margin': count_df['net_margin'].mean(),
                'avg_efficiency': count_df['efficiency_score'].mean(),
                'avg_velocity_decay': count_df['velocity_decay'].mean(),
                'sku_count': len(count_df),
                'synthetic_cogs_avg': count_df['synthetic_cogs'].mean() if 'synthetic_cogs' in count_df else 0,
                'synthetic_vol_cf_avg': count_df['synthetic_vol_cf'].mean() if 'synthetic_vol_cf' in count_df else 0,
                'dominant_zone': count_df.groupby('capital_zone')['weekly_sales_filled'].sum().idxmax() if not count_df.empty else 'Unknown',
                'health_status': _calculate_pod_health(count_df),
                # CI Metrics
                'avg_bb_share': count_avg_bb,
                'avg_price_gap': count_avg_gap,
                'ci_alert': _get_ci_alert(count_df, ci_context),
                # Forecast Metrics
                'forecast_change': count_forecast_change,
                'forecast_signal': count_forecast_signal,
                # Directives (now informed by forecast + 36M history)
                'ops_action': _get_count_directive(count_df, history_df, ci_context),
                'media_action': _get_media_directive(count_df, history_df, ci_context),
                # Actionable Intelligence (aggregated)
                **count_actionable,
                'asins': []
            }
            
            # L3: ASIN-level detail with CI
            for _, row in count_df.iterrows():
                asin_detail = {
                    'level': 'L3_ASIN',
                    'asin': row['asin'],
                    'title': row.get('title', '')[:50],
                    'revenue': row['weekly_sales_filled'],
                    'margin': row['net_margin'],
                    'efficiency': row['efficiency_score'],
                    'velocity_decay': row['velocity_decay'],
                    'capital_zone': row['capital_zone'],
                    'ad_action': row['ad_action'],
                    'ecom_action': row['ecom_action'],
                    'synthetic_cogs': row.get('synthetic_cogs', 0),
                    'synthetic_vol_cf': row.get('synthetic_vol_cf', 0),
                    'trend': row.get('Trend (36M)', []),
                    # CI Metrics
                    'bb_share': row.get('amazon_bb_share', 1.0),
                    'price_gap': row.get('price_gap', 0),
                    'comp_price': row.get('new_fba_price', 0)
                }
                count_metrics['asins'].append(asin_detail)
            
            flavor_metrics['counts'][count] = count_metrics
        
        hierarchy[flavor] = flavor_metrics
    
    return hierarchy


def _get_aggregate_forecast_signal(avg_change: float, surge_count: int, decline_count: int, total_count: int) -> str:
    """Get aggregate forecast signal for a Pod."""
    if total_count == 0:
        return "→ NO DATA"
    
    surge_pct = surge_count / total_count
    decline_pct = decline_count / total_count
    
    if avg_change > 0.15 or surge_pct > 0.5:
        return "📈 SURGE AHEAD"
    elif avg_change > 0.05:
        return "↗️ GROWING"
    elif avg_change < -0.15 or decline_pct > 0.5:
        return "📉 DECLINE AHEAD"
    elif avg_change < -0.05:
        return "↘️ SOFTENING"
    else:
        return "→ STABLE"


def _aggregate_actionable_intel(df: pd.DataFrame) -> dict:
    """
    Aggregate actionable intelligence for a Pod (Flavor or Count level).
    
    Uses revenue-weighted voting to determine dominant action for each category.
    """
    if df.empty:
        return {
            'inventory_intel': {'action': '✅ MAINTAIN LEVELS', 'reason': 'No data', 'urgency': 'low'},
            'media_intel': {'action': '⚖️ STEADY PACE', 'reason': 'No data', 'timing': 'now'},
            'pricing_intel': {'action': '✅ MAINTAIN PRICE', 'reason': 'No data', 'confidence': 'low'},
            'portfolio_intel': {'action': '📊 MONITOR', 'reason': 'No data', 'horizon': 'long'},
            'seasonal_alert': None,
            'critical_flags': []
        }
    
    total_rev = df['weekly_sales_filled'].sum()
    if total_rev == 0:
        total_rev = 1  # Avoid division by zero
    
    # Critical flags detection
    critical_flags = []
    
    # Check for trending to zero SKUs
    if 'trending_to_zero' in df.columns:
        zero_trending_rev = df[df['trending_to_zero'] == True]['weekly_sales_filled'].sum()
        if zero_trending_rev / total_rev > 0.20:
            critical_flags.append("💀 20%+ revenue trending to zero")
    
    # Check for accelerating growth
    if 'accelerating_growth' in df.columns:
        accel_rev = df[df['accelerating_growth'] == True]['weekly_sales_filled'].sum()
        if accel_rev / total_rev > 0.30:
            critical_flags.append("🚀 30%+ revenue in acceleration")
    
    # Seasonal alerts
    seasonal_alert = None
    if 'seasonal_peak' in df.columns:
        peak_rev = df[df['seasonal_peak'] == True]['weekly_sales_filled'].sum()
        if peak_rev / total_rev > 0.25:
            seasonal_alert = "📅 SEASONAL PEAK INCOMING"
    
    if 'seasonal_trough' in df.columns and seasonal_alert is None:
        trough_rev = df[df['seasonal_trough'] == True]['weekly_sales_filled'].sum()
        if trough_rev / total_rev > 0.25:
            seasonal_alert = "⏸️ SEASONAL TROUGH APPROACHING"
    
    # Revenue-weighted action voting
    def get_dominant_action(action_col, reason_col):
        """Get the most impactful action by revenue weight."""
        if action_col not in df.columns:
            return None, None
        
        action_weights = df.groupby(action_col)['weekly_sales_filled'].sum()
        if action_weights.empty:
            return None, None
        
        dominant = action_weights.idxmax()
        dominant_df = df[df[action_col] == dominant]
        reason = dominant_df[reason_col].iloc[0] if reason_col in dominant_df.columns and len(dominant_df) > 0 else ""
        return dominant, reason
    
    # Aggregate each intelligence category
    inv_action, inv_reason = get_dominant_action('inventory_action', 'inventory_reason')
    media_action, media_reason = get_dominant_action('media_timing', 'media_reason')
    price_action, price_reason = get_dominant_action('pricing_action', 'pricing_reason')
    port_action, port_reason = get_dominant_action('portfolio_action', 'portfolio_reason')
    
    # Determine urgency levels
    inv_urgency = 'critical' if inv_action and ('REORDER' in inv_action or 'STOP' in inv_action) else 'normal'
    media_timing = 'pre-position' if media_action and ('EARLY' in media_action or 'PRE-POSITION' in media_action) else 'now'
    price_confidence = 'high' if price_action and ('RAISE' in price_action or 'PROMO' in price_action) else 'moderate'
    port_horizon = 'immediate' if port_action and ('EXIT' in port_action or 'DOUBLE' in port_action) else 'strategic'
    
    return {
        'inventory_intel': {
            'action': inv_action or '✅ MAINTAIN LEVELS',
            'reason': inv_reason or 'Stable demand',
            'urgency': inv_urgency
        },
        'media_intel': {
            'action': media_action or '⚖️ STEADY PACE',
            'reason': media_reason or 'Standard allocation',
            'timing': media_timing
        },
        'pricing_intel': {
            'action': price_action or '✅ MAINTAIN PRICE',
            'reason': price_reason or 'No change needed',
            'confidence': price_confidence
        },
        'portfolio_intel': {
            'action': port_action or '📊 MONITOR',
            'reason': port_reason or 'Standard tracking',
            'horizon': port_horizon
        },
        'seasonal_alert': seasonal_alert,
        'critical_flags': critical_flags
    }


def _get_ci_alert(df: pd.DataFrame, ci_context: dict) -> str:
    """
    Generate Competitive Intelligence alert based on market signals.
    
    Analyzes: Buy Box share, Price gaps, Share velocity, Competitor pricing
    """
    if df.empty:
        return ""
    
    # Extract CI signals
    avg_bb = df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in df.columns else 1.0
    avg_gap = df['price_gap'].fillna(0).mean() if 'price_gap' in df.columns else 0
    share_delta = ci_context.get('share_delta', 0)
    is_losing_share = ci_context.get('is_losing_share', False)
    
    alerts = []
    
    # Buy Box Loss Alert
    if avg_bb < 0.70:
        alerts.append("🔴 BB CRISIS (<70%)")
    elif avg_bb < 0.85:
        alerts.append("🟡 BB EROSION")
    
    # Price War Detection
    if avg_gap > 0.12:
        alerts.append("⚔️ PRICE WAR (+12% gap)")
    elif avg_gap > 0.08:
        alerts.append("💰 PREMIUM RISK")
    elif avg_gap < -0.05:
        alerts.append("📉 UNDERPRICED")
    
    # Share Velocity Alert
    if share_delta < -0.15:
        alerts.append("🚨 SHARE COLLAPSE")
    elif share_delta < -0.05:
        alerts.append("📉 LOSING SHARE")
    elif share_delta > 0.05:
        alerts.append("📈 GAINING SHARE")
    
    return " | ".join(alerts) if alerts else "✅ CI STABLE"


def _calculate_pod_health(df: pd.DataFrame) -> str:
    """Calculate health status for a pod (Flavor or Count level)."""
    if df.empty:
        return "⚪ NO DATA"
    
    # Weight by revenue
    total_rev = df['weekly_sales_filled'].sum()
    if total_rev == 0:
        return "⚪ NO DATA"
    
    # Check zone distribution
    bleed_pct = df[df['capital_zone'].str.contains('BLEED', na=False)]['weekly_sales_filled'].sum() / total_rev
    drag_pct = df[df['capital_zone'].str.contains('DRAG', na=False)]['weekly_sales_filled'].sum() / total_rev
    fortress_pct = df[df['capital_zone'].str.contains('FORTRESS', na=False)]['weekly_sales_filled'].sum() / total_rev
    
    avg_decay = df['velocity_decay'].mean()
    
    if bleed_pct > 0.3 or avg_decay > 1.5:
        return "🔴 CRITICAL"
    elif drag_pct > 0.4 or avg_decay > 1.2:
        return "🟡 AT RISK"
    elif fortress_pct > 0.5 and avg_decay < 1.0:
        return "🟢 HEALTHY"
    else:
        return "🟡 MONITOR"


def _get_flavor_directive(df: pd.DataFrame, history_df: pd.DataFrame = None, ci_context: dict = None) -> str:
    """
    Get strategic directive for Flavor level (L1).
    Incorporates Competitive Intelligence + Demand Forecast + 36M Historical Context for market-aware decisions.
    """
    if df.empty:
        return "📊 ANALYZE"
    
    ci_context = ci_context or {}
    history_df = history_df if history_df is not None else pd.DataFrame()
    
    # Handle NaN values with sensible defaults
    avg_decay = df['velocity_decay'].fillna(1.0).mean()
    avg_margin = df['net_margin'].fillna(0).mean()
    total_rev = df['weekly_sales_filled'].sum()
    
    # === 36M HISTORICAL CONTEXT ===
    # Get representative ASIN for historical analysis
    historical_context_available = False
    revenue_is_anomaly = False
    historically_strong_revenue = False
    revenue_percentile = 50
    revenue_trend = 0
    
    if not history_df.empty and 'asin' in df.columns:
        # Use highest revenue ASIN as representative
        rep_asin = df.nlargest(1, 'weekly_sales_filled')['asin'].iloc[0] if len(df) > 0 else None
        
        if rep_asin:
            # Analyze revenue history (this exists in raw historical data)
            rev_ctx = get_historical_context(history_df, rep_asin, total_rev, 'weekly_sales_filled')
            if not rev_ctx.get('insufficient_data', False):
                historical_context_available = True
                revenue_percentile = rev_ctx['percentile']
                revenue_is_anomaly = rev_ctx['is_anomaly']
                revenue_trend = rev_ctx['trend_direction']
                # Consider historically strong if revenue is typically high
                historically_strong_revenue = rev_ctx['mean'] > rev_ctx['median'] * 1.2
    
    # CI Metrics
    avg_bb = df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in df.columns else 1.0
    avg_gap = df['price_gap'].fillna(0).mean() if 'price_gap' in df.columns else 0
    share_delta = ci_context.get('share_delta', 0)
    is_losing_share = ci_context.get('is_losing_share', False)
    
    # FORECAST METRICS
    avg_forecast_change = df['forecast_change'].fillna(0).mean() if 'forecast_change' in df.columns else 0
    forecast_surging = avg_forecast_change > 0.10
    forecast_declining = avg_forecast_change < -0.10
    forecast_strong = avg_forecast_change > 0.05
    forecast_weak = avg_forecast_change < -0.05
    
    # Zone distribution analysis
    bleed_rev = df[df['capital_zone'].str.contains('BLEED', na=False)]['weekly_sales_filled'].sum()
    drag_rev = df[df['capital_zone'].str.contains('DRAG', na=False)]['weekly_sales_filled'].sum()
    fortress_rev = df[df['capital_zone'].str.contains('FORTRESS', na=False)]['weekly_sales_filled'].sum()
    frontier_rev = df[df['capital_zone'].str.contains('FRONTIER', na=False)]['weekly_sales_filled'].sum()
    
    bleed_pct = bleed_rev / total_rev if total_rev > 0 else 0
    drag_pct = drag_rev / total_rev if total_rev > 0 else 0
    healthy_pct = (fortress_rev + frontier_rev) / total_rev if total_rev > 0 else 0
    
    # === 36M + FORECAST-ENHANCED DECISION TREE ===
    
    # CRITICAL: Decline forecast compounds exit signals
    # BUT: Historical context can override if revenue is anomalously low
    if bleed_pct > 0.25 and forecast_declining:
        if historical_context_available and revenue_percentile > 60 and revenue_is_anomaly:
            return "⚠️ INVESTIGATE DROP (historically top 40%, unusual dip)"
        return "💀 ACCELERATE EXIT (Forecast: Decline)"
    
    if bleed_pct > 0.30 or avg_decay > 1.5:
        if historical_context_available and historically_strong_revenue and revenue_trend > 0:
            return "⏳ HOLD & MONITOR (strong historical trend, temporary dip)"
        return "💀 EXIT FLAVOR LINE"
    
    # CI PRIORITY: Buy Box Crisis overrides other signals
    if avg_bb < 0.60:
        return "🚨 RECOVER BUY BOX"
    
    # CI PRIORITY: Severe share loss with poor margins
    if share_delta < -0.15 and avg_margin < 0.12 and not forecast_strong:
        return "💀 EXIT FLAVOR LINE"
    
    # FORECAST OPPORTUNITY: Surge forecast with healthy metrics
    if forecast_surging and healthy_pct > 0.50 and avg_margin > 0.12:
        return "🚀 PRE-SCALE (Forecast: Surge)"
    
    # CI: Active price war detection
    if avg_gap > 0.12:
        if forecast_declining:
            return "⚔️ DISENGAGE PRICE WAR"
        else:
            return "⚔️ PRICE WAR DEFENSE"
    
    # CI: Losing share but healthy fundamentals - defend
    if is_losing_share and healthy_pct > 0.50:
        if forecast_strong:
            return "💪 DEFEND & INVEST (Forecast: Growing)"
        elif avg_gap > 0.05:
            return "🎫 MATCH PRICE (DEFEND)"
        else:
            return "📢 INCREASE VISIBILITY"
    
    # Standard health checks with forecast + historical context
    if drag_pct > 0.40 or avg_decay > 1.3:
        if forecast_strong:
            return "⏳ HOLD (Forecast: Improving)"
        elif historical_context_available and revenue_percentile > 75:
            return "⏳ HOLD (Top 25% historical performance)"
        else:
            return "⚠️ HOLD & EVALUATE"
    elif avg_decay > 1.1:
        return "📉 DEFEND SHARE"
    
    # CI: Gaining share with strong fundamentals - aggressive growth
    if share_delta > 0.05 and healthy_pct > 0.60 and avg_margin > 0.15:
        if forecast_surging:
            return "🔥 MAX SCALE (Share + Forecast)"
        return "🚀 AGGRESSIVE EXPANSION"
    
    # Standard growth conditions (tightened)
    if healthy_pct > 0.70 and avg_margin > 0.18 and avg_decay < 0.85 and avg_bb > 0.90:
        return "🚀 EXPAND FLAVOR LINE"
    elif healthy_pct > 0.50 and avg_margin > 0.12:
        return "✅ MAINTAIN ALLOCATION"
    elif avg_margin < 0.10:
        return "📈 TEST PRICE INCREASE"
    else:
        return "📊 STRATEGIC REVIEW"


def _get_count_directive(df: pd.DataFrame, history_df: pd.DataFrame = None, ci_context: dict = None) -> str:
    """
    Get ops directive for Count level (L2).
    Incorporates Competitive Intelligence + Demand Forecast + 36M Historical Context.
    """
    if df.empty:
        return "📊 ANALYZE"
    
    ci_context = ci_context or {}
    history_df = history_df if history_df is not None else pd.DataFrame()
    
    # Handle NaN values
    avg_margin = df['net_margin'].fillna(0).mean()
    avg_decay = df['velocity_decay'].fillna(1.0).mean()
    avg_cogs = df['synthetic_cogs'].mean() if 'synthetic_cogs' in df.columns else 0
    avg_price = df['filled_price'].mean() if 'filled_price' in df.columns else 0
    total_rev = df['weekly_sales_filled'].sum()
    
    # === 36M HISTORICAL CONTEXT ===
    historical_context_available = False
    revenue_percentile = 50
    historically_strong = False
    
    if not history_df.empty and 'asin' in df.columns:
        rep_asin = df.nlargest(1, 'weekly_sales_filled')['asin'].iloc[0] if len(df) > 0 else None
        if rep_asin:
            rev_ctx = get_historical_context(history_df, rep_asin, total_rev, 'weekly_sales_filled')
            if not rev_ctx.get('insufficient_data', False):
                historical_context_available = True
                revenue_percentile = rev_ctx['percentile']
                historically_strong = rev_ctx['mean'] > rev_ctx['median'] * 1.15
    
    # CI Metrics
    avg_bb = df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in df.columns else 1.0
    avg_gap = df['price_gap'].fillna(0).mean() if 'price_gap' in df.columns else 0
    avg_comp_price = df['new_fba_price'].fillna(0).mean() if 'new_fba_price' in df.columns else 0
    share_delta = ci_context.get('share_delta', 0)
    
    # FORECAST METRICS
    avg_forecast_change = df['forecast_change'].fillna(0).mean() if 'forecast_change' in df.columns else 0
    forecast_surging = avg_forecast_change > 0.10
    forecast_declining = avg_forecast_change < -0.10
    forecast_strong = avg_forecast_change > 0.05
    forecast_weak = avg_forecast_change < -0.05
    
    cogs_ratio = avg_cogs / avg_price if avg_price > 0 else 0.25
    
    # Zone distribution
    bleed_rev = df[df['capital_zone'].str.contains('BLEED', na=False)]['weekly_sales_filled'].sum()
    drag_rev = df[df['capital_zone'].str.contains('DRAG', na=False)]['weekly_sales_filled'].sum()
    bleed_pct = bleed_rev / total_rev if total_rev > 0 else 0
    drag_pct = drag_rev / total_rev if total_rev > 0 else 0
    
    # === 36M + FORECAST-ENHANCED DECISION TREE ===
    
    # CRITICAL: Exit conditions (but check 36M history first)
    if avg_margin < 0.05 and forecast_declining:
        if historical_context_available and revenue_percentile > 70:
            return "⚠️ INVESTIGATE MARGIN (historically top 30%)"
        return "💀 ACCELERATE EXIT (Forecast: Decline)"
    if avg_margin < 0.05 or bleed_pct > 0.40:
        if historical_context_available and historically_strong:
            return "⏳ HOLD & FIX (historically strong performer)"
        return "💀 EXIT COUNT SIZE"
    
    # CI PRIORITY: Buy Box lost - immediate action needed
    if avg_bb < 0.50:
        if avg_gap > 0.10:
            return "🎫 EMERGENCY REPRICE"
        else:
            return "🚨 INVESTIGATE BB LOSS"
    
    # FORECAST: Surge predicted with good Buy Box - pre-stock
    if forecast_surging and avg_bb > 0.80 and avg_margin > 0.10:
        return "📦 PRE-STOCK INVENTORY (Surge)"
    
    # CI: Competitor significantly undercutting
    if avg_gap > 0.15:
        if forecast_strong and avg_margin > 0.15:
            return "🎫 DEFEND PRICE (Forecast Strong)"
        elif avg_margin > 0.15:
            return "🎫 CLIP COUPON (-10%)"
        else:
            return "⚔️ PRICE WAR ALERT"
    
    # CI: We're underpriced - opportunity to raise (especially if demand surging)
    if avg_gap < -0.08 and avg_bb > 0.90:
        if forecast_surging:
            return "📈 AGGRESSIVE PRICE (+12%)"
        return "📈 RAISE PRICE (+8%)"
    
    # CI: Moderate price gap with share loss
    if avg_gap > 0.08 and share_delta < -0.05:
        if forecast_weak:
            return "⚠️ REDUCE INVENTORY"
        return "🎫 MATCH COMPETITOR"
    
    # Supply chain check
    if cogs_ratio > 0.32:
        return "🏭 RENEGOTIATE COGS"
    
    # Standard margin check with forecast context
    if avg_margin < 0.10:
        if forecast_strong:
            return "📈 RAISE PRICE (+8%)"
        return "📈 TEST PRICE (+5%)"
    
    # Velocity issues
    if avg_decay > 1.4:
        return "🎫 PROMOTIONAL CLEAR"
    elif drag_pct > 0.30:
        return "⚠️ OPTIMIZE LISTINGS"
    
    # CI: Strong position with market gains
    if share_delta > 0.03 and avg_margin > 0.18 and avg_bb > 0.90:
        return "🚀 SCALE COUNT SIZE"
    
    # Standard success conditions
    if avg_margin > 0.20 and avg_decay < 0.85 and bleed_pct == 0 and avg_bb > 0.85:
        return "🚀 SCALE COUNT SIZE"
    elif avg_margin > 0.15 and avg_decay < 1.0:
        return "✅ MAINTAIN OPS"
    else:
        return "📊 REVIEW METRICS"


def _get_media_directive(df: pd.DataFrame, history_df: pd.DataFrame = None, ci_context: dict = None) -> str:
    """
    Get media/advertising directive for Pod level.
    Determines ad spend strategy based on margins, velocity, CI signals, demand forecast, and 36M history.
    """
    if df.empty:
        return "⏸️ HOLD SPEND"
    
    ci_context = ci_context or {}
    history_df = history_df if history_df is not None else pd.DataFrame()
    
    # Handle NaN values
    avg_margin = df['net_margin'].fillna(0).mean()
    avg_decay = df['velocity_decay'].fillna(1.0).mean()
    avg_efficiency = df['efficiency_score'].fillna(0).mean()
    
    # CI Metrics
    avg_bb = df['amazon_bb_share'].fillna(1.0).mean() if 'amazon_bb_share' in df.columns else 1.0
    avg_gap = df['price_gap'].fillna(0).mean() if 'price_gap' in df.columns else 0
    share_delta = ci_context.get('share_delta', 0)
    total_rev = df['weekly_sales_filled'].sum()
    
    # FORECAST METRICS
    avg_forecast_change = df['forecast_change'].fillna(0).mean() if 'forecast_change' in df.columns else 0
    forecast_surging = avg_forecast_change > 0.10
    forecast_declining = avg_forecast_change < -0.10
    forecast_strong = avg_forecast_change > 0.05
    
    # === 36M HISTORICAL CONTEXT ===
    historical_context_available = False
    historically_strong = False
    
    if not history_df.empty and 'asin' in df.columns:
        rep_asin = df.nlargest(1, 'weekly_sales_filled')['asin'].iloc[0] if len(df) > 0 else None
        if rep_asin:
            rev_ctx = get_historical_context(history_df, rep_asin, total_rev, 'weekly_sales_filled')
            if not rev_ctx.get('insufficient_data', False):
                historical_context_available = True
                historically_strong = rev_ctx['mean'] > rev_ctx['median'] * 1.15
    
    # Zone distribution
    bleed_rev = df[df['capital_zone'].str.contains('BLEED', na=False)]['weekly_sales_filled'].sum()
    fortress_rev = df[df['capital_zone'].str.contains('FORTRESS', na=False)]['weekly_sales_filled'].sum()
    bleed_pct = bleed_rev / total_rev if total_rev > 0 else 0
    fortress_pct = fortress_rev / total_rev if total_rev > 0 else 0
    
    # === 36M + FORECAST-ENHANCED MEDIA DECISION TREE ===
    
    # CRITICAL: Stop all spend on bleeding products (but check history first)
    if forecast_declining and (bleed_pct > 0.20 or avg_margin < 0.08):
        if historical_context_available and historically_strong:
            return "⚠️ REDUCE & MONITOR (historically strong)"
        return "🛑 IMMEDIATE PAUSE (Decline)"
    if bleed_pct > 0.30 or avg_margin < 0.05:
        if historical_context_available and historically_strong:
            return "⏸️ PAUSE & INVESTIGATE (historically strong)"
        return "🛑 HARD PAUSE ALL"
    
    # Buy Box crisis - no point advertising
    if avg_bb < 0.50:
        return "🛑 PAUSE (FIX BB FIRST)"
    
    # FORECAST: Surge coming with strong fundamentals - pre-invest
    if forecast_surging and avg_margin > 0.12 and avg_bb > 0.75:
        return "🚀 PRE-SCALE SPEND (Surge)"
    
    # Severe price gap - ads won't convert
    if avg_gap > 0.15:
        if forecast_declining:
            return "🛑 PAUSE (Price + Decline)"
        return "⏸️ REDUCE (-50%)"
    
    # Losing share with good fundamentals - increase visibility
    if share_delta < -0.10 and avg_margin > 0.12 and avg_bb > 0.80:
        if forecast_strong:
            return "🔥 MAX VISIBILITY (Recover Share)"
        return "📢 AGGRESSIVE VISIBILITY"
    
    # Velocity decay but strong margins - defensive spend
    if avg_decay > 1.2 and avg_margin > 0.15:
        if forecast_strong:
            return "💪 DEFEND + INVEST (Forecast Up)"
        return "🎯 DEFENSIVE ROAS"
    
    # FORECAST: Strong performance with surge forecast - maximum scale
    if forecast_surging and fortress_pct > 0.50 and avg_margin > 0.15:
        return "🔥 MAX SCALE (Surge Predicted)"
    
    # Strong performance - scale aggressively
    if fortress_pct > 0.60 and avg_margin > 0.18 and avg_decay < 0.9 and avg_bb > 0.85:
        if forecast_strong:
            return "🚀 SCALE SPEND (+40%)"
        return "🚀 SCALE SPEND (+25%)"
    
    # Good margins, gaining share - invest for growth
    if share_delta > 0.03 and avg_margin > 0.15:
        if forecast_strong:
            return "🔥 AGGRESSIVE INVEST"
        return "📈 INVEST FOR SHARE"
    
    # FORECAST: Declining forecast with moderate metrics - reduce proactively
    if forecast_declining and avg_margin < 0.15:
        return "⚠️ REDUCE (Forecast Soft)"
    
    # Moderate conditions - optimize
    if avg_margin > 0.12 and avg_bb > 0.75:
        return "⚖️ OPTIMIZE ROAS"
    
    # Default - maintain cautiously
    if avg_margin > 0.10:
        if forecast_declining:
            return "⏸️ CAUTIOUS HOLD"
        return "✅ MAINTAIN SPEND"
    else:
        return "⏸️ REDUCE SPEND"