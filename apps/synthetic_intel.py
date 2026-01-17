"""
ShelfGuard Synthetic Intelligence Module
=========================================
AI-enriched data fill for financial gaps and Keepa market interpolation.

This module replaces hardcoded defaults with predictive estimates based on:
1. Internal Synthetic Financials (Firmographic Fill)
2. Keepa Market Gaps (Interpolation Fill)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, Optional

# =============================================================================
# SECTION 1: INTERNAL SYNTHETIC FINANCIALS
# =============================================================================

# Starbucks K-Cup Portfolio Cost Structure (2026 Calibrated)
# Based on CPG industry benchmarks and K-Cup manufacturing economics

FLAVOR_COST_TIERS = {
    # Premium flavors have higher ingredient costs
    'premium': ['Caramel', 'Vanilla', 'Mocha', 'Pumpkin Spice', 'Peppermint'],
    'specialty': ['Sumatra', 'Caffe Verona', 'Espresso', 'Italian Roast', 'Komodo Dragon'],
    'core': ['Pike Place', 'Breakfast Blend', 'French Roast', 'House Blend', 'Veranda', 'Standard']
}

# Base COGS as percentage of price (before count adjustment)
COGS_BASE_RATES = {
    'premium': 0.32,     # Premium flavors: higher ingredient costs
    'specialty': 0.28,   # Single-origin/specialty: moderate premium
    'core': 0.22         # Core blends: economies of scale
}

# Count-based efficiency multipliers (bulk = lower per-unit cost)
COUNT_EFFICIENCY = {
    'bulk_large': (80, float('inf'), 0.85),   # 80+ pods: 15% efficiency gain
    'bulk_medium': (40, 79, 0.92),             # 40-79 pods: 8% efficiency gain
    'standard': (20, 39, 1.0),                 # 20-39 pods: baseline
    'small': (1, 19, 1.12)                     # <20 pods: 12% premium (small packs)
}

# K-Cup packaging volumetric standards (cubic feet)
PACKAGING_VOL_CF = {
    'bulk_large': 0.45,    # 80+ count box
    'bulk_medium': 0.28,   # 40-79 count box
    'standard': 0.15,      # 20-39 count box (typical 24-pack)
    'small': 0.08          # <20 count box (10-pack)
}

# Landed logistics indices (freight + palletization, 2026 rates)
LOGISTICS_INDEX = {
    'chicago': 0.018,      # Chicago hub: 1.8% of price
    'new_york': 0.022,     # NYC hub: 2.2% of price (higher last-mile)
    'default': 0.020       # National average: 2.0% of price
}


def get_flavor_tier(flavor: str) -> str:
    """Classify flavor into cost tier."""
    flavor_lower = flavor.lower() if flavor else ''
    
    for tier, flavors in FLAVOR_COST_TIERS.items():
        if any(f.lower() in flavor_lower for f in flavors):
            return tier
    return 'core'  # Default to core if unknown


def extract_pod_count(count_str: str) -> int:
    """Extract numeric pod count from count string."""
    if not count_str or count_str == 'Standard':
        return 24  # Default assumption for unlabeled products
    
    match = re.search(r'(\d+)', str(count_str))
    return int(match.group(1)) if match else 24


def get_count_tier(pod_count: int) -> str:
    """Get efficiency tier based on pod count."""
    for tier, (min_ct, max_ct, _) in COUNT_EFFICIENCY.items():
        if min_ct <= pod_count <= max_ct:
            return tier
    return 'standard'


def calculate_synthetic_cogs(row: pd.Series) -> float:
    """
    AI Synthetic COGS Calculation
    
    Replaces: price * 0.25 (hardcoded)
    
    Logic: Analyzes Flavor (Premium vs Core) and Count (Bulk efficiency)
    to estimate actual manufacturing + landed cost.
    """
    price = float(row.get('filled_price', 0) or 0)
    if price <= 0:
        return 0.0
    
    flavor = str(row.get('Flavor', 'Standard'))
    count_str = str(row.get('Count', 'Standard'))
    
    # 1. Get base COGS rate from flavor tier
    tier = get_flavor_tier(flavor)
    base_rate = COGS_BASE_RATES.get(tier, 0.25)
    
    # 2. Apply count-based efficiency multiplier
    pod_count = extract_pod_count(count_str)
    count_tier = get_count_tier(pod_count)
    
    for t, (_, _, multiplier) in COUNT_EFFICIENCY.items():
        if t == count_tier:
            efficiency_multiplier = multiplier
            break
    else:
        efficiency_multiplier = 1.0
    
    # 3. Calculate synthetic COGS
    synthetic_cogs = price * base_rate * efficiency_multiplier
    
    return round(synthetic_cogs, 2)


def calculate_synthetic_volume(row: pd.Series) -> float:
    """
    AI Dimensional Inference for Volumetric CF
    
    Replaces: default=0.05 (hardcoded)
    
    Logic: Predicts cubic feet based on K-Cup packaging standards
    to calculate accurate FBA storage fees.
    """
    count_str = str(row.get('Count', 'Standard'))
    pod_count = extract_pod_count(count_str)
    count_tier = get_count_tier(pod_count)
    
    # Get base volume from tier
    base_vol = PACKAGING_VOL_CF.get(count_tier, 0.15)
    
    # Fine-tune based on actual count within tier
    if count_tier == 'bulk_large':
        # Scale for very large packs (100+ vs 80)
        scale_factor = min(1.3, pod_count / 80)
        return round(base_vol * scale_factor, 4)
    elif count_tier == 'bulk_medium':
        scale_factor = 0.9 + (pod_count - 40) / 400
        return round(base_vol * scale_factor, 4)
    
    return round(base_vol, 4)


def calculate_landed_logistics(row: pd.Series, hub: str = 'default') -> float:
    """
    AI Landed Logistics Estimation
    
    New field (previously: None)
    
    Logic: Factors in current freight and palletization indices
    for Chicago/New York hubs to estimate FBA landing costs.
    """
    price = float(row.get('filled_price', 0) or 0)
    if price <= 0:
        return 0.0
    
    logistics_rate = LOGISTICS_INDEX.get(hub, LOGISTICS_INDEX['default'])
    
    # Heavier/larger items have higher logistics costs
    count_str = str(row.get('Count', 'Standard'))
    pod_count = extract_pod_count(count_str)
    
    # Weight factor: bulk items have economies but higher base freight
    if pod_count >= 80:
        weight_factor = 1.15
    elif pod_count >= 40:
        weight_factor = 1.05
    else:
        weight_factor = 1.0
    
    return round(price * logistics_rate * weight_factor, 2)


# =============================================================================
# SECTION 2: KEEPA MARKET GAP INTERPOLATION
# =============================================================================

def interpolate_bsr(
    current_bsr: float,
    historical_bsr: pd.Series,
    sister_sku_bsr: Optional[pd.Series] = None
) -> Tuple[float, str]:
    """
    AI BSR Interpolation (Shadow Rank)
    
    Fills: Missing Sales Rank points during stock-outs
    
    Logic: Uses 36-month historical velocity and sister-SKU performance
    to estimate "Shadow Rank" during data outages.
    
    Returns: (interpolated_bsr, confidence_level)
    """
    if pd.notna(current_bsr) and current_bsr > 0:
        return current_bsr, 'actual'
    
    # Strategy 1: Use rolling median of historical data
    if historical_bsr is not None and len(historical_bsr) > 0:
        valid_bsr = historical_bsr[historical_bsr > 0]
        if len(valid_bsr) >= 4:
            # Use recent 8-week median for trend continuity
            recent_median = valid_bsr.tail(8).median()
            long_term_median = valid_bsr.median()
            
            # Weighted blend: 70% recent, 30% long-term
            shadow_rank = (recent_median * 0.7) + (long_term_median * 0.3)
            return round(shadow_rank, 2), 'interpolated_historical'
    
    # Strategy 2: Use sister SKU performance if available
    if sister_sku_bsr is not None and len(sister_sku_bsr) > 0:
        valid_sister = sister_sku_bsr[sister_sku_bsr > 0]
        if len(valid_sister) > 0:
            return round(valid_sister.median(), 2), 'interpolated_sister'
    
    # Strategy 3: Category average fallback
    return 500.0, 'fallback_category_avg'


def estimate_buybox_floor(
    current_price: float,
    historical_prices: pd.Series,
    competitor_price: float,
    historical_gaps: Optional[pd.Series] = None
) -> Tuple[float, float]:
    """
    AI Buy Box Floor Estimation
    
    Fills: Suppressed Buy Box (no price shown on front-end)
    
    Logic: Analyzes historical Price Gaps and competitor behavior
    to predict the likely Buy Box clearing price.
    
    Returns: (estimated_floor_price, confidence_score 0-1)
    """
    # If we have current price, use it
    if pd.notna(current_price) and current_price > 0:
        return current_price, 1.0
    
    estimates = []
    weights = []
    
    # Strategy 1: Historical price median (most reliable)
    if historical_prices is not None and len(historical_prices) > 0:
        valid_prices = historical_prices[historical_prices > 0]
        if len(valid_prices) >= 4:
            recent_median = valid_prices.tail(8).median()
            estimates.append(recent_median)
            weights.append(0.5)
    
    # Strategy 2: Competitor price + typical gap
    if pd.notna(competitor_price) and competitor_price > 0:
        # Typical Starbucks premium over competitors: 5-10%
        if historical_gaps is not None and len(historical_gaps) > 0:
            avg_gap = historical_gaps.mean()
            estimated_from_comp = competitor_price * (1 + avg_gap)
        else:
            estimated_from_comp = competitor_price * 1.08  # Default 8% premium
        estimates.append(estimated_from_comp)
        weights.append(0.3)
    
    # Strategy 3: Historical minimum as absolute floor
    if historical_prices is not None and len(historical_prices) > 0:
        valid_prices = historical_prices[historical_prices > 0]
        if len(valid_prices) > 0:
            floor_price = valid_prices.quantile(0.1)  # 10th percentile
            estimates.append(floor_price)
            weights.append(0.2)
    
    if estimates:
        # Weighted average
        total_weight = sum(weights)
        weighted_price = sum(e * w for e, w in zip(estimates, weights)) / total_weight
        confidence = min(1.0, total_weight)
        return round(weighted_price, 2), round(confidence, 2)
    
    # Fallback: No data available
    return 0.0, 0.0


def predict_competitor_map(
    current_comp_price: float,
    historical_comp_prices: pd.Series,
    is_competitor_oos: bool = False
) -> Tuple[float, str]:
    """
    AI Competitor MAP Prediction
    
    Fills: Competitor floor prices when they are OOS
    
    Logic: Predicts a competitor's likely return price based on
    their previous 12-month MAP (Minimum Advertised Price) patterns.
    
    Returns: (predicted_map, prediction_type)
    """
    if not is_competitor_oos and pd.notna(current_comp_price) and current_comp_price > 0:
        return current_comp_price, 'actual'
    
    if historical_comp_prices is not None and len(historical_comp_prices) > 0:
        valid_prices = historical_comp_prices[historical_comp_prices > 0]
        
        if len(valid_prices) >= 8:
            # Find the "floor" - the minimum price they've historically held
            floor_10pct = valid_prices.quantile(0.10)
            floor_25pct = valid_prices.quantile(0.25)
            recent_min = valid_prices.tail(12).min()
            
            # MAP is typically between 10th and 25th percentile
            predicted_map = (floor_10pct + floor_25pct + recent_min) / 3
            return round(predicted_map, 2), 'predicted_map'
        
        elif len(valid_prices) > 0:
            # Limited history: use simple minimum
            return round(valid_prices.min(), 2), 'historical_min'
    
    # Fallback: use current if available, else 0
    if pd.notna(current_comp_price) and current_comp_price > 0:
        return current_comp_price, 'current_fallback'
    
    return 0.0, 'no_data'


# =============================================================================
# SECTION 3: BATCH PROCESSING FUNCTIONS
# =============================================================================

def enrich_synthetic_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch process: Add synthetic financial columns to DataFrame.
    
    New columns added:
    - synthetic_cogs: AI-estimated landed COGS
    - synthetic_vol_cf: AI-inferred volumetric cubic feet
    - landed_logistics: AI-estimated logistics cost
    """
    df = df.copy()
    
    # Apply synthetic calculations
    df['synthetic_cogs'] = df.apply(calculate_synthetic_cogs, axis=1)
    df['synthetic_vol_cf'] = df.apply(calculate_synthetic_volume, axis=1)
    df['landed_logistics'] = df.apply(lambda row: calculate_landed_logistics(row, 'default'), axis=1)
    
    return df


def interpolate_keepa_gaps(df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch process: Interpolate Keepa market gaps using historical data.
    
    Fills gaps in:
    - sales_rank_filled: BSR interpolation
    - filled_price: Buy Box floor estimation
    - new_fba_price: Competitor MAP prediction
    """
    df = df.copy()
    
    for idx, row in df.iterrows():
        asin = row.get('asin')
        asin_history = history_df[history_df['asin'] == asin] if history_df is not None else pd.DataFrame()
        
        # 1. BSR Interpolation
        if pd.isna(row.get('sales_rank_filled')) or row.get('sales_rank_filled', 0) == 0:
            historical_bsr = asin_history['sales_rank_filled'] if not asin_history.empty else pd.Series()
            interpolated_bsr, _ = interpolate_bsr(row.get('sales_rank_filled'), historical_bsr)
            df.at[idx, 'sales_rank_filled'] = interpolated_bsr
        
        # 2. Buy Box Floor Estimation
        if pd.isna(row.get('filled_price')) or row.get('filled_price', 0) == 0:
            historical_prices = asin_history['filled_price'] if not asin_history.empty else pd.Series()
            estimated_price, _ = estimate_buybox_floor(
                row.get('filled_price'),
                historical_prices,
                row.get('new_fba_price', 0)
            )
            df.at[idx, 'filled_price'] = estimated_price
        
        # 3. Competitor MAP Prediction
        if pd.isna(row.get('new_fba_price')) or row.get('new_fba_price', 0) == 0:
            historical_comp = asin_history['new_fba_price'] if not asin_history.empty else pd.Series()
            predicted_map, _ = predict_competitor_map(row.get('new_fba_price'), historical_comp, True)
            df.at[idx, 'new_fba_price'] = predicted_map
    
    return df


def get_synthetic_intel_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for synthetic intelligence fills.
    """
    total_rows = len(df)
    
    return {
        'total_skus': total_rows,
        'avg_synthetic_cogs_rate': (df['synthetic_cogs'] / df['filled_price'].replace(0, np.nan)).mean(),
        'avg_synthetic_vol_cf': df['synthetic_vol_cf'].mean(),
        'avg_landed_logistics': df['landed_logistics'].mean(),
        'cogs_range': (df['synthetic_cogs'].min(), df['synthetic_cogs'].max()),
        'vol_cf_range': (df['synthetic_vol_cf'].min(), df['synthetic_vol_cf'].max()),
    }
