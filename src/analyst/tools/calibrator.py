# src/analyst/tools/calibrator.py

"""
CALIBRATOR - The Bayesian Update Layer

This tool calculates REAL physics for a specific product based on observed data,
overriding the default priors in config.py.

Architecture:
- Input: df_weekly (product-level time series)
- Process: Calculate actual elasticities, impacts, and behaviors
- Output: CalibratedPhysics (product-specific physics overrides)

The Calibrator turns "Common Sense" (config.py) into "Empirical Reality" (this output).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from scipy import stats
import warnings

from ..config import KEEPA_CONFIG


@dataclass
class CalibratedPhysics:
    """
    Product-specific physics calculated from observed data.
    These override the defaults in KEEPA_CONFIG.
    
    The calibrator learns the TRUE physics for this specific product,
    turning generic assumptions into empirical reality.
    """
    asin: str
    
    # ==========================================================================
    # BUCKET 1: ELASTICITIES (Response to Inputs)
    # ==========================================================================
    
    # Price Elasticity: % change in units per % change in price
    price_elasticity: Optional[float] = None
    price_elasticity_confidence: str = "UNCALIBRATED"
    
    # Amazon Impact: % sales change when Amazon owns Buy Box
    amazon_impact_pct: Optional[float] = None
    amazon_impact_confidence: str = "UNCALIBRATED"
    
    # Competition Sensitivity: Impact per additional seller
    competition_sensitivity: Optional[float] = None
    competition_sensitivity_confidence: str = "UNCALIBRATED"
    
    # Review Impact: How much do reviews drive rank?
    review_elasticity: Optional[float] = None  # % rank change per % review change
    review_elasticity_confidence: str = "UNCALIBRATED"
    
    # ==========================================================================
    # BUCKET 2: BSR FORMULA CALIBRATION
    # Default: units = 145000 * BSR^-0.9
    # Calibrated: units = bsr_constant * BSR^bsr_exponent
    # ==========================================================================
    
    bsr_constant: Optional[float] = None  # Default: 145000
    bsr_exponent: Optional[float] = None  # Default: -0.9
    bsr_formula_confidence: str = "UNCALIBRATED"
    bsr_formula_error_pct: Optional[float] = None  # MAPE vs actual monthly_sold
    
    # ==========================================================================
    # BUCKET 3: SIGNIFICANCE THRESHOLDS (Noise vs Signal)
    # These should be relative to product's normal volatility
    # ==========================================================================
    
    # Adaptive significance: 2 std devs of normal volatility
    price_significance_threshold: Optional[float] = None  # Default: 0.03 (3%)
    rank_significance_threshold: Optional[float] = None   # Default: 0.10 (10%)
    revenue_significance_threshold: Optional[float] = None  # Default: 0.15 (15%)
    
    # Severity calibration: relative to this product's volatility
    severity_minor: Optional[float] = None     # Default: 0.05
    severity_moderate: Optional[float] = None  # Default: 0.15
    severity_major: Optional[float] = None     # Default: 0.30
    severity_critical: Optional[float] = None  # Default: 0.50
    
    # ==========================================================================
    # BUCKET 4: SEASONALITY
    # ==========================================================================
    
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None  # Auto-detected weeks (4=monthly, 13=quarterly)
    seasonal_amplitude: Optional[float] = None  # % swing from trough to peak
    peak_weeks: List[int] = field(default_factory=list)  # Week numbers (1-52)
    trough_weeks: List[int] = field(default_factory=list)
    
    # ==========================================================================
    # BUCKET 5: VOLATILITY PROFILE
    # ==========================================================================
    
    price_volatility: Optional[float] = None  # Std dev of price changes
    rank_volatility: Optional[float] = None   # Std dev of rank changes
    revenue_volatility: Optional[float] = None  # Std dev of revenue changes
    stability_score: Optional[float] = None   # 0-1 (higher = more stable)
    volatility_regime: str = "UNKNOWN"  # "STABLE", "MODERATE", "VOLATILE", "CHAOTIC"
    
    # ==========================================================================
    # BUCKET 6: TREND
    # ==========================================================================
    
    trend_direction: str = "STABLE"  # "GROWING", "DECLINING", "STABLE"
    trend_strength: Optional[float] = None  # Normalized slope coefficient
    trend_r_squared: Optional[float] = None  # How well does linear trend fit?
    
    # ==========================================================================
    # BUCKET 7: CATEGORY-SPECIFIC THRESHOLDS
    # ==========================================================================
    
    # Rating danger zone (category-specific)
    rating_danger_threshold: Optional[float] = None  # Default: 4.0
    rating_danger_confidence: str = "UNCALIBRATED"
    
    # OOS thresholds (product-specific recovery time)
    oos_warning_threshold: Optional[float] = None  # Default: 0.05
    oos_critical_threshold: Optional[float] = None  # Default: 0.20
    oos_recovery_weeks: Optional[int] = None  # How many weeks to recover from OOS?
    
    # Amazon threat thresholds
    amazon_warning_threshold: Optional[float] = None  # Default: 0.40
    amazon_critical_threshold: Optional[float] = None  # Default: 0.80
    
    # ==========================================================================
    # BUCKET 8: DATA QUALITY
    # ==========================================================================
    
    data_weeks: int = 0
    data_completeness: float = 0.0  # 0-1 (1 = no missing data)
    is_reliable: bool = False  # True if enough data for calibration
    
    # Warnings / Anomalies
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "elasticities": {
                "price": {
                    "value": self.price_elasticity,
                    "confidence": self.price_elasticity_confidence,
                },
                "amazon_impact": {
                    "value": self.amazon_impact_pct,
                    "confidence": self.amazon_impact_confidence,
                },
                "competition": {
                    "value": self.competition_sensitivity,
                    "confidence": self.competition_sensitivity_confidence,
                },
                "review": {
                    "value": self.review_elasticity,
                    "confidence": self.review_elasticity_confidence,
                },
            },
            "bsr_formula": {
                "constant": self.bsr_constant,
                "exponent": self.bsr_exponent,
                "confidence": self.bsr_formula_confidence,
                "error_pct": self.bsr_formula_error_pct,
            },
            "significance_thresholds": {
                "price": self.price_significance_threshold,
                "rank": self.rank_significance_threshold,
                "revenue": self.revenue_significance_threshold,
            },
            "severity_calibration": {
                "minor": self.severity_minor,
                "moderate": self.severity_moderate,
                "major": self.severity_major,
                "critical": self.severity_critical,
            },
            "seasonality": {
                "detected": self.has_seasonality,
                "period_weeks": self.seasonal_period,
                "amplitude_pct": self.seasonal_amplitude,
                "peak_weeks": self.peak_weeks,
                "trough_weeks": self.trough_weeks,
            },
            "volatility": {
                "price_std": self.price_volatility,
                "rank_std": self.rank_volatility,
                "revenue_std": self.revenue_volatility,
                "stability_score": self.stability_score,
                "regime": self.volatility_regime,
            },
            "trend": {
                "direction": self.trend_direction,
                "strength": self.trend_strength,
                "r_squared": self.trend_r_squared,
            },
            "category_thresholds": {
                "rating_danger": self.rating_danger_threshold,
                "oos_warning": self.oos_warning_threshold,
                "oos_critical": self.oos_critical_threshold,
                "oos_recovery_weeks": self.oos_recovery_weeks,
                "amazon_warning": self.amazon_warning_threshold,
                "amazon_critical": self.amazon_critical_threshold,
            },
            "data_quality": {
                "weeks": self.data_weeks,
                "completeness": self.data_completeness,
                "is_reliable": self.is_reliable,
            },
            "warnings": self.warnings,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"CALIBRATED PHYSICS FOR {self.asin}:"]
        lines.append("")
        
        # Elasticities
        lines.append("ELASTICITIES:")
        if self.price_elasticity is not None:
            direction = "INELASTIC (price cuts DON'T work)" if abs(self.price_elasticity) < 1.0 else "ELASTIC (price cuts work)"
            lines.append(f"  Price: {self.price_elasticity:.2f} ({direction}) [{self.price_elasticity_confidence}]")
        
        if self.amazon_impact_pct is not None:
            lines.append(f"  Amazon Impact: {self.amazon_impact_pct*100:.0f}% sales change when Amazon owns Buy Box [{self.amazon_impact_confidence}]")
        
        if self.competition_sensitivity is not None:
            lines.append(f"  Competition: {self.competition_sensitivity:.2f} sensitivity [{self.competition_sensitivity_confidence}]")
        
        # BSR Formula
        if self.bsr_constant is not None:
            lines.append("")
            lines.append("BSR FORMULA (Calibrated):")
            lines.append(f"  units = {self.bsr_constant:.0f} * BSR^{self.bsr_exponent:.2f}")
            if self.bsr_formula_error_pct:
                lines.append(f"  Error: {self.bsr_formula_error_pct:.0%} [{self.bsr_formula_confidence}]")
        
        # Significance Thresholds
        if self.price_significance_threshold is not None:
            lines.append("")
            lines.append("SIGNIFICANCE THRESHOLDS (Product-Specific):")
            lines.append(f"  Price change > {self.price_significance_threshold:.1%} = significant")
            if self.rank_significance_threshold:
                lines.append(f"  Rank change > {self.rank_significance_threshold:.1%} = significant")
        
        # Severity
        if self.severity_minor is not None:
            lines.append(f"  Severity levels: {self.severity_minor:.0%} / {self.severity_moderate:.0%} / {self.severity_major:.0%} / {self.severity_critical:.0%}")
        
        # Seasonality
        if self.has_seasonality:
            lines.append("")
            lines.append("SEASONALITY:")
            lines.append(f"  Period: {self.seasonal_period} weeks")
            if self.seasonal_amplitude:
                lines.append(f"  Amplitude: {self.seasonal_amplitude:.0%} swing")
            lines.append(f"  Peak weeks: {self.peak_weeks}, Trough weeks: {self.trough_weeks}")
        
        # Volatility
        lines.append("")
        lines.append(f"VOLATILITY REGIME: {self.volatility_regime}")
        if self.price_volatility:
            lines.append(f"  Price volatility: {self.price_volatility:.1%}")
        if self.rank_volatility:
            lines.append(f"  Rank volatility: {self.rank_volatility:.1%}")
        
        # Trend
        lines.append("")
        lines.append(f"TREND: {self.trend_direction}")
        if self.trend_strength:
            lines.append(f"  Strength: {self.trend_strength:.2f}")
        
        # Data Quality
        lines.append("")
        lines.append(f"DATA QUALITY: {self.data_weeks} weeks, {self.data_completeness:.0%} complete")
        lines.append(f"  Calibration reliable: {'YES' if self.is_reliable else 'NO (use defaults)'}")
        
        if self.warnings:
            lines.append("")
            # Defensive: ensure all warnings are strings
            warn_strs = [str(w) if not isinstance(w, str) else w for w in self.warnings]
            lines.append(f"WARNINGS: {', '.join(warn_strs)}")
        
        return "\n".join(lines)


def calibrate_physics(df_weekly: pd.DataFrame, asin: str = "UNKNOWN") -> CalibratedPhysics:
    """
    Calculate REAL elasticity and behaviors for a specific product.
    
    Args:
        df_weekly: Weekly time series data for the product
        asin: ASIN identifier
        
    Returns:
        CalibratedPhysics with product-specific overrides
    """
    result = CalibratedPhysics(asin=asin)
    
    # Ensure we have data
    if df_weekly is None or len(df_weekly) == 0:
        result.warnings.append("NO_DATA")
        return result
    
    # Sort by time
    df = df_weekly.copy()
    if 'week_start' in df.columns:
        df = df.sort_values('week_start')
    
    # Calculate data quality
    result.data_weeks = len(df)
    result.data_completeness = _calculate_completeness(df)
    result.is_reliable = result.data_weeks >= 4 and result.data_completeness >= 0.7
    
    if not result.is_reliable:
        result.warnings.append(f"INSUFFICIENT_DATA: {result.data_weeks} weeks, {result.data_completeness*100:.0f}% complete")
        return result
    
    # Calculate each calibration
    try:
        result = _calibrate_price_elasticity(df, result)
    except Exception as e:
        result.warnings.append(f"PRICE_ELASTICITY_FAILED: {str(e)}")
    
    try:
        result = _calibrate_amazon_impact(df, result)
    except Exception as e:
        result.warnings.append(f"AMAZON_IMPACT_FAILED: {str(e)}")
    
    try:
        result = _calibrate_competition_sensitivity(df, result)
    except Exception as e:
        result.warnings.append(f"COMPETITION_FAILED: {str(e)}")
    
    try:
        result = _calibrate_seasonality(df, result)
    except Exception as e:
        result.warnings.append(f"SEASONALITY_FAILED: {str(e)}")
    
    try:
        result = _calibrate_volatility(df, result)
    except Exception as e:
        result.warnings.append(f"VOLATILITY_FAILED: {str(e)}")
    
    try:
        result = _calibrate_trend(df, result)
    except Exception as e:
        result.warnings.append(f"TREND_FAILED: {str(e)}")
    
    # NEW CALIBRATIONS (Generation 3/4)
    try:
        result = _calibrate_bsr_formula(df, result)
    except Exception as e:
        result.warnings.append(f"BSR_FORMULA_FAILED: {str(e)}")
    
    try:
        result = _calibrate_significance_thresholds(df, result)
    except Exception as e:
        result.warnings.append(f"SIGNIFICANCE_FAILED: {str(e)}")
    
    try:
        result = _calibrate_severity_levels(df, result)
    except Exception as e:
        result.warnings.append(f"SEVERITY_FAILED: {str(e)}")
    
    try:
        result = _calibrate_category_thresholds(df, result)
    except Exception as e:
        result.warnings.append(f"CATEGORY_THRESHOLDS_FAILED: {str(e)}")
    
    return result


def _calculate_completeness(df: pd.DataFrame) -> float:
    """Calculate fraction of non-null values in key columns."""
    key_cols = ['filled_price', 'sales_rank', 'estimated_units']
    available_cols = [c for c in key_cols if c in df.columns]
    
    if not available_cols:
        return 0.0
    
    total_cells = len(df) * len(available_cols)
    non_null = sum(df[col].notna().sum() for col in available_cols)
    return non_null / total_cells if total_cells > 0 else 0.0


def _calibrate_price_elasticity(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Calculate real price elasticity: % change in units / % change in price.
    
    Elasticity interpretation:
    - |e| > 1: Elastic (price sensitive) - price cuts work
    - |e| < 1: Inelastic (price insensitive) - price cuts don't work
    - e should be negative (price up -> units down)
    """
    price_col = 'filled_price' if 'filled_price' in df.columns else 'buy_box_price'
    unit_col = 'estimated_units'
    
    if price_col not in df.columns or unit_col not in df.columns:
        result.warnings.append("MISSING_COLUMNS_FOR_ELASTICITY")
        return result
    
    # Calculate percentage changes
    price_change = df[price_col].pct_change()
    unit_change = df[unit_col].pct_change()
    
    # Filter for significant price moves (>3%) to avoid noise
    threshold = KEEPA_CONFIG["THRESHOLDS"]["price_change_pct"]
    mask = (abs(price_change) > threshold) & price_change.notna() & unit_change.notna()
    
    if mask.sum() < 3:
        result.warnings.append("INSUFFICIENT_PRICE_CHANGES")
        return result
    
    # Calculate elasticity using median to be robust to outliers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elasticity_values = unit_change[mask] / price_change[mask]
        elasticity_values = elasticity_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(elasticity_values) < 3:
        return result
    
    # Use median for robustness
    result.price_elasticity = float(np.median(elasticity_values))
    
    # Clip to reasonable range
    result.price_elasticity = np.clip(result.price_elasticity, -5.0, 5.0)
    
    # Set confidence based on sample size
    if mask.sum() >= 10:
        result.price_elasticity_confidence = "HIGH"
    elif mask.sum() >= 5:
        result.price_elasticity_confidence = "MEDIUM"
    else:
        result.price_elasticity_confidence = "LOW"
    
    return result


def _calibrate_amazon_impact(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Measure: Does Amazon taking Buy Box actually hurt sales?
    
    Compare sales when Amazon has high BB share vs low BB share.
    """
    amazon_col = 'bb_stats_amazon_90'
    sales_col = 'estimated_units'
    
    if amazon_col not in df.columns or sales_col not in df.columns:
        return result
    
    # Split into high-Amazon and low-Amazon periods
    high_amz = df[df[amazon_col] > 0.5][sales_col].mean()
    low_amz = df[df[amazon_col] < 0.1][sales_col].mean()
    
    # Need data in both buckets
    high_count = (df[amazon_col] > 0.5).sum()
    low_count = (df[amazon_col] < 0.1).sum()
    
    if pd.isna(high_amz) or pd.isna(low_amz) or high_count < 2 or low_count < 2:
        return result
    
    # Calculate impact: (sales with Amazon - sales without) / sales without
    if low_amz > 0:
        impact = (high_amz - low_amz) / low_amz
        result.amazon_impact_pct = float(np.clip(impact, -1.0, 0.5))
        
        # Confidence based on sample sizes
        if min(high_count, low_count) >= 5:
            result.amazon_impact_confidence = "HIGH"
        elif min(high_count, low_count) >= 3:
            result.amazon_impact_confidence = "MEDIUM"
        else:
            result.amazon_impact_confidence = "LOW"
    
    return result


def _calibrate_competition_sensitivity(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Measure: How much does adding a seller hurt our sales?
    """
    seller_col = 'new_offer_count'
    sales_col = 'estimated_units'
    
    if seller_col not in df.columns or sales_col not in df.columns:
        return result
    
    # Calculate correlation between seller count changes and sales changes
    seller_change = df[seller_col].diff()
    sales_change = df[sales_col].pct_change()
    
    # Remove NaN
    mask = seller_change.notna() & sales_change.notna()
    if mask.sum() < 5:
        return result
    
    # Use correlation as proxy for sensitivity
    try:
        corr, p_value = stats.pearsonr(seller_change[mask], sales_change[mask])
        if p_value < 0.10:  # Significant at 10%
            result.competition_sensitivity = float(corr)
            result.competition_sensitivity_confidence = "HIGH" if p_value < 0.05 else "MEDIUM"
    except:
        pass
    
    return result


def _calibrate_seasonality(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Detect seasonal patterns in sales.
    """
    sales_col = 'estimated_units'
    
    if sales_col not in df.columns or len(df) < 8:
        return result
    
    sales = df[sales_col].dropna()
    if len(sales) < 8:
        return result
    
    # Calculate autocorrelation at different lags
    try:
        from statsmodels.tsa.stattools import acf
        
        max_lag = min(len(sales) // 2, 13)  # Max 13 weeks (quarter)
        autocorr = acf(sales, nlags=max_lag, fft=True)
        
        # Look for peaks at 4 weeks (monthly) or 13 weeks (quarterly)
        for period in [4, 13]:
            if period < len(autocorr):
                if autocorr[period] > 0.3:  # Significant autocorrelation
                    result.has_seasonality = True
                    result.seasonal_period = period
                    
                    # Calculate amplitude as coefficient of variation
                    result.seasonal_amplitude = float(sales.std() / sales.mean())
                    break
    except ImportError:
        # statsmodels not available, skip seasonality detection
        pass
    
    return result


def _calibrate_volatility(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Measure price and rank volatility.
    """
    price_col = 'filled_price' if 'filled_price' in df.columns else 'buy_box_price'
    rank_col = 'sales_rank'
    
    # Price volatility (std of % changes)
    if price_col in df.columns:
        price_pct = df[price_col].pct_change().dropna()
        if len(price_pct) > 2:
            result.price_volatility = float(price_pct.std())
    
    # Rank volatility (std of % changes)
    if rank_col in df.columns:
        rank_pct = df[rank_col].pct_change().dropna()
        if len(rank_pct) > 2:
            result.rank_volatility = float(rank_pct.std())
    
    # Revenue volatility
    if 'weekly_revenue' in df.columns:
        rev_pct = df['weekly_revenue'].pct_change().dropna()
        if len(rev_pct) > 2:
            result.revenue_volatility = float(rev_pct.std())
    
    # Stability score: inverse of volatility (normalized)
    if result.price_volatility is not None and result.rank_volatility is not None:
        avg_vol = (result.price_volatility + result.rank_volatility) / 2
        # Map volatility to 0-1 stability score (higher = more stable)
        result.stability_score = float(1.0 / (1.0 + avg_vol * 10))
        
        # Set volatility regime
        if avg_vol < 0.05:
            result.volatility_regime = "STABLE"
        elif avg_vol < 0.15:
            result.volatility_regime = "MODERATE"
        elif avg_vol < 0.30:
            result.volatility_regime = "VOLATILE"
        else:
            result.volatility_regime = "CHAOTIC"
    
    return result


def _calibrate_trend(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Determine if the product is growing, declining, or stable.
    """
    rank_col = 'sales_rank'
    
    if rank_col not in df.columns or len(df) < 4:
        return result
    
    # Use linear regression on rank over time
    rank = df[rank_col].dropna()
    if len(rank) < 4:
        return result
    
    # Create time index
    x = np.arange(len(rank))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rank)
        
        # Normalize slope by mean rank for interpretation
        mean_rank = rank.mean()
        normalized_slope = slope / mean_rank if mean_rank > 0 else 0
        
        result.trend_strength = float(abs(normalized_slope))
        
        # Determine direction (remember: for rank, negative slope = GROWING)
        if normalized_slope < -0.02 and p_value < 0.10:
            result.trend_direction = "GROWING"  # Rank decreasing = good
        elif normalized_slope > 0.02 and p_value < 0.10:
            result.trend_direction = "DECLINING"  # Rank increasing = bad
        else:
            result.trend_direction = "STABLE"
        
        result.trend_r_squared = float(r_value ** 2)
            
    except:
        result.trend_direction = "UNKNOWN"
    
    return result


# =============================================================================
# NEW CALIBRATION FUNCTIONS (Generation 3/4)
# =============================================================================

def _calibrate_bsr_formula(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Calibrate the BSR-to-Units formula for this specific product/category.
    
    Default formula: units = 145000 * BSR^(-0.9)
    Calibrated formula: units = C * BSR^(E) where C and E are fitted
    
    Uses monthly_sold (Amazon's badge) as ground truth when available.
    """
    rank_col = 'sales_rank'
    actual_units_col = 'monthly_sold'
    
    if rank_col not in df.columns:
        return result
    
    # If we have actual monthly_sold data, calibrate against it
    if actual_units_col in df.columns:
        # Get rows where we have both rank and actual units
        valid = df[[rank_col, actual_units_col]].dropna()
        valid = valid[valid[rank_col] > 0]
        valid = valid[valid[actual_units_col] > 0]
        
        if len(valid) >= 3:
            try:
                # Log-log regression: log(units) = log(C) + E * log(BSR)
                log_rank = np.log(valid[rank_col])
                log_units = np.log(valid[actual_units_col])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, log_units)
                
                # Convert back: C = exp(intercept), E = slope
                result.bsr_constant = float(np.exp(intercept))
                result.bsr_exponent = float(slope)
                
                # Calculate error
                predicted = result.bsr_constant * (valid[rank_col] ** result.bsr_exponent)
                mape = np.mean(np.abs((valid[actual_units_col] - predicted) / valid[actual_units_col]))
                result.bsr_formula_error_pct = float(mape)
                
                # Set confidence
                if len(valid) >= 10 and mape < 0.30:
                    result.bsr_formula_confidence = "HIGH"
                elif len(valid) >= 5 and mape < 0.50:
                    result.bsr_formula_confidence = "MEDIUM"
                else:
                    result.bsr_formula_confidence = "LOW"
                    
            except Exception:
                # Fall back to defaults
                result.bsr_constant = 145000.0
                result.bsr_exponent = -0.9
                result.bsr_formula_confidence = "DEFAULT"
    else:
        # No ground truth - use default formula
        result.bsr_constant = 145000.0
        result.bsr_exponent = -0.9
        result.bsr_formula_confidence = "DEFAULT"
    
    return result


def _calibrate_significance_thresholds(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Calibrate significance thresholds based on product's normal volatility.
    
    A 10% change might be noise for a volatile product but signal for a stable one.
    We set thresholds at 2 standard deviations of typical changes.
    """
    price_col = 'filled_price' if 'filled_price' in df.columns else 'buy_box_price'
    rank_col = 'sales_rank'
    rev_col = 'weekly_revenue'
    
    # Price significance = 2 std of normal price changes
    if price_col in df.columns:
        price_changes = df[price_col].pct_change().dropna().abs()
        if len(price_changes) >= 4:
            # Use 2 std as significance threshold (captures 95% of normal variation)
            threshold = float(price_changes.std() * 2)
            result.price_significance_threshold = max(0.02, min(0.15, threshold))
    
    # Rank significance = 2 std of normal rank changes
    if rank_col in df.columns:
        rank_changes = df[rank_col].pct_change().dropna().abs()
        if len(rank_changes) >= 4:
            threshold = float(rank_changes.std() * 2)
            result.rank_significance_threshold = max(0.05, min(0.30, threshold))
    
    # Revenue significance = 2 std of normal revenue changes
    if rev_col in df.columns:
        rev_changes = df[rev_col].pct_change().dropna().abs()
        if len(rev_changes) >= 4:
            threshold = float(rev_changes.std() * 2)
            result.revenue_significance_threshold = max(0.10, min(0.40, threshold))
    
    return result


def _calibrate_severity_levels(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Calibrate severity levels relative to product's volatility profile.
    
    Default: minor=5%, moderate=15%, major=30%, critical=50%
    Calibrated: based on percentiles of observed changes
    """
    # Use the already-calculated volatility if available
    if result.price_volatility is None and result.rank_volatility is None:
        return result
    
    # Average volatility as baseline
    volatilities = [v for v in [result.price_volatility, result.rank_volatility, result.revenue_volatility] if v is not None]
    if not volatilities:
        return result
    
    avg_vol = np.mean(volatilities)
    
    # Set severity levels as multiples of volatility
    # Minor = 1x normal vol, Moderate = 2x, Major = 4x, Critical = 8x
    result.severity_minor = float(max(0.03, min(0.10, avg_vol * 1.0)))
    result.severity_moderate = float(max(0.08, min(0.25, avg_vol * 2.0)))
    result.severity_major = float(max(0.15, min(0.50, avg_vol * 4.0)))
    result.severity_critical = float(max(0.30, min(0.80, avg_vol * 8.0)))
    
    return result


def _calibrate_category_thresholds(df: pd.DataFrame, result: CalibratedPhysics) -> CalibratedPhysics:
    """
    Calibrate category-specific thresholds based on observed data patterns.
    
    - Rating danger zone: What rating threshold correlates with sales drop?
    - OOS thresholds: How much OOS actually hurts this product?
    - Amazon thresholds: What Amazon BB share level hurts this product?
    """
    # Rating danger threshold calibration
    if 'rating' in df.columns and 'sales_rank' in df.columns:
        ratings = df['rating'].dropna()
        if len(ratings) >= 4:
            # Find the rating below which rank tends to be worse
            # Default to 4.0, but adjust if data suggests otherwise
            result.rating_danger_threshold = 4.0  # Start with default
            
            # Check if there's a clear breakpoint
            median_rating = ratings.median()
            if median_rating < 4.0:
                # Products in this category typically have lower ratings
                result.rating_danger_threshold = max(3.5, median_rating - 0.3)
            
            result.rating_danger_confidence = "DEFAULT"
    
    # OOS threshold calibration
    if 'oos_pct_30' in df.columns:
        oos = df['oos_pct_30'].dropna()
        if len(oos) >= 4:
            # Use 75th percentile as "normal" OOS for this product
            p75 = float(oos.quantile(0.75))
            result.oos_warning_threshold = max(0.03, min(0.15, p75 * 0.5))
            result.oos_critical_threshold = max(0.10, min(0.40, p75 * 1.5))
            
            # Estimate recovery time from OOS events
            # Look for periods where OOS spiked then recovered
            oos_spikes = oos > result.oos_critical_threshold
            if oos_spikes.any():
                # Count consecutive non-spike weeks after a spike as recovery
                recovery_weeks = []
                in_recovery = False
                weeks = 0
                for is_spike in oos_spikes:
                    if is_spike:
                        if in_recovery and weeks > 0:
                            recovery_weeks.append(weeks)
                        in_recovery = True
                        weeks = 0
                    elif in_recovery:
                        if not is_spike:
                            weeks += 1
                        else:
                            in_recovery = False
                
                if recovery_weeks:
                    result.oos_recovery_weeks = int(np.median(recovery_weeks))
    
    # Amazon threat threshold calibration
    if 'bb_stats_amazon_90' in df.columns:
        amz_share = df['bb_stats_amazon_90'].dropna()
        if len(amz_share) >= 4:
            # If Amazon is typically present, adjust thresholds up
            median_amz = float(amz_share.median())
            max_amz = float(amz_share.max())
            
            if median_amz > 0.30:
                # Amazon is a regular presence - higher thresholds
                result.amazon_warning_threshold = min(0.60, median_amz + 0.15)
                result.amazon_critical_threshold = min(0.90, max_amz)
            else:
                # Amazon is occasional - lower thresholds
                result.amazon_warning_threshold = 0.30
                result.amazon_critical_threshold = 0.60
    
    return result
