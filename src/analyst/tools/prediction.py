# src/analyst/tools/prediction.py

"""
PREDICTION SENSOR - Forecasting Tool

Uses Holt-Winters Exponential Smoothing (statsmodels) for time series forecasting.
Provides:
- Point forecasts for key metrics
- Prediction intervals (confidence bounds)
- Trend detection
- Seasonal decomposition

Output: Structured ForecastSignal for the Orchestrator
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

from ..config import KEEPA_CONFIG, get_target_direction


@dataclass
class Forecast:
    """Forecast for a single metric."""
    metric: str
    horizon_weeks: int
    
    # Point forecasts
    forecasts: List[float] = field(default_factory=list)
    
    # Confidence intervals
    lower_bound: List[float] = field(default_factory=list)  # 95% lower
    upper_bound: List[float] = field(default_factory=list)  # 95% upper
    
    # Current state
    current_value: Optional[float] = None
    
    # Trend
    trend_direction: str = "STABLE"  # "UP", "DOWN", "STABLE"
    trend_strength: Optional[float] = None  # 0-1
    
    # Model quality
    model_type: str = "UNKNOWN"
    mae: Optional[float] = None  # Mean Absolute Error on holdout
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    
    # Interpretation
    forecast_summary: str = ""
    
    def expected_change_pct(self) -> Optional[float]:
        """Calculate expected % change from current to end of horizon."""
        if self.current_value and self.forecasts and self.current_value != 0:
            return (self.forecasts[-1] - self.current_value) / abs(self.current_value)
        return None


@dataclass
class SeasonalDecomposition:
    """Seasonal pattern breakdown."""
    metric: str
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None
    seasonal_strength: Optional[float] = None  # 0-1
    peak_weeks: List[int] = field(default_factory=list)
    trough_weeks: List[int] = field(default_factory=list)


@dataclass
class ForecastSignal:
    """
    Complete forecasting output.
    This is what gets passed to the Orchestrator.
    """
    asin: str
    analysis_timestamp: str
    horizon_weeks: int
    
    # Forecasts by metric
    forecasts: Dict[str, Forecast] = field(default_factory=dict)
    
    # Seasonal patterns
    seasonality: Dict[str, SeasonalDecomposition] = field(default_factory=dict)
    
    # Summary
    metrics_forecasted: int = 0
    overall_outlook: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN"
    confidence_level: str = "MEDIUM"  # "HIGH", "MEDIUM", "LOW"
    
    # Key predictions
    key_predictions: List[str] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "timestamp": self.analysis_timestamp,
            "horizon_weeks": self.horizon_weeks,
            "summary": {
                "metrics_forecasted": self.metrics_forecasted,
                "outlook": self.overall_outlook,
                "confidence": self.confidence_level,
            },
            "forecasts": {
                metric: {
                    "current": f.current_value,
                    "forecast_end": f.forecasts[-1] if f.forecasts else None,
                    "expected_change_pct": f.expected_change_pct(),
                    "trend": f.trend_direction,
                    "model": f.model_type,
                    "mape": f.mape,
                }
                for metric, f in self.forecasts.items()
            },
            "seasonality": {
                metric: {
                    "detected": s.has_seasonality,
                    "period": s.seasonal_period,
                    "strength": s.seasonal_strength,
                }
                for metric, s in self.seasonality.items()
            },
            "key_predictions": self.key_predictions,
            "warnings": self.warnings,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"=== {self.horizon_weeks}-WEEK FORECAST FOR {self.asin} ==="]
        lines.append(f"Outlook: {self.overall_outlook} (Confidence: {self.confidence_level})")
        lines.append("")
        
        lines.append("FORECASTS:")
        for metric, f in self.forecasts.items():
            change = f.expected_change_pct()
            change_str = f"{change:+.1%}" if change else "N/A"
            direction = get_target_direction(metric)
            
            # Interpret if forecast is good or bad
            if change:
                if direction == "lower_is_better":
                    good_bad = "GOOD" if change < 0 else "BAD"
                else:
                    good_bad = "GOOD" if change > 0 else "BAD"
            else:
                good_bad = "NEUTRAL"
            
            lines.append(f"  - {metric}: {f.current_value:.0f} → {f.forecasts[-1]:.0f} ({change_str}, {good_bad})")
            lines.append(f"    Trend: {f.trend_direction}, Model: {f.model_type}")
        
        if self.key_predictions:
            lines.append("")
            lines.append("KEY PREDICTIONS:")
            for pred in self.key_predictions:
                lines.append(f"  • {pred}")
        
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS: " + " | ".join(self.warnings))
        
        return "\n".join(lines)


def forecast_metrics(
    df_weekly: pd.DataFrame,
    asin: str = "UNKNOWN",
    metrics: Optional[List[str]] = None,
    horizon_weeks: int = 4,
    seasonal_period: int = 4
) -> ForecastSignal:
    """
    Generate forecasts for key metrics.
    
    Args:
        df_weekly: Weekly time series data
        asin: ASIN identifier
        metrics: List of metrics to forecast (defaults to key targets)
        horizon_weeks: Number of weeks to forecast ahead
        seasonal_period: Expected seasonal period in weeks
        
    Returns:
        ForecastSignal with all forecasts
    """
    signal = ForecastSignal(
        asin=asin,
        analysis_timestamp=datetime.now().isoformat(),
        horizon_weeks=horizon_weeks
    )
    
    if df_weekly is None or len(df_weekly) < 4:
        signal.warnings.append("INSUFFICIENT_DATA")
        signal.confidence_level = "LOW"
        return signal
    
    # Sort by time
    df = df_weekly.copy()
    if 'week_start' in df.columns:
        df = df.sort_values('week_start')
    
    # Default metrics
    if metrics is None:
        metrics = ["estimated_units", "sales_rank", "filled_price", "weekly_revenue"]
    
    # Filter to available
    metrics = [m for m in metrics if m in df.columns]
    signal.metrics_forecasted = len(metrics)
    
    bullish_count = 0
    bearish_count = 0
    
    for metric in metrics:
        try:
            forecast = _forecast_metric(df, metric, horizon_weeks, seasonal_period)
            signal.forecasts[metric] = forecast
            
            # Also get seasonal decomposition
            signal.seasonality[metric] = _decompose_seasonality(df, metric, seasonal_period)
            
            # Track outlook
            change = forecast.expected_change_pct()
            direction = get_target_direction(metric)
            if change:
                if direction == "lower_is_better":
                    if change < -0.05:
                        bullish_count += 1
                    elif change > 0.05:
                        bearish_count += 1
                else:
                    if change > 0.05:
                        bullish_count += 1
                    elif change < -0.05:
                        bearish_count += 1
            
            # Generate key prediction
            if change and abs(change) > 0.10:
                direction_word = "improve" if (
                    (direction == "lower_is_better" and change < 0) or
                    (direction != "lower_is_better" and change > 0)
                ) else "decline"
                signal.key_predictions.append(
                    f"{metric} expected to {direction_word} {abs(change):.0%} over {horizon_weeks} weeks"
                )
                
        except Exception as e:
            signal.warnings.append(f"FORECAST_FAILED_{metric.upper()}: {str(e)}")
    
    # Determine overall outlook
    if bullish_count > bearish_count and bullish_count >= 2:
        signal.overall_outlook = "BULLISH"
    elif bearish_count > bullish_count and bearish_count >= 2:
        signal.overall_outlook = "BEARISH"
    elif bullish_count == 0 and bearish_count == 0:
        signal.overall_outlook = "NEUTRAL"
    else:
        signal.overall_outlook = "MIXED"
    
    # Determine confidence
    avg_mape = np.mean([
        f.mape for f in signal.forecasts.values() 
        if f.mape is not None
    ])
    if np.isnan(avg_mape):
        signal.confidence_level = "LOW"
    elif avg_mape < 0.10:
        signal.confidence_level = "HIGH"
    elif avg_mape < 0.25:
        signal.confidence_level = "MEDIUM"
    else:
        signal.confidence_level = "LOW"
    
    return signal


def _forecast_metric(
    df: pd.DataFrame,
    metric: str,
    horizon: int,
    seasonal_period: int
) -> Forecast:
    """Generate forecast for a single metric using Holt-Winters."""
    forecast = Forecast(metric=metric, horizon_weeks=horizon)
    
    series = df[metric].dropna()
    if len(series) < 4:
        forecast.model_type = "INSUFFICIENT_DATA"
        return forecast
    
    forecast.current_value = float(series.iloc[-1])
    
    # Try different models in order of complexity
    try:
        return _try_holt_winters(series, forecast, horizon, seasonal_period)
    except:
        pass
    
    try:
        return _try_exponential_smoothing(series, forecast, horizon)
    except:
        pass
    
    # Fallback to simple trend
    return _simple_trend_forecast(series, forecast, horizon)


def _try_holt_winters(
    series: pd.Series,
    forecast: Forecast,
    horizon: int,
    seasonal_period: int
) -> Forecast:
    """Try Holt-Winters Exponential Smoothing."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Need at least 2 full seasonal cycles
    if len(series) < seasonal_period * 2:
        raise ValueError("Insufficient data for seasonal model")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Fit Holt-Winters with additive seasonality
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_period,
            initialization_method='estimated'
        )
        fitted = model.fit(optimized=True)
        
        # Generate forecast
        predictions = fitted.forecast(horizon)
        forecast.forecasts = [float(x) for x in predictions]
        
        # Confidence intervals (approximate using residual std)
        residuals = series - fitted.fittedvalues
        std_error = float(residuals.std())
        
        forecast.lower_bound = [max(0, p - 1.96 * std_error) for p in forecast.forecasts]
        forecast.upper_bound = [p + 1.96 * std_error for p in forecast.forecasts]
        
        # Calculate MAPE on in-sample
        mape = np.mean(np.abs((series - fitted.fittedvalues) / series.replace(0, np.nan)))
        forecast.mape = float(mape) if not np.isnan(mape) else None
        forecast.mae = float(np.mean(np.abs(residuals)))
        
        # Determine trend from smoothed level
        if fitted.level.iloc[-1] > fitted.level.iloc[-min(4, len(fitted.level))]:
            forecast.trend_direction = "UP"
        elif fitted.level.iloc[-1] < fitted.level.iloc[-min(4, len(fitted.level))]:
            forecast.trend_direction = "DOWN"
        else:
            forecast.trend_direction = "STABLE"
        
        forecast.model_type = "HOLT_WINTERS"
        forecast.forecast_summary = f"Holt-Winters forecast with {seasonal_period}-week seasonality"
    
    return forecast


def _try_exponential_smoothing(
    series: pd.Series,
    forecast: Forecast,
    horizon: int
) -> Forecast:
    """Try simple exponential smoothing with trend."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Fit Holt's linear trend model (no seasonality)
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal=None,
            initialization_method='estimated'
        )
        fitted = model.fit(optimized=True)
        
        predictions = fitted.forecast(horizon)
        forecast.forecasts = [float(x) for x in predictions]
        
        residuals = series - fitted.fittedvalues
        std_error = float(residuals.std())
        
        forecast.lower_bound = [max(0, p - 1.96 * std_error) for p in forecast.forecasts]
        forecast.upper_bound = [p + 1.96 * std_error for p in forecast.forecasts]
        
        mape = np.mean(np.abs((series - fitted.fittedvalues) / series.replace(0, np.nan)))
        forecast.mape = float(mape) if not np.isnan(mape) else None
        forecast.mae = float(np.mean(np.abs(residuals)))
        
        if fitted.level.iloc[-1] > fitted.level.iloc[-min(4, len(fitted.level))]:
            forecast.trend_direction = "UP"
        elif fitted.level.iloc[-1] < fitted.level.iloc[-min(4, len(fitted.level))]:
            forecast.trend_direction = "DOWN"
        else:
            forecast.trend_direction = "STABLE"
        
        forecast.model_type = "EXPONENTIAL_SMOOTHING"
        forecast.forecast_summary = "Exponential smoothing with linear trend"
    
    return forecast


def _simple_trend_forecast(
    series: pd.Series,
    forecast: Forecast,
    horizon: int
) -> Forecast:
    """Fallback: simple linear trend extrapolation."""
    from scipy import stats
    
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    
    # Generate forecasts
    future_x = np.arange(len(series), len(series) + horizon)
    predictions = intercept + slope * future_x
    
    forecast.forecasts = [float(max(0, p)) for p in predictions]
    
    # Confidence intervals
    residuals = series - (intercept + slope * x)
    std_error = float(residuals.std())
    
    forecast.lower_bound = [max(0, p - 1.96 * std_error) for p in forecast.forecasts]
    forecast.upper_bound = [p + 1.96 * std_error for p in forecast.forecasts]
    
    forecast.mae = float(np.mean(np.abs(residuals)))
    mape = np.mean(np.abs(residuals / series.replace(0, np.nan)))
    forecast.mape = float(mape) if not np.isnan(mape) else None
    
    if slope > 0:
        forecast.trend_direction = "UP"
    elif slope < 0:
        forecast.trend_direction = "DOWN"
    else:
        forecast.trend_direction = "STABLE"
    
    forecast.trend_strength = float(abs(r_value))
    forecast.model_type = "LINEAR_TREND"
    forecast.forecast_summary = "Linear trend extrapolation (fallback model)"
    
    return forecast


def _decompose_seasonality(
    df: pd.DataFrame,
    metric: str,
    period: int
) -> SeasonalDecomposition:
    """Decompose time series to extract seasonal patterns."""
    result = SeasonalDecomposition(metric=metric)
    
    series = df[metric].dropna()
    if len(series) < period * 2:
        return result
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(
            series,
            period=period,
            extrapolate_trend='freq'
        )
        
        seasonal = decomposition.seasonal.dropna()
        
        if len(seasonal) > 0:
            # Check if seasonality is significant
            seasonal_var = seasonal.var()
            total_var = series.var()
            
            if total_var > 0:
                result.seasonal_strength = float(seasonal_var / total_var)
                
                if result.seasonal_strength > 0.1:  # 10% of variance
                    result.has_seasonality = True
                    result.seasonal_period = period
                    
                    # Find peak and trough weeks within the period
                    if len(seasonal) >= period:
                        period_pattern = seasonal.iloc[-period:]
                        peak_idx = int(period_pattern.argmax())
                        trough_idx = int(period_pattern.argmin())
                        result.peak_weeks = [peak_idx + 1]  # 1-indexed
                        result.trough_weeks = [trough_idx + 1]
                        
    except Exception:
        pass
    
    return result
