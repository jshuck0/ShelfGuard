# src/analyst/profiler.py

"""
PROFILER - The Triage Nurse

This is the first step in the Semantic Stack. It reads the raw data
USING config.py to produce a Vitals JSON that correctly interprets
"Good" vs "Bad" based on directionality.

Architecture:
- Step 0: Config (loaded) - tells us what metrics mean
- Step 1: Raw Data (input) - no meaning yet
- Step 2: Profiler (this) - produces interpreted Vitals
- Step 3: Tools - use Vitals to guide analysis
- Step 4: Orchestrator - synthesizes into narrative
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import warnings

from .config import KEEPA_CONFIG, is_improvement, calculate_change_severity, get_target_direction


@dataclass
class MetricVital:
    """Health status for a single metric."""
    name: str
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    change_pct: Optional[float] = None
    direction: str = "higher_is_better"  # From config
    status: str = "UNKNOWN"  # IMPROVING, DECLINING, STABLE, CRITICAL, UNKNOWN
    severity: str = "NEGLIGIBLE"  # NEGLIGIBLE, MINOR, MODERATE, MAJOR, CRITICAL
    is_good: Optional[bool] = None  # True = improving in the right direction
    alert: Optional[str] = None  # Alert message if any


@dataclass
class DataHealth:
    """Data quality assessment."""
    total_weeks: int = 0
    valid_weeks: int = 0
    completeness_pct: float = 0.0
    missing_columns: List[str] = field(default_factory=list)
    is_sufficient: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class StationarityResult:
    """Is the data stable enough to analyze?"""
    metric: str
    is_stationary: bool = False
    adf_statistic: Optional[float] = None
    p_value: Optional[float] = None
    interpretation: str = "UNKNOWN"


@dataclass
class SeasonalityResult:
    """Are there predictable patterns?"""
    metric: str
    has_seasonality: bool = False
    period_weeks: Optional[int] = None
    amplitude_pct: Optional[float] = None
    peak_weeks: List[int] = field(default_factory=list)


@dataclass
class DistributionResult:
    """Are there outliers?"""
    metric: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    outlier_count: int = 0
    outlier_indices: List[int] = field(default_factory=list)


@dataclass  
class TrendResult:
    """What direction is the product heading?"""
    metric: str
    direction: str = "STABLE"  # GROWING, DECLINING, STABLE
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    is_significant: bool = False


@dataclass
class ProfilerVitals:
    """
    Complete health assessment of the data.
    This is the output of the Profiler that feeds into the Tools.
    """
    asin: str
    profile_timestamp: str
    
    # Data Quality
    data_health: DataHealth = field(default_factory=DataHealth)
    
    # Metric-by-Metric Status
    target_vitals: Dict[str, MetricVital] = field(default_factory=dict)
    lever_vitals: Dict[str, MetricVital] = field(default_factory=dict)
    
    # Statistical Properties
    stationarity: Dict[str, StationarityResult] = field(default_factory=dict)
    seasonality: Dict[str, SeasonalityResult] = field(default_factory=dict)
    distributions: Dict[str, DistributionResult] = field(default_factory=dict)
    trends: Dict[str, TrendResult] = field(default_factory=dict)
    
    # Flags & Alerts
    active_flags: List[str] = field(default_factory=list)
    critical_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Overall Assessment
    overall_health: str = "UNKNOWN"  # HEALTHY, AT_RISK, CRITICAL, INSUFFICIENT_DATA
    confidence_score: float = 0.0  # 0-1 (how much can we trust the analysis)

    # Archetype Context (optional, set by caller)
    archetype: Optional[Any] = None  # ProductArchetype from src.models.product_archetype
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "timestamp": self.profile_timestamp,
            "data_health": {
                "weeks": self.data_health.total_weeks,
                "completeness": self.data_health.completeness_pct,
                "is_sufficient": self.data_health.is_sufficient,
                "warnings": self.data_health.warnings,
            },
            "target_vitals": {k: self._vital_to_dict(v) for k, v in self.target_vitals.items()},
            "lever_vitals": {k: self._vital_to_dict(v) for k, v in self.lever_vitals.items()},
            "trends": {k: {"direction": v.direction, "significant": v.is_significant} for k, v in self.trends.items()},
            "active_flags": self.active_flags,
            "critical_alerts": self.critical_alerts,
            "overall_health": self.overall_health,
            "confidence": self.confidence_score,
            "archetype": self.archetype.to_dict() if self.archetype and hasattr(self.archetype, 'to_dict') else None,
        }
    
    def _vital_to_dict(self, v: MetricVital) -> Dict:
        return {
            "current": v.current_value,
            "change_pct": v.change_pct,
            "status": v.status,
            "severity": v.severity,
            "is_good": v.is_good,
            "alert": v.alert,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"=== PROFILER VITALS FOR {self.asin} ==="]
        lines.append(f"Overall Health: {self.overall_health} (Confidence: {self.confidence_score:.0%})")
        lines.append(f"Data: {self.data_health.total_weeks} weeks, {self.data_health.completeness_pct:.0%} complete")
        lines.append("")
        
        lines.append("TARGET METRICS:")
        for name, vital in self.target_vitals.items():
            icon = "✅" if vital.is_good else "⚠️" if vital.severity in ["MINOR", "MODERATE"] else "🚨" if vital.severity in ["MAJOR", "CRITICAL"] else "➖"
            change_str = f"{vital.change_pct:+.1%}" if vital.change_pct else "N/A"
            lines.append(f"  {icon} {name}: {vital.status} ({change_str}) [{vital.severity}]")
        
        lines.append("")
        lines.append("TRENDS:")
        for name, trend in self.trends.items():
            lines.append(f"  - {name}: {trend.direction}")
        
        if self.critical_alerts:
            lines.append("")
            lines.append("🚨 CRITICAL ALERTS:")
            for alert in self.critical_alerts:
                lines.append(f"  - {alert}")
        
        if self.active_flags:
            lines.append("")
            # Defensive: ensure all flags are strings
            flag_strs = [str(f) if not isinstance(f, str) else f for f in self.active_flags]
            lines.append("FLAGS: " + ", ".join(flag_strs))

        # Include archetype context if available
        if self.archetype and hasattr(self.archetype, 'to_prompt_context'):
            lines.append("")
            lines.append(self.archetype.to_prompt_context())

        return "\n".join(lines)


def run_profiler(
    df_weekly: pd.DataFrame,
    asin: str = "UNKNOWN",
    archetype: Optional[Any] = None,
) -> ProfilerVitals:
    """
    Main entry point: Profile the data and produce Vitals.

    This runs through the checks defined in KEEPA_CONFIG["PROFILER_CHECKS"]
    and produces a comprehensive health assessment.

    Args:
        df_weekly: Weekly time series data for the product
        asin: ASIN identifier
        archetype: Optional ProductArchetype for context-aware interpretation

    Returns:
        ProfilerVitals with complete health assessment
    """
    from datetime import datetime

    vitals = ProfilerVitals(
        asin=asin,
        profile_timestamp=datetime.now().isoformat(),
        archetype=archetype,
    )
    
    # Step 1: Data Health Check
    vitals.data_health = _check_data_health(df_weekly)
    
    if not vitals.data_health.is_sufficient:
        vitals.overall_health = "INSUFFICIENT_DATA"
        vitals.confidence_score = 0.1
        return vitals
    
    # Sort data by time
    df = df_weekly.copy()
    if 'week_start' in df.columns:
        df = df.sort_values('week_start')
    
    # Step 2: Profile Target Metrics
    for metric_name, metric_config in KEEPA_CONFIG["TARGETS"].items():
        col = metric_config.get("col", metric_name)
        if col in df.columns:
            vital = _profile_metric(df, col, metric_config)
            vital.name = metric_name
            vitals.target_vitals[metric_name] = vital
    
    # Step 3: Profile Lever Metrics
    for lever_name, lever_config in KEEPA_CONFIG["LEVERS"].items():
        col = lever_config.get("col", lever_name)
        if col in df.columns:
            vital = _profile_metric(df, col, {"direction": "neutral"})
            vital.name = lever_name
            vitals.lever_vitals[lever_name] = vital
    
    # Step 4: Stationarity Tests (on key targets)
    for metric_name in ["sales_rank", "estimated_units"]:
        col = KEEPA_CONFIG["TARGETS"].get(metric_name, {}).get("col", metric_name)
        if col in df.columns:
            vitals.stationarity[metric_name] = _test_stationarity(df, col)
    
    # Step 5: Seasonality Detection
    for metric_name in ["estimated_units"]:
        col = KEEPA_CONFIG["TARGETS"].get(metric_name, {}).get("col", metric_name)
        if col in df.columns:
            vitals.seasonality[metric_name] = _detect_seasonality(df, col)
    
    # Step 6: Distribution Analysis
    for metric_name, metric_config in KEEPA_CONFIG["TARGETS"].items():
        col = metric_config.get("col", metric_name)
        if col in df.columns:
            vitals.distributions[metric_name] = _analyze_distribution(df, col)
    
    # Step 7: Trend Detection
    for metric_name, metric_config in KEEPA_CONFIG["TARGETS"].items():
        col = metric_config.get("col", metric_name)
        direction = metric_config.get("direction", "higher_is_better")
        if col in df.columns:
            vitals.trends[metric_name] = _detect_trend(df, col, direction)
    
    # Step 8: Check Flags
    vitals.active_flags = _check_flags(df)
    
    # Step 9: Generate Alerts
    vitals.critical_alerts = _generate_alerts(vitals, df)
    
    # Step 10: Overall Assessment
    vitals.overall_health, vitals.confidence_score = _assess_overall_health(vitals)
    
    return vitals


def _check_data_health(df: pd.DataFrame) -> DataHealth:
    """Check if we have enough clean data to analyze."""
    health = DataHealth()
    config = KEEPA_CONFIG["PROFILER_CHECKS"]["data_health"]
    
    if df is None or len(df) == 0:
        health.warnings.append("NO_DATA")
        return health
    
    health.total_weeks = len(df)
    
    # Check for required columns
    required_cols = ["sales_rank", "filled_price", "estimated_units"]
    health.missing_columns = [c for c in required_cols if c not in df.columns]
    
    # Calculate completeness
    available_cols = [c for c in required_cols if c in df.columns]
    if available_cols:
        total_cells = len(df) * len(available_cols)
        non_null = sum(df[col].notna().sum() for col in available_cols)
        health.completeness_pct = non_null / total_cells if total_cells > 0 else 0.0
    
    # Count valid weeks (at least sales_rank present)
    if "sales_rank" in df.columns:
        health.valid_weeks = df["sales_rank"].notna().sum()
    else:
        health.valid_weeks = health.total_weeks
    
    # Determine if sufficient
    min_weeks = config.get("min_weeks", 4)
    max_missing = config.get("max_missing_pct", 0.20)
    
    health.is_sufficient = (
        health.valid_weeks >= min_weeks and
        health.completeness_pct >= (1 - max_missing) and
        len(health.missing_columns) <= 1
    )
    
    if health.valid_weeks < min_weeks:
        health.warnings.append(f"INSUFFICIENT_WEEKS: {health.valid_weeks} < {min_weeks}")
    
    if health.completeness_pct < (1 - max_missing):
        health.warnings.append(f"LOW_COMPLETENESS: {health.completeness_pct:.0%}")
    
    return health


def _profile_metric(df: pd.DataFrame, col: str, config: Dict) -> MetricVital:
    """Profile a single metric column."""
    vital = MetricVital(name=col)
    vital.direction = config.get("direction", "higher_is_better")
    
    series = df[col].dropna()
    if len(series) < 2:
        vital.status = "UNKNOWN"
        return vital
    
    # Get current and previous values
    vital.current_value = float(series.iloc[-1])
    vital.previous_value = float(series.iloc[0])
    
    # Calculate change
    if vital.previous_value != 0:
        vital.change_pct = (vital.current_value - vital.previous_value) / abs(vital.previous_value)
    else:
        vital.change_pct = 0.0
    
    # Determine status using config directionality
    vital.is_good = is_improvement(col, vital.previous_value, vital.current_value)
    
    if vital.is_good:
        vital.status = "IMPROVING"
    elif vital.change_pct is not None:
        if abs(vital.change_pct) < 0.02:
            vital.status = "STABLE"
        else:
            vital.status = "DECLINING"
    else:
        vital.status = "STABLE"
    
    # Calculate severity
    if vital.change_pct is not None:
        vital.severity = calculate_change_severity(vital.change_pct)
    
    # Check for critical thresholds
    critical = config.get("critical_threshold")
    if critical is not None:
        if vital.direction == "higher_is_better" and vital.current_value < critical:
            vital.alert = f"{col} below critical threshold ({vital.current_value} < {critical})"
            vital.status = "CRITICAL"
        elif vital.direction == "lower_is_better" and vital.current_value > critical:
            vital.alert = f"{col} above critical threshold ({vital.current_value} > {critical})"
            vital.status = "CRITICAL"
    
    return vital


def _test_stationarity(df: pd.DataFrame, col: str) -> StationarityResult:
    """Test if the time series is stationary (stable mean/variance)."""
    result = StationarityResult(metric=col)
    
    series = df[col].dropna()
    if len(series) < 8:
        result.interpretation = "INSUFFICIENT_DATA"
        return result
    
    try:
        from statsmodels.tsa.stattools import adfuller
        
        adf_result = adfuller(series, autolag='AIC')
        result.adf_statistic = float(adf_result[0])
        result.p_value = float(adf_result[1])
        
        p_threshold = KEEPA_CONFIG["PROFILER_CHECKS"]["stationarity"]["p_threshold"]
        result.is_stationary = result.p_value < p_threshold
        
        if result.is_stationary:
            result.interpretation = "STATIONARY - Mean-reverting, predictable"
        else:
            result.interpretation = "NON_STATIONARY - Trending or random walk"
            
    except ImportError:
        result.interpretation = "STATSMODELS_NOT_AVAILABLE"
    except Exception as e:
        result.interpretation = f"ERROR: {str(e)}"
    
    return result


def _detect_seasonality(df: pd.DataFrame, col: str) -> SeasonalityResult:
    """Detect seasonal patterns."""
    result = SeasonalityResult(metric=col)
    
    series = df[col].dropna()
    if len(series) < 8:
        return result
    
    try:
        from statsmodels.tsa.stattools import acf
        
        config = KEEPA_CONFIG["PROFILER_CHECKS"]["seasonality"]
        period = config.get("period", 4)
        min_amplitude = config.get("min_amplitude", 0.10)
        
        max_lag = min(len(series) // 2, period + 2)
        if max_lag <= period:
            return result
            
        autocorr = acf(series, nlags=max_lag, fft=True)
        
        if autocorr[period] > 0.3:
            result.has_seasonality = True
            result.period_weeks = period
            result.amplitude_pct = float(series.std() / series.mean()) if series.mean() != 0 else 0.0
            
    except:
        pass
    
    return result


def _analyze_distribution(df: pd.DataFrame, col: str) -> DistributionResult:
    """Analyze distribution and detect outliers."""
    result = DistributionResult(metric=col)
    
    series = df[col].dropna()
    if len(series) < 4:
        return result
    
    result.mean = float(series.mean())
    result.std = float(series.std())
    result.min = float(series.min())
    result.max = float(series.max())
    
    try:
        result.skewness = float(stats.skew(series))
    except:
        pass
    
    # Detect outliers using IQR method
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    result.outlier_count = int(outliers.sum())
    result.outlier_indices = list(series[outliers].index)
    
    return result


def _detect_trend(df: pd.DataFrame, col: str, direction: str) -> TrendResult:
    """Detect trend direction."""
    result = TrendResult(metric=col)
    
    series = df[col].dropna()
    if len(series) < 4:
        return result
    
    x = np.arange(len(series))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        result.slope = float(slope)
        result.r_squared = float(r_value ** 2)
        result.is_significant = p_value < 0.10
        
        # Normalize slope for interpretation
        mean_val = series.mean()
        normalized_slope = slope / mean_val if mean_val != 0 else 0
        
        # For rank: negative slope = improving (GROWING)
        # For others: positive slope = improving (GROWING)
        if direction == "lower_is_better":
            if normalized_slope < -0.02 and result.is_significant:
                result.direction = "GROWING"
            elif normalized_slope > 0.02 and result.is_significant:
                result.direction = "DECLINING"
            else:
                result.direction = "STABLE"
        else:
            if normalized_slope > 0.02 and result.is_significant:
                result.direction = "GROWING"
            elif normalized_slope < -0.02 and result.is_significant:
                result.direction = "DECLINING"
            else:
                result.direction = "STABLE"
                
    except:
        result.direction = "UNKNOWN"
    
    return result


def _check_flags(df: pd.DataFrame) -> List[str]:
    """Check binary flags from config."""
    flags = []
    
    for flag_name, flag_config in KEEPA_CONFIG.get("FLAGS", {}).items():
        col = flag_config.get("col", flag_name)
        if col in df.columns:
            # Check most recent value
            latest = df[col].iloc[-1] if len(df) > 0 else None
            if latest:
                flags.append(flag_name.upper())
    
    return flags


def _generate_alerts(vitals: ProfilerVitals, df: pd.DataFrame) -> List[str]:
    """Generate critical alerts based on thresholds."""
    alerts = []
    thresholds = KEEPA_CONFIG.get("THRESHOLDS", {})
    
    # Check target vitals for critical status
    for name, vital in vitals.target_vitals.items():
        if vital.status == "CRITICAL":
            alerts.append(vital.alert or f"{name} in CRITICAL state")
        elif vital.severity == "CRITICAL":
            direction = "drop" if not vital.is_good else "surge"
            alerts.append(f"{name} {direction}: {vital.change_pct:+.0%}")
    
    # Check specific thresholds
    if "rating" in vitals.target_vitals:
        rating_vital = vitals.target_vitals["rating"]
        if rating_vital.current_value and rating_vital.current_value < thresholds.get("rating_danger_zone", 4.0):
            alerts.append(f"Rating below 4.0 danger zone: {rating_vital.current_value:.1f}")
    
    # Check OOS
    if "oos_pct_30" in df.columns:
        oos = df["oos_pct_30"].iloc[-1] if len(df) > 0 else 0
        if oos and oos > thresholds.get("oos_critical", 0.20):
            alerts.append(f"Critical OOS: {oos:.0%} in last 30 days")
        elif oos and oos > thresholds.get("oos_warning", 0.05):
            alerts.append(f"OOS Warning: {oos:.0%} in last 30 days")
    
    # Check Amazon dominance
    if "bb_stats_amazon_90" in df.columns:
        amz_share = df["bb_stats_amazon_90"].iloc[-1] if len(df) > 0 else 0
        if amz_share and amz_share > thresholds.get("amazon_threat_critical", 0.80):
            alerts.append(f"Amazon Dominance: {amz_share:.0%} Buy Box share")
        elif amz_share and amz_share > thresholds.get("amazon_threat_warning", 0.40):
            alerts.append(f"Amazon Competition: {amz_share:.0%} Buy Box share")
    
    return alerts


def _assess_overall_health(vitals: ProfilerVitals) -> Tuple[str, float]:
    """Determine overall health status and confidence score."""
    
    # Count critical/declining targets
    critical_count = 0
    declining_count = 0
    improving_count = 0
    
    for vital in vitals.target_vitals.values():
        if vital.status == "CRITICAL":
            critical_count += 1
        elif vital.status == "DECLINING":
            declining_count += 1
        elif vital.status == "IMPROVING":
            improving_count += 1
    
    # Determine overall health
    if critical_count > 0 or len(vitals.critical_alerts) > 2:
        health = "CRITICAL"
    elif declining_count > improving_count:
        health = "AT_RISK"
    elif improving_count > declining_count:
        health = "HEALTHY"
    else:
        health = "STABLE"
    
    # Calculate confidence score based on data quality
    confidence = vitals.data_health.completeness_pct
    
    # Adjust for data volume
    if vitals.data_health.total_weeks >= 12:
        confidence *= 1.0
    elif vitals.data_health.total_weeks >= 8:
        confidence *= 0.9
    elif vitals.data_health.total_weeks >= 4:
        confidence *= 0.7
    else:
        confidence *= 0.4
    
    # Adjust for missing columns
    confidence *= (1.0 - 0.1 * len(vitals.data_health.missing_columns))
    
    return health, max(0.0, min(1.0, confidence))
