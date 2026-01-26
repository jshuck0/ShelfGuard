# src/analyst/tools/volatility.py

"""
VOLATILITY SENSOR - Anomaly Detection Tool

Detects unusual movements in metrics using statistical methods:
- Z-Score detection for normally distributed metrics
- IQR (Interquartile Range) for robust outlier detection
- Rolling volatility for regime change detection

Output: Structured AnomalySignal for the Orchestrator
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..config import KEEPA_CONFIG, get_target_direction


@dataclass
class Anomaly:
    """A single detected anomaly."""
    metric: str
    week: str  # Week identifier
    value: float
    expected_value: float
    deviation: float  # In standard deviations or IQR units
    direction: str  # "SPIKE" or "CRASH"
    severity: str  # "MINOR", "MODERATE", "MAJOR", "EXTREME"
    is_good: bool  # Based on config directionality
    context: str  # Human-readable explanation


@dataclass
class VolatilityProfile:
    """Volatility characteristics for a metric."""
    metric: str
    current_volatility: float  # Recent std dev
    historical_volatility: float  # Long-term std dev
    regime: str  # "STABLE", "VOLATILE", "EXTREME"
    regime_change_detected: bool
    regime_change_week: Optional[str] = None


@dataclass
class AnomalySignal:
    """
    Complete anomaly detection output.
    This is what gets passed to the Orchestrator.
    """
    asin: str
    analysis_timestamp: str
    
    # Detected anomalies by metric
    anomalies: List[Anomaly] = field(default_factory=list)
    
    # Volatility profiles
    volatility_profiles: Dict[str, VolatilityProfile] = field(default_factory=dict)
    
    # Summary stats
    total_anomalies: int = 0
    critical_anomalies: int = 0
    metrics_analyzed: int = 0
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    # Warnings (for consistency with other signals)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "timestamp": self.analysis_timestamp,
            "summary": {
                "total_anomalies": self.total_anomalies,
                "critical_anomalies": self.critical_anomalies,
                "metrics_analyzed": self.metrics_analyzed,
            },
            "anomalies": [
                {
                    "metric": a.metric,
                    "week": a.week,
                    "value": a.value,
                    "expected": a.expected_value,
                    "deviation": a.deviation,
                    "direction": a.direction,
                    "severity": a.severity,
                    "is_good": a.is_good,
                    "context": a.context,
                }
                for a in self.anomalies
            ],
            "volatility": {
                metric: {
                    "current": vp.current_volatility,
                    "historical": vp.historical_volatility,
                    "regime": vp.regime,
                    "regime_change": vp.regime_change_detected,
                }
                for metric, vp in self.volatility_profiles.items()
            },
            "alerts": self.alerts,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"=== ANOMALY DETECTION FOR {self.asin} ==="]
        lines.append(f"Found {self.total_anomalies} anomalies ({self.critical_anomalies} critical)")
        lines.append("")
        
        if self.anomalies:
            lines.append("ANOMALIES DETECTED:")
            for a in sorted(self.anomalies, key=lambda x: x.deviation, reverse=True)[:10]:
                icon = "🚨" if a.severity in ["MAJOR", "EXTREME"] else "⚠️"
                good_bad = "GOOD" if a.is_good else "BAD"
                lines.append(f"  {icon} {a.metric} {a.direction} ({a.severity}, {good_bad})")
                lines.append(f"     Week: {a.week}, Value: {a.value:.2f} (Expected: {a.expected_value:.2f})")
                lines.append(f"     {a.context}")
        else:
            lines.append("No significant anomalies detected.")
        
        lines.append("")
        lines.append("VOLATILITY REGIMES:")
        for metric, vp in self.volatility_profiles.items():
            change = " ⚡ REGIME CHANGE" if vp.regime_change_detected else ""
            lines.append(f"  - {metric}: {vp.regime}{change}")
        
        if self.alerts:
            lines.append("")
            # Defensive: ensure all alerts are strings
            alert_strs = [str(a) if not isinstance(a, str) else a for a in self.alerts]
            lines.append("ALERTS: " + " | ".join(alert_strs))
        
        return "\n".join(lines)


def detect_anomalies(
    df_weekly: pd.DataFrame,
    asin: str = "UNKNOWN",
    metrics: Optional[List[str]] = None,
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5
) -> AnomalySignal:
    """
    Detect anomalies in the time series data.
    
    Args:
        df_weekly: Weekly time series data
        asin: ASIN identifier
        metrics: List of metrics to check (defaults to all targets)
        z_threshold: Z-score threshold for anomaly detection
        iqr_multiplier: IQR multiplier for robust detection
        
    Returns:
        AnomalySignal with all detected anomalies
    """
    signal = AnomalySignal(
        asin=asin,
        analysis_timestamp=datetime.now().isoformat()
    )
    
    if df_weekly is None or len(df_weekly) < 4:
        signal.alerts.append("INSUFFICIENT_DATA")
        return signal
    
    # Sort by time
    df = df_weekly.copy()
    if 'week_start' in df.columns:
        df = df.sort_values('week_start')
    
    # Default to target metrics from config
    if metrics is None:
        metrics = [
            cfg.get("col", name) 
            for name, cfg in KEEPA_CONFIG["TARGETS"].items()
        ]
    
    # Filter to available columns
    metrics = [m for m in metrics if m in df.columns]
    signal.metrics_analyzed = len(metrics)
    
    for metric in metrics:
        # Detect anomalies using both methods
        z_anomalies = _detect_z_score_anomalies(df, metric, z_threshold)
        iqr_anomalies = _detect_iqr_anomalies(df, metric, iqr_multiplier)
        
        # Merge and deduplicate (keep the more severe)
        all_anomalies = _merge_anomalies(z_anomalies, iqr_anomalies)
        signal.anomalies.extend(all_anomalies)
        
        # Calculate volatility profile
        signal.volatility_profiles[metric] = _calculate_volatility_profile(df, metric)
    
    # Update summary
    signal.total_anomalies = len(signal.anomalies)
    signal.critical_anomalies = sum(
        1 for a in signal.anomalies if a.severity in ["MAJOR", "EXTREME"]
    )
    
    # Generate alerts
    if signal.critical_anomalies > 0:
        signal.alerts.append(f"{signal.critical_anomalies} CRITICAL ANOMALIES DETECTED")
    
    regime_changes = [m for m, v in signal.volatility_profiles.items() if v.regime_change_detected]
    if regime_changes:
        signal.alerts.append(f"REGIME CHANGE in: {', '.join(regime_changes)}")
    
    return signal


def _detect_z_score_anomalies(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 2.5
) -> List[Anomaly]:
    """Detect anomalies using Z-score method."""
    anomalies = []
    
    series = df[metric].dropna()
    if len(series) < 4:
        return anomalies
    
    # Calculate rolling statistics for context-aware detection
    window = min(len(series) // 2, 8)
    rolling_mean = series.rolling(window=window, min_periods=2).mean()
    rolling_std = series.rolling(window=window, min_periods=2).std()
    
    # Calculate Z-scores
    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    # Get direction config
    direction = get_target_direction(metric)
    
    # Find anomalies
    for idx in series.index:
        if pd.isna(z_scores.loc[idx]):
            continue
            
        z = z_scores.loc[idx]
        if abs(z) > threshold:
            value = series.loc[idx]
            expected = rolling_mean.loc[idx]
            
            # Determine if spike or crash
            if z > 0:
                anomaly_direction = "SPIKE"
            else:
                anomaly_direction = "CRASH"
            
            # Determine if good or bad based on config
            if direction == "lower_is_better":
                is_good = (anomaly_direction == "CRASH")  # Lower is better
            else:
                is_good = (anomaly_direction == "SPIKE")  # Higher is better
            
            # Severity based on Z-score magnitude
            severity = _classify_severity(abs(z), threshold)
            
            # Get week string
            week_str = str(df.loc[idx, 'week_start']) if 'week_start' in df.columns else str(idx)
            
            anomalies.append(Anomaly(
                metric=metric,
                week=week_str,
                value=float(value),
                expected_value=float(expected),
                deviation=float(abs(z)),
                direction=anomaly_direction,
                severity=severity,
                is_good=is_good,
                context=f"{metric} is {abs(z):.1f} std devs {'above' if z > 0 else 'below'} expected"
            ))
    
    return anomalies


def _detect_iqr_anomalies(
    df: pd.DataFrame,
    metric: str,
    multiplier: float = 1.5
) -> List[Anomaly]:
    """Detect anomalies using IQR (robust to non-normal distributions)."""
    anomalies = []
    
    series = df[metric].dropna()
    if len(series) < 4:
        return anomalies
    
    # Calculate IQR bounds
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        return anomalies  # No variance
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Get direction config
    direction = get_target_direction(metric)
    median = series.median()
    
    # Find outliers
    for idx in series.index:
        value = series.loc[idx]
        
        if value < lower_bound or value > upper_bound:
            # Calculate deviation in IQR units
            if value > upper_bound:
                deviation = (value - Q3) / IQR
                anomaly_direction = "SPIKE"
            else:
                deviation = (Q1 - value) / IQR
                anomaly_direction = "CRASH"
            
            # Determine if good or bad
            if direction == "lower_is_better":
                is_good = (anomaly_direction == "CRASH")
            else:
                is_good = (anomaly_direction == "SPIKE")
            
            severity = _classify_severity(deviation, 1.5)
            week_str = str(df.loc[idx, 'week_start']) if 'week_start' in df.columns else str(idx)
            
            anomalies.append(Anomaly(
                metric=metric,
                week=week_str,
                value=float(value),
                expected_value=float(median),
                deviation=float(deviation),
                direction=anomaly_direction,
                severity=severity,
                is_good=is_good,
                context=f"{metric} is {deviation:.1f} IQR units from typical range"
            ))
    
    return anomalies


def _merge_anomalies(z_anomalies: List[Anomaly], iqr_anomalies: List[Anomaly]) -> List[Anomaly]:
    """Merge anomalies from both methods, keeping the more severe."""
    merged = {}
    
    for a in z_anomalies:
        key = (a.metric, a.week)
        merged[key] = a
    
    for a in iqr_anomalies:
        key = (a.metric, a.week)
        if key in merged:
            # Keep the more severe
            existing = merged[key]
            severity_order = {"MINOR": 1, "MODERATE": 2, "MAJOR": 3, "EXTREME": 4}
            if severity_order.get(a.severity, 0) > severity_order.get(existing.severity, 0):
                merged[key] = a
        else:
            merged[key] = a
    
    return list(merged.values())


def _classify_severity(deviation: float, base_threshold: float) -> str:
    """Classify anomaly severity based on deviation magnitude."""
    ratio = deviation / base_threshold
    
    if ratio >= 3.0:
        return "EXTREME"
    elif ratio >= 2.0:
        return "MAJOR"
    elif ratio >= 1.5:
        return "MODERATE"
    else:
        return "MINOR"


def _calculate_volatility_profile(df: pd.DataFrame, metric: str) -> VolatilityProfile:
    """Calculate volatility characteristics for a metric."""
    series = df[metric].dropna()
    
    if len(series) < 4:
        return VolatilityProfile(
            metric=metric,
            current_volatility=0.0,
            historical_volatility=0.0,
            regime="UNKNOWN",
            regime_change_detected=False
        )
    
    # Calculate returns/changes for volatility
    pct_changes = series.pct_change().dropna()
    
    if len(pct_changes) < 2:
        return VolatilityProfile(
            metric=metric,
            current_volatility=0.0,
            historical_volatility=0.0,
            regime="UNKNOWN",
            regime_change_detected=False
        )
    
    # Historical volatility (full period)
    historical_vol = float(pct_changes.std())
    
    # Recent volatility (last 4 weeks or half the data)
    recent_window = min(4, len(pct_changes) // 2)
    current_vol = float(pct_changes.tail(recent_window).std())
    
    # Classify regime
    if current_vol < 0.05:
        regime = "STABLE"
    elif current_vol < 0.15:
        regime = "MODERATE"
    elif current_vol < 0.30:
        regime = "VOLATILE"
    else:
        regime = "EXTREME"
    
    # Detect regime change (volatility doubled or halved)
    regime_change = False
    regime_change_week = None
    
    if historical_vol > 0:
        vol_ratio = current_vol / historical_vol
        if vol_ratio > 2.0 or vol_ratio < 0.5:
            regime_change = True
            # Find when the change occurred
            rolling_vol = pct_changes.rolling(window=recent_window).std()
            for i in range(len(rolling_vol) - 1, recent_window, -1):
                if abs(rolling_vol.iloc[i] - rolling_vol.iloc[i-1]) > historical_vol:
                    if 'week_start' in df.columns:
                        regime_change_week = str(df.iloc[i]['week_start'])
                    break
    
    return VolatilityProfile(
        metric=metric,
        current_volatility=current_vol,
        historical_volatility=historical_vol,
        regime=regime,
        regime_change_detected=regime_change,
        regime_change_week=regime_change_week
    )
