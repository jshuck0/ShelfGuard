# src/analyst/tools/causal.py

"""
CAUSAL SENSOR - Causality Analysis Tool

Answers the "WHY" question by testing causal relationships:
- Granger Causality tests (does X predict Y?)
- Lagged regression analysis
- Validates causal chains from config.py

Output: Structured CausalSignal for the Orchestrator
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

from ..config import KEEPA_CONFIG, get_target_direction


@dataclass
class CausalRelationship:
    """A detected or validated causal relationship."""
    cause_metric: str
    effect_metric: str
    
    # Direction and strength
    relationship_type: str  # "POSITIVE", "NEGATIVE", "NONE"
    strength: float  # Correlation or regression coefficient
    p_value: Optional[float] = None
    is_significant: bool = False
    
    # Granger causality
    granger_f_stat: Optional[float] = None
    granger_p_value: Optional[float] = None
    optimal_lag: int = 1
    
    # Interpretation
    interpretation: str = ""
    confidence: str = "LOW"  # "HIGH", "MEDIUM", "LOW"
    
    # Match to config
    matches_config_chain: Optional[str] = None  # Name of matching causal chain


@dataclass
class ChainValidation:
    """Validation result for a causal chain from config."""
    chain_name: str
    chain_desc: str
    
    is_validated: bool = False
    observed_cause: bool = False  # Did the cause event occur?
    observed_effect: bool = False  # Did the effect occur?
    
    expected_direction: str = ""
    actual_direction: str = ""
    
    confidence: str = "LOW"
    explanation: str = ""


@dataclass
class CausalSignal:
    """
    Complete causal analysis output.
    This is what gets passed to the Orchestrator.
    """
    asin: str
    analysis_timestamp: str
    
    # Discovered relationships
    relationships: List[CausalRelationship] = field(default_factory=list)
    
    # Validated chains from config
    chain_validations: Dict[str, ChainValidation] = field(default_factory=dict)
    
    # Primary drivers (strongest causal links)
    primary_drivers: List[str] = field(default_factory=list)
    
    # Summary
    relationships_tested: int = 0
    significant_relationships: int = 0
    validated_chains: int = 0
    
    # Key findings
    key_findings: List[str] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asin": self.asin,
            "timestamp": self.analysis_timestamp,
            "summary": {
                "relationships_tested": self.relationships_tested,
                "significant_found": self.significant_relationships,
                "chains_validated": self.validated_chains,
            },
            "relationships": [
                {
                    "cause": r.cause_metric,
                    "effect": r.effect_metric,
                    "type": r.relationship_type,
                    "strength": r.strength,
                    "significant": r.is_significant,
                    "optimal_lag": r.optimal_lag,
                    "confidence": r.confidence,
                }
                for r in self.relationships
            ],
            "chain_validations": {
                name: {
                    "validated": v.is_validated,
                    "cause_observed": v.observed_cause,
                    "effect_observed": v.observed_effect,
                    "confidence": v.confidence,
                }
                for name, v in self.chain_validations.items()
            },
            "primary_drivers": self.primary_drivers,
            "key_findings": self.key_findings,
            "warnings": self.warnings,
        }
    
    def to_prompt_string(self) -> str:
        """Format for LLM injection."""
        lines = [f"=== CAUSAL ANALYSIS FOR {self.asin} ==="]
        lines.append(f"Tested {self.relationships_tested} relationships, found {self.significant_relationships} significant")
        lines.append("")
        
        if self.primary_drivers:
            lines.append("PRIMARY DRIVERS OF PERFORMANCE:")
            for driver in self.primary_drivers:
                lines.append(f"  • {driver}")
        
        lines.append("")
        lines.append("SIGNIFICANT RELATIONSHIPS:")
        for r in sorted(self.relationships, key=lambda x: abs(x.strength), reverse=True):
            if r.is_significant:
                icon = "↑" if r.relationship_type == "POSITIVE" else "↓"
                lines.append(f"  {icon} {r.cause_metric} → {r.effect_metric}")
                lines.append(f"    {r.interpretation} (Confidence: {r.confidence})")
        
        if self.chain_validations:
            lines.append("")
            lines.append("CAUSAL CHAIN VALIDATIONS:")
            for name, v in self.chain_validations.items():
                status = "✅ VALIDATED" if v.is_validated else "❌ NOT VALIDATED"
                lines.append(f"  {status}: {v.chain_desc}")
                if v.explanation:
                    lines.append(f"    → {v.explanation}")
        
        if self.key_findings:
            lines.append("")
            lines.append("KEY FINDINGS:")
            for finding in self.key_findings:
                lines.append(f"  • {finding}")
        
        if self.warnings:
            lines.append("")
            # Defensive: ensure all warnings are strings
            warn_strs = [str(w) if not isinstance(w, str) else w for w in self.warnings]
            lines.append("WARNINGS: " + " | ".join(warn_strs))
        
        return "\n".join(lines)


def analyze_causality(
    df_weekly: pd.DataFrame,
    asin: str = "UNKNOWN",
    target_metrics: Optional[List[str]] = None,
    lever_metrics: Optional[List[str]] = None,
    max_lag: int = 4
) -> CausalSignal:
    """
    Analyze causal relationships in the data.
    
    Args:
        df_weekly: Weekly time series data
        asin: ASIN identifier
        target_metrics: Outcome metrics to analyze
        lever_metrics: Potential driver metrics
        max_lag: Maximum lag to test in Granger causality
        
    Returns:
        CausalSignal with causal analysis results
    """
    signal = CausalSignal(
        asin=asin,
        analysis_timestamp=datetime.now().isoformat()
    )
    
    if df_weekly is None or len(df_weekly) < 8:
        signal.warnings.append("INSUFFICIENT_DATA_FOR_CAUSAL_ANALYSIS")
        return signal
    
    # Sort by time
    df = df_weekly.copy()
    if 'week_start' in df.columns:
        df = df.sort_values('week_start')
    
    # Get metrics from config if not specified
    if target_metrics is None:
        target_metrics = [
            cfg.get("col", name)
            for name, cfg in KEEPA_CONFIG["TARGETS"].items()
        ]
    
    if lever_metrics is None:
        lever_metrics = [
            cfg.get("col", name)
            for name, cfg in KEEPA_CONFIG["LEVERS"].items()
        ]
    
    # Filter to available columns
    target_metrics = [m for m in target_metrics if m in df.columns]
    lever_metrics = [m for m in lever_metrics if m in df.columns]
    
    # Test all lever -> target relationships
    for target in target_metrics:
        for lever in lever_metrics:
            if target == lever:
                continue
            
            signal.relationships_tested += 1
            
            try:
                relationship = _test_causal_relationship(df, lever, target, max_lag)
                if relationship:
                    signal.relationships.append(relationship)
                    if relationship.is_significant:
                        signal.significant_relationships += 1
            except Exception as e:
                signal.warnings.append(f"FAILED_{lever}_{target}: {str(e)[:50]}")
    
    # Validate causal chains from config
    for chain_name, chain_config in KEEPA_CONFIG.get("CAUSAL_CHAINS", {}).items():
        validation = _validate_causal_chain(df, chain_name, chain_config)
        signal.chain_validations[chain_name] = validation
        if validation.is_validated:
            signal.validated_chains += 1
    
    # Identify primary drivers
    signal.primary_drivers = _identify_primary_drivers(signal.relationships, target_metrics)
    
    # Generate key findings
    signal.key_findings = _generate_key_findings(signal)
    
    return signal


def _test_causal_relationship(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int
) -> Optional[CausalRelationship]:
    """Test if cause_col Granger-causes effect_col."""
    
    cause = df[cause_col].dropna()
    effect = df[effect_col].dropna()
    
    # Align series
    common_idx = cause.index.intersection(effect.index)
    if len(common_idx) < 10:
        return None
    
    cause = cause.loc[common_idx]
    effect = effect.loc[common_idx]
    
    relationship = CausalRelationship(
        cause_metric=cause_col,
        effect_metric=effect_col
    )
    
    # 1. Simple correlation
    try:
        from scipy.stats import pearsonr, spearmanr
        
        corr, p_val = pearsonr(cause, effect)
        relationship.strength = float(corr)
        relationship.p_value = float(p_val)
        relationship.is_significant = p_val < 0.10
        
        if corr > 0.1:
            relationship.relationship_type = "POSITIVE"
        elif corr < -0.1:
            relationship.relationship_type = "NEGATIVE"
        else:
            relationship.relationship_type = "NONE"
            
    except:
        return None
    
    # 2. Granger causality test
    try:
        relationship = _run_granger_test(cause, effect, relationship, max_lag)
    except:
        pass  # Granger test optional
    
    # 3. Lagged correlation to find optimal lag
    try:
        optimal_lag, lag_corr = _find_optimal_lag(cause, effect, max_lag)
        relationship.optimal_lag = optimal_lag
        if abs(lag_corr) > abs(relationship.strength):
            relationship.strength = float(lag_corr)
    except:
        pass
    
    # Set confidence
    if relationship.granger_p_value is not None and relationship.granger_p_value < 0.05:
        relationship.confidence = "HIGH"
        relationship.is_significant = True
    elif relationship.p_value is not None and relationship.p_value < 0.10:
        relationship.confidence = "MEDIUM"
    else:
        relationship.confidence = "LOW"
    
    # Generate interpretation
    relationship.interpretation = _interpret_relationship(relationship)
    
    # Check if matches a config chain
    relationship.matches_config_chain = _match_to_config_chain(
        cause_col, effect_col, relationship.relationship_type
    )
    
    return relationship


def _run_granger_test(
    cause: pd.Series,
    effect: pd.Series,
    relationship: CausalRelationship,
    max_lag: int
) -> CausalRelationship:
    """Run Granger causality test."""
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Combine into DataFrame
    data = pd.DataFrame({
        'effect': effect.values,
        'cause': cause.values
    }).dropna()
    
    if len(data) < max_lag + 10:
        return relationship
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Run Granger test
        # Note: grangercausalitytests expects [effect, cause] order
        results = grangercausalitytests(data[['effect', 'cause']], maxlag=max_lag, verbose=False)
        
        # Find best lag (lowest p-value)
        best_lag = 1
        best_p = 1.0
        best_f = 0.0
        
        for lag in range(1, max_lag + 1):
            if lag in results:
                f_stat = results[lag][0]['ssr_ftest'][0]
                p_val = results[lag][0]['ssr_ftest'][1]
                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
                    best_lag = lag
        
        relationship.granger_f_stat = float(best_f)
        relationship.granger_p_value = float(best_p)
        relationship.optimal_lag = best_lag
    
    return relationship


def _find_optimal_lag(cause: pd.Series, effect: pd.Series, max_lag: int) -> Tuple[int, float]:
    """Find the lag with the strongest correlation."""
    best_lag = 0
    best_corr = 0.0
    
    for lag in range(0, max_lag + 1):
        if lag >= len(cause) - 2:
            break
            
        lagged_cause = cause.shift(lag).dropna()
        aligned_effect = effect.loc[lagged_cause.index]
        
        if len(lagged_cause) < 5:
            continue
        
        try:
            corr = lagged_cause.corr(aligned_effect)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        except:
            pass
    
    return best_lag, best_corr


def _interpret_relationship(rel: CausalRelationship) -> str:
    """Generate human-readable interpretation."""
    if not rel.is_significant:
        return f"No significant relationship between {rel.cause_metric} and {rel.effect_metric}"
    
    direction = "increases" if rel.relationship_type == "POSITIVE" else "decreases"
    lag_text = f" (with {rel.optimal_lag}-week delay)" if rel.optimal_lag > 0 else ""
    
    return f"When {rel.cause_metric} increases, {rel.effect_metric} {direction}{lag_text}"


def _match_to_config_chain(cause: str, effect: str, rel_type: str) -> Optional[str]:
    """Check if relationship matches a causal chain from config."""
    for chain_name, chain_config in KEEPA_CONFIG.get("CAUSAL_CHAINS", {}).items():
        chain_cause = chain_config.get("cause", ("", "", 0))[0]
        chain_effect = chain_config.get("effect", ("", "", 0))[0]
        
        if cause == chain_cause and effect == chain_effect:
            return chain_name
    
    return None


def _validate_causal_chain(
    df: pd.DataFrame,
    chain_name: str,
    chain_config: Dict
) -> ChainValidation:
    """Validate whether a causal chain from config is present in the data."""
    validation = ChainValidation(
        chain_name=chain_name,
        chain_desc=chain_config.get("desc", chain_name)
    )
    
    cause_spec = chain_config.get("cause", ("", "", 0))
    effect_spec = chain_config.get("effect", ("", "", 0))
    
    if len(cause_spec) < 3 or len(effect_spec) < 3:
        validation.explanation = "Invalid chain specification"
        return validation
    
    cause_col, cause_direction, cause_threshold = cause_spec
    effect_col, effect_direction, effect_threshold = effect_spec
    
    if cause_col not in df.columns or effect_col not in df.columns:
        validation.explanation = f"Missing columns: {cause_col} or {effect_col}"
        return validation
    
    # Check if cause event occurred recently
    cause_series = df[cause_col].dropna()
    if len(cause_series) < 4:
        validation.explanation = "Insufficient data"
        return validation
    
    # Calculate recent changes
    cause_change = cause_series.pct_change().iloc[-4:].mean()
    effect_change = df[effect_col].dropna().pct_change().iloc[-4:].mean()
    
    # Check cause direction
    if cause_direction == "increase":
        validation.observed_cause = cause_change > cause_threshold
    elif cause_direction == "decrease":
        validation.observed_cause = cause_change < -cause_threshold
    else:
        validation.observed_cause = abs(cause_change) < cause_threshold  # "stable"
    
    # Check effect direction
    validation.expected_direction = effect_direction
    if effect_direction == "increase":
        validation.observed_effect = effect_change > effect_threshold
        validation.actual_direction = "increase" if effect_change > 0 else "decrease"
    elif effect_direction == "decrease":
        validation.observed_effect = effect_change < -effect_threshold
        validation.actual_direction = "decrease" if effect_change < 0 else "increase"
    else:
        validation.observed_effect = abs(effect_change) < effect_threshold
        validation.actual_direction = "stable"
    
    # Validate chain
    if validation.observed_cause and validation.observed_effect:
        validation.is_validated = True
        validation.confidence = "HIGH"
        validation.explanation = f"Chain validated: {cause_col} {cause_direction} → {effect_col} {effect_direction}"
    elif validation.observed_cause and not validation.observed_effect:
        validation.confidence = "MEDIUM"
        validation.explanation = f"Cause observed ({cause_col} {cause_direction}) but effect not seen"
    else:
        validation.confidence = "LOW"
        validation.explanation = "Cause event not observed in recent data"
    
    return validation


def _identify_primary_drivers(
    relationships: List[CausalRelationship],
    targets: List[str]
) -> List[str]:
    """Identify the primary drivers of target metrics."""
    drivers = {}
    
    for rel in relationships:
        if rel.effect_metric in targets and rel.is_significant:
            key = rel.cause_metric
            if key not in drivers:
                drivers[key] = 0
            drivers[key] += abs(rel.strength)
    
    # Sort by total impact and return top 3
    sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_drivers[:3]]


def _generate_key_findings(signal: CausalSignal) -> List[str]:
    """Generate key findings from the analysis."""
    findings = []
    
    # Finding 1: Strongest relationships
    significant = [r for r in signal.relationships if r.is_significant]
    if significant:
        strongest = max(significant, key=lambda x: abs(x.strength))
        findings.append(f"Strongest driver: {strongest.cause_metric} → {strongest.effect_metric} (r={strongest.strength:.2f})")
    
    # Finding 2: Validated chains
    validated = [v for v in signal.chain_validations.values() if v.is_validated]
    if validated:
        findings.append(f"Validated causal patterns: {', '.join(v.chain_name for v in validated)}")
    
    # Finding 3: Broken chains (expected but not seen)
    broken = [
        v for v in signal.chain_validations.values() 
        if v.observed_cause and not v.observed_effect
    ]
    if broken:
        findings.append(f"Broken chains (cause without effect): {', '.join(v.chain_name for v in broken)}")
    
    # Finding 4: Primary drivers
    if signal.primary_drivers:
        findings.append(f"Primary performance drivers: {', '.join(signal.primary_drivers)}")
    
    return findings
