# src/analyst/tools/__init__.py

"""
Analytical Tools for the Self-Driving Analyst

These tools perform deterministic Python calculations and return
structured JSON briefs to the Orchestrator (LLM).

Architecture:
- Each tool reads data + config
- Each tool outputs a DiagnosticSignal
- The Orchestrator combines signals into a narrative
"""

from .calibrator import calibrate_physics, CalibratedPhysics
from .volatility import detect_anomalies, AnomalySignal
from .prediction import forecast_metrics, ForecastSignal
from .causal import analyze_causality, CausalSignal
from .cluster import segment_products, ClusterSignal

__all__ = [
    "calibrate_physics",
    "CalibratedPhysics",
    "detect_anomalies",
    "AnomalySignal",
    "forecast_metrics",
    "ForecastSignal",
    "analyze_causality",
    "CausalSignal",
    "segment_products",
    "ClusterSignal",
]
