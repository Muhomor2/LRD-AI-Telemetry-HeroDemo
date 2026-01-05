"""
LRD-AI-Telemetry: Seismic Precursors of LLM Degradation

Applies fractal/seismic methodology (LRD, Hurst, QO3/FIO) to detect
early-warning signals in long-form LLM generation.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from .metrics import (
    token_entropy_from_logits,
    repetition_rate,
    hurst_exponent,
    rolling_hurst,
    entropy_deficit_sid,
)
from .generate import generate_with_metrics, RunResult

__version__ = "1.0.0"
__author__ = "Igor Chechelnitsky"

__all__ = [
    "token_entropy_from_logits",
    "repetition_rate", 
    "hurst_exponent",
    "rolling_hurst",
    "entropy_deficit_sid",
    "generate_with_metrics",
    "RunResult",
]
