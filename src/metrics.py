"""
Fractal and information-theoretic metrics for LLM telemetry.

Implements:
- Token entropy (Shannon)
- Repetition rate (loop detection)
- Hurst exponent (R/S analysis for LRD)
- SID (Seismic Information Deficit)

Based on QO3/FIO framework:
https://doi.org/10.5281/zenodo.18145167
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np
from scipy import stats
import torch


def token_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Token-level Shannon entropy for a single-step next-token distribution.
    
    Lower entropy = model is "too confident" (potential precursor to collapse)
    
    Args:
        logits: Tensor of shape [vocab_size]
    
    Returns:
        Entropy in nats
    """
    probs = torch.softmax(logits, dim=-1)
    # Add small epsilon to avoid log(0)
    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
    return float(ent)


def repetition_rate(token_ids: List[int], window: int = 200) -> float:
    """
    Simple loop metric: fraction of tokens in the last `window` 
    that have appeared earlier in the sequence.
    
    This is a LATE signal — by the time this rises, collapse is visible.
    
    Args:
        token_ids: Full sequence of generated token IDs
        window: Size of the tail window to check
    
    Returns:
        Fraction [0, 1] of repeated tokens
    """
    if len(token_ids) < 20:
        return 0.0
    
    tail = token_ids[-min(window, len(token_ids)):]
    seen_before = set(token_ids[:-len(tail)])
    
    if not seen_before:
        return 0.0
    
    rep = sum(1 for t in tail if t in seen_before) / len(tail)
    return float(rep)


def hurst_exponent(ts: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Hurst exponent via R/S (Rescaled Range) analysis.
    
    This is a KEY EARLY WARNING metric:
    - H ≈ 0.5: Random walk (healthy generation)
    - H > 0.65: Persistent / trending (early warning)
    - H > 0.80: Near-deterministic (collapse imminent)
    
    Based on methodology from:
    https://doi.org/10.5281/zenodo.18018292
    
    Args:
        ts: Time series (e.g., entropy values)
        max_lag: Maximum lag for R/S calculation
    
    Returns:
        Hurst exponent H ∈ [0, 1]
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    if len(ts) < 20:
        return np.nan
    
    if max_lag is None:
        max_lag = len(ts) // 4
    
    max_lag = max(max_lag, 10)
    lags = np.arange(10, min(max_lag + 1, len(ts) // 2))
    
    if len(lags) < 3:
        return np.nan
    
    rs_values = []
    valid_lags = []
    
    for lag in lags:
        n_blocks = len(ts) // lag
        if n_blocks < 2:
            continue
        
        rs_block = []
        for i in range(n_blocks):
            block = ts[i * lag:(i + 1) * lag]
            
            if len(block) < 2:
                continue
                
            # Mean-adjusted cumulative deviation
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            
            # Range
            R = np.max(cumdev) - np.min(cumdev)
            
            # Standard deviation
            S = np.std(block, ddof=1)
            
            if S > 1e-10:
                rs_block.append(R / S)
        
        if rs_block:
            rs_values.append(np.mean(rs_block))
            valid_lags.append(lag)
    
    if len(rs_values) < 3:
        return np.nan
    
    # Linear regression on log-log scale
    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
    
    # Hurst exponent is the slope
    return float(np.clip(slope, 0, 1))


def rolling_hurst(
    ts: np.ndarray,
    window: int = 200,
    step: int = 50
) -> Tuple[List[int], List[float]]:
    """
    Calculate Hurst exponent on rolling windows.
    
    Returns positions and corresponding Hurst values for plotting.
    
    Args:
        ts: Time series
        window: Window size for each Hurst calculation
        step: Step between windows
    
    Returns:
        Tuple of (positions, hurst_values)
    """
    positions = []
    hurst_values = []
    
    ts = np.asarray(ts)
    
    for i in range(window, len(ts), step):
        window_data = ts[i - window:i]
        h = hurst_exponent(window_data)
        
        if not np.isnan(h):
            positions.append(i)
            hurst_values.append(h)
    
    return positions, hurst_values


def entropy_deficit_sid(
    entropy_series: np.ndarray,
    baseline_window: int = 100
) -> np.ndarray:
    """
    Seismic Information Deficit (SID) — cumulative entropy deficit.
    
    Measures how much "information" the model is "losing" compared
    to its initial behavior. Rising SID = model "running out of ideas".
    
    Based on QO3/FIO framework:
    https://doi.org/10.5281/zenodo.18145167
    
    Args:
        entropy_series: Token entropy values
        baseline_window: Initial window to establish baseline entropy
    
    Returns:
        Cumulative SID at each step
    """
    entropy_series = np.asarray(entropy_series)
    
    if len(entropy_series) < baseline_window + 10:
        return np.zeros_like(entropy_series)
    
    # Baseline: mean entropy in first N tokens
    baseline = np.mean(entropy_series[:baseline_window])
    
    # Deficit at each step (positive when entropy is below baseline)
    deficit = baseline - entropy_series
    
    # Cumulative deficit (SID)
    sid = np.cumsum(np.maximum(deficit, 0))
    
    return sid


def varentropy(entropy_series: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Variance of entropy (rolling window).
    
    High varentropy = model is "unstable" before collapse.
    
    Args:
        entropy_series: Token entropy values
        window: Rolling window size
    
    Returns:
        Rolling variance of entropy
    """
    entropy_series = np.asarray(entropy_series)
    
    if len(entropy_series) < window:
        return np.zeros_like(entropy_series)
    
    result = np.zeros(len(entropy_series))
    
    for i in range(window, len(entropy_series)):
        result[i] = np.var(entropy_series[i - window:i])
    
    return result


def detect_regime_change(
    entropy_series: np.ndarray,
    hurst_series: List[float],
    hurst_positions: List[int],
    entropy_threshold: float = 0.7,  # fraction of initial entropy
    hurst_threshold: float = 0.65,
) -> Optional[int]:
    """
    Detect early regime change using combined entropy + Hurst signals.
    
    Returns the EARLIEST step where warning signals appear.
    
    Args:
        entropy_series: Token entropy values
        hurst_series: Hurst values from rolling_hurst
        hurst_positions: Positions from rolling_hurst
        entropy_threshold: Trigger when entropy falls below this fraction of baseline
        hurst_threshold: Trigger when Hurst exceeds this value
    
    Returns:
        First step with warning signal, or None
    """
    # Check entropy drop
    baseline = np.mean(entropy_series[:100]) if len(entropy_series) > 100 else np.mean(entropy_series)
    
    entropy_warning = None
    for i, ent in enumerate(entropy_series):
        if ent < baseline * entropy_threshold:
            entropy_warning = i
            break
    
    # Check Hurst increase
    hurst_warning = None
    for pos, h in zip(hurst_positions, hurst_series):
        if h > hurst_threshold:
            hurst_warning = pos
            break
    
    # Return earliest signal
    warnings = [w for w in [entropy_warning, hurst_warning] if w is not None]
    
    return min(warnings) if warnings else None
