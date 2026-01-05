"""
Advanced Fractal Metrics for LLM Telemetry v2.1

Implements:
- Token Entropy (Shannon)
- Repetition Rate (token-level, fast)
- N-gram Repetition Rate (strict loop detection)
- KL Divergence Drift (regime change indicator)
- Hurst Exponent (R/S Analysis) — heuristic for persistence
- Higuchi Fractal Dimension
- Spectral Entropy
- 1/f Exponent (empirical spectral slope)
- Bootstrap Confidence Intervals
- Seismic Information Deficit (SID)

IMPORTANT NOTES:
- Hurst via R/S is a quick heuristic; DFA recommended for rigorous analysis
- β ≈ 2H-1 relation holds for fGn; we use β as empirical slope
- D ≈ 2-H is a sanity check for self-affine processes

Based on QO3/FIO Framework:
https://doi.org/10.5281/zenodo.18145167

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.signal import welch
from typing import List, Optional, Tuple, Dict
import torch


# =============================================================================
# SEED AND REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# BASIC METRICS
# =============================================================================

def token_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Token-level Shannon entropy for next-token distribution.
    
    H = -Σ p(x) log p(x)
    
    Lower entropy = model is "too confident" (potential precursor)
    """
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
    return float(ent)


def get_token_distribution(logits: torch.Tensor) -> np.ndarray:
    """Get probability distribution from logits."""
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


def repetition_rate(token_ids: List[int], window: int = 200) -> float:
    """
    Fraction of tokens in last window that appeared earlier.
    
    NOTE: This is a FAST but COARSE metric. Common tokens like "the", ","
    will inflate this even without true loops. Use ngram_repetition_rate
    for stricter loop detection.
    """
    if len(token_ids) < 20:
        return 0.0
    tail = token_ids[-min(window, len(token_ids)):]
    seen_before = set(token_ids[:-len(tail)])
    if not seen_before:
        return 0.0
    return sum(1 for t in tail if t in seen_before) / len(tail)


def ngram_repetition_rate(token_ids: List[int], n: int = 4, window: int = 200) -> float:
    """
    Strict loop detection: fraction of n-grams in window that are duplicates.
    
    Returns 0 = fully diverse, 1 = completely looping
    
    This is more reliable than token_repetition for detecting actual loops,
    as it's not fooled by common tokens.
    """
    if len(token_ids) < n + 10:
        return 0.0
    
    tail = token_ids[-min(window, len(token_ids)):]
    
    if len(tail) < n:
        return 0.0
    
    ngrams = [tuple(tail[i:i+n]) for i in range(len(tail) - n + 1)]
    
    if not ngrams:
        return 0.0
    
    unique = len(set(ngrams))
    total = len(ngrams)
    
    # Return 1 - diversity (so 0 = diverse, 1 = looping)
    return 1.0 - (unique / total)


# =============================================================================
# KL DIVERGENCE DRIFT
# =============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    KL divergence D_KL(P || Q).
    
    Measures how distribution P diverges from Q.
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sum(p * np.log(p / q)))


def compute_kl_drift(
    distributions: List[np.ndarray],
    step: int = 10,
) -> Tuple[List[int], List[float]]:
    """
    Compute KL divergence between consecutive distributions.
    
    Rising KL drift indicates regime change / instability.
    
    Returns:
        positions: step indices
        kl_values: KL divergence values
    """
    positions = []
    kl_values = []
    
    for i in range(step, len(distributions), step):
        p = distributions[i]
        q = distributions[i - step]
        
        kl = kl_divergence(p, q)
        
        if np.isfinite(kl):
            positions.append(i)
            kl_values.append(kl)
    
    return positions, kl_values


# =============================================================================
# HURST EXPONENT (R/S ANALYSIS)
# =============================================================================

def hurst_exponent(ts: np.ndarray, max_lag: Optional[int] = None) -> float:
    """
    Hurst exponent via R/S (Rescaled Range) analysis.
    
    IMPORTANT: R/S is a quick heuristic for persistence detection.
    It can overestimate H on non-stationary data with trends.
    For rigorous analysis, DFA (Detrended Fluctuation Analysis) is recommended.
    
    Interpretation (heuristic):
        H ≈ 0.5: Random walk (no memory)
        H > 0.65: Persistent / trending (warning)
        H > 0.80: Near-deterministic (collapse likely)
    
    Reference: https://doi.org/10.5281/zenodo.18018292
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    if len(ts) < 20:
        return np.nan
    
    # Remove NaN/Inf
    ts = ts[np.isfinite(ts)]
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
            
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(block, ddof=1)
            
            if S > 1e-10:
                rs_block.append(R / S)
        
        if rs_block:
            rs_values.append(np.mean(rs_block))
            valid_lags.append(lag)
    
    if len(rs_values) < 3:
        return np.nan
    
    slope, _, _, _, _ = stats.linregress(np.log(valid_lags), np.log(rs_values))
    return float(np.clip(slope, 0, 1))


# =============================================================================
# HIGUCHI FRACTAL DIMENSION
# =============================================================================

def higuchi_fractal_dimension(ts: np.ndarray, k_max: Optional[int] = None) -> float:
    """
    Higuchi Fractal Dimension.
    
    Measures complexity of a time series in the time domain.
    
    Interpretation (heuristic):
        D ≈ 1.5: Complex, healthy signal
        D < 1.3: Increasing regularity (warning)
        D → 1.0: Near-periodic (collapse)
    
    Theoretical relation to Hurst: D ≈ 2 - H (for self-affine processes)
    Deviations indicate non-stationarity or regime transitions.
    
    Reference: Higuchi, T. (1988). Physica D, 31(2), 277-283.
    """
    ts = np.asarray(ts, dtype=np.float64)
    N = len(ts)
    
    if N < 20:
        return np.nan
    
    ts = ts[np.isfinite(ts)]
    N = len(ts)
    if N < 20:
        return np.nan
    
    if k_max is None:
        k_max = min(N // 4, 64)
    k_max = max(k_max, 4)
    
    L = []
    k_values = []
    
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, N, k)
            if len(indices) < 2:
                continue
            
            Xm = ts[indices]
            n = len(Xm)
            if n < 2:
                continue
            
            Lmk = np.sum(np.abs(np.diff(Xm))) * (N - 1) / (k * (n - 1) * k)
            Lk.append(Lmk)
        
        if Lk:
            L.append(np.mean(Lk))
            k_values.append(k)
    
    if len(L) < 3:
        return np.nan
    
    log_k = np.log(1.0 / np.array(k_values))
    log_L = np.log(L)
    
    slope, _, _, _, _ = stats.linregress(log_k, log_L)
    
    return float(np.clip(slope, 1.0, 2.0))


# =============================================================================
# SPECTRAL ENTROPY
# =============================================================================

def spectral_entropy(ts: np.ndarray, sf: float = 1.0, nperseg: Optional[int] = None) -> float:
    """
    Spectral Entropy — measures flatness of power spectral density.
    
    High spectral entropy = broadband signal (complex, healthy)
    Low spectral entropy = narrowband/periodic (warning)
    
    Normalized to [0, 1]:
        1.0 = white noise (uniform PSD)
        0.0 = pure sinusoid (single frequency)
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    if len(ts) < 16:
        return np.nan
    
    ts = ts[np.isfinite(ts)]
    if len(ts) < 16:
        return np.nan
    
    if nperseg is None:
        nperseg = min(len(ts), 256)
    
    freqs, psd = welch(ts, fs=sf, nperseg=nperseg)
    
    psd = psd[freqs > 0]
    
    if len(psd) < 2 or np.sum(psd) < 1e-10:
        return np.nan
    
    psd_norm = psd / np.sum(psd)
    H = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    H_max = np.log2(len(psd_norm))
    
    if H_max < 1e-10:
        return np.nan
    
    return float(H / H_max)


# =============================================================================
# 1/f EXPONENT
# =============================================================================

def one_over_f_exponent(ts: np.ndarray, sf: float = 1.0, nperseg: Optional[int] = None) -> float:
    """
    1/f exponent (β) from power spectral density.
    
    S(f) ∝ 1/f^β  →  log(S) = -β·log(f) + const
    
    Interpretation (empirical):
        β ≈ 0: White noise (uncorrelated)
        β ≈ 1: Pink noise (1/f, healthy complexity)
        β ≈ 2: Brownian noise (random walk)
        β > 2: Strong correlations (warning)
    
    NOTE: The relation β ≈ 2H - 1 holds for fractional Gaussian noise (fGn).
    For fBm, β ≈ 2H + 1. We treat β as an empirical spectral slope indicator.
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    if len(ts) < 32:
        return np.nan
    
    ts = ts[np.isfinite(ts)]
    if len(ts) < 32:
        return np.nan
    
    if nperseg is None:
        nperseg = min(len(ts), 256)
    
    freqs, psd = welch(ts, fs=sf, nperseg=nperseg)
    
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]
    
    if len(freqs) < 5:
        return np.nan
    
    valid = (psd > 0) & np.isfinite(psd)
    freqs = freqs[valid]
    psd = psd[valid]
    
    if len(freqs) < 5:
        return np.nan
    
    log_f = np.log10(freqs)
    log_psd = np.log10(psd)
    
    slope, _, _, _, _ = stats.linregress(log_f, log_psd)
    
    beta = -slope
    
    return float(beta)


# =============================================================================
# SEISMIC INFORMATION DEFICIT (SID)
# =============================================================================

def entropy_deficit_sid(
    entropy_series: np.ndarray,
    baseline_window: int = 100
) -> np.ndarray:
    """
    Seismic Information Deficit (SID) — cumulative entropy deficit.
    
    Measures how much "information capacity" the model is losing
    compared to its initial behavior.
    
    Rising SID = model "running out of ideas"
    
    Based on QO3/FIO framework.
    """
    entropy_series = np.asarray(entropy_series)
    
    if len(entropy_series) < baseline_window + 10:
        return np.zeros_like(entropy_series)
    
    baseline = np.mean(entropy_series[:baseline_window])
    deficit = baseline - entropy_series
    sid = np.cumsum(np.maximum(deficit, 0))
    
    return sid


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    ts: np.ndarray,
    metric_func,
    n_samples: int = 100,
    ci_level: float = 0.95,
    block_size: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    
    Uses block bootstrap to preserve temporal structure.
    
    Returns:
        (estimate, ci_lower, ci_upper)
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    
    if n < 20:
        val = metric_func(ts)
        return val, np.nan, np.nan
    
    if block_size is None:
        block_size = max(10, n // 10)
    
    estimates = []
    
    for _ in range(n_samples):
        # Block bootstrap
        n_blocks = n // block_size + 1
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        
        resampled = []
        for start in block_starts:
            resampled.extend(ts[start:start + block_size])
        resampled = np.array(resampled[:n])
        
        val = metric_func(resampled)
        if np.isfinite(val):
            estimates.append(val)
    
    if len(estimates) < 10:
        val = metric_func(ts)
        return val, np.nan, np.nan
    
    estimates = np.array(estimates)
    alpha = 1 - ci_level
    
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    point_estimate = metric_func(ts)
    
    return float(point_estimate), float(ci_lower), float(ci_upper)


# =============================================================================
# ROLLING METRICS
# =============================================================================

def rolling_metrics(
    ts: np.ndarray,
    window: int = 200,
    step: int = 50,
    compute_ci: bool = False,
    ci_n_samples: int = 50,
) -> Dict:
    """
    Compute all fractal metrics on rolling windows.
    
    Returns dict with positions and metric values.
    If compute_ci=True, also returns confidence intervals (expensive).
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    results = {
        'positions': [],
        'hurst': [],
        'higuchi': [],
        'spectral_entropy': [],
        'one_over_f': [],
    }
    
    if compute_ci:
        results['hurst_ci_lower'] = []
        results['hurst_ci_upper'] = []
        results['higuchi_ci_lower'] = []
        results['higuchi_ci_upper'] = []
    
    for i in range(window, len(ts), step):
        window_data = ts[i - window:i]
        
        if compute_ci:
            h, h_lo, h_hi = bootstrap_ci(window_data, hurst_exponent, ci_n_samples)
            d, d_lo, d_hi = bootstrap_ci(window_data, higuchi_fractal_dimension, ci_n_samples)
        else:
            h = hurst_exponent(window_data)
            d = higuchi_fractal_dimension(window_data)
        
        se = spectral_entropy(window_data)
        beta = one_over_f_exponent(window_data)
        
        if not np.isnan(h):
            results['positions'].append(i)
            results['hurst'].append(h)
            results['higuchi'].append(d if not np.isnan(d) else 1.5)
            results['spectral_entropy'].append(se if not np.isnan(se) else 0.5)
            results['one_over_f'].append(beta if not np.isnan(beta) else 1.0)
            
            if compute_ci:
                results['hurst_ci_lower'].append(h_lo if not np.isnan(h_lo) else h)
                results['hurst_ci_upper'].append(h_hi if not np.isnan(h_hi) else h)
                results['higuchi_ci_lower'].append(d_lo if not np.isnan(d_lo) else d)
                results['higuchi_ci_upper'].append(d_hi if not np.isnan(d_hi) else d)
    
    return results


# =============================================================================
# FIBONACCI WINDOWS (EXPERIMENTAL)
# =============================================================================

def fibonacci_sequence(n: int) -> List[int]:
    """Generate first n Fibonacci numbers > 1."""
    fibs = [1, 2]
    while len(fibs) < n + 10:
        fibs.append(fibs[-1] + fibs[-2])
    return [f for f in fibs if f > 1][:n]


def compute_metrics_fibonacci_windows(
    ts: np.ndarray,
    max_windows: int = 8
) -> Dict:
    """
    Compute Hurst and Higuchi using Fibonacci-sized windows.
    
    EXPERIMENTAL: Exploring quasicrystalline/aperiodic sampling.
    Scientific validation is ongoing.
    """
    ts = np.asarray(ts, dtype=np.float64)
    
    fibs = fibonacci_sequence(max_windows)
    results = {
        'windows': [],
        'hurst': [],
        'higuchi': [],
    }
    
    for fib in fibs:
        if fib > len(ts) // 2:
            break
        
        window = ts[-fib:] if fib <= len(ts) else ts
        
        h = hurst_exponent(window)
        d = higuchi_fractal_dimension(window)
        
        if not np.isnan(h) and not np.isnan(d):
            results['windows'].append(fib)
            results['hurst'].append(h)
            results['higuchi'].append(d)
    
    return results


# =============================================================================
# COMPOSITE WARNING SCORE
# =============================================================================

def compute_warning_score(
    hurst: float,
    higuchi: float,
    spectral_ent: float,
    one_over_f: float,
    ngram_rep: float = 0.0,
    kl_drift: float = 0.0,
    thresholds: Optional[Dict] = None,
) -> Tuple[float, str]:
    """
    Compute composite early warning score from all metrics.
    
    Weights: Hurst (30), Higuchi (25), Spectral (15), 1/f (15), N-gram (10), KL (5)
    Total: 100 points
    
    Returns:
        (score 0-100, level: 'normal'/'warning'/'critical')
    """
    if thresholds is None:
        thresholds = {
            'hurst_warning': 0.65,
            'hurst_critical': 0.80,
            'higuchi_warning': 1.30,
            'higuchi_critical': 1.20,
            'spectral_warning': 0.50,
            'spectral_critical': 0.30,
            'one_over_f_warning': 1.50,
            'one_over_f_critical': 2.00,
            'ngram_warning': 0.30,
            'ngram_critical': 0.50,
            'kl_warning': 0.50,
        }
    
    score = 0.0
    
    # Hurst contribution (0-30 points)
    if hurst > thresholds['hurst_critical']:
        score += 30
    elif hurst > thresholds['hurst_warning']:
        score += 15 + 15 * (hurst - thresholds['hurst_warning']) / (thresholds['hurst_critical'] - thresholds['hurst_warning'])
    elif hurst > 0.5:
        score += 15 * (hurst - 0.5) / (thresholds['hurst_warning'] - 0.5)
    
    # Higuchi contribution (0-25 points) — inverted
    if higuchi < thresholds['higuchi_critical']:
        score += 25
    elif higuchi < thresholds['higuchi_warning']:
        score += 12.5 + 12.5 * (thresholds['higuchi_warning'] - higuchi) / (thresholds['higuchi_warning'] - thresholds['higuchi_critical'])
    elif higuchi < 1.5:
        score += 12.5 * (1.5 - higuchi) / (1.5 - thresholds['higuchi_warning'])
    
    # Spectral entropy contribution (0-15 points) — inverted
    if spectral_ent < thresholds['spectral_critical']:
        score += 15
    elif spectral_ent < thresholds['spectral_warning']:
        score += 7.5 + 7.5 * (thresholds['spectral_warning'] - spectral_ent) / (thresholds['spectral_warning'] - thresholds['spectral_critical'])
    elif spectral_ent < 0.7:
        score += 7.5 * (0.7 - spectral_ent) / (0.7 - thresholds['spectral_warning'])
    
    # 1/f contribution (0-15 points)
    if one_over_f > thresholds['one_over_f_critical']:
        score += 15
    elif one_over_f > thresholds['one_over_f_warning']:
        score += 7.5 + 7.5 * (one_over_f - thresholds['one_over_f_warning']) / (thresholds['one_over_f_critical'] - thresholds['one_over_f_warning'])
    elif one_over_f > 1.0:
        score += 7.5 * (one_over_f - 1.0) / (thresholds['one_over_f_warning'] - 1.0)
    
    # N-gram repetition contribution (0-10 points)
    if ngram_rep > thresholds['ngram_critical']:
        score += 10
    elif ngram_rep > thresholds['ngram_warning']:
        score += 5 + 5 * (ngram_rep - thresholds['ngram_warning']) / (thresholds['ngram_critical'] - thresholds['ngram_warning'])
    elif ngram_rep > 0.1:
        score += 5 * (ngram_rep - 0.1) / (thresholds['ngram_warning'] - 0.1)
    
    # KL drift contribution (0-5 points)
    if kl_drift > thresholds['kl_warning']:
        score += min(5, 5 * kl_drift / thresholds['kl_warning'])
    
    # Determine level
    if score >= 70:
        level = 'critical'
    elif score >= 40:
        level = 'warning'
    else:
        level = 'normal'
    
    return float(min(score, 100)), level
