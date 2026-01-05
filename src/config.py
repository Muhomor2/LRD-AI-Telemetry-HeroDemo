"""
Configuration and thresholds for LRD-AI-Telemetry v2.1

All "magic numbers" are centralized here with documentation.
Thresholds are empirical heuristics, not absolute truths.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from dataclasses import dataclass, field
from typing import Optional
import sys


@dataclass
class Thresholds:
    """
    Detection thresholds.
    
    NOTE: These are empirical heuristics derived from preliminary experiments.
    They should be calibrated for specific models and use cases.
    Bootstrap confidence intervals are recommended for rigorous analysis.
    """
    
    # Hurst exponent thresholds
    hurst_warning: float = 0.65      # H > this → early warning
    hurst_critical: float = 0.80     # H > this → collapse imminent
    hurst_baseline: float = 0.50     # Random walk reference
    
    # Higuchi dimension thresholds (inverted: lower = more warning)
    higuchi_warning: float = 1.30    # D < this → warning
    higuchi_critical: float = 1.20   # D < this → critical
    higuchi_healthy: float = 1.50    # Complex, healthy signal
    
    # Spectral entropy thresholds (inverted: lower = more warning)
    spectral_warning: float = 0.50
    spectral_critical: float = 0.30
    
    # 1/f exponent thresholds
    one_over_f_warning: float = 1.50  # β > this → strong correlations
    one_over_f_critical: float = 2.00
    one_over_f_pink: float = 1.00     # Pink noise reference
    
    # Repetition thresholds
    repetition_warning: float = 0.55
    repetition_critical: float = 0.70
    
    # N-gram repetition (stricter loop detection)
    ngram_warning: float = 0.30
    ngram_critical: float = 0.50
    
    # KL drift threshold
    kl_drift_warning: float = 0.5
    kl_drift_critical: float = 1.0


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    sliding_window: int = 800        # Context window for generation
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 2000
    
    # Early stopping
    early_stop_enabled: bool = True
    early_stop_ngram_threshold: float = 0.50
    early_stop_entropy_ratio: float = 0.40  # Stop if entropy < baseline * ratio
    early_stop_patience: int = 100          # Steps below threshold before stopping


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    
    # Rolling window settings
    rolling_window: int = 200
    rolling_step: int = 50
    
    # Hurst R/S settings
    hurst_min_lag: int = 10
    
    # Higuchi settings
    higuchi_k_max: int = 64
    
    # Spectral settings
    spectral_nperseg: int = 256
    
    # N-gram settings
    ngram_n: int = 4
    ngram_window: int = 200
    
    # Baseline window for SID and calibration
    baseline_window: int = 100
    
    # Bootstrap settings (for confidence intervals)
    bootstrap_n_samples: int = 100
    bootstrap_ci_level: float = 0.95


@dataclass
class RunConfig:
    """Complete configuration for a single run."""
    
    # Model
    model_name: str = "distilgpt2"
    
    # Seed for reproducibility
    seed: Optional[int] = None
    
    # Sub-configs
    thresholds: Thresholds = field(default_factory=Thresholds)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Experimental features
    use_fibonacci_windows: bool = False
    use_early_stopping: bool = True
    compute_kl_drift: bool = True
    compute_bootstrap_ci: bool = False  # Expensive, disabled by default
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'seed': self.seed,
            'thresholds': {
                'hurst_warning': self.thresholds.hurst_warning,
                'hurst_critical': self.thresholds.hurst_critical,
                'higuchi_warning': self.thresholds.higuchi_warning,
                'higuchi_critical': self.thresholds.higuchi_critical,
                'spectral_warning': self.thresholds.spectral_warning,
                'one_over_f_warning': self.thresholds.one_over_f_warning,
                'repetition_warning': self.thresholds.repetition_warning,
                'ngram_warning': self.thresholds.ngram_warning,
                'kl_drift_warning': self.thresholds.kl_drift_warning,
            },
            'generation': {
                'sliding_window': self.generation.sliding_window,
                'temperature': self.generation.temperature,
                'top_p': self.generation.top_p,
                'max_tokens': self.generation.max_tokens,
                'early_stop_enabled': self.generation.early_stop_enabled,
            },
            'metrics': {
                'rolling_window': self.metrics.rolling_window,
                'rolling_step': self.metrics.rolling_step,
                'ngram_n': self.metrics.ngram_n,
                'baseline_window': self.metrics.baseline_window,
            },
            'experimental': {
                'use_fibonacci_windows': self.use_fibonacci_windows,
                'compute_kl_drift': self.compute_kl_drift,
                'compute_bootstrap_ci': self.compute_bootstrap_ci,
            },
        }


def get_environment_info() -> dict:
    """Collect environment information for reproducibility."""
    import torch
    import transformers
    import numpy as np
    import scipy
    
    env = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        env['cuda_version'] = torch.version.cuda
        env['gpu_name'] = torch.cuda.get_device_name(0)
    
    return env


# Scientific disclaimers
DISCLAIMERS = {
    'hurst_method': (
        "Hurst exponent estimated via R/S analysis. "
        "R/S is used as a quick heuristic for persistence detection; "
        "for rigorous analysis, DFA(1/2) with confidence intervals is recommended."
    ),
    'beta_h_relation': (
        "The relation β ≈ 2H - 1 holds for fractional Gaussian noise (fGn). "
        "For fractional Brownian motion (fBm), β ≈ 2H + 1. "
        "We treat β as an empirical spectral slope indicator."
    ),
    'd_h_relation': (
        "The relation D ≈ 2 - H holds for idealized self-affine processes. "
        "Deviations indicate non-stationarity or regime transitions. "
        "Used as a sanity check / heuristic, not strict validation."
    ),
    'fibonacci_windows': (
        "Fibonacci-based window sizes are an experimental feature "
        "exploring quasicrystalline/aperiodic sampling. "
        "Scientific validation is ongoing."
    ),
    'thresholds': (
        "All thresholds are empirical heuristics derived from preliminary experiments. "
        "They should be calibrated for specific models, prompts, and use cases. "
        "Auto-calibration based on baseline statistics is recommended."
    ),
}
