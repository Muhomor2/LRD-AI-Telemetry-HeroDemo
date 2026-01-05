"""
Visualization utilities for LLM telemetry.

Creates publication-quality plots showing entropy, repetition,
Hurst exponent, and early warning signals.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
    })


def plot_entropy(
    entropy_series: List[float],
    output_path: Optional[Path] = None,
    first_exceed_step: Optional[int] = None,
    first_warning_step: Optional[int] = None,
    title: str = "Token Entropy Over Generation Steps",
) -> None:
    """
    Plot token entropy with collapse/warning markers.
    
    Args:
        entropy_series: Entropy values per step
        output_path: Where to save the plot
        first_exceed_step: Step where repetition exceeded threshold
        first_warning_step: Step where early warning was triggered
        title: Plot title
    """
    setup_style()
    
    fig, ax = plt.subplots()
    
    ax.plot(entropy_series, color='#2E86AB', linewidth=0.8, alpha=0.8)
    
    # Rolling mean
    window = min(50, len(entropy_series) // 10)
    if window > 5:
        rolling_mean = np.convolve(entropy_series, np.ones(window)/window, mode='valid')
        ax.plot(
            range(window-1, len(entropy_series)),
            rolling_mean,
            color='#E94F37',
            linewidth=2,
            label=f'Rolling mean ({window} steps)'
        )
    
    # Markers
    if first_warning_step is not None:
        ax.axvline(first_warning_step, color='#F6AE2D', linestyle='--', linewidth=2, label='Early warning')
    
    if first_exceed_step is not None:
        ax.axvline(first_exceed_step, color='#E94F37', linestyle='-', linewidth=2, label='Collapse')
    
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_repetition(
    repetition_series: List[float],
    threshold: float = 0.55,
    output_path: Optional[Path] = None,
    first_exceed_step: Optional[int] = None,
    title: str = "Repetition Rate Over Generation Steps",
) -> None:
    """
    Plot repetition rate with threshold line.
    
    Args:
        repetition_series: Repetition rate values per step
        threshold: Threshold for collapse detection
        output_path: Where to save the plot
        first_exceed_step: Step where threshold was exceeded
        title: Plot title
    """
    setup_style()
    
    fig, ax = plt.subplots()
    
    ax.plot(repetition_series, color='#2E86AB', linewidth=0.8, alpha=0.8)
    ax.axhline(threshold, color='#E94F37', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    if first_exceed_step is not None:
        ax.axvline(first_exceed_step, color='#E94F37', linestyle='-', linewidth=2, alpha=0.5)
        ax.annotate(
            f'Collapse: step {first_exceed_step}',
            xy=(first_exceed_step, threshold),
            xytext=(first_exceed_step + 100, threshold + 0.1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#E94F37'),
        )
    
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Repetition Rate')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hurst(
    hurst_positions: List[int],
    hurst_series: List[float],
    warning_threshold: float = 0.65,
    output_path: Optional[Path] = None,
    first_exceed_step: Optional[int] = None,
    title: str = "Hurst Exponent (Fractal Memory) Over Generation",
) -> None:
    """
    Plot Hurst exponent with interpretation zones.
    
    Args:
        hurst_positions: Steps where Hurst was calculated
        hurst_series: Hurst exponent values
        warning_threshold: Threshold for early warning
        output_path: Where to save the plot
        first_exceed_step: Step where repetition collapse occurred
        title: Plot title
    """
    setup_style()
    
    fig, ax = plt.subplots()
    
    # Background zones
    ax.axhspan(0, 0.5, alpha=0.1, color='green', label='Anti-persistent (H < 0.5)')
    ax.axhspan(0.5, 0.65, alpha=0.1, color='yellow', label='Normal (0.5 ≤ H < 0.65)')
    ax.axhspan(0.65, 0.8, alpha=0.1, color='orange', label='Warning (0.65 ≤ H < 0.8)')
    ax.axhspan(0.8, 1.0, alpha=0.1, color='red', label='Critical (H ≥ 0.8)')
    
    ax.plot(hurst_positions, hurst_series, 'o-', color='#2E86AB', linewidth=2, markersize=4)
    ax.axhline(warning_threshold, color='#F6AE2D', linestyle='--', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    
    if first_exceed_step is not None:
        ax.axvline(first_exceed_step, color='#E94F37', linestyle='-', linewidth=2, alpha=0.5, label='Collapse')
    
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Hurst Exponent (H)')
    ax.set_title(title)
    ax.set_ylim(0.3, 1.0)
    
    # Legend
    handles = [
        mpatches.Patch(color='green', alpha=0.3, label='Anti-persistent'),
        mpatches.Patch(color='yellow', alpha=0.3, label='Normal'),
        mpatches.Patch(color='orange', alpha=0.3, label='Warning'),
        mpatches.Patch(color='red', alpha=0.3, label='Critical'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sid(
    sid_series: np.ndarray,
    output_path: Optional[Path] = None,
    first_exceed_step: Optional[int] = None,
    first_warning_step: Optional[int] = None,
    title: str = "Seismic Information Deficit (SID)",
) -> None:
    """
    Plot cumulative entropy deficit (SID).
    
    Args:
        sid_series: SID values per step
        output_path: Where to save the plot
        first_exceed_step: Step where collapse occurred
        first_warning_step: Step where early warning was triggered
        title: Plot title
    """
    setup_style()
    
    fig, ax = plt.subplots()
    
    ax.fill_between(range(len(sid_series)), sid_series, alpha=0.3, color='#E94F37')
    ax.plot(sid_series, color='#E94F37', linewidth=1.5)
    
    if first_warning_step is not None:
        ax.axvline(first_warning_step, color='#F6AE2D', linestyle='--', linewidth=2, label='Early warning')
    
    if first_exceed_step is not None:
        ax.axvline(first_exceed_step, color='#E94F37', linestyle='-', linewidth=2, label='Collapse')
    
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Cumulative Information Deficit')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_dashboard(
    entropy_series: List[float],
    repetition_series: List[float],
    hurst_positions: List[int],
    hurst_series: List[float],
    sid_series: Optional[np.ndarray] = None,
    rep_threshold: float = 0.55,
    hurst_warning: float = 0.65,
    first_exceed_step: Optional[int] = None,
    first_warning_step: Optional[int] = None,
    output_path: Optional[Path] = None,
    title: str = "LLM Degradation Telemetry Dashboard",
) -> None:
    """
    Create a combined dashboard with all metrics.
    
    Args:
        entropy_series: Entropy values
        repetition_series: Repetition rate values
        hurst_positions: Steps where Hurst was calculated
        hurst_series: Hurst values
        sid_series: SID values (optional)
        rep_threshold: Repetition threshold
        hurst_warning: Hurst warning threshold
        first_exceed_step: Collapse step
        first_warning_step: Warning step
        output_path: Where to save
        title: Main title
    """
    setup_style()
    
    n_plots = 4 if sid_series is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    
    # Entropy
    ax = axes[0]
    ax.plot(entropy_series, color='#2E86AB', linewidth=0.8, alpha=0.7)
    window = min(50, len(entropy_series) // 10)
    if window > 5:
        rolling = np.convolve(entropy_series, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(entropy_series)), rolling, color='#E94F37', linewidth=2)
    ax.set_ylabel('Entropy')
    ax.set_title('Token Entropy')
    
    # Repetition
    ax = axes[1]
    ax.plot(repetition_series, color='#2E86AB', linewidth=0.8)
    ax.axhline(rep_threshold, color='#E94F37', linestyle='--', linewidth=2)
    ax.set_ylabel('Repetition Rate')
    ax.set_ylim(0, 1)
    ax.set_title('Repetition Rate')
    
    # Hurst
    ax = axes[2]
    if hurst_positions and hurst_series:
        ax.plot(hurst_positions, hurst_series, 'o-', color='#2E86AB', linewidth=2, markersize=4)
        ax.axhline(hurst_warning, color='#F6AE2D', linestyle='--', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Hurst (H)')
    ax.set_ylim(0.3, 1.0)
    ax.set_title('Hurst Exponent (Fractal Memory)')
    
    # SID (if available)
    if sid_series is not None:
        ax = axes[3]
        ax.fill_between(range(len(sid_series)), sid_series, alpha=0.3, color='#E94F37')
        ax.plot(sid_series, color='#E94F37', linewidth=1.5)
        ax.set_ylabel('SID')
        ax.set_title('Seismic Information Deficit')
    
    # Add markers to all plots
    for ax in axes:
        if first_warning_step is not None:
            ax.axvline(first_warning_step, color='#F6AE2D', linestyle='--', linewidth=2, alpha=0.7)
        if first_exceed_step is not None:
            ax.axvline(first_exceed_step, color='#E94F37', linestyle='-', linewidth=2, alpha=0.7)
    
    axes[-1].set_xlabel('Generation Step')
    
    # Legend
    legend_elements = []
    if first_warning_step is not None:
        legend_elements.append(plt.Line2D([0], [0], color='#F6AE2D', linestyle='--', linewidth=2, label=f'Early Warning (step {first_warning_step})'))
    if first_exceed_step is not None:
        legend_elements.append(plt.Line2D([0], [0], color='#E94F37', linestyle='-', linewidth=2, label=f'Collapse (step {first_exceed_step})'))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
