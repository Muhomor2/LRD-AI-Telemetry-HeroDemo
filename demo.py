#!/usr/bin/env python3
"""
Seismic Precursors of LLM Degradation — Demo Script

Generates long text with a small LLM and tracks fractal telemetry:
- Token entropy
- Repetition rate  
- Hurst exponent (fractal memory)
- SID (Seismic Information Deficit)

Detects early warning signals before visible degradation.

Based on QO3/FIO framework:
https://doi.org/10.5281/zenodo.18145167

Usage:
    python demo.py --model distilgpt2 --max_new_tokens 2500
    
Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate import generate_with_metrics
from src.visualization import (
    plot_entropy,
    plot_repetition,
    plot_hurst,
    plot_sid,
    plot_combined_dashboard,
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect seismic precursors of LLM degradation"
    )
    
    # Model settings
    parser.add_argument(
        "--model", 
        default="distilgpt2",
        help="HuggingFace model name (default: distilgpt2)"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=2500,
        help="Maximum tokens to generate (default: 2500)"
    )
    parser.add_argument(
        "--sliding_window", 
        type=int, 
        default=800,
        help="Context window size (default: 800)"
    )
    
    # Sampling settings
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="Nucleus sampling threshold (default: 0.95)"
    )
    
    # Detection thresholds
    parser.add_argument(
        "--rep_threshold", 
        type=float, 
        default=0.55,
        help="Repetition rate threshold for collapse (default: 0.55)"
    )
    parser.add_argument(
        "--hurst_warning", 
        type=float, 
        default=0.65,
        help="Hurst exponent warning threshold (default: 0.65)"
    )
    
    # Hurst calculation settings
    parser.add_argument(
        "--hurst_window", 
        type=int, 
        default=200,
        help="Window size for Hurst calculation (default: 200)"
    )
    parser.add_argument(
        "--hurst_step", 
        type=int, 
        default=100,
        help="Steps between Hurst calculations (default: 100)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Output directory for plots (default: outputs)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress output"
    )
    
    # Custom prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (default: uses built-in prompt designed to trigger loops)"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    
    # Default prompt designed to trigger degradation
    if args.prompt is None:
        prompt = (
            "Repeat the explanation with more detail and examples, "
            "do not stop, keep expanding.\n\n"
            "Explain time series, regime shifts, and why early warning "
            "signals matter. Describe how complex systems can suddenly "
            "transition from one state to another, and what precursors "
            "might indicate an approaching transition.\n\n"
        )
    else:
        prompt = args.prompt
    
    print(f"\nGenerating {args.max_new_tokens} tokens...")
    print("-" * 60)
    
    # Generate with telemetry
    result = generate_with_metrics(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        rep_threshold=args.rep_threshold,
        sliding_window=args.sliding_window,
        temperature=args.temperature,
        top_p=args.top_p,
        hurst_window=args.hurst_window,
        hurst_step=args.hurst_step,
        hurst_warning=args.hurst_warning,
        verbose=not args.quiet,
    )
    
    print("-" * 60)
    print(result.summary())
    
    # Show last part of generated text
    print("\n=== LAST 1200 CHARS (to see looping) ===")
    print(result.text[-1200:])
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_entropy(
        result.entropy_series,
        output_path=output_dir / "entropy.png",
        first_exceed_step=result.first_exceed_step,
        first_warning_step=result.first_warning_step,
    )
    
    plot_repetition(
        result.repetition_series,
        threshold=args.rep_threshold,
        output_path=output_dir / "repetition.png",
        first_exceed_step=result.first_exceed_step,
    )
    
    if result.hurst_positions and result.hurst_series:
        plot_hurst(
            result.hurst_positions,
            result.hurst_series,
            warning_threshold=args.hurst_warning,
            output_path=output_dir / "hurst.png",
            first_exceed_step=result.first_exceed_step,
        )
    
    if result.sid_series is not None:
        plot_sid(
            result.sid_series,
            output_path=output_dir / "sid.png",
            first_exceed_step=result.first_exceed_step,
            first_warning_step=result.first_warning_step,
        )
    
    plot_combined_dashboard(
        result.entropy_series,
        result.repetition_series,
        result.hurst_positions,
        result.hurst_series,
        result.sid_series,
        rep_threshold=args.rep_threshold,
        hurst_warning=args.hurst_warning,
        first_exceed_step=result.first_exceed_step,
        first_warning_step=result.first_warning_step,
        output_path=output_dir / "combined_dashboard.png",
    )
    
    print("\n✅ Saved figures:")
    for png in sorted(output_dir.glob("*.png")):
        print(f"   - {png}")
    
    # Summary for quick reference
    if result.warning_lead_time is not None:
        print(f"\n🎯 KEY RESULT: Early warning detected {result.warning_lead_time} tokens before collapse!")
    elif result.first_warning_step is not None and result.first_exceed_step is None:
        print("\n⚠️  Warning signal detected but no collapse occurred (generation may have been too short)")
    elif result.first_exceed_step is not None:
        print("\n⚠️  Collapse detected but no early warning (threshold may need tuning)")
    else:
        print("\n✅ No degradation detected in this run")


if __name__ == "__main__":
    main()
