"""
Long-form text generation with fractal telemetry.

Generates text while tracking entropy, repetition, and Hurst exponent
to detect degradation early.

Based on QO3/FIO framework:
https://doi.org/10.5281/zenodo.18145167
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import (
    token_entropy_from_logits,
    repetition_rate,
    hurst_exponent,
    rolling_hurst,
    entropy_deficit_sid,
    detect_regime_change,
)


@dataclass
class RunResult:
    """Results from a generation run with telemetry."""
    
    # Generated text
    text: str
    
    # Per-step metrics
    entropy_series: List[float]
    repetition_series: List[float]
    
    # Hurst analysis (computed at intervals)
    hurst_positions: List[int] = field(default_factory=list)
    hurst_series: List[float] = field(default_factory=list)
    
    # SID (cumulative entropy deficit)
    sid_series: Optional[np.ndarray] = None
    
    # Detection results
    first_exceed_step: Optional[int] = None  # First repetition > threshold
    first_warning_step: Optional[int] = None  # First Hurst/entropy warning
    
    # Summary stats
    mean_entropy_before: Optional[float] = None
    hurst_before_collapse: Optional[float] = None
    warning_lead_time: Optional[int] = None  # Tokens between warning and collapse
    
    def summary(self) -> str:
        """Human-readable summary of the run."""
        lines = []
        lines.append("=" * 60)
        lines.append("GENERATION TELEMETRY SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"Total tokens generated: {len(self.entropy_series)}")
        lines.append(f"Mean entropy: {np.mean(self.entropy_series):.4f}")
        
        if self.hurst_series:
            lines.append(f"Final Hurst exponent: {self.hurst_series[-1]:.4f}")
        
        if self.first_exceed_step is not None:
            lines.append(f"\n⚠️  Repetition collapse at step: {self.first_exceed_step}")
            
            if self.mean_entropy_before is not None:
                lines.append(f"   Mean entropy before collapse: {self.mean_entropy_before:.4f}")
            
            if self.hurst_before_collapse is not None:
                lines.append(f"   Hurst before collapse: {self.hurst_before_collapse:.4f}")
        
        if self.first_warning_step is not None:
            lines.append(f"\n🔔 Early warning at step: {self.first_warning_step}")
            
            if self.warning_lead_time is not None:
                lines.append(f"   Lead time: {self.warning_lead_time} tokens before collapse")
        
        if self.first_exceed_step is None:
            lines.append("\n✅ No repetition collapse detected")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def nucleus_sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> int:
    """
    Nucleus (top-p) sampling for next token.
    
    Args:
        logits: Tensor of shape [vocab_size]
        temperature: Sampling temperature
        top_p: Cumulative probability threshold
    
    Returns:
        Selected token ID
    """
    logits = logits / max(temperature, 1e-6)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens after cumulative prob > top_p
    cutoff = cum > top_p
    cutoff[0] = False  # Always keep at least one token
    sorted_logits[cutoff] = -1e10

    filtered_probs = torch.softmax(sorted_logits, dim=-1)
    pick = torch.multinomial(filtered_probs, 1).item()
    return int(sorted_idx[pick].item())


def generate_with_metrics(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2500,
    rep_threshold: float = 0.55,
    sliding_window: int = 800,
    temperature: float = 1.0,
    top_p: float = 0.95,
    hurst_window: int = 200,
    hurst_step: int = 100,
    hurst_warning: float = 0.65,
    verbose: bool = True,
) -> RunResult:
    """
    Generate text with full fractal telemetry.
    
    Uses sliding context window to bypass small model context limits.
    Records entropy, repetition rate, and Hurst exponent per step.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Initial prompt
        max_new_tokens: Maximum tokens to generate
        rep_threshold: Repetition rate threshold for collapse detection
        sliding_window: Context window size for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        hurst_window: Window for Hurst calculation
        hurst_step: Steps between Hurst calculations
        hurst_warning: Hurst threshold for early warning
        verbose: Print progress
    
    Returns:
        RunResult with all telemetry data
    """
    device = model.device
    model.eval()

    full_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

    entropy_series: List[float] = []
    repetition_series: List[float] = []
    hurst_positions: List[int] = []
    hurst_series: List[float] = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Sliding window context
            ctx = full_ids[-sliding_window:] if len(full_ids) > sliding_window else full_ids
            out = model(ctx.unsqueeze(0))
            logits = out.logits[0, -1, :]

            # Record entropy
            ent = token_entropy_from_logits(logits)
            entropy_series.append(ent)

            # Sample next token
            next_id = nucleus_sample_next_token(
                logits=logits,
                temperature=temperature,
                top_p=top_p,
            )

            full_ids = torch.cat([full_ids, torch.tensor([next_id], device=device)])
            
            # Record repetition
            rep = repetition_rate(full_ids.tolist(), window=200)
            repetition_series.append(rep)

            # Calculate Hurst at intervals
            if step > hurst_window and step % hurst_step == 0:
                recent_ent = np.array(entropy_series[-hurst_window:])
                h = hurst_exponent(recent_ent)
                
                if not np.isnan(h):
                    hurst_positions.append(step)
                    hurst_series.append(h)
                    
                    if verbose and h > hurst_warning:
                        print(f"⚠️  Step {step}: Hurst = {h:.3f} (warning threshold: {hurst_warning})")

            # Progress
            if verbose and step % 500 == 0 and step > 0:
                print(f"Step {step}/{max_new_tokens} | Entropy: {ent:.2f} | Rep: {rep:.3f}")

    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    
    # Calculate SID
    sid_series = entropy_deficit_sid(np.array(entropy_series))

    # Find first repetition collapse
    first_exceed = next((i for i, v in enumerate(repetition_series) if v > rep_threshold), None)
    
    # Calculate stats before collapse
    mean_ent_before = None
    hurst_before = None
    
    if first_exceed is not None:
        start = max(0, first_exceed - 100)
        if first_exceed > start:
            mean_ent_before = float(np.mean(entropy_series[start:first_exceed]))
        
        # Find last Hurst before collapse
        for pos, h in zip(reversed(hurst_positions), reversed(hurst_series)):
            if pos < first_exceed:
                hurst_before = h
                break
    
    # Find early warning
    first_warning = detect_regime_change(
        np.array(entropy_series),
        hurst_series,
        hurst_positions,
        hurst_threshold=hurst_warning,
    )
    
    # Calculate lead time
    lead_time = None
    if first_warning is not None and first_exceed is not None:
        lead_time = first_exceed - first_warning if first_exceed > first_warning else None

    return RunResult(
        text=text,
        entropy_series=entropy_series,
        repetition_series=repetition_series,
        hurst_positions=hurst_positions,
        hurst_series=hurst_series,
        sid_series=sid_series,
        first_exceed_step=first_exceed,
        first_warning_step=first_warning,
        mean_entropy_before=mean_ent_before,
        hurst_before_collapse=hurst_before,
        warning_lead_time=lead_time,
    )
