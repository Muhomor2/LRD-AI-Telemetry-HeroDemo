"""
Text Generation with Fractal Telemetry v2.1

Features:
- Sliding window context
- Per-step entropy and distribution tracking
- N-gram repetition monitoring
- KL drift computation
- Early stopping on degradation signals
- Reproducible via seed

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import (
    token_entropy_from_logits,
    get_token_distribution,
    repetition_rate,
    ngram_repetition_rate,
    seed_everything,
)
from config import RunConfig, GenerationConfig


@dataclass
class TelemetryData:
    """Raw telemetry collected during generation."""
    
    entropy_series: List[float] = field(default_factory=list)
    repetition_series: List[float] = field(default_factory=list)
    ngram_rep_series: List[float] = field(default_factory=list)
    distributions: List[np.ndarray] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    
    # Generation metadata
    total_steps: int = 0
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    stop_step: Optional[int] = None


@dataclass
class GenerationResult:
    """Complete result from a generation run."""
    
    text: str
    telemetry: TelemetryData
    config: Dict[str, Any]
    
    def get_entropy_array(self) -> np.ndarray:
        return np.array(self.telemetry.entropy_series)
    
    def get_token_ids(self) -> List[int]:
        return self.telemetry.token_ids


def nucleus_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> int:
    """
    Nucleus (top-p) sampling for next token.
    
    Deterministic given fixed seed.
    """
    logits = logits / max(temperature, 1e-6)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cum > top_p
    cutoff[0] = False
    sorted_logits[cutoff] = -1e10

    filtered_probs = torch.softmax(sorted_logits, dim=-1)
    pick = torch.multinomial(filtered_probs, 1).item()
    return int(sorted_idx[pick].item())


class Generator:
    """
    Text generator with telemetry collection and early stopping.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        config: Optional[RunConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or RunConfig()
        
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback=None,
        store_distributions: bool = False,
    ) -> GenerationResult:
        """
        Generate text with full telemetry.
        
        Args:
            prompt: Initial text
            max_tokens: Override config max_tokens
            seed: Random seed for reproducibility
            progress_callback: Function(step, max_steps) for progress updates
            store_distributions: Whether to store full token distributions (memory intensive)
        
        Returns:
            GenerationResult with text and telemetry
        """
        gen_config = self.config.generation
        
        if max_tokens is None:
            max_tokens = gen_config.max_tokens
        
        # Set seed if provided
        actual_seed = seed if seed is not None else self.config.seed
        if actual_seed is not None:
            seed_everything(actual_seed)
        
        # Initialize
        full_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        sliding_window = gen_config.sliding_window
        
        telemetry = TelemetryData()
        telemetry.token_ids = full_ids.tolist()
        
        # For early stopping
        baseline_entropy = None
        low_entropy_count = 0
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Context window
                ctx = full_ids[-sliding_window:] if len(full_ids) > sliding_window else full_ids
                out = self.model(ctx.unsqueeze(0))
                logits = out.logits[0, -1, :]
                
                # Record entropy
                ent = token_entropy_from_logits(logits)
                telemetry.entropy_series.append(ent)
                
                # Store distribution if requested
                if store_distributions:
                    dist = get_token_distribution(logits)
                    # Compress: keep only top-k for memory
                    top_k = 1000
                    top_indices = np.argsort(dist)[-top_k:]
                    sparse_dist = np.zeros_like(dist)
                    sparse_dist[top_indices] = dist[top_indices]
                    telemetry.distributions.append(sparse_dist)
                
                # Sample next token
                next_id = nucleus_sample(
                    logits,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                )
                
                full_ids = torch.cat([full_ids, torch.tensor([next_id], device=self.device)])
                telemetry.token_ids.append(next_id)
                
                # Record repetition metrics
                token_list = full_ids.tolist()
                telemetry.repetition_series.append(
                    repetition_rate(token_list, window=200)
                )
                telemetry.ngram_rep_series.append(
                    ngram_repetition_rate(token_list, n=4, window=200)
                )
                
                # Establish baseline
                if step == self.config.metrics.baseline_window - 1:
                    baseline_entropy = np.mean(telemetry.entropy_series)
                
                # Early stopping check
                if gen_config.early_stop_enabled and baseline_entropy is not None:
                    # Check n-gram repetition
                    if telemetry.ngram_rep_series[-1] > gen_config.early_stop_ngram_threshold:
                        telemetry.stopped_early = True
                        telemetry.stop_reason = f"N-gram repetition exceeded {gen_config.early_stop_ngram_threshold}"
                        telemetry.stop_step = step
                        break
                    
                    # Check entropy collapse
                    if ent < baseline_entropy * gen_config.early_stop_entropy_ratio:
                        low_entropy_count += 1
                        if low_entropy_count >= gen_config.early_stop_patience:
                            telemetry.stopped_early = True
                            telemetry.stop_reason = f"Entropy below {gen_config.early_stop_entropy_ratio}x baseline for {gen_config.early_stop_patience} steps"
                            telemetry.stop_step = step
                            break
                    else:
                        low_entropy_count = 0
                
                # Progress callback
                if progress_callback and step % 50 == 0:
                    progress_callback(step, max_tokens)
        
        telemetry.total_steps = len(telemetry.entropy_series)
        
        # Decode text
        text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        
        # Build config dict
        config_dict = {
            'prompt_length': len(prompt),
            'max_tokens': max_tokens,
            'actual_tokens': telemetry.total_steps,
            'seed': actual_seed,
            'temperature': gen_config.temperature,
            'top_p': gen_config.top_p,
            'sliding_window': sliding_window,
            'early_stop_enabled': gen_config.early_stop_enabled,
            'stopped_early': telemetry.stopped_early,
            'stop_reason': telemetry.stop_reason,
        }
        
        return GenerationResult(
            text=text,
            telemetry=telemetry,
            config=config_dict,
        )


# Model cache
_MODEL_CACHE = {}


def get_model(model_name: str, device: Optional[str] = None):
    """Load model with caching."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = (model_name, device)
    
    if cache_key not in _MODEL_CACHE:
        print(f"Loading {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[cache_key] = (model, tokenizer)
        print(f"✅ {model_name} loaded")
    
    return _MODEL_CACHE[cache_key][0], _MODEL_CACHE[cache_key][1], device


def create_generator(model_name: str, config: Optional[RunConfig] = None) -> Generator:
    """Create a Generator instance with loaded model."""
    model, tokenizer, device = get_model(model_name)
    return Generator(model, tokenizer, device, config)
