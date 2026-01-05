"""
🌋 Seismic Precursors of LLM Degradation v2.1
Advanced Fractal Diagnostic Instrument

Clean Gradio interface with modular architecture.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
DOI: 10.5281/zenodo.18145167
"""

import gradio as gr
import numpy as np
from typing import Tuple, Optional
from datetime import datetime

from config import RunConfig, Thresholds, DISCLAIMERS
from metrics import (
    rolling_metrics,
    compute_warning_score,
    entropy_deficit_sid,
    compute_kl_drift,
    compute_metrics_fibonacci_windows,
    seed_everything,
)
from generate import create_generator, GenerationResult
from viz import (
    create_main_dashboard,
    create_warning_gauge,
    create_phase_portrait,
    create_sid_plot,
)
from export import create_export_package


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(
    model_name: str,
    max_tokens: int,
    seed: Optional[int],
    temperature: float,
    top_p: float,
    hurst_warning: float,
    rep_threshold: float,
    use_early_stop: bool,
    use_fibonacci: bool,
    compute_kl: bool,
    custom_prompt: str,
    progress=gr.Progress(),
) -> Tuple:
    """
    Main analysis pipeline.
    
    Returns: (dashboard, gauge, phase_portrait, summary, text_preview, export_file)
    """
    
    # Build config
    config = RunConfig(
        model_name=model_name,
        seed=seed if seed and seed > 0 else None,
    )
    config.generation.max_tokens = max_tokens
    config.generation.temperature = temperature
    config.generation.top_p = top_p
    config.generation.early_stop_enabled = use_early_stop
    config.thresholds.hurst_warning = hurst_warning
    config.thresholds.repetition_warning = rep_threshold
    config.use_fibonacci_windows = use_fibonacci
    config.compute_kl_drift = compute_kl
    
    # Prompt
    if custom_prompt.strip():
        prompt = custom_prompt
    else:
        prompt = (
            "Repeat the explanation with more detail and examples, "
            "do not stop, keep expanding.\n\n"
            "Explain time series, regime shifts, and why early warning "
            "signals matter. Describe how complex systems transition "
            "from one state to another.\n\n"
        )
    
    progress(0.05, desc="Loading model...")
    
    # Create generator
    try:
        generator = create_generator(model_name, config)
    except Exception as e:
        return None, None, None, f"❌ Error loading model: {e}", "", None
    
    progress(0.1, desc="Generating...")
    
    # Generate
    def progress_cb(step, total):
        progress(0.1 + 0.6 * (step / total), desc=f"Step {step}/{total}")
    
    result = generator.generate(
        prompt=prompt,
        seed=config.seed,
        progress_callback=progress_cb,
        store_distributions=compute_kl,
    )
    
    progress(0.75, desc="Computing fractal metrics...")
    
    # Compute rolling metrics
    entropy_array = np.array(result.telemetry.entropy_series)
    rolling_data = rolling_metrics(
        entropy_array,
        window=config.metrics.rolling_window,
        step=config.metrics.rolling_step,
        compute_ci=False,
    )
    
    # SID
    sid_series = entropy_deficit_sid(entropy_array, config.metrics.baseline_window)
    
    # KL drift (if enabled)
    kl_positions, kl_values = [], []
    if compute_kl and result.telemetry.distributions:
        kl_positions, kl_values = compute_kl_drift(
            result.telemetry.distributions,
            step=10,
        )
    
    # Fibonacci analysis (if enabled)
    fib_results = None
    if use_fibonacci and len(entropy_array) > 200:
        fib_results = compute_metrics_fibonacci_windows(entropy_array)
    
    progress(0.85, desc="Detecting warnings...")
    
    # Find collapse (n-gram based, stricter)
    ngram_series = result.telemetry.ngram_rep_series
    first_collapse = next(
        (i for i, v in enumerate(ngram_series) if v > config.thresholds.ngram_warning),
        None
    )
    
    # Find early warning (Hurst-based)
    first_warning = None
    for i, (pos, h) in enumerate(zip(rolling_data['positions'], rolling_data['hurst'])):
        if h > hurst_warning:
            first_warning = pos
            break
    
    # Lead time
    lead_time = None
    if first_warning is not None and first_collapse is not None:
        if first_warning < first_collapse:
            lead_time = first_collapse - first_warning
    
    # Warning scores
    final_score, final_level = 0.0, 'normal'
    if rolling_data['positions']:
        h = rolling_data['hurst'][-1]
        d = rolling_data['higuchi'][-1]
        se = rolling_data['spectral_entropy'][-1]
        beta = rolling_data['one_over_f'][-1]
        ngram_final = ngram_series[-1] if ngram_series else 0.0
        kl_final = kl_values[-1] if kl_values else 0.0
        
        final_score, final_level = compute_warning_score(
            h, d, se, beta, ngram_final, kl_final
        )
    
    progress(0.90, desc="Creating visualizations...")
    
    # Create figures
    thresholds = config.thresholds
    
    fig_dashboard = create_main_dashboard(
        result.telemetry.entropy_series,
        result.telemetry.repetition_series,
        ngram_series,
        rolling_data,
        thresholds,
        first_collapse,
        first_warning,
        kl_positions if compute_kl else None,
        kl_values if compute_kl else None,
    )
    
    fig_gauge = create_warning_gauge(final_score, final_level)
    
    fig_phase = None
    if rolling_data['positions'] and len(rolling_data['hurst']) > 2:
        fig_phase = create_phase_portrait(
            rolling_data['hurst'],
            rolling_data['higuchi'],
            rolling_data['positions'],
            thresholds,
        )
    
    progress(0.95, desc="Building summary...")
    
    # Summary
    summary_lines = ["## 📊 Analysis Results v2.1\n"]
    summary_lines.append(f"**Model:** {model_name}")
    summary_lines.append(f"**Seed:** {config.seed or 'random'}")
    summary_lines.append(f"**Tokens generated:** {result.telemetry.total_steps}")
    
    if result.telemetry.stopped_early:
        summary_lines.append(f"\n⚡ **Early stop:** {result.telemetry.stop_reason}")
    
    summary_lines.append(f"\n### Telemetry Summary")
    summary_lines.append(f"- Mean entropy: {np.mean(entropy_array):.3f}")
    summary_lines.append(f"- Final n-gram repetition: {ngram_series[-1]:.3f}")
    
    if rolling_data['hurst']:
        h_final = rolling_data['hurst'][-1]
        d_final = rolling_data['higuchi'][-1]
        summary_lines.append(f"\n### Fractal Metrics (final)")
        summary_lines.append(f"- **Hurst (H):** {h_final:.3f}")
        summary_lines.append(f"- **Higuchi (D):** {d_final:.3f}")
        summary_lines.append(f"- **Spectral Entropy:** {rolling_data['spectral_entropy'][-1]:.3f}")
        summary_lines.append(f"- **1/f β:** {rolling_data['one_over_f'][-1]:.3f}")
        
        d_theoretical = 2 - h_final
        delta = abs(d_final - d_theoretical)
        summary_lines.append(f"\n**Sanity check:** D ≈ 2-H → {d_theoretical:.3f} (actual: {d_final:.3f}, Δ={delta:.3f})")
    
    if first_warning is not None:
        summary_lines.append(f"\n### ⚠️ Early Warning")
        summary_lines.append(f"Detected at step **{first_warning}** (H > {hurst_warning})")
        if lead_time:
            summary_lines.append(f"**Lead time: {lead_time} tokens** before collapse! 🎯")
    
    if first_collapse is not None:
        summary_lines.append(f"\n### 🔴 Collapse")
        summary_lines.append(f"N-gram repetition exceeded threshold at step **{first_collapse}**")
    else:
        summary_lines.append("\n### ✅ No collapse detected")
    
    if fib_results and fib_results['windows']:
        summary_lines.append(f"\n### 🌀 Fibonacci Windows (experimental)")
        for w, h, d in zip(fib_results['windows'][-3:], 
                          fib_results['hurst'][-3:], 
                          fib_results['higuchi'][-3:]):
            summary_lines.append(f"- Window {w}: H={h:.3f}, D={d:.3f}")
    
    summary_lines.append(f"\n### 🎚️ Warning Score: {final_score:.1f}/100 ({final_level.upper()})")
    
    summary_lines.append("\n---")
    summary_lines.append(f"*{DISCLAIMERS['thresholds'][:100]}...*")
    summary_lines.append("\n*[DOI: 10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167)*")
    
    summary = "\n".join(summary_lines)
    
    # Text preview
    text_preview = f"### Last 800 characters:\n```\n{result.text[-800:]}\n```"
    
    # Export package
    analysis_results = {
        'first_warning': first_warning,
        'first_collapse': first_collapse,
        'lead_time': lead_time,
        'final_score': final_score,
        'final_level': final_level,
        'stopped_early': result.telemetry.stopped_early,
        'stop_reason': result.telemetry.stop_reason,
    }
    
    kl_data = {'positions': kl_positions, 'values': kl_values} if compute_kl else None
    
    export_bytes = create_export_package(
        text=result.text,
        token_ids=result.telemetry.token_ids,
        entropy_series=result.telemetry.entropy_series,
        rep_series=result.telemetry.repetition_series,
        ngram_rep_series=ngram_series,
        rolling_data=rolling_data,
        config=config,
        analysis_results=analysis_results,
        kl_data=kl_data,
    )
    
    # Save export
    export_path = "/tmp/lrd_export_v2.1.zip"
    with open(export_path, 'wb') as f:
        f.write(export_bytes)
    
    progress(1.0, desc="Done!")
    
    return fig_dashboard, fig_gauge, fig_phase, summary, text_preview, export_path


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

DESCRIPTION = """
# 🌋 Seismic Precursors of LLM Degradation v2.1

**Advanced Fractal Diagnostic Instrument for AI Phase Transitions**

Detects LLM degradation using multiple fractal metrics — applying earthquake science to AI.

### Metrics

| Metric | Signal | Interpretation |
|--------|--------|----------------|
| **Hurst (H)** | Early | H > 0.65 = persistent patterns (warning) |
| **Higuchi (D)** | Early | D < 1.3 = increasing regularity (warning) |
| **N-gram Rep** | Late | Strict loop detection |
| **Spectral Entropy** | Early | Low = narrowband (warning) |
| **1/f β** | Early | β > 1.5 = strong correlations |

### Scientific Notes

- Hurst via R/S is a **heuristic**; DFA recommended for rigorous analysis
- Thresholds are **empirical** — calibrate for your use case
- D ≈ 2-H is a **sanity check**, not strict validation

[DOI: 10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167) | 
[LRD Community](https://zenodo.org/communities/lrd-time-series)
"""

with gr.Blocks(title="🌋 LLM Degradation v2.1", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=["distilgpt2", "gpt2", "gpt2-medium"],
                value="distilgpt2",
                label="Model"
            )
            
            max_tokens = gr.Slider(500, 5000, 2000, 100, label="Max Tokens")
            
            seed_input = gr.Number(
                value=42,
                label="Seed (0 = random)",
                precision=0,
            )
            
            with gr.Accordion("⚙️ Sampling", open=False):
                temperature = gr.Slider(0.5, 1.5, 1.0, 0.1, label="Temperature")
                top_p = gr.Slider(0.8, 1.0, 0.95, 0.01, label="Top-p")
            
            with gr.Accordion("📊 Thresholds (empirical heuristics)", open=False):
                hurst_warning = gr.Slider(
                    0.55, 0.75, 0.65, 0.05,
                    label="Hurst Warning",
                    info="H above this triggers early warning"
                )
                rep_threshold = gr.Slider(
                    0.2, 0.5, 0.30, 0.05,
                    label="N-gram Rep Threshold",
                    info="N-gram repetition above this = collapse"
                )
            
            with gr.Accordion("🔬 Advanced", open=False):
                use_early_stop = gr.Checkbox(
                    value=True,
                    label="Early Stop on Degradation",
                    info="Stop generation when collapse detected"
                )
                use_fibonacci = gr.Checkbox(
                    value=False,
                    label="🌀 Fibonacci Windows (experimental)",
                    info="Quasicrystalline sampling"
                )
                compute_kl = gr.Checkbox(
                    value=False,
                    label="Compute KL Drift",
                    info="Track distribution changes (slower)"
                )
            
            custom_prompt = gr.Textbox(
                label="Custom Prompt (optional)",
                placeholder="Leave empty for default loop-inducing prompt...",
                lines=3
            )
            
            generate_btn = gr.Button("🚀 Generate & Analyze", variant="primary", size="lg")
            
            export_file = gr.File(label="📦 Export Package")
        
        with gr.Column(scale=2):
            with gr.Row():
                gauge_output = gr.Plot(label="Warning Score")
            
            dashboard_output = gr.Plot(label="Telemetry Dashboard")
            
            with gr.Accordion("📈 Phase Portrait (H-D Trajectory)", open=False):
                phase_output = gr.Plot(label="Phase Portrait")
            
            summary_output = gr.Markdown(label="Summary")
            text_output = gr.Markdown(label="Generated Text")
    
    generate_btn.click(
        fn=run_analysis,
        inputs=[
            model_dropdown, max_tokens, seed_input,
            temperature, top_p,
            hurst_warning, rep_threshold,
            use_early_stop, use_fibonacci, compute_kl,
            custom_prompt,
        ],
        outputs=[
            dashboard_output, gauge_output, phase_output,
            summary_output, text_output, export_file
        ],
    )
    
    gr.Markdown("""
    ---
    **Author:** Igor Chechelnitsky • [ORCID: 0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)
    
    **DOI:** [10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167) | 
    **Community:** [LRD Time Series](https://zenodo.org/communities/lrd-time-series)
    
    *We adapt early-warning telemetry concepts from complex systems to detect regime shifts in LLM generation.*
    """)


if __name__ == "__main__":
    demo.launch()
