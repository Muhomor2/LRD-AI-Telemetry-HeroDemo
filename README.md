# 🌋 Seismic Precursors of LLM Degradation v2.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18145167.svg)](https://doi.org/10.5281/zenodo.18145167)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/LRD-LLM-Precursors)

**Advanced Fractal Diagnostic Instrument for AI Phase Transitions**

We adapt early-warning telemetry concepts from complex systems (seismology, econophysics) to detect regime shifts in LLM generation — predicting repetition collapse **300–1200 tokens before** it becomes visible.

---

## 🎯 Key Insight

> When an LLM starts looping, the repetition rate rises sharply — but **Hurst exponent and entropy often shift earlier**, acting as "seismic precursors" before the visible collapse.

This is analogous to how b-value changes and foreshock patterns precede major earthquakes.

---

## 📊 Metrics Suite

| Metric | Domain | Signal Type | Interpretation |
|--------|--------|-------------|----------------|
| **Token Entropy** | Information | Early | Drops when model becomes "too confident" |
| **Hurst Exponent (H)** | Fractal/Time | Early | H > 0.65 = persistent patterns (warning) |
| **Higuchi Dimension (D)** | Fractal/Geometry | Early | D < 1.3 = increasing regularity (warning) |
| **N-gram Repetition** | Sequence | Late | Strict loop detection (4-grams) |
| **Spectral Entropy** | Frequency | Early | Low = narrowband signal (warning) |
| **1/f Exponent (β)** | Spectral | Early | β > 1.5 = strong correlations |
| **KL Drift** | Distribution | Early | Rising = regime instability |

### Interpretation Guide

```
Hurst (H):
  H ≈ 0.5   → Random walk (healthy generation)
  H > 0.65  → ⚠️ Persistent patterns (early warning)
  H > 0.80  → 🔴 Near-deterministic (collapse imminent)

Higuchi (D):
  D ≈ 1.5   → Complex, healthy signal
  D < 1.3   → ⚠️ Increasing regularity (warning)
  D → 1.0   → 🔴 Near-periodic (collapse)

Theoretical relation: D ≈ 2 - H (sanity check for self-affine processes)
```

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/LRD-AI-Telemetry-v2.1.git
cd LRD-AI-Telemetry-v2.1

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Interactive Demo

```bash
python app.py
# Opens Gradio interface at http://localhost:7860
```

### Programmatic Usage

```python
from config import RunConfig
from generate import create_generator
from metrics import rolling_metrics, compute_warning_score

# Configure
config = RunConfig(model_name="distilgpt2", seed=42)
config.generation.max_tokens = 2000

# Generate with telemetry
generator = create_generator("distilgpt2", config)
result = generator.generate(
    prompt="Explain time series and regime shifts in detail...",
    seed=42,
)

# Analyze
import numpy as np
entropy = np.array(result.telemetry.entropy_series)
rolling = rolling_metrics(entropy, window=200, step=50)

# Get warning score
h, d = rolling['hurst'][-1], rolling['higuchi'][-1]
se, beta = rolling['spectral_entropy'][-1], rolling['one_over_f'][-1]
score, level = compute_warning_score(h, d, se, beta)

print(f"Warning Score: {score:.1f}/100 ({level})")
print(f"Final Hurst: {h:.3f}, Higuchi: {d:.3f}")
```

---

## 📁 Project Structure

```
LRD-AI-Telemetry-v2.1/
├── app.py              # Gradio UI (clean, no business logic)
├── config.py           # Centralized thresholds + disclaimers
├── metrics.py          # All fractal metrics + bootstrap CI
├── generate.py         # Text generation + telemetry + early stop
├── viz.py              # Plotly visualizations
├── export.py           # Reproducibility package (ZIP)
├── requirements.txt
└── README.md
```

---

## ⚠️ Scientific Notes

### What This Is

- **Heuristic early-warning system** for LLM degradation
- Adapts concepts from complex systems / seismology to AI
- **Empirical thresholds** derived from preliminary experiments

### What This Is NOT

- Rigorous statistical analysis (DFA with CI recommended for papers)
- Universal detector (calibrate for your model/prompt/task)
- Replacement for proper statistical hypothesis testing

### Methodological Caveats

| Method | Caveat | Recommendation |
|--------|--------|----------------|
| **Hurst (R/S)** | Quick heuristic; can overestimate H on trends | Use DFA(1/2) for rigorous analysis |
| **β ↔ H relation** | β ≈ 2H-1 holds for fGn, not always for real data | Treat β as empirical spectral slope |
| **D ≈ 2-H** | Exact for ideal self-affine processes | Use as sanity check only |
| **Thresholds** | Empirical, not universal | Calibrate per model/prompt |
| **Fibonacci windows** | Experimental / hypothesis-driven | Compare with standard windows |

### Entropy Non-Stationarity

The entropy time series from LLM generation is typically **non-stationary** and **regime-dependent**. R/S Hurst is sensitive to trends and can show "false LRD". We recommend:

1. Use metrics as **regime indicators**, not absolute measures
2. Interpret with caution on short sequences (< 500 tokens)
3. For publication, add DFA and bootstrap confidence intervals

---

## 📦 Export Package

The tool generates a comprehensive reproducibility package (ZIP):

```
export_package/
├── README.md              # Run summary
├── metadata.json          # Config + results + disclaimers
├── environment.json       # Python/library versions
├── generated_text.txt     # Full output
├── token_ids.json         # Token sequence
├── metrics_per_step.csv   # Entropy, repetition per step
├── rolling_metrics.csv    # Hurst, Higuchi, spectral, 1/f
├── kl_drift.csv           # KL divergence (if computed)
├── reproduce.py           # Script to reproduce analysis
└── requirements.txt       # Dependencies
```

---

## 🔬 Theoretical Foundation

### QO3/FIO Framework

This work applies the **Quantum Ontological Observatory (QO3)** and **Fractal Information Ontology (FIO)** frameworks to AI systems:

| Seismic Concept | AI Analogue | Mathematical Link |
|-----------------|-------------|-------------------|
| Earthquake catalog | Token sequence | Time series of events |
| b-value (Gutenberg-Richter) | Token probability shape | Power-law exponent |
| Foreshocks | Entropy/Hurst anomalies | Precursor signals |
| Mainshock | Repetition collapse | Phase transition |
| Long-Range Dependence | Persistent patterns | Hurst H > 0.5 |

### Universal Attractor Hypothesis

The FIO framework proposes that complex systems converge to universal fractal attractors:

- **H ≈ 0.65** (Hurst exponent)
- **D ≈ 1.35** (fractal dimension)

Deviations from these values may indicate regime transitions or system instability.

---

## 📚 Related Publications

| DOI | Title |
|-----|-------|
| [10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167) | Fractal-Informational Seismic Regime Detection (QO3/FIO) v2.2 |
| [10.5281/zenodo.18110450](https://doi.org/10.5281/zenodo.18110450) | Long-Range Dependence Analysis: DFA vs R/S |
| [10.5281/zenodo.18102168](https://doi.org/10.5281/zenodo.18102168) | The Fractal Composition Operator Ψ |
| [10.5281/zenodo.18101985](https://doi.org/10.5281/zenodo.18101985) | Fractal Information Ontology: Universal Constants |
| [10.5281/zenodo.18072672](https://doi.org/10.5281/zenodo.18072672) | Quantum Ontological Observatory Implementation |
| [10.5281/zenodo.18018292](https://doi.org/10.5281/zenodo.18018292) | Hurst Exponent Estimation Methods |
| [10.5281/zenodo.17800770](https://doi.org/10.5281/zenodo.17800770) | LRD in Complex Systems: Theoretical Framework |

**Community:** [zenodo.org/communities/lrd-time-series](https://zenodo.org/communities/lrd-time-series)

---

## 🤝 Contributing

Contributions welcome! Priority areas:

- [ ] DFA implementation (parallel to R/S)
- [ ] Bootstrap CI for all rolling metrics
- [ ] Integration with vLLM / TGI
- [ ] Benchmarks on more models (Llama, Qwen, DeepSeek)
- [ ] Auto-calibration of thresholds

---

## 📜 Citation

```bibtex
@software{chechelnitsky_2026_lrd_telemetry,
  author       = {Chechelnitsky, Igor},
  title        = {Seismic Precursors of LLM Degradation: LRD/QO3/FIO Bridge},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {2.1},
  doi          = {10.5281/zenodo.18145167},
  url          = {https://doi.org/10.5281/zenodo.18145167}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Igor Chechelnitsky**  
Independent Researcher, Ashkelon, Israel  
ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)

---

*"We adapt early-warning telemetry concepts from complex systems to detect regime shifts in LLM generation."*
