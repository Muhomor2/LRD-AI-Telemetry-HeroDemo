# Seismic Precursors of LLM Degradation: LRD/QO3/FIO Bridge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18145167.svg)](https://doi.org/10.5281/zenodo.18145167)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Minimal reproducible demo:** Token-level entropy, Hurst exponent, and repetition rate as **early-warning signals** of degradation in long-form LLM generation — applying fractal/seismic methodology to AI telemetry.

## Why This Matters (2026)

| Problem | Solution |
|---------|----------|
| LLM long-context sessions degrade unpredictably (loops, repetition, hallucinations) | Fractal metrics detect degradation **300–1200 tokens early** |
| Test-time compute scaling lacks stop-criteria | Hurst exponent + entropy provide cheap real-time monitoring |
| No theoretical framework for LLM "phase transitions" | QO3/FIO bridges seismology, fractals, and AI dynamics |

## Key Insight

> **When the model starts looping, repetition rate rises sharply — but Hurst exponent and entropy often shift earlier, acting as "seismic precursors" before the visible collapse.**

This is analogous to how b-value changes and foreshock patterns precede major earthquakes.

---

## Quickstart

### Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/LRD-AI-Telemetry-HeroDemo.git
cd LRD-AI-Telemetry-HeroDemo

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python demo.py --model distilgpt2 --max_new_tokens 2500
```

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/LRD-AI-Telemetry-HeroDemo/blob/main/notebooks/hero_demo.ipynb)

Open `notebooks/hero_demo.ipynb` and run all cells.

---

## Output Examples

After running `demo.py`, you'll find in `outputs/`:

| File | Description |
|------|-------------|
| `entropy.png` | Token entropy over generation steps |
| `repetition.png` | Repetition rate with threshold line |
| `hurst.png` | Rolling Hurst exponent (fractal memory indicator) |
| `combined_dashboard.png` | All metrics in one view |

**Console output:**
- First step where repetition > threshold
- Mean entropy in 100 steps before that point
- Hurst exponent trend before collapse
- Last 1200 chars of generated text (to see loops)

---

## Metrics Explained

| Metric | What It Measures | Early Warning Signal |
|--------|------------------|---------------------|
| **Token Entropy** | Uncertainty in next-token distribution | Drops before repetition (model becomes "too confident") |
| **Repetition Rate** | Fraction of tokens seen before in window | Direct measure of looping — but late signal |
| **Hurst Exponent (H)** | Long-range dependence / fractal memory | H > 0.65 indicates persistent behavior → precursor |
| **SID (Entropy Deficit)** | Cumulative information compression | Rising SID = model "running out of ideas" |

### Interpretation Guide

```
H ≈ 0.5  →  Random walk (healthy generation)
H > 0.65 →  Persistent / trending (early warning)
H > 0.80 →  Near-deterministic (collapse imminent)
```

---

## Theoretical Foundation: QO3/FIO Framework

This demo applies concepts from the **Fractal Information Ontology (FIO)** and **Quantum Ontological Observatory (QO3)** frameworks to AI systems:

| Seismic Concept | AI Analogue |
|-----------------|-------------|
| b-value (Gutenberg-Richter) | Token probability distribution shape |
| Foreshocks | Entropy/Hurst anomalies before collapse |
| Mainshock | Repetition loop onset |
| Long-Range Dependence | Persistent patterns in hidden states |

The Fractal Composition Operator **Ψ** provides theoretical grounding for how multi-scale information combines in complex systems — applicable to both geological and neural dynamics.

---

## Related Publications (Zenodo)

This work is part of the **LRD Time Series** research community. Full methodology and theoretical foundations:

| Record | Title | DOI |
|--------|-------|-----|
| Core Method | Fractal-Informational Seismic Regime Detection (QO3/FIO) v2.2 | [10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167) |
| LRD Analysis | Long-Range Dependence Analysis: DFA vs R/S Comparison | [10.5281/zenodo.18110450](https://doi.org/10.5281/zenodo.18110450) |
| Fractal Operator | The Fractal Composition Operator Ψ: Definition and Axiomatic Foundation | [10.5281/zenodo.18102168](https://doi.org/10.5281/zenodo.18102168) |
| FIO Theory | Fractal Information Ontology: Universal Constants | [10.5281/zenodo.18101985](https://doi.org/10.5281/zenodo.18101985) |
| QO3 Framework | Quantum Ontological Observatory Implementation | [10.5281/zenodo.18072672](https://doi.org/10.5281/zenodo.18072672) |
| Hurst Analysis | Hurst Exponent Estimation Methods | [10.5281/zenodo.18018292](https://doi.org/10.5281/zenodo.18018292) |
| Foundation | LRD in Complex Systems: Theoretical Framework | [10.5281/zenodo.17800770](https://doi.org/10.5281/zenodo.17800770) |

**Community:** [zenodo.org/communities/lrd-time-series](https://zenodo.org/communities/lrd-time-series/records)

---

## Project Structure

```
LRD-AI-Telemetry-HeroDemo/
├── README.md
├── requirements.txt
├── demo.py                    # Main entry point
├── CITATION.cff               # GitHub citation
├── zenodo.json                # Zenodo metadata
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── metrics.py             # Entropy, repetition, Hurst
│   ├── generate.py            # Generation with telemetry
│   └── visualization.py       # Plotting utilities
├── notebooks/
│   └── hero_demo.ipynb        # Colab-ready notebook
├── outputs/                   # Generated plots (gitignored)
└── docs/
    └── METHODOLOGY.md         # Detailed methodology
```

---

## Advanced Usage

### Custom Models

```bash
# Test with larger models (requires GPU)
python demo.py --model Qwen/Qwen2.5-7B-Instruct --max_new_tokens 5000

# Adjust sensitivity
python demo.py --rep_threshold 0.50 --hurst_warning 0.68
```

### Programmatic API

```python
from src.generate import generate_with_metrics
from src.metrics import hurst_exponent

result = generate_with_metrics(
    model=model,
    tokenizer=tokenizer,
    prompt="Your prompt here...",
    max_new_tokens=3000,
    rep_threshold=0.55,
)

# Access metrics
print(f"Collapse at step: {result.first_exceed_step}")
print(f"Hurst before collapse: {result.hurst_before}")
print(f"Entropy trend: {result.entropy_trend}")
```

---

## Contributing

Contributions welcome! Especially:
- Testing on more models (DeepSeek-R1, Llama-3.3, Gemma-2)
- Additional fractal metrics (DFA α, spectral entropy)
- Integration with inference frameworks (vLLM, TGI)

---

## Citation

If you use this work, please cite:

```bibtex
@software{chechelnitsky_2026_llm_precursors,
  author       = {Chechelnitsky, Igor},
  title        = {Seismic Precursors of LLM Degradation: LRD/QO3/FIO Bridge},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18145167},
  url          = {https://doi.org/10.5281/zenodo.18145167}
}
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Author

**Igor Chechelnitsky**  
Independent Researcher, Ashkelon, Israel  
ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)

---

*"The same fractal patterns that predict earthquakes can predict when an AI will start repeating itself."*
