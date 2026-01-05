---
title: "🌋 Seismic Precursors of LLM Degradation v2.1"
emoji: 🌋
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
tags:
  - llm
  - telemetry
  - fractal
  - hurst-exponent
  - higuchi-dimension
  - spectral-entropy
  - early-warning
  - long-range-dependence
  - qo3-fio
short_description: "Advanced fractal diagnostics for LLM collapse detection v2.1"
---

# 🌋 Seismic Precursors of LLM Degradation v2.1

**Advanced Fractal Diagnostic Instrument for AI Phase Transitions**

## What's New in v2.1

| Feature | Description |
|---------|-------------|
| 🎲 **Reproducible Seeds** | Full determinism with seed control |
| 🔄 **N-gram Repetition** | Stricter loop detection (not fooled by common tokens) |
| 📊 **KL Drift** | Distribution change tracking |
| ⚡ **Early Stopping** | Auto-stop on degradation signals |
| 📈 **Phase Portrait** | H-D trajectory visualization |
| 📦 **Full Export** | Comprehensive reproducibility package |
| ⚠️ **Scientific Disclaimers** | Honest about heuristics vs rigorous methods |

## Metrics

| Metric | Type | Signal | Notes |
|--------|------|--------|-------|
| Token Entropy | Information | Early | Shannon H |
| Hurst (H) | Fractal Memory | Early | R/S heuristic; DFA recommended |
| Higuchi (D) | Complexity | Early | D ≈ 2-H (sanity check) |
| N-gram Rep | Loop | Late | Strict (4-gram) |
| Spectral Entropy | Frequency | Early | PSD flatness |
| 1/f β | Scale-free | Early | Empirical slope |
| KL Drift | Distribution | Early | Optional |

## Scientific Notes

### What This Is
- **Heuristic early-warning system** for LLM degradation
- Adapts concepts from complex systems / seismology
- Empirical thresholds, not absolute truths

### What This Is NOT
- Rigorous statistical analysis (no DFA, no CI by default)
- Universal detector (calibrate for your model/prompt)
- Replacement for proper statistical testing

### Key Relations (Theoretical)
- **D ≈ 2 - H** for self-affine processes (sanity check)
- **β ≈ 2H - 1** for fGn (we use β as empirical slope)

## Related Publications

| DOI | Title |
|-----|-------|
| [10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167) | QO3/FIO v2.2 |
| [10.5281/zenodo.18110450](https://doi.org/10.5281/zenodo.18110450) | LRD Analysis |
| [10.5281/zenodo.18102168](https://doi.org/10.5281/zenodo.18102168) | Fractal Operator Ψ |
| [10.5281/zenodo.18101985](https://doi.org/10.5281/zenodo.18101985) | FIO Universal Constants |
| [10.5281/zenodo.18072672](https://doi.org/10.5281/zenodo.18072672) | QO3 Implementation |
| [10.5281/zenodo.18018292](https://doi.org/10.5281/zenodo.18018292) | Hurst Estimation |
| [10.5281/zenodo.17800770](https://doi.org/10.5281/zenodo.17800770) | LRD Framework |

**Community:** [zenodo.org/communities/lrd-time-series](https://zenodo.org/communities/lrd-time-series)

---

**Author:** Igor Chechelnitsky • [ORCID: 0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)

*"We adapt early-warning telemetry concepts from complex systems to detect regime shifts in LLM generation."*
