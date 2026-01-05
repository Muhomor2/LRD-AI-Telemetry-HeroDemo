# Methodology: Seismic Precursors of LLM Degradation

## Theoretical Foundation

### The QO3/FIO Framework

This work applies the **Quantum Ontological Observatory (QO3)** and **Fractal Information Ontology (FIO)** frameworks to AI systems. Originally developed for seismic regime detection, these frameworks provide a rigorous mathematical basis for understanding phase transitions in complex systems.

**Key publications:**
- [Fractal-Informational Seismic Regime Detection v2.2](https://doi.org/10.5281/zenodo.18145167)
- [Fractal Information Ontology: Universal Constants](https://doi.org/10.5281/zenodo.18101985)
- [The Fractal Composition Operator Ψ](https://doi.org/10.5281/zenodo.18102168)

### The Seismic-AI Analogy

| Seismic Domain | AI Domain | Mathematical Link |
|----------------|-----------|-------------------|
| Earthquake catalog | Token sequence | Time series of events |
| b-value (Gutenberg-Richter) | Token probability distribution | Power-law exponent |
| Foreshocks | Entropy/Hurst anomalies | Precursor signals |
| Mainshock | Repetition collapse | Regime transition |
| Long-Range Dependence | Persistent patterns | Hurst exponent H > 0.5 |
| Seismic Information Deficit | Entropy deficit | Cumulative "lost" information |

---

## Metrics

### 1. Token Entropy

**Definition:** Shannon entropy of the next-token probability distribution.

```
H(t) = -Σ p(x) log p(x)
```

where `p(x)` is the softmax probability for token `x` at step `t`.

**Interpretation:**
- High entropy (~8-10 nats): Model is uncertain, exploring many options
- Low entropy (<5 nats): Model is confident, narrowing focus
- Dropping entropy before collapse: Model becoming "too certain" about limited options

**Why it matters:** Entropy drop often precedes visible repetition by 200-800 tokens.

### 2. Repetition Rate

**Definition:** Fraction of tokens in the last window that appeared earlier in the sequence.

```
R(t) = |{x ∈ tail(t) : x ∈ prefix(t)}| / |tail(t)|
```

**Parameters:**
- Window size: 200 tokens (default)
- Threshold: 0.55 (default)

**Interpretation:**
- R < 0.3: Normal generation
- R > 0.5: Significant repetition
- R > 0.7: Severe looping

**Limitation:** This is a **late** signal — by the time R rises, collapse is visible.

### 3. Hurst Exponent (H)

**Definition:** Measure of long-range dependence in a time series, estimated via R/S (Rescaled Range) analysis.

```
E[R(n)/S(n)] ~ C · n^H
```

where:
- R(n) = range of cumulative deviations from mean
- S(n) = standard deviation
- n = time scale

**Interpretation:**

| H value | Meaning | LLM State |
|---------|---------|-----------|
| H < 0.5 | Anti-persistent (mean-reverting) | Rare in LLM generation |
| H ≈ 0.5 | Random walk | Healthy, diverse generation |
| 0.5 < H < 0.65 | Weak persistence | Normal |
| H > 0.65 | Strong persistence | **Early warning** |
| H > 0.80 | Near-deterministic | Collapse imminent |

**Why it's an early signal:** H measures the "memory" in the entropy time series. When the model starts trending toward collapse, entropy becomes more autocorrelated — H rises before repetition becomes visible.

**Reference:** [Hurst Exponent Estimation Methods](https://doi.org/10.5281/zenodo.18018292)

### 4. Seismic Information Deficit (SID)

**Definition:** Cumulative entropy deficit relative to baseline.

```
SID(t) = Σ max(H_baseline - H(τ), 0) for τ = 1 to t
```

where `H_baseline` is the mean entropy in the first 100 tokens.

**Interpretation:**
- Rising SID: Model is "losing information capacity"
- Sharp SID increase: Rapid transition toward collapse
- SID plateau: Stable generation regime

**Origin:** Adapted from seismic information deficit metrics in QO3/FIO framework.

---

## Algorithm

### Generation Loop

```python
for step in range(max_new_tokens):
    # 1. Get next-token distribution
    logits = model(context)[-1]
    
    # 2. Record entropy
    entropy = shannon_entropy(softmax(logits))
    
    # 3. Sample next token
    next_token = nucleus_sample(logits, top_p=0.95)
    
    # 4. Update sequence
    sequence.append(next_token)
    
    # 5. Calculate repetition rate
    rep_rate = repetition_rate(sequence, window=200)
    
    # 6. Every N steps: calculate Hurst on entropy series
    if step % 100 == 0 and step > 200:
        H = hurst_exponent(entropy_series[-200:])
```

### Early Warning Detection

```python
def detect_warning(entropy_series, hurst_series):
    # Check for entropy drop below 70% of baseline
    baseline = mean(entropy_series[:100])
    entropy_warning = first(i for i, e in enumerate(entropy_series) 
                            if e < 0.7 * baseline)
    
    # Check for Hurst exceeding 0.65
    hurst_warning = first(pos for pos, h in zip(positions, hurst_series)
                          if h > 0.65)
    
    # Return earliest signal
    return min(entropy_warning, hurst_warning)
```

---

## Sliding Window Context

For models with limited context (GPT-2: 1024 tokens), we use a sliding window:

```python
sliding_window = 800

if len(sequence) > sliding_window:
    context = sequence[-sliding_window:]
else:
    context = sequence
```

This allows generating arbitrarily long sequences while staying within model limits.

---

## Empirical Observations

Based on experiments with distilgpt2, gpt2-medium, and larger models:

### Typical Lead Times

| Model | Early Warning → Collapse | Observation |
|-------|-------------------------|-------------|
| distilgpt2 | 300-800 tokens | Collapses quickly |
| gpt2-medium | 500-1200 tokens | More gradual |
| Llama-3.2-3B | 800-2000 tokens | Often avoids collapse |
| Qwen2.5-7B | 1000-3000 tokens | Robust, longer context |

### Characteristic Patterns

1. **Pre-collapse entropy profile:**
   - Gradual decrease over 500-1000 tokens
   - Increased variance (varentropy) before drop
   - Rolling mean crosses below baseline

2. **Pre-collapse Hurst profile:**
   - Steady rise from ~0.5 to >0.65
   - Acceleration in final 200-400 tokens
   - Often crosses 0.8 just before visible repetition

3. **SID behavior:**
   - Slow accumulation during healthy generation
   - Rapid increase during collapse

---

## Threshold Selection

### Default Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Repetition | 0.55 | Visible looping typically starts here |
| Hurst warning | 0.65 | 2σ above random walk (0.5) |
| Hurst critical | 0.80 | Near-deterministic regime |
| Entropy drop | 0.70 × baseline | Significant confidence increase |

### Tuning Guidelines

- **More sensitive (more false positives):** Lower Hurst to 0.60, raise entropy ratio to 0.75
- **Less sensitive (fewer false positives):** Raise Hurst to 0.70, lower entropy ratio to 0.65
- **Model-specific:** Larger models may need higher thresholds

---

## Limitations

1. **Stochastic sampling:** Results vary between runs due to nucleus sampling
2. **Prompt sensitivity:** Some prompts don't trigger collapse
3. **Model architecture:** Behavior differs significantly across architectures
4. **Computational cost:** Hurst calculation adds ~5% overhead

## Future Work

1. **DFA (Detrended Fluctuation Analysis):** Alternative to R/S for Hurst estimation
2. **Spectral entropy:** Power-law fit on entropy spectrum
3. **Real-time integration:** vLLM/TGI middleware for production monitoring
4. **Larger models:** Systematic study of scaling behavior

---

## References

1. Chechelnitsky, I. (2026). Fractal-Informational Seismic Regime Detection (QO3/FIO) v2.2. Zenodo. https://doi.org/10.5281/zenodo.18145167

2. Chechelnitsky, I. (2026). Long-Range Dependence Analysis: DFA vs R/S Comparison. Zenodo. https://doi.org/10.5281/zenodo.18110450

3. Chechelnitsky, I. (2026). The Fractal Composition Operator Ψ: Definition and Axiomatic Foundation. Zenodo. https://doi.org/10.5281/zenodo.18102168

4. Chechelnitsky, I. (2026). Fractal Information Ontology: Universal Constants. Zenodo. https://doi.org/10.5281/zenodo.18101985

5. Chechelnitsky, I. (2026). Quantum Ontological Observatory Implementation. Zenodo. https://doi.org/10.5281/zenodo.18072672

6. Chechelnitsky, I. (2026). Hurst Exponent Estimation Methods. Zenodo. https://doi.org/10.5281/zenodo.18018292

7. Chechelnitsky, I. (2025). LRD in Complex Systems: Theoretical Framework. Zenodo. https://doi.org/10.5281/zenodo.17800770

---

**Community:** [zenodo.org/communities/lrd-time-series](https://zenodo.org/communities/lrd-time-series/records)
