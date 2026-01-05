"""
Export Module for LRD-AI-Telemetry v2.1

Creates comprehensive reproducibility packages with:
- All metrics data (CSV)
- Configuration (JSON)
- Environment info (JSON)
- Generated text
- Figures (PNG via Plotly)
- Reproduction script

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

import json
import zipfile
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from config import RunConfig, get_environment_info, DISCLAIMERS


def create_export_package(
    text: str,
    token_ids: List[int],
    entropy_series: List[float],
    rep_series: List[float],
    ngram_rep_series: List[float],
    rolling_data: Dict,
    config: RunConfig,
    analysis_results: Dict[str, Any],
    figures: Optional[Dict[str, bytes]] = None,
    kl_data: Optional[Dict] = None,
) -> bytes:
    """
    Create comprehensive ZIP package for reproducibility.
    
    Contents:
    - README.md
    - metadata.json (run config + results)
    - environment.json (versions)
    - generated_text.txt
    - token_ids.json
    - metrics_per_step.csv
    - rolling_metrics.csv
    - figures/*.png (if provided)
    - reproduce.py (script to reproduce analysis)
    """
    
    buffer = io.BytesIO()
    timestamp = datetime.now().isoformat()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # =====================================================================
        # Metadata
        # =====================================================================
        metadata = {
            'timestamp': timestamp,
            'doi': '10.5281/zenodo.18145167',
            'author_orcid': '0009-0007-4607-1946',
            'version': '2.1',
            'config': config.to_dict(),
            'results': {
                'total_tokens': len(entropy_series),
                'first_warning': analysis_results.get('first_warning'),
                'first_collapse': analysis_results.get('first_collapse'),
                'warning_lead_time': analysis_results.get('lead_time'),
                'final_warning_score': analysis_results.get('final_score'),
                'final_warning_level': analysis_results.get('final_level'),
                'stopped_early': analysis_results.get('stopped_early', False),
                'stop_reason': analysis_results.get('stop_reason'),
            },
            'disclaimers': DISCLAIMERS,
        }
        zf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        # =====================================================================
        # Environment
        # =====================================================================
        env_info = get_environment_info()
        zf.writestr('environment.json', json.dumps(env_info, indent=2))
        
        # =====================================================================
        # Generated text
        # =====================================================================
        zf.writestr('generated_text.txt', text)
        
        # =====================================================================
        # Token IDs
        # =====================================================================
        zf.writestr('token_ids.json', json.dumps(token_ids))
        
        # =====================================================================
        # Per-step metrics (CSV)
        # =====================================================================
        df_metrics = pd.DataFrame({
            'step': range(len(entropy_series)),
            'entropy': entropy_series,
            'repetition_rate': rep_series,
            'ngram_repetition': ngram_rep_series if ngram_rep_series else [0.0] * len(entropy_series),
        })
        zf.writestr('metrics_per_step.csv', df_metrics.to_csv(index=False))
        
        # =====================================================================
        # Rolling metrics (CSV)
        # =====================================================================
        if rolling_data.get('positions'):
            df_rolling = pd.DataFrame({
                'position': rolling_data['positions'],
                'hurst': rolling_data['hurst'],
                'higuchi': rolling_data['higuchi'],
                'spectral_entropy': rolling_data['spectral_entropy'],
                'one_over_f': rolling_data['one_over_f'],
            })
            
            # Add CI if available
            if 'hurst_ci_lower' in rolling_data:
                df_rolling['hurst_ci_lower'] = rolling_data['hurst_ci_lower']
                df_rolling['hurst_ci_upper'] = rolling_data['hurst_ci_upper']
                df_rolling['higuchi_ci_lower'] = rolling_data['higuchi_ci_lower']
                df_rolling['higuchi_ci_upper'] = rolling_data['higuchi_ci_upper']
            
            zf.writestr('rolling_metrics.csv', df_rolling.to_csv(index=False))
        
        # =====================================================================
        # KL Drift (if available)
        # =====================================================================
        if kl_data and kl_data.get('positions'):
            df_kl = pd.DataFrame({
                'position': kl_data['positions'],
                'kl_divergence': kl_data['values'],
            })
            zf.writestr('kl_drift.csv', df_kl.to_csv(index=False))
        
        # =====================================================================
        # Figures (PNG)
        # =====================================================================
        if figures:
            for name, png_bytes in figures.items():
                zf.writestr(f'figures/{name}.png', png_bytes)
        
        # =====================================================================
        # Reproduction script
        # =====================================================================
        reproduce_script = f'''#!/usr/bin/env python3
"""
Reproduction script for LRD-AI-Telemetry analysis.

Generated: {timestamp}
Original DOI: 10.5281/zenodo.18145167

To reproduce:
    pip install -r requirements.txt
    python reproduce.py
"""

import json
import pandas as pd
import numpy as np

# Load configuration
with open('metadata.json') as f:
    metadata = json.load(f)

config = metadata['config']
print(f"Original run timestamp: {{metadata['timestamp']}}")
print(f"Model: {{config['model_name']}}")
print(f"Seed: {{config['seed']}}")
print(f"Max tokens: {{config['generation']['max_tokens']}}")

# Load metrics
df_step = pd.read_csv('metrics_per_step.csv')
print(f"\\nTotal steps: {{len(df_step)}}")
print(f"Mean entropy: {{df_step['entropy'].mean():.4f}}")

# Load rolling metrics
try:
    df_rolling = pd.read_csv('rolling_metrics.csv')
    print(f"\\nFinal Hurst: {{df_rolling['hurst'].iloc[-1]:.4f}}")
    print(f"Final Higuchi: {{df_rolling['higuchi'].iloc[-1]:.4f}}")
except FileNotFoundError:
    print("Rolling metrics not available (run was too short)")

# Results
results = metadata['results']
print(f"\\n=== RESULTS ===")
print(f"Warning score: {{results['final_warning_score']:.1f}} ({{results['final_warning_level']}})")
if results['first_warning']:
    print(f"First warning at step: {{results['first_warning']}}")
if results['first_collapse']:
    print(f"First collapse at step: {{results['first_collapse']}}")
if results['warning_lead_time']:
    print(f"Lead time: {{results['warning_lead_time']}} tokens")

print("\\n=== To fully reproduce, run the full pipeline with the same seed ===")
'''
        zf.writestr('reproduce.py', reproduce_script)
        
        # =====================================================================
        # Requirements
        # =====================================================================
        requirements = '''# Requirements for LRD-AI-Telemetry v2.1
numpy>=1.21.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
transformers>=4.35.0
plotly>=5.18.0
gradio>=4.44.0
'''
        zf.writestr('requirements.txt', requirements)
        
        # =====================================================================
        # README
        # =====================================================================
        readme = f'''# LRD-AI-Telemetry Export Package v2.1

## Run Information

- **Timestamp:** {timestamp}
- **Model:** {config.model_name}
- **Seed:** {config.seed}
- **Tokens generated:** {len(entropy_series)}

## Results

- **Warning Score:** {analysis_results.get('final_score', 'N/A'):.1f}/100 ({analysis_results.get('final_level', 'N/A')})
- **First Warning:** Step {analysis_results.get('first_warning', 'N/A')}
- **First Collapse:** Step {analysis_results.get('first_collapse', 'N/A')}
- **Lead Time:** {analysis_results.get('lead_time', 'N/A')} tokens

## Files

| File | Description |
|------|-------------|
| `metadata.json` | Run configuration, results, and disclaimers |
| `environment.json` | Python/library versions |
| `generated_text.txt` | Full generated text |
| `token_ids.json` | Token ID sequence |
| `metrics_per_step.csv` | Entropy, repetition per step |
| `rolling_metrics.csv` | Hurst, Higuchi, spectral, 1/f |
| `kl_drift.csv` | KL divergence between steps (if computed) |
| `figures/` | Visualization PNGs |
| `reproduce.py` | Script to reproduce analysis |
| `requirements.txt` | Python dependencies |

## Citation

```bibtex
@software{{chechelnitsky_2026_lrd_telemetry,
  author       = {{Chechelnitsky, Igor}},
  title        = {{Seismic Precursors of LLM Degradation: LRD/QO3/FIO Bridge}},
  year         = 2026,
  publisher    = {{Zenodo}},
  doi          = {{10.5281/zenodo.18145167}},
}}
```

## Scientific Notes

{DISCLAIMERS['thresholds']}

{DISCLAIMERS['hurst_method']}

## Links

- **DOI:** [10.5281/zenodo.18145167](https://doi.org/10.5281/zenodo.18145167)
- **Community:** [LRD Time Series](https://zenodo.org/communities/lrd-time-series)
- **Author ORCID:** [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)
'''
        zf.writestr('README.md', readme)
    
    buffer.seek(0)
    return buffer.getvalue()


def figure_to_png(fig) -> bytes:
    """Convert Plotly figure to PNG bytes."""
    return fig.to_image(format="png", width=1200, height=800, scale=2)
