"""
Visualization Module for LRD-AI-Telemetry v2.1

Creates interactive Plotly dashboards for LLM telemetry analysis.

Author: Igor Chechelnitsky
ORCID: 0009-0007-4607-1946
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Thresholds


# Color palette
COLORS = {
    'blue': '#2E86AB',
    'red': '#E94F37',
    'orange': '#F6AE2D',
    'green': '#4CAF50',
    'purple': '#9C27B0',
    'gray': '#888888',
}


def create_main_dashboard(
    entropy_series: List[float],
    rep_series: List[float],
    ngram_rep_series: List[float],
    rolling_data: Dict,
    thresholds: Optional[Thresholds] = None,
    first_collapse: Optional[int] = None,
    first_warning: Optional[int] = None,
    kl_positions: Optional[List[int]] = None,
    kl_values: Optional[List[float]] = None,
    show_ci: bool = False,
) -> go.Figure:
    """
    Create 6-panel interactive dashboard.
    
    Panels:
    1. Token Entropy
    2. Hurst Exponent
    3. Repetition (token + n-gram)
    4. Higuchi Dimension
    5. Spectral Entropy
    6. 1/f Exponent (+ KL drift if available)
    """
    if thresholds is None:
        thresholds = Thresholds()
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📉 Token Entropy",
            "📊 Hurst Exponent (H) — persistence indicator",
            "🔁 Repetition Rate",
            "🔬 Higuchi Dimension (D) — complexity",
            "📡 Spectral Entropy",
            "🎚️ 1/f Exponent (β) / KL Drift",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    
    positions = rolling_data.get('positions', [])
    
    # =========================================================================
    # Panel 1: Entropy (row 1, col 1)
    # =========================================================================
    fig.add_trace(
        go.Scatter(
            y=entropy_series,
            mode='lines',
            line=dict(color=COLORS['blue'], width=1),
            opacity=0.6,
            name='Entropy',
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # Rolling mean
    if len(entropy_series) > 50:
        window = 50
        rolling_ent = np.convolve(entropy_series, np.ones(window)/window, mode='valid')
        fig.add_trace(
            go.Scatter(
                x=list(range(window-1, len(entropy_series))),
                y=rolling_ent,
                mode='lines',
                line=dict(color=COLORS['red'], width=2),
                name='Rolling Mean',
                showlegend=False,
            ),
            row=1, col=1
        )
    
    # =========================================================================
    # Panel 2: Hurst (row 1, col 2)
    # =========================================================================
    if positions and rolling_data.get('hurst'):
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rolling_data['hurst'],
                mode='lines+markers',
                line=dict(color=COLORS['blue'], width=2),
                marker=dict(size=5),
                name='Hurst',
                showlegend=False,
            ),
            row=1, col=2
        )
        
        # Confidence intervals if available
        if show_ci and 'hurst_ci_lower' in rolling_data:
            fig.add_trace(
                go.Scatter(
                    x=positions + positions[::-1],
                    y=rolling_data['hurst_ci_upper'] + rolling_data['hurst_ci_lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(46, 134, 171, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI',
                    showlegend=False,
                ),
                row=1, col=2
            )
    
    # Reference lines
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS['gray'], row=1, col=2)
    fig.add_hline(y=thresholds.hurst_warning, line_dash="dash", line_color=COLORS['orange'], row=1, col=2)
    fig.add_hline(y=thresholds.hurst_critical, line_dash="dash", line_color=COLORS['red'], row=1, col=2)
    
    # =========================================================================
    # Panel 3: Repetition (row 2, col 1)
    # =========================================================================
    fig.add_trace(
        go.Scatter(
            y=rep_series,
            mode='lines',
            line=dict(color=COLORS['blue'], width=1),
            name='Token Rep',
            opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # N-gram repetition (stricter)
    if ngram_rep_series:
        fig.add_trace(
            go.Scatter(
                y=ngram_rep_series,
                mode='lines',
                line=dict(color=COLORS['purple'], width=2),
                name='N-gram Rep',
                showlegend=False,
            ),
            row=2, col=1
        )
    
    fig.add_hline(y=thresholds.repetition_warning, line_dash="dash", line_color=COLORS['orange'], row=2, col=1)
    fig.add_hline(y=thresholds.ngram_warning, line_dash="dash", line_color=COLORS['purple'], 
                  annotation_text="n-gram", row=2, col=1)
    
    # =========================================================================
    # Panel 4: Higuchi (row 2, col 2)
    # =========================================================================
    if positions and rolling_data.get('higuchi'):
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rolling_data['higuchi'],
                mode='lines+markers',
                line=dict(color=COLORS['purple'], width=2),
                marker=dict(size=5),
                name='Higuchi D',
                showlegend=False,
            ),
            row=2, col=2
        )
    
    fig.add_hline(y=1.5, line_dash="dot", line_color=COLORS['gray'], row=2, col=2)
    fig.add_hline(y=thresholds.higuchi_warning, line_dash="dash", line_color=COLORS['orange'], row=2, col=2)
    fig.add_hline(y=thresholds.higuchi_critical, line_dash="dash", line_color=COLORS['red'], row=2, col=2)
    
    # =========================================================================
    # Panel 5: Spectral Entropy (row 3, col 1)
    # =========================================================================
    if positions and rolling_data.get('spectral_entropy'):
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rolling_data['spectral_entropy'],
                mode='lines+markers',
                line=dict(color=COLORS['green'], width=2),
                marker=dict(size=5),
                name='Spectral Ent',
                showlegend=False,
            ),
            row=3, col=1
        )
    
    fig.add_hline(y=thresholds.spectral_warning, line_dash="dash", line_color=COLORS['orange'], row=3, col=1)
    fig.add_hline(y=thresholds.spectral_critical, line_dash="dash", line_color=COLORS['red'], row=3, col=1)
    
    # =========================================================================
    # Panel 6: 1/f + KL Drift (row 3, col 2)
    # =========================================================================
    if positions and rolling_data.get('one_over_f'):
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=rolling_data['one_over_f'],
                mode='lines+markers',
                line=dict(color=COLORS['orange'], width=2),
                marker=dict(size=5),
                name='1/f β',
                showlegend=False,
            ),
            row=3, col=2
        )
    
    # KL drift on secondary y-axis (if available)
    if kl_positions and kl_values:
        fig.add_trace(
            go.Scatter(
                x=kl_positions,
                y=kl_values,
                mode='lines',
                line=dict(color=COLORS['red'], width=1, dash='dot'),
                name='KL Drift',
                opacity=0.7,
                showlegend=False,
            ),
            row=3, col=2
        )
    
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS['gray'], row=3, col=2)
    fig.add_hline(y=thresholds.one_over_f_warning, line_dash="dash", line_color=COLORS['orange'], row=3, col=2)
    fig.add_hline(y=thresholds.one_over_f_critical, line_dash="dash", line_color=COLORS['red'], row=3, col=2)
    
    # =========================================================================
    # Warning/Collapse markers on all panels
    # =========================================================================
    for row in [1, 2, 3]:
        for col in [1, 2]:
            if first_warning is not None:
                fig.add_vline(
                    x=first_warning,
                    line_dash="dash",
                    line_color=COLORS['orange'],
                    line_width=2,
                    row=row, col=col
                )
            if first_collapse is not None:
                fig.add_vline(
                    x=first_collapse,
                    line_dash="solid",
                    line_color=COLORS['red'],
                    line_width=2,
                    row=row, col=col
                )
    
    # =========================================================================
    # Layout
    # =========================================================================
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="🌋 LLM Degradation Telemetry Dashboard v2.1",
        title_x=0.5,
        title_font_size=18,
    )
    
    # Axis labels
    fig.update_yaxes(title_text="Entropy", row=1, col=1)
    fig.update_yaxes(title_text="H", range=[0.3, 1.0], row=1, col=2)
    fig.update_yaxes(title_text="Rep Rate", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="D", range=[1.0, 2.0], row=2, col=2)
    fig.update_yaxes(title_text="Spec Ent", range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text="β / KL", row=3, col=2)
    
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_xaxes(title_text="Step", row=3, col=2)
    
    return fig


def create_warning_gauge(score: float, level: str) -> go.Figure:
    """Create a gauge chart for warning score."""
    
    colors = {
        'normal': COLORS['green'],
        'warning': COLORS['orange'],
        'critical': COLORS['red'],
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Warning Score: {level.upper()}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(level, COLORS['gray'])},
            'steps': [
                {'range': [0, 40], 'color': '#E8F5E9'},
                {'range': [40, 70], 'color': '#FFF3E0'},
                {'range': [70, 100], 'color': '#FFEBEE'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig


def create_phase_portrait(
    hurst_series: List[float],
    higuchi_series: List[float],
    positions: List[int],
    thresholds: Optional[Thresholds] = None,
) -> go.Figure:
    """
    Create H-D phase portrait showing degradation trajectory.
    
    Healthy zone: center (H~0.5-0.6, D~1.4-1.5)
    Degradation trajectory: upper-left (H↑, D↓)
    """
    if thresholds is None:
        thresholds = Thresholds()
    
    fig = go.Figure()
    
    # Background zones
    # Healthy zone (green)
    fig.add_shape(
        type="rect",
        x0=0.45, x1=0.65, y0=1.35, y1=1.55,
        fillcolor="rgba(76, 175, 80, 0.2)",
        line=dict(width=0),
    )
    
    # Warning zone (orange)
    fig.add_shape(
        type="rect",
        x0=0.65, x1=0.80, y0=1.20, y1=1.35,
        fillcolor="rgba(246, 174, 45, 0.2)",
        line=dict(width=0),
    )
    
    # Critical zone (red)
    fig.add_shape(
        type="rect",
        x0=0.80, x1=1.0, y0=1.0, y1=1.20,
        fillcolor="rgba(233, 79, 55, 0.2)",
        line=dict(width=0),
    )
    
    # Trajectory
    if hurst_series and higuchi_series:
        # Color by time (early=blue, late=red)
        n = len(hurst_series)
        colors = [f'rgb({int(255*i/n)}, {int(100*(1-i/n))}, {int(255*(1-i/n))})' for i in range(n)]
        
        fig.add_trace(go.Scatter(
            x=hurst_series,
            y=higuchi_series,
            mode='lines+markers',
            marker=dict(
                size=8,
                color=list(range(n)),
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Step"),
            ),
            line=dict(color='gray', width=1),
            text=[f"Step {p}" for p in positions],
            hoverinfo='text+x+y',
            name='Trajectory',
        ))
        
        # Start marker
        fig.add_trace(go.Scatter(
            x=[hurst_series[0]],
            y=[higuchi_series[0]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Start',
        ))
        
        # End marker
        fig.add_trace(go.Scatter(
            x=[hurst_series[-1]],
            y=[higuchi_series[-1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='End',
        ))
    
    # Reference lines
    fig.add_hline(y=2-0.5, line_dash="dot", line_color="gray", 
                  annotation_text="D=2-H (H=0.5)")
    fig.add_vline(x=thresholds.hurst_warning, line_dash="dash", line_color=COLORS['orange'])
    fig.add_hline(y=thresholds.higuchi_warning, line_dash="dash", line_color=COLORS['orange'])
    
    fig.update_layout(
        title="Phase Portrait: H-D Trajectory",
        xaxis_title="Hurst Exponent (H)",
        yaxis_title="Higuchi Dimension (D)",
        xaxis=dict(range=[0.3, 1.0]),
        yaxis=dict(range=[1.0, 2.0]),
        height=500,
        showlegend=True,
    )
    
    return fig


def create_sid_plot(
    sid_series: np.ndarray,
    first_collapse: Optional[int] = None,
    first_warning: Optional[int] = None,
) -> go.Figure:
    """Create Seismic Information Deficit plot."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=sid_series,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(233, 79, 55, 0.3)',
        line=dict(color=COLORS['red'], width=2),
        name='SID',
    ))
    
    if first_warning is not None:
        fig.add_vline(x=first_warning, line_dash="dash", line_color=COLORS['orange'], line_width=2)
    
    if first_collapse is not None:
        fig.add_vline(x=first_collapse, line_dash="solid", line_color=COLORS['red'], line_width=2)
    
    fig.update_layout(
        title="Seismic Information Deficit (SID)",
        xaxis_title="Step",
        yaxis_title="Cumulative Entropy Deficit",
        height=300,
    )
    
    return fig
