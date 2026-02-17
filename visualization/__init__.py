"""
Visualization module for neural audio codec comparison.
"""

from .music_comparison import (
    plot_waveform_comparison,
    plot_reconstruction_error,
    plot_frequency_band_error,
    plot_error_distribution,
    plot_metric_comparison
)

__all__ = [
    'plot_waveform_comparison',
    'plot_reconstruction_error',
    'plot_frequency_band_error',
    'plot_error_distribution',
    'plot_metric_comparison'
]
