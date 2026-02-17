"""
Visualization functions for comparing neural audio codecs.

This module provides publication-quality plotting functions for:
1. Bitrate vs Quality trade-off curves
2. Embedding quality comparison (FAD correlation)
3. Spectrogram comparisons showing tonal artifacts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Color scheme for different codecs
CODEC_COLORS = {
    'encodec': '#E63946',      # Red
    'dac': '#457B9D',          # Blue
    'soundstream': '#2A9D8F',  # Teal
    'chimera': '#F77F00',      # Orange
    'lyra': '#9B59B6'          # Purple
}

CODEC_MARKERS = {
    'encodec': 'o',
    'dac': 's',
    'soundstream': '^',
    'chimera': 'D',
    'lyra': 'v'
}


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2,
        'lines.markersize': 8
    })


def plot_bitrate_vs_quality(
    data: Dict[str, List[Tuple[float, float]]],
    metric_name: str = "ViSQOL",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True
) -> None:
    """
    Plot bitrate vs quality trade-off curves for multiple codecs.
    
    Args:
        data: Dictionary mapping codec names to list of (bitrate, quality_score) tuples
              Example: {'dac': [(8, 3.2), (16, 3.9), (32, 4.28)], 'encodec': [...]}
        metric_name: Name of quality metric (e.g., "ViSQOL", "SI-SDR", "MOS")
        output_path: Path to save figure. If None, displays interactively.
        title: Custom title for the plot
        show_legend: Whether to show the legend
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for codec_name, points in data.items():
        codec_key = codec_name.lower()
        bitrates = [p[0] for p in points]
        scores = [p[1] for p in points]
        
        color = CODEC_COLORS.get(codec_key, '#333333')
        marker = CODEC_MARKERS.get(codec_key, 'o')
        
        ax.plot(bitrates, scores, 
                marker=marker, 
                color=color, 
                label=codec_name.upper() if codec_key in CODEC_COLORS else codec_name,
                linewidth=2.5,
                markersize=9,
                markeredgewidth=1.5,
                markeredgecolor='white')
    
    ax.set_xlabel('Bitrate (kbps)', fontweight='bold')
    ax.set_ylabel(f'{metric_name} Score', fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Codec Performance: Bitrate vs {metric_name}', fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_legend:
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    
    # Add arrow annotation showing "better" direction
    ax.annotate('', xy=(0.05, 0.95), xytext=(0.05, 0.85),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
                annotation_clip=False)
    ax.text(0.08, 0.90, 'Better', transform=ax.transAxes, 
            fontsize=9, color='green', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved bitrate vs quality plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_quality(
    data: Dict[str, float],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    include_specialist_models: bool = True
) -> None:
    """
    Plot embedding quality comparison (FAD Spearman correlation).
    
    Args:
        data: Dictionary mapping model names to FAD Spearman correlation values
              Example: {'EnCodec': 0.66, 'DAC': 0.81, 'Chimera': 0.82, 'CLAP': 0.88}
        output_path: Path to save figure. If None, displays interactively.
        title: Custom title for the plot
        include_specialist_models: If True, highlights specialist models (CLAP, OpenL3)
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by correlation value
    sorted_items = sorted(data.items(), key=lambda x: x[1])
    models = [item[0] for item in sorted_items]
    correlations = [item[1] for item in sorted_items]
    
    # Determine colors: specialist models in different color
    specialist_models = {'clap', 'openl3', 'vggish', 'panns'}
    colors = []
    for model in models:
        model_lower = model.lower().replace('-', '').replace('_', '')
        if any(spec in model_lower for spec in specialist_models):
            colors.append('#9B59B6')  # Purple for specialist models
        else:
            # Try to match codec colors
            codec_key = model_lower.split()[0]
            colors.append(CODEC_COLORS.get(codec_key, '#457B9D'))
    
    bars = ax.barh(models, correlations, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('FAD Spearman Correlation ($R_s$)', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title('Embedding Quality for Audio Assessment', fontweight='bold', pad=15)
    
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(corr + 0.02, i, f'{corr:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Add legend if specialist models are included
    if include_specialist_models and any('#9B59B6' in c for c in colors):
        codec_patch = mpatches.Patch(color='#457B9D', label='Neural Audio Codecs')
        specialist_patch = mpatches.Patch(color='#9B59B6', label='Specialist Models')
        ax.legend(handles=[codec_patch, specialist_patch], loc='lower right', 
                 framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding quality plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram_comparison(
    audio_files: Dict[str, str],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    sr: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    time_range: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create side-by-side spectrogram comparison of different codec reconstructions.
    
    Args:
        audio_files: Dictionary mapping labels to audio file paths
                     Example: {'Original': 'original.wav', 'DAC': 'dac_recon.wav', ...}
        output_path: Path to save figure. If None, displays interactively.
        title: Custom title for the plot
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel frequency bins
        time_range: Optional (start_sec, end_sec) to zoom into a specific time range
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        raise ImportError("librosa is required for spectrogram plotting. Install with: pip install librosa")
    
    setup_plot_style()
    
    n_audio = len(audio_files)
    fig, axes = plt.subplots(n_audio, 1, figsize=(12, 3 * n_audio))
    
    if n_audio == 1:
        axes = [axes]
    
    for idx, (label, audio_path) in enumerate(audio_files.items()):
        # Load audio
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        
        # Extract time range if specified
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            y = y[start_sample:end_sample]
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot
        img = librosa.display.specshow(
            S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel',
            ax=axes[idx], cmap='viridis'
        )
        
        axes[idx].set_title(label, fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Frequency (Hz)', fontweight='bold')
        
        # Only show x-label on bottom plot
        if idx < n_audio - 1:
            axes[idx].set_xlabel('')
        else:
            axes[idx].set_xlabel('Time (s)', fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')
        cbar.set_label('dB', fontweight='bold')
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.995)
    else:
        fig.suptitle('Spectrogram Comparison: Tonal Content', fontweight='bold', fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrogram comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_codebook_entropy(
    data: Dict[str, float],
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot codebook entropy comparison across codecs.
    
    Args:
        data: Dictionary mapping codec names to codebook entropy values
              Example: {'EnCodec': 6.2, 'DAC': 7.4, 'Chimera': 8.6}
        output_path: Path to save figure. If None, displays interactively.
        title: Custom title for the plot
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by entropy value
    sorted_items = sorted(data.items(), key=lambda x: x[1])
    codecs = [item[0] for item in sorted_items]
    entropies = [item[1] for item in sorted_items]
    
    # Determine colors
    colors = []
    for codec in codecs:
        codec_key = codec.lower().split()[0]
        colors.append(CODEC_COLORS.get(codec_key, '#457B9D'))
    
    bars = ax.barh(codecs, entropies, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Codebook Entropy (bits)', fontweight='bold')
    ax.set_ylabel('Codec', fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title('Codebook Utilization Comparison', fontweight='bold', pad=15)
    
    # Set x-axis to start from 0 and go slightly beyond max
    ax.set_xlim(0, max(entropies) * 1.1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels on bars
    for i, (bar, entropy) in enumerate(zip(bars, entropies)):
        ax.text(entropy + 0.1, i, f'{entropy:.1f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Add annotation
    ax.text(0.98, 0.02, 'Higher entropy = Better codebook utilization', 
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved codebook entropy plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def load_metrics_from_json(json_path: str) -> Dict:
    """Load metrics data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate plots using actual codec evaluation data
    
    import sys
    from pathlib import Path as PathLib
    
    # Try to load real metrics data
    metrics_files = list(Path('outputs/evaluations').glob('*_metrics.json'))
    
    if not metrics_files:
        print("No metrics files found in outputs/evaluations/")
        print("Run evaluate_codecs.py first to generate real data.")
        sys.exit(1)
    
    print(f"Found {len(metrics_files)} evaluation result(s). Using the most recent...")
    latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_metrics}")
    
    # Load the metrics
    metrics_data = load_metrics_from_json(str(latest_metrics))
    
    # Create output directory for figures
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots with real codec data...")
    
    # 1. Bitrate vs Quality (ViSQOL) - if ViSQOL data is available
    has_visqol = any('ViSQOL' in metrics for metrics in metrics_data.values() if isinstance(metrics, dict))
    
    if has_visqol:
        bitrate_quality_data = {}
        
        # Extract bitrate from codec names and ViSQOL scores
        for codec_name, metrics in metrics_data.items():
            if 'chimera' in codec_name.lower():
                # Extract bitrate from name like "chimera_24kbps"
                if '24' in codec_name:
                    bitrate = 24
                elif '30' in codec_name:
                    bitrate = 30
                else:
                    continue
                    
                if 'ViSQOL' in metrics:
                    if 'Chimera' not in bitrate_quality_data:
                        bitrate_quality_data['Chimera'] = []
                    bitrate_quality_data['Chimera'].append((bitrate, metrics['ViSQOL']))
            elif codec_name.lower() == 'encodec':
                if 'ViSQOL' in metrics:
                    bitrate_quality_data['EnCodec'] = [(24, metrics['ViSQOL'])]
            elif codec_name.lower() == 'dac':
                if 'ViSQOL' in metrics:
                    bitrate_quality_data['DAC'] = [(32, metrics['ViSQOL'])]
        
        if bitrate_quality_data:
            plot_bitrate_vs_quality(
                bitrate_quality_data,
                metric_name="ViSQOL",
                output_path=str(output_dir / "bitrate_vs_visqol.png"),
                title="Codec Performance: Bitrate vs ViSQOL Score (Real Data)"
            )
    
    # 2. SI-SDR comparison (always available)
    si_sdr_data = {}
    for codec_name, metrics in metrics_data.items():
        if 'SI-SDR (dB)' in metrics:
            # Clean up codec names for display
            if 'chimera' in codec_name.lower():
                if '24' in codec_name:
                    display_name = 'Chimera 24kbps'
                elif '30' in codec_name:
                    display_name = 'Chimera 30kbps'
                else:
                    display_name = codec_name
            else:
                display_name = codec_name.upper()
            
            si_sdr_data[display_name] = metrics['SI-SDR (dB)']
    
    if si_sdr_data:
        # Plot as horizontal bar chart (similar to embedding quality style)
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by SI-SDR value
        sorted_items = sorted(si_sdr_data.items(), key=lambda x: x[1])
        codecs = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        # Determine colors
        colors = []
        for codec in codecs:
            codec_key = codec.lower().split()[0]
            colors.append(CODEC_COLORS.get(codec_key, '#457B9D'))
        
        bars = ax.barh(codecs, scores, color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('SI-SDR (dB)', fontweight='bold')
        ax.set_ylabel('Codec', fontweight='bold')
        ax.set_title('Audio Reconstruction Quality (SI-SDR) - Real Data', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.3, i, f'{score:.2f} dB', 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "si_sdr_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved SI-SDR comparison to {output_dir / 'si_sdr_comparison.png'}")
        plt.close()
    
    # 3. Spectrogram comparison - if audio files exist
    base_name = latest_metrics.stem.replace('_metrics', '')
    audio_dir = latest_metrics.parent
    
    audio_files = {}
    # Check for original
    orig_path = audio_dir / f"{base_name}_original.wav"
    if orig_path.exists():
        audio_files['Original'] = str(orig_path)
    
    # Check for codec reconstructions
    for codec in ['dac', 'encodec', 'soundstream', 'chimera_24kbps', 'chimera_30kbps']:
        codec_path = audio_dir / f"{base_name}_{codec}.wav"
        if codec_path.exists():
            display_name = codec.upper() if codec in ['dac', 'encodec', 'soundstream'] else codec.replace('_', ' ').title()
            audio_files[display_name] = str(codec_path)
    
    if len(audio_files) > 1:
        print(f"\nGenerating spectrogram comparison with {len(audio_files)} audio files...")
        plot_spectrogram_comparison(
            audio_files,
            output_path=str(output_dir / "spectrogram_comparison.png"),
            title=f"Spectrogram Comparison: {base_name}",
            time_range=(5.0, 7.0)  # 2-second window starting at 5s
        )
    
    print("\n✓ All plots generated successfully using real codec data!")
    print(f"📁 Saved to: {output_dir.absolute()}")
