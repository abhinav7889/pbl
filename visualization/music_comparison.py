"""
Music-specific visualization for comparing neural audio codec reconstructions.

Focuses on waveform differences and reconstruction quality analysis for real music samples.
"""

import matplotlib.pyplot as plt
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
    'original': '#2D3142'      # Dark gray
}


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
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
        'lines.linewidth': 1.5
    })


def plot_waveform_comparison(
    audio_data: Dict[str, np.ndarray],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot waveform comparison showing original and codec reconstructions.
    
    Args:
        audio_data: Dictionary mapping labels to audio waveforms (numpy arrays)
                    Example: {'Original': original_wav, 'DAC': dac_recon, 'EnCodec': encodec_recon}
        sr: Sample rate
        time_range: Optional (start_sec, end_sec) to zoom into a specific time range
        output_path: Path to save figure
        title: Custom title
    """
    setup_plot_style()
    
    n_audio = len(audio_data)
    fig, axes = plt.subplots(n_audio, 1, figsize=(14, 2.5 * n_audio), sharex=True)
    
    if n_audio == 1:
        axes = [axes]
    
    for idx, (label, waveform) in enumerate(audio_data.items()):
        # Extract time range if specified
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            waveform = waveform[start_sample:end_sample]
            time_offset = time_range[0]
        else:
            time_offset = 0
        
        # Create time axis
        time = np.arange(len(waveform)) / sr + time_offset
        
        # Determine color
        color = CODEC_COLORS.get(label.lower(), '#333333')
        
        # Plot waveform
        axes[idx].plot(time, waveform, color=color, linewidth=0.5, alpha=0.8)
        axes[idx].set_ylabel('Amplitude', fontweight='bold')
        axes[idx].set_title(label, fontweight='bold', loc='left', pad=10)
        axes[idx].set_ylim(-1.0, 1.0)
        axes[idx].grid(True, alpha=0.3)
        
        # Add RMS value
        rms = np.sqrt(np.mean(waveform**2))
        axes[idx].text(0.98, 0.95, f'RMS: {rms:.4f}', 
                      transform=axes[idx].transAxes,
                      ha='right', va='top',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                      fontsize=9)
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.998)
    else:
        fig.suptitle('Waveform Comparison', fontweight='bold', fontsize=14, y=0.998)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved waveform comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_reconstruction_error(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot reconstruction error (original - reconstruction) for each codec.
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary mapping codec names to reconstructed waveforms
        sr: Sample rate
        time_range: Optional (start_sec, end_sec) to zoom
        output_path: Path to save figure
        title: Custom title
        normalize: If True, normalize error signals for better visualization
    """
    setup_plot_style()
    
    n_codecs = len(reconstructions)
    fig, axes = plt.subplots(n_codecs, 1, figsize=(14, 2.5 * n_codecs), sharex=True)
    
    if n_codecs == 1:
        axes = [axes]
    
    # Extract time range if specified
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        original_segment = original[start_sample:end_sample]
        time_offset = time_range[0]
    else:
        original_segment = original
        time_offset = 0
    
    for idx, (codec_name, reconstruction) in enumerate(reconstructions.items()):
        # Extract same time range from reconstruction
        if time_range:
            recon_segment = reconstruction[start_sample:end_sample]
        else:
            recon_segment = reconstruction
        
        # Compute error (residual) - ensure same length
        min_len = min(len(original_segment), len(recon_segment))
        error = original_segment[:min_len] - recon_segment[:min_len]
        
        # Normalize if requested
        if normalize and np.max(np.abs(error)) > 0:
            error = error / np.max(np.abs(error))
        
        # Create time axis
        time = np.arange(len(error)) / sr + time_offset
        
        # Determine color
        color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
        
        # Plot error
        axes[idx].plot(time, error, color=color, linewidth=0.5, alpha=0.8)
        axes[idx].set_ylabel('Error', fontweight='bold')
        axes[idx].set_title(f'{codec_name} - Reconstruction Error', fontweight='bold', loc='left', pad=10)
        axes[idx].grid(True, alpha=0.3)
        
        # Compute metrics
        mse = np.mean(error**2)
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        # Add metrics text
        metrics_text = f'MSE: {mse:.6f} | MAE: {mae:.6f} | Max: {max_error:.4f}'
        axes[idx].text(0.98, 0.95, metrics_text,
                      transform=axes[idx].transAxes,
                      ha='right', va='top',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                      fontsize=9)
        
        # Set y-limits
        if normalize:
            axes[idx].set_ylim(-1.1, 1.1)
        else:
            max_val = np.max(np.abs(error)) * 1.1
            axes[idx].set_ylim(-max_val, max_val)
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.998)
    else:
        fig.suptitle('Reconstruction Error Analysis', fontweight='bold', fontsize=14, y=0.998)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction error plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_frequency_band_error(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    sr: int = 48000,
    bands: Optional[List[Tuple[int, int]]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot reconstruction error broken down by frequency bands.
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary mapping codec names to reconstructed waveforms
        sr: Sample rate
        bands: List of (low_freq, high_freq) tuples in Hz. 
               Default: [(20, 250), (250, 2000), (2000, 8000), (8000, 24000)]
        output_path: Path to save figure
        title: Custom title
    """
    try:
        from scipy import signal
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")
    
    setup_plot_style()
    
    if bands is None:
        bands = [(20, 250), (250, 2000), (2000, 8000), (8000, 24000)]
        band_labels = ['Bass\n(20-250 Hz)', 'Low-Mid\n(250-2k Hz)', 
                      'High-Mid\n(2-8k Hz)', 'Treble\n(8-24k Hz)']
    else:
        band_labels = [f'{low}-{high} Hz' for low, high in bands]
    
    n_bands = len(bands)
    codec_names = list(reconstructions.keys())
    n_codecs = len(codec_names)
    
    # Compute error per band for each codec
    errors = np.zeros((n_codecs, n_bands))
    
    for codec_idx, (codec_name, reconstruction) in enumerate(reconstructions.items()):
        # Ensure same length before computing error
        min_len = min(len(original), len(reconstruction))
        error_signal = original[:min_len] - reconstruction[:min_len]
        
        for band_idx, (low_freq, high_freq) in enumerate(bands):
            # Design bandpass filter
            nyquist = sr / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are valid
            low = max(0.001, min(low, 0.999))
            high = max(low + 0.001, min(high, 0.999))
            
            # Bandpass filter
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered_error = signal.sosfilt(sos, error_signal)
            
            # Compute RMS error in this band
            errors[codec_idx, band_idx] = np.sqrt(np.mean(filtered_error**2))
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_bands)
    width = 0.8 / n_codecs
    
    for codec_idx, codec_name in enumerate(codec_names):
        offset = (codec_idx - n_codecs/2 + 0.5) * width
        color = CODEC_COLORS.get(codec_name.lower(), '#333333')
        
        bars = ax.bar(x + offset, errors[codec_idx], width, 
                     label=codec_name, color=color, 
                     edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Frequency Band', fontweight='bold', fontsize=12)
    ax.set_ylabel('RMS Error', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title('Reconstruction Error by Frequency Band', fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved frequency band error plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 100
) -> None:
    """
    Plot histogram of reconstruction error distributions.
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary mapping codec names to reconstructed waveforms
        output_path: Path to save figure
        title: Custom title
        bins: Number of histogram bins
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for codec_name, reconstruction in reconstructions.items():
        # Ensure same length before computing error
        min_len = min(len(original), len(reconstruction))
        error = original[:min_len] - reconstruction[:min_len]
        color = CODEC_COLORS.get(codec_name.lower(), '#333333')
        
        ax.hist(error, bins=bins, alpha=0.6, label=codec_name, 
               color=color, edgecolor='white', linewidth=0.5, density=True)
    
    ax.set_xlabel('Error Value', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title('Error Distribution Comparison', fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved error distribution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot grouped bar chart comparing multiple metrics across codecs.
    
    Args:
        metrics: Nested dictionary: {codec_name: {metric_name: value}}
                 Example: {'DAC': {'SI-SDR': 16.8, 'ViSQOL': 4.28}, 'EnCodec': {...}}
        output_path: Path to save figure
        title: Custom title
    """
    setup_plot_style()
    
    codec_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    
    n_codecs = len(codec_names)
    n_metrics = len(metric_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_codecs
    
    for codec_idx, codec_name in enumerate(codec_names):
        offset = (codec_idx - n_codecs/2 + 0.5) * width
        color = CODEC_COLORS.get(codec_name.lower(), '#333333')
        
        values = [metrics[codec_name][metric] for metric in metric_names]
        
        bars = ax.bar(x + offset, values, width,
                     label=codec_name, color=color,
                     edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    else:
        ax.set_title('Objective Metric Comparison', fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram_comparison(
    audio_data: Dict[str, np.ndarray],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    n_mels: int = 128,
    fmax: int = 8000
) -> None:
    """
    Plot mel spectrogram comparison for multiple audio signals.
    
    This is crucial for identifying tonal artifacts visually, as tonal weaknesses
    manifest as discontinuities, noise, or incorrect harmonic structures in the
    frequency domain.
    
    Args:
        audio_data: Dictionary mapping labels to audio waveforms
        sr: Sample rate
        time_range: Optional (start_sec, end_sec) to zoom
        output_path: Path to save figure
        title: Custom title
        n_mels: Number of mel bands
        fmax: Maximum frequency to display
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        raise ImportError("librosa is required for spectrogram visualization. Install with: pip install librosa")
    
    setup_plot_style()
    
    n_audio = len(audio_data)
    fig, axes = plt.subplots(n_audio, 1, figsize=(14, 3 * n_audio))
    
    if n_audio == 1:
        axes = [axes]
    
    for idx, (label, waveform) in enumerate(audio_data.items()):
        # Extract time range if specified
        if time_range:
            start_sample = int(time_range[0] * sr)
            end_sample = int(time_range[1] * sr)
            waveform = waveform[start_sample:end_sample]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=sr, 
            n_mels=n_mels,
            fmax=fmax,
            hop_length=512
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            fmax=fmax,
            ax=axes[idx],
            cmap='viridis',
            hop_length=512
        )
        
        # Determine color
        color = CODEC_COLORS.get(label.lower(), '#333333')
        
        axes[idx].set_title(label, fontweight='bold', loc='left', pad=10, color=color)
        axes[idx].set_ylabel('Frequency (Hz)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=axes[idx], format='%+2.0f dB')
        cbar.set_label('Power (dB)', fontweight='bold')
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.998)
    else:
        fig.suptitle('Mel Spectrogram Comparison', fontweight='bold', fontsize=14, y=0.998)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrogram comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_spectrogram(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    n_mels: int = 128,
    fmax: int = 8000
) -> None:
    """
    Plot error spectrograms showing time-frequency reconstruction errors.
    
    This reveals WHERE errors occur in the time-frequency plane, which is
    critical for understanding codec weaknesses (e.g., high-frequency artifacts,
    tonal instability at specific frequencies).
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary mapping codec names to reconstructed waveforms
        sr: Sample rate
        time_range: Optional (start_sec, end_sec) to zoom
        output_path: Path to save figure
        title: Custom title
        n_mels: Number of mel bands
        fmax: Maximum frequency to display
    """
    try:
        import librosa
        import librosa.display
    except ImportError:
        raise ImportError("librosa is required for error spectrogram. Install with: pip install librosa")
    
    setup_plot_style()
    
    # Extract time range if specified
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        original_segment = original[start_sample:end_sample]
    else:
        original_segment = original
    
    # Compute original mel spectrogram
    mel_original = librosa.feature.melspectrogram(
        y=original_segment,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        hop_length=512
    )
    mel_original_db = librosa.power_to_db(mel_original, ref=np.max)
    
    n_codecs = len(reconstructions)
    fig, axes = plt.subplots(n_codecs, 1, figsize=(14, 3 * n_codecs))
    
    if n_codecs == 1:
        axes = [axes]
    
    for idx, (codec_name, reconstruction) in enumerate(reconstructions.items()):
        # Extract time range
        if time_range:
            recon_segment = reconstruction[start_sample:end_sample]
        else:
            recon_segment = reconstruction
        
        # Ensure same length
        min_len = min(len(original_segment), len(recon_segment))
        
        # Compute reconstruction mel spectrogram
        mel_recon = librosa.feature.melspectrogram(
            y=recon_segment[:min_len],
            sr=sr,
            n_mels=n_mels,
            fmax=fmax,
            hop_length=512
        )
        mel_recon_db = librosa.power_to_db(mel_recon, ref=np.max)
        
        # Compute error (difference in dB)
        min_frames = min(mel_original_db.shape[1], mel_recon_db.shape[1])
        error_spec = np.abs(mel_original_db[:, :min_frames] - mel_recon_db[:, :min_frames])
        
        # Plot error spectrogram
        img = librosa.display.specshow(
            error_spec,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            fmax=fmax,
            ax=axes[idx],
            cmap='hot',
            hop_length=512
        )
        
        # Determine color
        color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
        
        axes[idx].set_title(f'{codec_name} - Error Spectrogram', fontweight='bold', loc='left', pad=10, color=color)
        axes[idx].set_ylabel('Frequency (Hz)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=axes[idx], format='%.1f dB')
        cbar.set_label('Absolute Error (dB)', fontweight='bold')
        
        # Compute mean error per frequency band
        mean_error_per_band = np.mean(error_spec, axis=1)
        max_error_band = np.argmax(mean_error_per_band)
        max_error_freq = librosa.mel_to_hz(max_error_band * (fmax / n_mels))
        
        # Add text annotation
        axes[idx].text(
            0.02, 0.95,
            f'Peak error at {max_error_freq:.0f} Hz\nMean error: {np.mean(error_spec):.2f} dB',
            transform=axes[idx].transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            fontsize=9
        )
    
    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.998)
    else:
        fig.suptitle('Reconstruction Error Spectrogram', fontweight='bold', fontsize=14, y=0.998)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved error spectrogram to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_harmonic_stability(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    n_fft: int = 4096,
    hop_length: int = 512
) -> None:
    """
    Plot harmonic stability analysis showing pitch tracking and harmonic content.
    
    This is specifically designed to reveal the "tonal weakness" problem:
    - Pitch instability (wobbling fundamental frequency)
    - Harmonic degradation (loss of overtones)
    - Temporal coherence issues
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary mapping codec names to reconstructed waveforms
        sr: Sample rate
        time_range: Optional (start_sec, end_sec) to zoom
        output_path: Path to save figure
        title: Custom title
        n_fft: FFT size for frequency resolution
        hop_length: Hop length for STFT
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for harmonic analysis. Install with: pip install librosa")
    
    setup_plot_style()
    
    # Extract time range if specified
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        original_segment = original[start_sample:end_sample]
        time_offset = time_range[0]
    else:
        original_segment = original
        time_offset = 0
    
    # Create figure with 2 subplots per codec (pitch tracking + harmonic strength)
    n_codecs = len(reconstructions)
    fig, axes = plt.subplots(n_codecs, 2, figsize=(16, 3 * n_codecs))
    
    if n_codecs == 1:
        axes = axes.reshape(1, -1)
    
    # Compute pitch track for original
    try:
        f0_original, voiced_flag_original, voiced_probs_original = librosa.pyin(
            original_segment,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )
        times = librosa.times_like(f0_original, sr=sr, hop_length=hop_length) + time_offset
    except:
        # Fallback if pyin fails
        f0_original = None
        times = np.arange(len(original_segment)) / sr + time_offset
    
    for idx, (codec_name, reconstruction) in enumerate(reconstructions.items()):
        # Extract time range
        if time_range:
            recon_segment = reconstruction[start_sample:end_sample]
        else:
            recon_segment = reconstruction
        
        # Ensure same length
        min_len = min(len(original_segment), len(recon_segment))
        
        # Plot 1: Pitch tracking
        ax_pitch = axes[idx, 0]
        
        if f0_original is not None:
            # Plot original pitch
            ax_pitch.plot(times, f0_original, 'k-', alpha=0.3, linewidth=2, label='Original')
            
            # Compute pitch track for reconstruction
            try:
                f0_recon, voiced_flag_recon, voiced_probs_recon = librosa.pyin(
                    recon_segment[:min_len],
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    hop_length=hop_length
                )
                
                # Determine color
                color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
                
                # Plot reconstruction pitch
                ax_pitch.plot(times, f0_recon, color=color, linewidth=2, label=codec_name)
                
                # Compute pitch deviation
                valid_mask = ~np.isnan(f0_original) & ~np.isnan(f0_recon)
                if np.any(valid_mask):
                    pitch_deviation = np.abs(f0_original[valid_mask] - f0_recon[valid_mask])
                    mean_deviation = np.mean(pitch_deviation)
                    max_deviation = np.max(pitch_deviation)
                else:
                    mean_deviation = 0
                    max_deviation = 0
                
                ax_pitch.text(
                    0.98, 0.95,
                    f'Mean dev: {mean_deviation:.2f} Hz\nMax dev: {max_deviation:.2f} Hz',
                    transform=ax_pitch.transAxes,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                    fontsize=9
                )
            except:
                ax_pitch.text(
                    0.5, 0.5,
                    'Pitch tracking failed',
                    transform=ax_pitch.transAxes,
                    ha='center', va='center',
                    fontsize=10
                )
        
        ax_pitch.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax_pitch.set_xlabel('Time (s)', fontweight='bold')
        ax_pitch.set_title(f'{codec_name} - Pitch Tracking', fontweight='bold', loc='left', pad=10)
        ax_pitch.legend(loc='upper right', framealpha=0.9)
        ax_pitch.grid(True, alpha=0.3)
        
        # Plot 2: Harmonic strength over time
        ax_harmonic = axes[idx, 1]
        
        # Compute chromagram to show harmonic content
        chroma_original = librosa.feature.chroma_cqt(
            y=original_segment[:min_len],
            sr=sr,
            hop_length=hop_length
        )
        chroma_recon = librosa.feature.chroma_cqt(
            y=recon_segment[:min_len],
            sr=sr,
            hop_length=hop_length
        )
        
        # Compute harmonic similarity
        chroma_times = librosa.times_like(chroma_original, sr=sr, hop_length=hop_length) + time_offset
        harmonic_similarity = np.sum(chroma_original * chroma_recon, axis=0) / (
            np.sqrt(np.sum(chroma_original**2, axis=0) + 1e-8) * 
            np.sqrt(np.sum(chroma_recon**2, axis=0) + 1e-8)
        )
        
        color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
        ax_harmonic.plot(chroma_times, harmonic_similarity, color=color, linewidth=2)
        ax_harmonic.axhline(y=np.mean(harmonic_similarity), color='gray', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(harmonic_similarity):.3f}')
        ax_harmonic.fill_between(chroma_times, 0, harmonic_similarity, color=color, alpha=0.3)
        
        ax_harmonic.set_ylabel('Harmonic Similarity', fontweight='bold')
        ax_harmonic.set_xlabel('Time (s)', fontweight='bold')
        ax_harmonic.set_title(f'{codec_name} - Harmonic Preservation', fontweight='bold', loc='left', pad=10)
        ax_harmonic.set_ylim([0, 1])
        ax_harmonic.legend(loc='lower right', framealpha=0.9)
        ax_harmonic.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=14, y=0.998)
    else:
        fig.suptitle('Harmonic Stability Analysis', fontweight='bold', fontsize=14, y=0.998)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved harmonic stability plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comprehensive_analysis(
    original: np.ndarray,
    reconstructions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    sr: int = 48000,
    time_range: Optional[Tuple[float, float]] = None,
    include_theoretical: bool = True,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    song_title: Optional[str] = None
) -> None:
    """
    Create a comprehensive single-page visualization with all analysis plots.
    
    This generates a publication-quality multi-panel figure containing ALL codecs:
    - Spectrograms (Original + ALL Codecs)
    - Error spectrograms (ALL Codecs)
    - Harmonic stability analysis (ALL Codecs)
    - Frequency band error analysis (ALL Codecs)
    - Bitrate vs quality curves
    - Metric comparison
    
    Args:
        original: Original audio waveform
        reconstructions: Dictionary of reconstructed audio (excluding 'Original')
        metrics: Dictionary of metrics for each codec
        sr: Sample rate
        time_range: Optional time range to focus on
        include_theoretical: Whether to include Chimera theoretical data
        output_path: Path to save the comprehensive figure
        title: Custom title for the entire figure
        artist: Artist name (optional, for title generation)
        song_title: Song title (optional, for title generation)
    """
    try:
        import librosa
        import librosa.display
        from scipy import signal as scipy_signal
    except ImportError:
        raise ImportError("librosa and scipy are required. Install with: pip install librosa scipy")
    
    setup_plot_style()
    
    # Create comprehensive figure with dynamic GridSpec based on number of codecs
    from matplotlib.gridspec import GridSpec
    
    n_codecs = len(reconstructions)
    n_cols = min(4, n_codecs + 1)  # Max 4 columns (original + 3 codecs per row)
    n_rows_spectrograms = int(np.ceil((n_codecs + 1) / n_cols))  # +1 for original
    n_rows_errors = int(np.ceil(n_codecs / n_cols))
    n_rows_harmonics = int(np.ceil(n_codecs / n_cols))
    n_rows_freqbands = 1  # Frequency band analysis
    
    total_rows = n_rows_spectrograms + n_rows_errors + n_rows_harmonics + n_rows_freqbands + 2  # +2 for metrics
    
    fig = plt.figure(figsize=(6 * n_cols, 4 * total_rows))
    gs = GridSpec(total_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Extract time range
    if time_range:
        start_sample = int(time_range[0] * sr)
        end_sample = int(time_range[1] * sr)
        original_segment = original[start_sample:end_sample]
        time_offset = time_range[0]
    else:
        original_segment = original
        start_sample = 0
        end_sample = len(original)
        time_offset = 0
    
    current_row = 0
    
    # === SECTION 1: SPECTROGRAMS (Original + ALL Codecs) ===
    print("    Creating spectrograms for all codecs...")
    
    # Compute original spectrogram once
    mel_orig = librosa.feature.melspectrogram(y=original_segment, sr=sr, n_mels=128, fmax=8000, hop_length=512)
    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
    
    # Plot original
    ax_orig = fig.add_subplot(gs[current_row, 0])
    img = librosa.display.specshow(mel_orig_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000,
                                   ax=ax_orig, cmap='viridis', hop_length=512)
    ax_orig.set_title('Original', fontweight='bold', fontsize=11, pad=8)
    ax_orig.set_ylabel('Freq (Hz)', fontweight='bold', fontsize=10)
    plt.colorbar(img, ax=ax_orig, format='%+2.0f dB', pad=0.01)
    
    # Plot all codec spectrograms
    codec_items = list(reconstructions.items())
    for idx, (codec_name, reconstruction) in enumerate(codec_items):
        row = current_row + (idx + 1) // n_cols
        col = (idx + 1) % n_cols
        
        ax_spec = fig.add_subplot(gs[row, col])
        
        recon_segment = reconstruction[start_sample:end_sample] if time_range else reconstruction
        mel_recon = librosa.feature.melspectrogram(y=recon_segment, sr=sr, n_mels=128, fmax=8000, hop_length=512)
        mel_recon_db = librosa.power_to_db(mel_recon, ref=np.max)
        
        color = CODEC_COLORS.get(codec_name.lower(), '#333333')
        img = librosa.display.specshow(mel_recon_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000,
                                       ax=ax_spec, cmap='viridis', hop_length=512)
        ax_spec.set_title(f'{codec_name}', fontweight='bold', fontsize=11, pad=8, color=color)
        ax_spec.set_ylabel('Freq (Hz)', fontweight='bold', fontsize=10)
        plt.colorbar(img, ax=ax_spec, format='%+2.0f dB', pad=0.01)
    
    current_row += n_rows_spectrograms
    
    # === SECTION 2: ERROR SPECTROGRAMS (ALL Codecs) ===
    print("    Creating error spectrograms for all codecs...")
    
    for idx, (codec_name, reconstruction) in enumerate(codec_items):
        row = current_row + idx // n_cols
        col = idx % n_cols
        
        ax_err = fig.add_subplot(gs[row, col])
        
        recon_segment = reconstruction[start_sample:end_sample] if time_range else reconstruction
        min_len = min(len(original_segment), len(recon_segment))
        
        mel_recon = librosa.feature.melspectrogram(y=recon_segment[:min_len], sr=sr, n_mels=128, fmax=8000, hop_length=512)
        mel_recon_db = librosa.power_to_db(mel_recon, ref=np.max)
        
        min_frames = min(mel_orig_db.shape[1], mel_recon_db.shape[1])
        error_spec = np.abs(mel_orig_db[:, :min_frames] - mel_recon_db[:, :min_frames])
        
        img = librosa.display.specshow(error_spec, sr=sr, x_axis='time', y_axis='mel', fmax=8000,
                                       ax=ax_err, cmap='hot', hop_length=512)
        
        color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
        ax_err.set_title(f'{codec_name} - Error', fontweight='bold', fontsize=11, pad=8, color=color)
        ax_err.set_ylabel('Freq (Hz)', fontweight='bold', fontsize=10)
        plt.colorbar(img, ax=ax_err, format='%.1f', pad=0.01)
        
        mean_err = np.mean(error_spec)
        ax_err.text(0.02, 0.98, f'Avg: {mean_err:.2f}dB',
                   transform=ax_err.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                   fontsize=8)
    
    current_row += n_rows_errors
    
    # === SECTION 3: HARMONIC STABILITY (ALL Codecs) ===
    print("    Creating harmonic stability analysis for all codecs...")
    
    for idx, (codec_name, reconstruction) in enumerate(codec_items):
        row = current_row + idx // n_cols
        col = idx % n_cols
        
        ax_harm = fig.add_subplot(gs[row, col])
        
        recon_segment = reconstruction[start_sample:end_sample] if time_range else reconstruction
        min_len = min(len(original_segment), len(recon_segment))
        
        # Compute chromagram for harmonic similarity
        chroma_orig = librosa.feature.chroma_cqt(y=original_segment[:min_len], sr=sr, hop_length=512)
        chroma_recon = librosa.feature.chroma_cqt(y=recon_segment[:min_len], sr=sr, hop_length=512)
        
        chroma_times = librosa.times_like(chroma_orig, sr=sr, hop_length=512) + time_offset
        harmonic_sim = np.sum(chroma_orig * chroma_recon, axis=0) / (
            np.sqrt(np.sum(chroma_orig**2, axis=0) + 1e-8) * 
            np.sqrt(np.sum(chroma_recon**2, axis=0) + 1e-8)
        )
        
        color = CODEC_COLORS.get(codec_name.lower(), '#E63946')
        ax_harm.plot(chroma_times, harmonic_sim, color=color, linewidth=2)
        ax_harm.fill_between(chroma_times, 0, harmonic_sim, color=color, alpha=0.3)
        ax_harm.axhline(y=np.mean(harmonic_sim), color='gray', linestyle='--', alpha=0.5)
        
        ax_harm.set_title(f'{codec_name} - Harmonic Similarity', fontweight='bold', fontsize=11, 
                         pad=8, color=color)
        ax_harm.set_ylabel('Similarity', fontweight='bold', fontsize=10)
        ax_harm.set_xlabel('Time (s)', fontweight='bold', fontsize=10)
        ax_harm.set_ylim([0, 1])
        ax_harm.grid(True, alpha=0.3)
        
        ax_harm.text(0.98, 0.95, f'Mean: {np.mean(harmonic_sim):.3f}',
                    transform=ax_harm.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                    fontsize=8)
    
    current_row += n_rows_harmonics
    
    # === SECTION 4: FREQUENCY BAND ANALYSIS (ALL Codecs) ===
    print("    Creating frequency band error analysis for all codecs...")
    
    # Define frequency bands
    bands = [(20, 250), (250, 2000), (2000, 8000), (8000, 24000)]
    band_labels = ['Bass\n(20-250 Hz)', 'Low-Mid\n(250-2k Hz)', 
                   'High-Mid\n(2-8k Hz)', 'Treble\n(8-24k Hz)']
    n_bands = len(bands)
    
    # Compute error per band for each codec
    errors = np.zeros((n_codecs, n_bands))
    
    for codec_idx, (codec_name, reconstruction) in enumerate(codec_items):
        # Ensure same length before computing error
        min_len = min(len(original), len(reconstruction))
        error_signal = original[:min_len] - reconstruction[:min_len]
        
        for band_idx, (low_freq, high_freq) in enumerate(bands):
            # Design bandpass filter
            nyquist = sr / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are valid
            low = max(0.001, min(low, 0.999))
            high = max(low + 0.001, min(high, 0.999))
            
            # Bandpass filter
            sos = scipy_signal.butter(4, [low, high], btype='band', output='sos')
            filtered_error = scipy_signal.sosfilt(sos, error_signal)
            
            # Compute RMS error in this band
            errors[codec_idx, band_idx] = np.sqrt(np.mean(filtered_error**2))
    
    # Add frequency band section to GridSpec
    n_rows_freqbands = 1
    ax_freqbands = fig.add_subplot(gs[current_row, :])
    
    x_bands = np.arange(n_bands)
    width = 0.8 / n_codecs
    
    for codec_idx, codec_name in enumerate([name for name, _ in codec_items]):
        offset = (codec_idx - n_codecs/2 + 0.5) * width
        color = CODEC_COLORS.get(codec_name.lower(), '#333333')
        
        bars = ax_freqbands.bar(x_bands + offset, errors[codec_idx], width, 
                     label=codec_name, color=color, 
                     edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Add value labels on bars (compact format)
        for bar, err_val in zip(bars, errors[codec_idx]):
            height = bar.get_height()
            ax_freqbands.text(bar.get_x() + bar.get_width()/2., height,
                   f'{err_val:.4f}',
                   ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax_freqbands.set_xlabel('Frequency Band', fontweight='bold', fontsize=11)
    ax_freqbands.set_ylabel('RMS Error', fontweight='bold', fontsize=11)
    ax_freqbands.set_title('Reconstruction Error by Frequency Band', fontweight='bold', fontsize=12, pad=10)
    ax_freqbands.set_xticks(x_bands)
    ax_freqbands.set_xticklabels(band_labels, fontsize=9)
    ax_freqbands.legend(loc='upper right', framealpha=0.95, fontsize=9, ncol=min(n_codecs, 5))
    ax_freqbands.grid(True, alpha=0.3, axis='y')
    
    current_row += n_rows_freqbands
    
    # === SECTION 5: METRICS (Bitrate vs Quality + SI-SDR Bars) ===
    print("    Creating metric comparison plots...")
    
    ax_bitrate = fig.add_subplot(gs[current_row:, :int(n_cols*0.6)])
    ax_metrics = fig.add_subplot(gs[current_row:, int(n_cols*0.6):])
    
    if include_theoretical:
        import sys
        from pathlib import Path
        eval_dir = Path(__file__).parent.parent / 'evaluation'
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        from simulate_chimera import get_chimera_theoretical_comparison
        theoretical_data = get_chimera_theoretical_comparison()
    else:
        theoretical_data = {}
    
    bitrate_mapping = {'dac': 32, 'encodec': 24, 'soundstream': 16, 'chimera_24kbps': 24, 'chimera_30kbps': 30}
    
    # Plot SI-SDR vs Bitrate
    for codec_name, codec_metrics in metrics.items():
        if 'SI-SDR (dB)' not in codec_metrics:
            continue
        
        codec_key = codec_name.lower()
        bitrate = bitrate_mapping.get(codec_key, 24)
        value = codec_metrics['SI-SDR (dB)']
        
        color = CODEC_COLORS.get(codec_key, '#333333')
        marker = 'D' if 'chimera' in codec_key else 'o'
        label = codec_name.replace('_', ' ').upper()
        
        ax_bitrate.scatter(bitrate, value, s=150, marker=marker, color=color,
                          edgecolor='white', linewidth=2, label=label, zorder=3, alpha=0.9)
    
    # Plot theoretical lines (as reference only)
    if include_theoretical:
        theoretical_points = {}
        for codec_full_name, theo_metrics in theoretical_data.items():
            if 'SI-SDR (dB)' not in theo_metrics:
                continue
            codec_base = codec_full_name.split('(')[0].strip().lower()
            bitrate = theo_metrics['Bitrate']
            value = theo_metrics['SI-SDR (dB)']
            
            if codec_base not in theoretical_points:
                theoretical_points[codec_base] = []
            theoretical_points[codec_base].append((bitrate, value))
        
        for codec_base, points in theoretical_points.items():
            # Only plot theoretical lines for Chimera to compare against simulation
            if 'chimera' in codec_base:
                points = sorted(points)
                bitrates = [p[0] for p in points]
                values = [p[1] for p in points]
                color = CODEC_COLORS.get(codec_base, '#333333')
                ax_bitrate.plot(bitrates, values, color=color, linestyle=':', linewidth=1.5, alpha=0.5, zorder=1, label='Chimera (Projected)')
    
    ax_bitrate.set_xlabel('Bitrate (kbps)', fontweight='bold', fontsize=11)
    ax_bitrate.set_ylabel('SI-SDR (dB)', fontweight='bold', fontsize=11)
    ax_bitrate.set_title('Codec Efficiency: Bitrate vs Quality', fontweight='bold', fontsize=12, pad=10)
    ax_bitrate.legend(loc='best', framealpha=0.95, fontsize=9)
    ax_bitrate.grid(True, alpha=0.3)
    
    # SI-SDR comparison bars
    codec_names = list(metrics.keys())
    x = np.arange(len(codec_names))
    values = [metrics[c].get('SI-SDR (dB)', 0) for c in codec_names]
    colors = [CODEC_COLORS.get(c.lower(), '#333333') for c in codec_names]
    
    bars = ax_metrics.bar(x, values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax_metrics.set_ylabel('SI-SDR (dB)', fontweight='bold', fontsize=11)
    ax_metrics.set_title('SI-SDR Comparison', fontweight='bold', fontsize=12, pad=10)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels([c.replace('_', ' ').upper() for c in codec_names], 
                               rotation=45, ha='right', fontsize=8)
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Main title
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=18, y=0.995)
    elif artist and song_title:
        fig.suptitle(f'Codec Analysis: {artist} - {song_title}', fontweight='bold', fontsize=18, y=0.995)
    elif song_title:
        fig.suptitle(f'Codec Analysis: {song_title}', fontweight='bold', fontsize=18, y=0.995)
    else:
        fig.suptitle('Comprehensive Neural Audio Codec Analysis', fontweight='bold', fontsize=18, y=0.995)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive analysis to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_bitrate_quality_curve(
    experimental_metrics: Dict[str, Dict[str, float]],
    include_theoretical: bool = True,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot bitrate vs quality trade-off curves including theoretical Chimera projections.
    
    This visualization is crucial for understanding codec efficiency and comparing
    real experimental results with theoretical Chimera performance.
    
    Args:
        experimental_metrics: Dictionary mapping codec names to their metrics
                            Example: {'dac': {'SI-SDR (dB)': 16.8, 'Mel Distance': 0.58}, ...}
        include_theoretical: Whether to include theoretical data from presentation.md
        output_path: Path to save figure
        title: Custom title
    """
    import sys
    from pathlib import Path
    # Add evaluation directory to path
    eval_dir = Path(__file__).parent.parent / 'evaluation'
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from simulate_chimera import get_chimera_theoretical_comparison
    
    setup_plot_style()
    
    # Create figure with 3 subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Metric configurations
    metrics_config = [
        {
            'key': 'SI-SDR (dB)',
            'title': 'SI-SDR vs Bitrate',
            'ylabel': 'SI-SDR (dB)',
            'higher_better': True
        },
        {
            'key': 'Mel Distance',
            'title': 'Mel Distance vs Bitrate',
            'ylabel': 'Mel Distance',
            'higher_better': False
        },
        {
            'key': 'ViSQOL',
            'title': 'ViSQOL vs Bitrate',
            'ylabel': 'ViSQOL Score',
            'higher_better': True
        }
    ]
    
    # Get theoretical data if requested
    if include_theoretical:
        theoretical_data = get_chimera_theoretical_comparison()
    else:
        theoretical_data = {}
    
    # Define bitrate mapping for experimental codecs (you may need to adjust these)
    bitrate_mapping = {
        'dac': 32,
        'encodec': 24,
        'soundstream': 16
    }
    
    for ax_idx, config in enumerate(metrics_config):
        ax = axes[ax_idx]
        metric_key = config['key']
        
        # Plot experimental results
        for codec_name, metrics in experimental_metrics.items():
            if metric_key not in metrics:
                continue
            
            codec_key = codec_name.lower()
            bitrate = bitrate_mapping.get(codec_key, 24)
            value = metrics[metric_key]
            
            color = CODEC_COLORS.get(codec_key, '#333333')
            marker = 'o'
            
            ax.scatter(bitrate, value, s=150, marker=marker, color=color, 
                      edgecolor='white', linewidth=2, label=f'{codec_name.upper()} (exp)',
                      zorder=3, alpha=0.9)
        
        # Plot theoretical results
        if include_theoretical:
            theoretical_points = {}  # codec -> list of (bitrate, value)
            
            for codec_full_name, theo_metrics in theoretical_data.items():
                if metric_key not in theo_metrics:
                    continue
                
                # Extract codec base name (e.g., "Chimera" from "Chimera (30kbps)")
                codec_base = codec_full_name.split('(')[0].strip().lower()
                bitrate = theo_metrics['Bitrate']
                value = theo_metrics[metric_key]
                
                if codec_base not in theoretical_points:
                    theoretical_points[codec_base] = []
                theoretical_points[codec_base].append((bitrate, value))
            
            # Plot lines connecting theoretical points for each codec
            for codec_base, points in theoretical_points.items():
                if 'chimera' not in codec_base.lower():
                    continue
                    
                points = sorted(points)  # Sort by bitrate
                bitrates = [p[0] for p in points]
                values = [p[1] for p in points]
                
                color = CODEC_COLORS.get(codec_base, '#333333')
                
                # Plot line
                ax.plot(bitrates, values, color=color, linestyle=':', linewidth=1.5, 
                       alpha=0.5, zorder=1, label=f'{codec_base} (Projected)')
                
                # Plot points
                for br, val in points:
                    if 'chimera' in codec_base.lower():
                        ax.scatter(br, val, s=80, marker='x', color=color,
                                 linewidth=1.5, 
                                 label=f'{codec_base.upper()} Target' if points.index((br, val)) == 0 else '',
                                 zorder=2, alpha=0.6)
        
        ax.set_xlabel('Bitrate (kbps)', fontweight='bold', fontsize=12)
        ax.set_ylabel(config['ylabel'], fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontweight='bold', fontsize=13, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=9)
        
        # Set appropriate y-axis limits
        if config['higher_better']:
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(bottom=0)
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    else:
        fig.suptitle('Codec Performance: Bitrate vs Quality', fontweight='bold', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved bitrate-quality curve to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Music comparison visualization module loaded.")
    print("\nExample usage:")
    print("""
    import soundfile as sf
    from visualization.music_comparison import *
    
    # Load audio files
    original, sr = sf.read('original.wav')
    dac_recon, _ = sf.read('dac_reconstruction.wav')
    encodec_recon, _ = sf.read('encodec_reconstruction.wav')
    
    # 1. Waveform comparison
    plot_waveform_comparison({
        'Original': original,
        'DAC': dac_recon,
        'EnCodec': encodec_recon
    }, sr=sr, time_range=(10, 12), output_path='waveform_comp.png')
    
    # 2. Reconstruction error
    plot_reconstruction_error(
        original,
        {'DAC': dac_recon, 'EnCodec': encodec_recon},
        sr=sr,
        time_range=(10, 12),
        output_path='error_comp.png'
    )
    
    # 3. Frequency band analysis
    plot_frequency_band_error(
        original,
        {'DAC': dac_recon, 'EnCodec': encodec_recon},
        sr=sr,
        output_path='freq_band_error.png'
    )
    
    # 4. Metric comparison
    plot_metric_comparison({
        'DAC': {'SI-SDR': 16.8, 'ViSQOL': 4.28, 'Mel Dist': 0.58},
        'EnCodec': {'SI-SDR': 12.3, 'ViSQOL': 3.82, 'Mel Dist': 0.89}
    }, output_path='metrics.png')
    """)
