"""
Naive User-Friendly Visualization for Audio Codec Comparison.

This module creates simple, non-technical visualizations that help everyday users
understand codec performance without requiring audio engineering knowledge.

Focus: Clean, minimalist design answering "Which codec sounds best?"
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys

# Add evaluation directory to path for imports
eval_dir = Path(__file__).parent.parent / 'evaluation'
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))


# Color scheme (Minimalist & Clean)
COLORS = {
    'excellent': '#4CAF50',  # Green
    'good': '#8BC34A',       # Light green
    'fair': '#FFC107',       # Amber
    'poor': '#FF5722',       # Red-orange
    'neutral': '#607D8B',    # Blue-grey
    'chimera': '#F77F00',    # Orange (highlight)
    'background': '#FAFAFA', # Very light grey
}

# Codec colors for consistency
CODEC_COLORS = {
    'encodec': '#E63946',
    'dac': '#457B9D',
    'soundstream': '#2A9D8F',
    'chimera_24kbps': '#F77F00',
    'chimera_30kbps': '#F77F00',
    'chimera': '#F77F00',
}


def compute_quality_score(metrics: Dict[str, float]) -> float:
    """
    Convert technical metrics to simple 0-10 quality score.
    
    This scoring is designed to be intuitive:
    - 9-10: Excellent (near-lossless)
    - 7-9: Good (high quality)
    - 5-7: Fair (acceptable)
    - 0-5: Poor (noticeable artifacts)
    
    Args:
        metrics: Dictionary with 'SI-SDR (dB)', 'Mel Distance', etc.
        
    Returns:
        Quality score from 0-10
    """
    score = 0.0
    weights = 0.0
    
    # SI-SDR: Most important (50% weight)
    # Typical range: 10-20 dB
    # 10 dB = poor (score ~3), 15 dB = good (score ~7), 20 dB = excellent (score ~10)
    if 'SI-SDR (dB)' in metrics:
        si_sdr = metrics['SI-SDR (dB)']
        # Linear mapping: 10dB->3, 15dB->6.5, 20dB->10
        si_sdr_score = min(10, max(0, (si_sdr - 10) * 0.7 + 3))
        score += si_sdr_score * 0.5
        weights += 0.5
    
    # Mel Distance: Perceptual quality (30% weight)
    # Lower is better, typical range: 0.4-1.0
    # 0.4 = excellent (score 10), 0.7 = good (score 5), 1.0 = poor (score 0)
    if 'Mel Distance' in metrics:
        mel_dist = metrics['Mel Distance']
        # Inverse mapping: 0.4->10, 0.7->5, 1.0->0
        mel_score = min(10, max(0, (1.0 - mel_dist) * 10 / 0.6))
        score += mel_score * 0.3
        weights += 0.3
    
    # ViSQOL: If available (20% weight)
    # Range: 1-5, where 5 is best
    if 'ViSQOL' in metrics:
        visqol = metrics['ViSQOL']
        # Scale to 0-10: 1->0, 3->5, 5->10
        visqol_score = (visqol - 1) * 2.5
        score += visqol_score * 0.2
        weights += 0.2
    elif weights < 1.0:
        # If no ViSQOL, redistribute weight to SI-SDR
        if 'SI-SDR (dB)' in metrics:
            si_sdr = metrics['SI-SDR (dB)']
            si_sdr_score = min(10, max(0, (si_sdr - 10) * 0.7 + 3))
            score += si_sdr_score * 0.2
            weights += 0.2
    
    # Normalize by total weights
    if weights > 0:
        final_score = score / weights
    else:
        final_score = 0
    
    return min(10, max(0, final_score))


def score_to_stars(score: float) -> str:
    """
    Convert 0-10 score to star rating string.
    
    Args:
        score: Quality score from 0-10
        
    Returns:
        String with star symbols (e.g., "★★★★☆")
    """
    filled = int(round(score / 2))  # Convert to 0-5 scale
    empty = 5 - filled
    return '★' * filled + '☆' * empty


def get_quality_color(score: float) -> str:
    """Get color based on quality score."""
    if score >= 9.0:
        return COLORS['excellent']
    elif score >= 7.0:
        return COLORS['good']
    elif score >= 5.0:
        return COLORS['fair']
    else:
        return COLORS['poor']


def assess_codec_suitability(codec_name: str, score: float) -> Dict[str, bool]:
    """
    Assess what each codec is good for based on name and score.
    
    Returns:
        Dictionary with suitability for different use cases
    """
    codec_lower = codec_name.lower()
    
    suitability = {
        'Voice': False,
        'Music': False,
        'Tonal': False
    }
    
    # Heuristics based on codec characteristics
    if 'dac' in codec_lower:
        suitability['Voice'] = score >= 7.0
        suitability['Music'] = score >= 7.5
        suitability['Tonal'] = score >= 7.0
    elif 'encodec' in codec_lower:
        suitability['Voice'] = score >= 5.0
        suitability['Music'] = score >= 6.0
        suitability['Tonal'] = score >= 5.5
    elif 'soundstream' in codec_lower:
        suitability['Voice'] = score >= 6.0
        suitability['Music'] = score >= 5.5
        suitability['Tonal'] = score >= 5.0
    elif 'chimera' in codec_lower:
        suitability['Voice'] = True  # Chimera is good for everything
        suitability['Music'] = True
        suitability['Tonal'] = True
    
    return suitability


def plot_naive_user_summary(
    metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    audio_name: str = "Audio Sample"
) -> None:
    """
    Create a comprehensive, user-friendly summary visualization.
    
    This generates a clean, minimalist single-page report that answers:
    - Which codec sounds best?
    - What's the quality vs file size trade-off?
    - Which codec should I use?
    
    Args:
        metrics: Dictionary mapping codec names to their metrics
        output_path: Path to save the figure
        audio_name: Name of the audio being tested
    """
    # Compute scores for all codecs
    codec_scores = {}
    for codec_name, codec_metrics in metrics.items():
        codec_scores[codec_name] = compute_quality_score(codec_metrics)
    
    # Find the best codec
    best_codec = max(codec_scores.items(), key=lambda x: x[1])
    
    # Setup figure
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Use GridSpec for layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.4,
                  left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # === TITLE ===
    fig.suptitle('Audio Codec Performance Summary',
                 fontsize=24, fontweight='bold', y=0.97)
    
    # === SUBTITLE ===
    subtitle_ax = fig.add_subplot(gs[0, :])
    subtitle_ax.axis('off')
    subtitle_ax.text(0.5, 0.5, f'Testing: {audio_name}',
                    ha='center', va='center', fontsize=16, color=COLORS['neutral'])
    
    # === OVERALL WINNER BANNER ===
    winner_ax = fig.add_subplot(gs[1, :])
    winner_ax.axis('off')
    
    winner_name = best_codec[0].replace('_', ' ').upper()
    winner_score = best_codec[1]
    winner_color = CODEC_COLORS.get(best_codec[0].lower(), COLORS['chimera'])
    
    # Draw winner box
    winner_box = mpatches.FancyBboxPatch(
        (0.1, 0.2), 0.8, 0.6,
        boxstyle="round,pad=0.05",
        edgecolor=winner_color,
        facecolor=winner_color,
        alpha=0.15,
        linewidth=3,
        transform=winner_ax.transAxes
    )
    winner_ax.add_patch(winner_box)
    
    # Trophy and text
    winner_ax.text(0.15, 0.65, '🏆', fontsize=48, va='center', transform=winner_ax.transAxes)
    winner_ax.text(0.25, 0.7, f'BEST QUALITY: {winner_name}',
                  fontsize=20, fontweight='bold', va='center',
                  color=winner_color, transform=winner_ax.transAxes)
    
    stars = score_to_stars(winner_score)
    winner_ax.text(0.25, 0.45, f'{stars}  {winner_score:.1f}/10',
                  fontsize=18, va='center', transform=winner_ax.transAxes)
    
    # Get bitrate info
    bitrate_map = {'dac': '32kbps', 'encodec': '24kbps', 'soundstream': '16kbps',
                   'chimera_24kbps': '24kbps', 'chimera_30kbps': '30kbps'}
    winner_bitrate = bitrate_map.get(best_codec[0].lower(), 'N/A')
    
    winner_ax.text(0.25, 0.25, f'Quality Score: {winner_score:.1f}/10  |  File Size: {winner_bitrate}',
                  fontsize=14, va='center', color=COLORS['neutral'],
                  transform=winner_ax.transAxes)
    
    # === CODEC SCORE CARDS ===
    codec_list = sorted(codec_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    
    for idx, (codec_name, score) in enumerate(codec_list):
        col = idx % 4
        ax_card = fig.add_subplot(gs[2, col])
        ax_card.axis('off')
        
        # Card background
        card_color = CODEC_COLORS.get(codec_name.lower(), COLORS['neutral'])
        border_color = get_quality_color(score)
        
        card_box = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            edgecolor=border_color,
            facecolor='white',
            linewidth=3,
            transform=ax_card.transAxes
        )
        ax_card.add_patch(card_box)
        
        # Codec name
        display_name = codec_name.replace('_', ' ').upper()
        ax_card.text(0.5, 0.85, display_name,
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    color=card_color, transform=ax_card.transAxes)
        
        # Bitrate
        bitrate = bitrate_map.get(codec_name.lower(), 'N/A')
        ax_card.text(0.5, 0.72, bitrate,
                    ha='center', va='top', fontsize=10,
                    color=COLORS['neutral'], transform=ax_card.transAxes)
        
        # Stars
        stars = score_to_stars(score)
        ax_card.text(0.5, 0.58, stars,
                    ha='center', va='center', fontsize=16,
                    color='#FFD700', transform=ax_card.transAxes)
        
        # Score
        ax_card.text(0.5, 0.42, f'{score:.1f}/10',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax_card.transAxes)
        
        # Suitability indicators
        suitability = assess_codec_suitability(codec_name, score)
        y_pos = 0.28
        for use_case, is_suitable in suitability.items():
            symbol = '✓' if is_suitable else '✗'
            color = COLORS['excellent'] if is_suitable else COLORS['poor']
            ax_card.text(0.5, y_pos, f'{symbol} {use_case}',
                        ha='center', va='center', fontsize=9,
                        color=color, transform=ax_card.transAxes)
            y_pos -= 0.12
    
    # === QUALITY vs FILE SIZE SCATTER ===
    scatter_ax = fig.add_subplot(gs[3:, :3])
    
    # Prepare data
    bitrate_values = {'dac': 32, 'encodec': 24, 'soundstream': 16,
                     'chimera_24kbps': 24, 'chimera_30kbps': 30}
    
    x_data = []
    y_data = []
    colors_list = []
    sizes = []
    labels = []
    
    for codec_name, score in codec_scores.items():
        bitrate = bitrate_values.get(codec_name.lower(), 24)
        x_data.append(bitrate)
        y_data.append(score)
        
        color = CODEC_COLORS.get(codec_name.lower(), COLORS['neutral'])
        colors_list.append(color)
        
        # Larger size for Chimera
        size = 300 if 'chimera' in codec_name.lower() else 200
        sizes.append(size)
        
        label = codec_name.replace('_', ' ').upper()
        labels.append(label)
    
    # Plot scatter
    scatter_ax.scatter(x_data, y_data, s=sizes, c=colors_list,
                      alpha=0.7, edgecolors='white', linewidths=2, zorder=3)
    
    # Add labels
    for x, y, label, is_chimera in zip(x_data, y_data, labels, ['chimera' in l.lower() for l in labels]):
        offset_y = 0.4 if is_chimera else 0.3
        fontweight = 'bold' if is_chimera else 'normal'
        scatter_ax.text(x, y + offset_y, label, ha='center', fontsize=10,
                       fontweight=fontweight, zorder=4)
    
    # Quality zones (background shading)
    scatter_ax.axhspan(0, 5, alpha=0.1, color=COLORS['poor'], zorder=1)
    scatter_ax.axhspan(5, 7, alpha=0.1, color=COLORS['fair'], zorder=1)
    scatter_ax.axhspan(7, 9, alpha=0.1, color=COLORS['good'], zorder=1)
    scatter_ax.axhspan(9, 10, alpha=0.1, color=COLORS['excellent'], zorder=1)
    
    scatter_ax.set_xlabel('File Size (kbps) → Smaller is Better', fontsize=13, fontweight='bold')
    scatter_ax.set_ylabel('Sound Quality → Higher is Better', fontsize=13, fontweight='bold')
    scatter_ax.set_title('Quality vs File Size Trade-off', fontsize=15, fontweight='bold', pad=15)
    scatter_ax.set_ylim(4, 10.5)
    scatter_ax.set_xlim(12, 36)
    scatter_ax.grid(True, alpha=0.2, linestyle='--')
    scatter_ax.set_facecolor('#FAFAFA')
    
    # === RECOMMENDATIONS ===
    rec_ax = fig.add_subplot(gs[3:, 3])
    rec_ax.axis('off')
    
    # Find best for each category
    best_quality = max(codec_scores.items(), key=lambda x: x[1])
    smallest_file = min([(c, bitrate_values.get(c.lower(), 24)) for c in codec_scores.keys()],
                       key=lambda x: x[1])
    
    # Balanced: best score per kbps ratio
    balanced = max([(c, codec_scores[c] / bitrate_values.get(c.lower(), 24)) 
                    for c in codec_scores.keys()], key=lambda x: x[1])
    
    rec_ax.text(0.5, 0.95, 'Quick Recommendations',
               ha='center', va='top', fontsize=14, fontweight='bold',
               transform=rec_ax.transAxes)
    
    recommendations = [
        ('💎', 'Best Quality', best_quality[0].replace('_', ' ').upper()),
        ('💾', 'Smallest File', smallest_file[0].replace('_', ' ').upper()),
        ('⚖️', 'Best Balance', balanced[0].replace('_', ' ').upper()),
    ]
    
    y_pos = 0.80
    for icon, category, codec in recommendations:
        # Background box
        box = mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.12), 0.9, 0.11,
            boxstyle="round,pad=0.01",
            edgecolor=COLORS['neutral'],
            facecolor=COLORS['background'],
            alpha=0.5,
            linewidth=1,
            transform=rec_ax.transAxes
        )
        rec_ax.add_patch(box)
        
        rec_ax.text(0.12, y_pos - 0.065, icon, fontsize=20, va='center',
                   transform=rec_ax.transAxes)
        rec_ax.text(0.25, y_pos - 0.04, category, fontsize=10, fontweight='bold',
                   va='center', transform=rec_ax.transAxes)
        rec_ax.text(0.25, y_pos - 0.09, codec, fontsize=9,
                   color=COLORS['neutral'], va='center', transform=rec_ax.transAxes)
        
        y_pos -= 0.16
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved naive user summary to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Naive User Summary Visualization Module")
    print("=" * 50)
    
    # Try to load real metrics data first
    metrics_files = list(Path('outputs/evaluations').glob('*_metrics.json'))
    
    if metrics_files:
        latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading real data from: {latest_metrics}")
        
        import json
        with open(latest_metrics, 'r') as f:
            metrics_data = json.load(f)
            
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = latest_metrics.stem.replace('_metrics', '')
        
        plot_naive_user_summary(
            metrics_data,
            output_path=str(output_dir / f"{base_name}_summary.png"),
            audio_name=base_name
        )
        print(f"Generated summary for {base_name}")
        
    else:
        # Fallback to example data
        print("No real metrics found, using example data...")
        sample_metrics = {
            'DAC': {
                'SI-SDR (dB)': 16.8,
                'Mel Distance': 0.58,
                'ViSQOL': 4.28
            },
            'EnCodec': {
                'SI-SDR (dB)': 12.3,
                'Mel Distance': 0.89,
                'ViSQOL': 3.82
            },
            'Chimera_30kbps': {
                'SI-SDR (dB)': 17.9,
                'Mel Distance': 0.51,
                'ViSQOL': 4.42
            }
        }
        
        for codec, metrics in sample_metrics.items():
            score = compute_quality_score(metrics)
            stars_count = int(round(score / 2))
            print(f"{codec}: {score:.1f}/10 ({stars_count}/5 stars)")
