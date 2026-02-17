# Neural Audio Codec Comparison Tool

Music-specific visualization and evaluation toolkit for comparing neural audio codecs (DAC, EnCodec, SoundStream) on real music samples.

## Overview

This toolkit provides:
- **Waveform difference visualization** - See reconstruction quality at sample level
- **Reconstruction error analysis** - Identify where artifacts occur
- **Frequency band error breakdown** - Compare performance across bass/mid/treble
- **Objective metrics computation** - SI-SDR, Mel Distance, MSE, MAE
- **Automated evaluation pipeline** - Process multiple songs and codecs

## Installation

### Option 1: Quick Install (Recommended)

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Verify installation
python check_dependencies.py
```

### Option 2: Manual Install

1. **Install core dependencies:**
```bash
pip install numpy scipy matplotlib soundfile librosa
```

2. **Install PyTorch (with CUDA for GPU acceleration):**
```bash
# For CUDA 12.1 (matches your environment)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Or for CPU-only (slower)
pip install torch torchvision torchaudio
```

3. **Install audio codecs:**
```bash
pip install descript-audio-codec encodec audiotools
```

4. **Verify installation:**
```bash
python check_dependencies.py
```

### What Gets Installed

**Core Dependencies:**
- `numpy` - Numerical computing
- `scipy` - Signal processing
- `matplotlib` - Plotting and visualization
- `soundfile` - Audio file I/O
- `librosa` - Audio analysis

**Deep Learning:**
- `torch` - PyTorch framework
- `torchvision` - Vision utilities
- `torchaudio` - Audio utilities

**Audio Codecs:**
- `descript-audio-codec` - DAC codec
- `encodec` - EnCodec codec
- `audiotools` - Audio processing utilities

**Optional:**
- `pyyaml` - Configuration file support

## Quick Start

### 1. Basic Usage - Single Audio File

```bash
python evaluation/evaluate_codecs.py \
  --input data/test_audio/your_song.wav \
  --codecs dac encodec \
  --output-dir outputs/evaluations
```

This will:
- Encode and decode your song with DAC and EnCodec
- Compute objective metrics (SI-SDR, Mel Distance, etc.)
- Generate comparison visualizations
- Save all results to `outputs/evaluations/`

### 2. Custom Codec Settings

**DAC with specific model:**
```bash
python evaluation/evaluate_codecs.py \
  --input song.wav \
  --codecs dac \
  --dac-model 44khz \
  --output-dir results/
```

**EnCodec with specific bandwidth:**
```bash
python evaluation/evaluate_codecs.py \
  --input song.wav \
  --codecs encodec \
  --encodec-bandwidth 12.0 \
  --output-dir results/
```

### 3. Using Python API

```python
import soundfile as sf
from evaluation.evaluate_codecs import evaluate_codec
from visualization.music_comparison import (
    plot_waveform_comparison,
    plot_reconstruction_error,
    plot_frequency_band_error
)

# Load your song
original, sr = sf.read('song.wav')

# Evaluate DAC
dac_recon, dac_metrics = evaluate_codec('dac', original, sr, model_type='44khz')

# Evaluate EnCodec
encodec_recon, encodec_metrics = evaluate_codec('encodec', original, sr, bandwidth=24.0)

# Visualize waveform differences (zoom to 2-second window)
plot_waveform_comparison({
    'Original': original,
    'DAC': dac_recon,
    'EnCodec': encodec_recon
}, sr=sr, time_range=(10, 12), output_path='waveforms.png')

# Analyze reconstruction errors
plot_reconstruction_error(
    original,
    {'DAC': dac_recon, 'EnCodec': encodec_recon},
    sr=sr,
    output_path='errors.png'
)

# Compare performance across frequency bands
plot_frequency_band_error(
    original,
    {'DAC': dac_recon, 'EnCodec': encodec_recon},
    sr=sr,
    output_path='freq_bands.png'
)

print("DAC Metrics:", dac_metrics)
print("EnCodec Metrics:", encodec_metrics)
```

## Output Files

After running evaluation, you'll get:

```
outputs/evaluations/
├── your_song_original.wav          # Original audio
├── your_song_dac.wav               # DAC reconstruction
├── your_song_encodec.wav           # EnCodec reconstruction
├── your_song_metrics.json          # All metrics in JSON format
└── figures/
    ├── your_song_waveforms.png     # Waveform comparison
    ├── your_song_errors.png        # Reconstruction error plots
    ├── your_song_freq_bands.png    # Frequency band error analysis
    └── your_song_metrics.png       # Metric comparison bar chart
```

## Understanding the Graphs

### 1. Waveform Comparison
- Shows original vs reconstructions side-by-side
- **What to look for:** Visual differences in waveform shape
- **Good sign:** Reconstructed waveforms closely match original shape

### 2. Reconstruction Error (Residual)
- Shows `original - reconstruction` for each codec
- **What to look for:** Lower amplitude = better reconstruction
- **Metrics shown:** MSE, MAE, Max Error
- **Good sign:** Error signal has low amplitude and no clear patterns

### 3. Frequency Band Error
- Breaks down RMS error by frequency range
- **Bass (20-250 Hz):** Low frequency performance
- **Low-Mid (250-2k Hz):** Vocal and instrument fundamentals
- **High-Mid (2-8k Hz):** Clarity and presence
- **Treble (8-24k Hz):** Air and sparkle
- **What to look for:** Which codec performs better in which frequency range
- **Use case:** If DAC has lower error in treble but higher in bass, you know its strengths/weaknesses

### 4. Metric Comparison
- Shows SI-SDR, Mel Distance, MSE, MAE across codecs
- **Higher SI-SDR = Better** (signal vs distortion ratio)
- **Lower Mel Distance = Better** (perceptual similarity)
- **Lower MSE/MAE = Better** (reconstruction accuracy)

## Best Graphs for Your Presentation

Based on your use case (comparing DAC with other codecs on real music):

1. **Frequency Band Error** - Shows if DAC struggles with specific frequencies
2. **Reconstruction Error** - Visual proof of where artifacts occur
3. **Metric Comparison** - Objective performance summary
4. **Waveform Comparison** - Easy for audiences to understand

**NOT recommended for real music:**
- Spectrograms (too much detail, hard to see differences)
- MUSHRA scores (requires subjective listening tests)
- Tonal-specific metrics (only relevant for synthetic bell/piano tones)

## Configuration

Edit `config/comparison_config.yaml` to customize:
- Which codecs to test
- Bitrate/bandwidth settings
- Which visualizations to generate
- Frequency band definitions
- Output paths

## Tips for Best Results

1. **Use consistent sample rate:** 48kHz works best for both DAC and EnCodec
2. **Test diverse music:** Different genres may show different codec strengths
3. **Focus on 2-4 second windows:** Easier to see waveform differences
4. **Compare at matched bitrates:** DAC 32kbps vs EnCodec 24kbps isn't fair
5. **Listen to reconstructions:** Objective metrics don't tell the whole story

## Troubleshooting

**"Import error: dac not found"**
```bash
pip install descript-audio-codec
```

**"Import error: encodec not found"**
```bash
pip install encodec
```

**"librosa not available, skipping mel distance"**
```bash
pip install librosa
```

**"CUDA out of memory"**
- Add `--cpu` flag (if implemented) or reduce audio file length
- Process shorter segments of your song

**Waveforms look identical but metrics show differences**
- Zoom into a smaller time window with `time_range=(start, end)`
- Look at the error plots instead - they're more sensitive

## Project Structure

```
pbl_project/
├── visualization/
│   ├── codec_comparison.py      # Original plotting functions (unused for music)
│   └── music_comparison.py      # Music-specific visualizations
├── evaluation/
│   └── evaluate_codecs.py       # Main evaluation script
├── config/
│   └── comparison_config.yaml   # Configuration file
├── data/
│   ├── test_audio/             # Your music files go here
│   └── sample_metrics.json     # Example metrics structure
└── outputs/
    ├── evaluations/            # Codec reconstructions
    └── figures/                # Generated plots
```

## Citation

If you use this toolkit in your research, please cite the relevant codec papers:

**DAC:**
```
Kumar et al. "High-Fidelity Audio Compression with Improved RVQGAN" (2023)
```

**EnCodec:**
```
Défossez et al. "High Fidelity Neural Audio Compression" (2022)
```

## License

This evaluation toolkit is provided as-is for research and educational purposes.
