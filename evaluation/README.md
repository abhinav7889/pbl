# Neural Audio Codec Evaluation

This directory contains tools for evaluating and comparing neural audio codecs on music files, including a **theoretical simulation of Project Chimera**.

## Supported Codecs

### Real Codecs
- **DAC (Descript Audio Codec)**: High-quality general-purpose codec
  - Native sample rates: 16kHz, 24kHz, 44.1kHz
  - Automatic resampling handles any input sample rate
- **EnCodec**: Meta's neural audio codec optimized for music
  - Native sample rates: 24kHz, 48kHz
- **SoundStream**: Google's foundational neural audio codec
  - Native sample rate: 16kHz
  - Automatic resampling handles any input sample rate

### Simulated Codec (Theoretical)
- **Chimera**: Simulated next-generation codec based on theoretical projections
  - Two variants: 24kbps and 30kbps
  - Uses DAC as base, reports theoretical metrics from presentation.md
  - Demonstrates projected improvements in tonal fidelity
  - Enable with `--include-chimera` flag

## Installation

### Prerequisites
```bash
pip install torch numpy soundfile librosa scipy matplotlib
```

**Note**: `librosa` is **required** for automatic resampling between different sample rates. Without it, you must ensure your input audio matches the codec's native sample rate.

### Codec Libraries

#### DAC (Descript Audio Codec)
```bash
pip install descript-audio-codec
```

#### EnCodec
```bash
pip install encodec
```

#### SoundStream
```bash
pip install soundstream
```

Note: SoundStream uses pre-trained models from the `haydenshively/soundstream` implementation.

## Usage

### Basic Usage

Evaluate a single audio file with multiple codecs:

```bash
python evaluate_codecs.py --input path/to/audio.wav --codecs dac encodec soundstream
```

### With Chimera Simulation

Include theoretical Chimera codec comparison:

```bash
python evaluate_codecs.py \
    --input path/to/audio.wav \
    --codecs dac encodec \
    --include-chimera
```

This will generate:
- Real codec reconstructions (DAC, EnCodec, etc.)
- Simulated Chimera 24kbps and 30kbps reconstructions
- Bitrate vs quality curves comparing all codecs

### Advanced Options

```bash
python evaluate_codecs.py \
    --input path/to/audio.wav \
    --codecs dac encodec soundstream \
    --output-dir results/my_evaluation \
    --sr 48000 \
    --dac-model 44khz \
    --encodec-bandwidth 24.0 \
    --soundstream-bandwidth 6.0 \
    --include-chimera
```

### Arguments

- `--input`: Path to input audio file (required)
- `--codecs`: List of codecs to evaluate (default: `dac encodec`)
  - Options: `dac`, `encodec`, `soundstream`
- `--output-dir`: Output directory for results (default: `outputs/evaluations`)
- `--sr`: Sample rate in Hz (default: `48000`)
- `--include-chimera`: Include simulated Chimera codec (both 24kbps and 30kbps variants)

#### Codec-Specific Options

**DAC:**
- `--dac-model`: Model type (choices: `16khz`, `24khz`, `44khz`; default: `44khz`)
  - `16khz`: Optimized for 16kHz audio (speech)
  - `24khz`: Optimized for 24kHz audio (balanced)
  - `44khz`: Optimized for 44.1kHz audio (music, CD quality)
  - **Note**: Audio is automatically resampled to match the model's native rate

**EnCodec:**
- `--encodec-bandwidth`: Target bandwidth in kbps (choices: `1.5`, `3.0`, `6.0`, `12.0`, `24.0`; default: `24.0`)

**SoundStream:**
- `--soundstream-bandwidth`: Target bandwidth in kbps (choices: `3.0`, `6.0`, `12.0`, `18.0`; default: `6.0`)
  - **Note**: Operates at 16kHz; input audio is automatically resampled

## Output

The evaluation script produces:

### Audio Files
- `{filename}_original.wav`: Original input audio
- `{filename}_{codec}.wav`: Reconstructed audio for each codec
- `{filename}_chimera_24kbps.wav`: Simulated Chimera at 24kbps (if `--include-chimera`)
- `{filename}_chimera_30kbps.wav`: Simulated Chimera at 30kbps (if `--include-chimera`)

### Metrics
- `{filename}_metrics.json`: Objective quality metrics for each codec
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - Mel Distance
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - ViSQOL (for Chimera simulation)

### Visualizations (in `figures/` subdirectory)

#### For Technical Users
- **`{filename}_comprehensive_analysis.png`**: **All-in-one publication-quality multi-panel figure**
  - **4x3 grid layout** containing:
    - **Row 1**: Spectrograms (Original + Codecs) - Visual frequency content comparison
    - **Row 2**: Error Spectrograms - Time-frequency error distribution
    - **Row 3**: Harmonic Stability - Harmonic similarity over time
    - **Row 4**: Bitrate-Quality Curves + SI-SDR Comparison
  
  This single figure provides a complete technical analysis at a glance!

#### For Everyone (Non-Technical Users) 👤
- **`{filename}_summary.png`**: **User-friendly summary with simple scoring**
  - **Clean, minimalist design** answering "Which codec sounds best?"
  - **5-star quality ratings** (converted from technical metrics)
  - **Simple 0-10 scores** for easy comparison
  - **Quick recommendations**: Best quality, smallest file, best balance
  - **Quality vs File Size scatter plot** with clear zones
  - **Use case indicators**: ✓ Good for Music, Voice, Tonal content
  
  Perfect for sharing with non-technical stakeholders!

#### Supplementary Figures
- `{filename}_freq_bands.png`: Detailed frequency band error breakdown (Bass, Low-Mid, High-Mid, Treble)

## Understanding the Visualizations

### For Technical Analysis (`_comprehensive_analysis.png`)

The consolidated visualization shows:

1. **Spectrograms (Row 1)**: 
   - Visual inspection of tonal artifacts, harmonic structure
   - Discontinuities, missing harmonics, spectral smearing
   - Side-by-side comparison of original vs codecs

2. **Error Spectrograms (Row 2)**:
   - WHERE errors concentrate in time-frequency space
   - High-frequency noise, problematic frequency bands
   - Quantified average error per codec

3. **Harmonic Stability (Row 3)**:
   - Harmonic content preservation over time
   - The "tonal weakness" problem visualization
   - Mean harmonic similarity scores

4. **Codec Efficiency (Row 4)**:
   - Bitrate vs Quality trade-off curves
   - Theoretical Chimera projections overlaid
   - SI-SDR direct comparison bars

### For User-Friendly Summary (`_summary.png`)

The naive user visualization provides:

1. **Overall Winner Banner**: 
   - 🏆 Clearly identifies the best codec
   - Shows quality score (0-10) and star rating
   - File size and key strengths

2. **Codec Score Cards**:
   - 5-star ratings for quick comparison
   - Simple 0-10 quality scores
   - Color-coded borders (Green=Excellent, Yellow=Good, Red=Fair)
   - Suitability indicators: ✓ Good for Music/Voice/Tonal

3. **Quality vs File Size Plot**:
   - Easy-to-read scatter plot
   - Clear quality zones (color-coded background)
   - Each codec clearly labeled
   - Shows trade-off at a glance

4. **Quick Recommendations**:
   - 💎 Best Quality
   - 💾 Smallest File
   - ⚖️ Best Balance

### How Scoring Works

**Technical metrics are converted to 0-10 scores:**
- **9-10**: Excellent (near-lossless quality)
- **7-9**: Good (high quality, minimal artifacts)
- **5-7**: Fair (acceptable quality)
- **0-5**: Poor (noticeable artifacts)

**Scoring formula** (simplified):
- SI-SDR (50%): Sound quality metric
- Mel Distance (30%): Perceptual difference
- ViSQOL (20%): Audio quality assessment

## Example

```bash
# Evaluate a music file with all codecs and Chimera simulation
python evaluate_codecs.py \
    --input samples/mozart.wav \
    --codecs dac encodec soundstream \
    --output-dir outputs/evaluations/mozart \
    --include-chimera

# Evaluate tonal content (bells, piano) to test "tonal weakness"
python evaluate_codecs.py \
    --input samples/bells.wav \
    --codecs dac encodec \
    --include-chimera \
    --output-dir outputs/evaluations/bells_tonal

# Evaluate at different bitrates
python evaluate_codecs.py \
    --input samples/jazz.wav \
    --codecs encodec \
    --encodec-bandwidth 6.0 \
    --output-dir outputs/evaluations/jazz_6kbps
```

## Metrics Explained

### SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- Higher is better (measured in dB)
- Measures signal quality independent of scale
- Typical values: 12-20 dB for high-quality codecs

### Mel Distance
- Lower is better
- Measures perceptual spectral difference
- Based on mel-scale frequency representation

### MSE/MAE
- Lower is better
- Direct waveform difference measures
- MSE penalizes large errors more heavily

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory errors, the models will automatically fall back to CPU. You can also:
- Process shorter audio clips
- Use lower sample rates
- Evaluate codecs one at a time

### Missing librosa
**Critical**: librosa is required for automatic sample rate conversion. Without it:
- DAC will fail if input sample rate doesn't match the model (e.g., 48kHz input with 44khz model)
- SoundStream will fail (requires 16kHz)
- Mel distance metric will be skipped

Install librosa for full functionality:
```bash
pip install librosa
```

### Sample Rate Mismatch
The evaluation script automatically handles sample rate conversion:
- **DAC 44khz model** expects 44.1kHz but will work with any input (e.g., 48kHz)
- **SoundStream** expects 16kHz but will work with any input
- **EnCodec** works best with 24kHz or 48kHz

For best results:
- Use 48kHz for general music evaluation
- Use 44.1kHz with `--dac-model 44khz` for CD-quality music
- Use 24kHz with `--dac-model 24khz` for balanced quality/efficiency

## Project Structure

Related to Project Chimera - see `../presentation.md` for the full proposal on advancing neural audio codecs and solving the "tonal weakness" problem.
