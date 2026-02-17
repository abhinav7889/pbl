# Installation Guide

## Prerequisites

- Python 3.11 (recommended - matches your conda environment)
- CUDA 12.1 (optional, for GPU acceleration)
- At least 4GB RAM
- 10GB free disk space (for models and audio files)

## Installation Steps

### Step 1: Set Up Environment

**Option A: Using your existing conda environment**
```bash
# Activate your existing DAC environment
conda activate dac

# Install additional dependencies
pip install -r requirements.txt
```

**Option B: Create new conda environment**
```bash
# Create environment from your existing setup
conda create -n codec_comparison python=3.11
conda activate codec_comparison

# Install CUDA toolkit (optional, for GPU)
conda install cuda-toolkit=12.1 cudnn=8.9

# Install all dependencies
pip install -r requirements.txt
```

**Option C: Using pip only**
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python check_dependencies.py
```

**Expected output:**
```
============================================================
Checking dependencies for Music Codec Comparison Tool
============================================================

Core Dependencies:
------------------------------------------------------------
✓ NumPy                          - OK
✓ SciPy                          - OK
✓ Matplotlib                     - OK
✓ SoundFile                      - OK
✓ Librosa                        - OK

Deep Learning:
------------------------------------------------------------
✓ PyTorch                        - OK
✓ TorchVision                    - OK
✓ TorchAudio                     - OK

✓ CUDA available - GPU: NVIDIA GeForce RTX 3080

Audio Codecs:
------------------------------------------------------------
✓ DAC (Descript Audio Codec)     - OK
✓ EnCodec                        - OK

Optional Dependencies:
------------------------------------------------------------
✓ PyYAML                         - OK
✓ AudioTools                     - OK

============================================================
Summary:
============================================================
✓ All required dependencies are installed!
```

### Step 3: Test with Sample Audio

```bash
# Download a short test audio (or use your own)
# Place it in data/test_audio/

# Run evaluation
python evaluation/evaluate_codecs.py \
  --input data/test_audio/test.wav \
  --codecs dac encodec \
  --output-dir outputs/test_run

# Check the output
ls outputs/test_run/figures/
```

## Troubleshooting

### Issue: "Import error: dac not found"

**Solution:**
```bash
pip install descript-audio-codec
```

### Issue: "Import error: encodec not found"

**Solution:**
```bash
pip install encodec
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Use shorter audio clips (< 30 seconds)
2. Process one codec at a time
3. Use CPU instead:
   ```bash
   # The scripts will automatically fall back to CPU if GPU is unavailable
   ```

### Issue: "librosa not available, skipping mel distance"

**Solution:**
```bash
pip install librosa
```

### Issue: PyTorch installation fails

**For CUDA 12.1:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### Issue: "No module named 'yaml'"

**Solution:**
```bash
pip install pyyaml
```

## Package Versions

If you encounter version conflicts, here are the tested versions:

```
numpy==1.26.4
scipy==1.11.4
matplotlib==3.8.2
soundfile==0.12.1
librosa==0.10.1
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
descript-audio-codec==1.0.0
encodec==0.1.1
audiotools==0.7.3
pyyaml==6.0.1
```

## Alternative: Use Your Existing DAC Environment

Since you already have a DAC environment set up, you can just add the missing packages:

```bash
conda activate dac  # or your environment name

pip install matplotlib scipy librosa encodec audiotools pyyaml
```

Then verify:
```bash
python check_dependencies.py
```

## Next Steps

Once installation is complete:
1. Read the [README.md](README.md) for usage instructions
2. Place your music files in `data/test_audio/`
3. Run your first comparison!

```bash
python evaluation/evaluate_codecs.py \
  --input data/test_audio/your_song.wav \
  --codecs dac encodec \
  --output-dir outputs/evaluations
```
