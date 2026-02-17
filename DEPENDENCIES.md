# Dependencies Summary

## Required Packages

### Core Scientific Computing
| Package | Purpose | Version |
|---------|---------|---------|
| `numpy` | Numerical arrays and operations | ≥1.24.0, <2.0.0 |
| `scipy` | Signal processing (bandpass filters) | ≥1.10.0 |
| `matplotlib` | Plotting and visualization | ≥3.7.0 |
| `soundfile` | Audio file I/O (WAV, FLAC, etc.) | ≥0.12.0 |
| `librosa` | Audio analysis (mel spectrograms, resampling) | ≥0.10.0 |

### Deep Learning Framework
| Package | Purpose | Version |
|---------|---------|---------|
| `torch` | PyTorch framework (codec models) | ≥2.1.0 |
| `torchvision` | Vision utilities (codec dependencies) | ≥0.16.0 |
| `torchaudio` | Audio utilities (codec dependencies) | ≥2.1.0 |

### Audio Codecs
| Package | Purpose | Version |
|---------|---------|---------|
| `descript-audio-codec` | DAC codec implementation | ≥1.0.0 |
| `encodec` | EnCodec codec implementation | ≥0.1.1 |
| `audiotools` | Audio processing utilities | ≥0.7.0 |

### Optional
| Package | Purpose | Version |
|---------|---------|---------|
| `pyyaml` | Configuration file support | ≥6.0 |

## Installation Commands

### Quick Install (All at Once)
```bash
pip install -r requirements.txt
```

### Manual Install by Category

**Core dependencies:**
```bash
pip install numpy scipy matplotlib soundfile librosa
```

**PyTorch (with CUDA 12.1):**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

**PyTorch (CPU only):**
```bash
pip install torch torchvision torchaudio
```

**Audio codecs:**
```bash
pip install descript-audio-codec encodec audiotools
```

**Optional:**
```bash
pip install pyyaml
```

## Using Your Existing Environment

If you already have a DAC environment from your conda setup, you likely already have:
- ✓ Python 3.11
- ✓ torch, torchvision, torchaudio (with CUDA 12.1)
- ✓ descript-audio-codec
- ✓ numpy, scipy
- ✓ flask, flask-cors (not needed for this project)

**You only need to add:**
```bash
conda activate dac  # or your environment name

pip install matplotlib soundfile librosa encodec audiotools pyyaml
```

## Verification

Check which packages are installed:
```bash
python check_dependencies.py
```

Expected output if all packages are installed:
```
============================================================
Checking dependencies for Music Codec Comparison Tool
============================================================

Core Dependencies:
------------------------------------------------------------
[OK] NumPy                         
[OK] SciPy                         
[OK] Matplotlib                    
[OK] SoundFile                     
[OK] Librosa                       

Deep Learning:
------------------------------------------------------------
[OK] PyTorch                       
[OK] TorchVision                   
[OK] TorchAudio                    

[OK] CUDA available - GPU: NVIDIA ...

Audio Codecs:
------------------------------------------------------------
[OK] DAC (Descript Audio Codec)    
[OK] EnCodec                       

Optional Dependencies:
------------------------------------------------------------
[OK] PyYAML                        
[OK] AudioTools                    

============================================================
Summary:
============================================================
[OK] All required dependencies are installed!
```

## Package Sizes (Approximate)

| Package | Size | Notes |
|---------|------|-------|
| numpy | ~20 MB | |
| scipy | ~50 MB | |
| matplotlib | ~30 MB | |
| soundfile | ~1 MB | Lightweight |
| librosa | ~10 MB | |
| torch (with CUDA) | ~2.5 GB | Large! GPU support |
| torch (CPU only) | ~200 MB | Smaller |
| torchvision | ~10 MB | |
| torchaudio | ~5 MB | |
| descript-audio-codec | ~5 MB | + model downloads (~100MB) |
| encodec | ~10 MB | + model downloads (~50MB) |
| audiotools | ~5 MB | |

**Total install size:** ~3-4 GB with CUDA, ~500 MB CPU-only

## System Requirements

- **Python:** 3.11 (recommended) or 3.9-3.12
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 10GB free space (for packages and audio files)
- **GPU:** Optional but recommended (NVIDIA with CUDA 12.1)
- **OS:** Windows, Linux, or macOS

## Why Each Package is Needed

### Core Processing
- **numpy:** All audio is processed as numpy arrays
- **scipy:** Bandpass filtering for frequency band analysis
- **matplotlib:** Generate all comparison plots
- **soundfile:** Read/write WAV files
- **librosa:** Compute mel spectrograms, resample audio

### Deep Learning
- **torch/torchvision/torchaudio:** Required by both DAC and EnCodec
- CUDA support enables 10-100x faster processing

### Codecs
- **descript-audio-codec:** The DAC codec you want to compare
- **encodec:** Facebook's EnCodec for comparison
- **audiotools:** Utilities used by DAC

### Optional
- **pyyaml:** Read the config file (can skip if not using config)

## Troubleshooting

### "Librosa not loading"
```bash
# Librosa needs additional dependencies
pip install numba resampy
```

### "SoundFile can't find library"
```bash
# On Ubuntu/Debian
sudo apt-get install libsndfile1

# On macOS
brew install libsndfile

# On Windows (usually auto-installs)
pip install soundfile --force-reinstall
```

### "PyTorch CUDA version mismatch"
Make sure you install the CUDA-matching version:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio
```

### "Out of memory" errors
- Use CPU instead of GPU for small files
- Process shorter audio clips
- Process one codec at a time

## Next Steps

Once all dependencies are installed:
1. ✓ Run `python check_dependencies.py` to verify
2. ✓ Place your music files in `data/test_audio/`
3. ✓ Run your first comparison!

```bash
python evaluation/evaluate_codecs.py \
  --input data/test_audio/song.wav \
  --codecs dac encodec \
  --output-dir outputs/test
```
