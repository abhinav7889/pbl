# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Option A: Using your existing DAC environment (recommended)
conda activate dac
pip install matplotlib soundfile librosa encodec audiotools pyyaml

# Option B: Fresh install
pip install -r requirements.txt

# Verify installation
python check_dependencies.py
```

### Step 2: Prepare Your Music Files
```bash
# Place your music files in data/test_audio/
# Supported formats: WAV, FLAC, MP3, etc.
```

### Step 3: Run Comparison
```bash
python evaluation/evaluate_codecs.py \
  --input data/test_audio/your_song.wav \
  --codecs dac encodec \
  --output-dir outputs/my_comparison
```

## 📊 View Results

After running, check:
```
outputs/my_comparison/
├── figures/
│   ├── your_song_waveforms.png      ← Waveform comparison
│   ├── your_song_errors.png         ← Reconstruction errors
│   ├── your_song_freq_bands.png     ← Frequency band analysis ⭐
│   └── your_song_metrics.png        ← Metric comparison
├── your_song_dac.wav                ← DAC reconstruction
├── your_song_encodec.wav            ← EnCodec reconstruction
└── your_song_metrics.json           ← All metrics
```

## 🎯 Best Graphs for Your Presentation

For comparing DAC vs EnCodec/SoundStream on **real music**:

1. **Frequency Band Error** (`freq_bands.png`) - Shows which codec performs better in bass/mid/treble
2. **Reconstruction Error** (`errors.png`) - Visual proof of artifacts
3. **Metric Comparison** (`metrics.png`) - Objective performance summary

## ⚡ Quick Commands

**Compare at different bitrates:**
```bash
# DAC at 24kbps
python evaluation/evaluate_codecs.py \
  --input song.wav --codecs dac \
  --dac-model 24khz

# EnCodec at 12kbps
python evaluation/evaluate_codecs.py \
  --input song.wav --codecs encodec \
  --encodec-bandwidth 12.0
```

**Process multiple songs:**
```bash
for song in data/test_audio/*.wav; do
  python evaluation/evaluate_codecs.py \
    --input "$song" --codecs dac encodec \
    --output-dir outputs/batch_results
done
```

## 📖 Documentation

- **DEPENDENCIES.md** - Complete list of required packages
- **INSTALL.md** - Detailed installation guide
- **README.md** - Full usage documentation
- **presentation.md** - Your original presentation document

## 🔧 Common Issues

**Missing packages?**
```bash
python check_dependencies.py
pip install -r requirements.txt
```

**CUDA out of memory?**
- Use shorter audio clips (< 30 seconds)
- Process one codec at a time

**Import errors?**
```bash
# Make sure you're in the project directory
cd C:\Users\abh80\Downloads\pbl\pbl_project

# Activate your environment
conda activate dac
```

## 💡 Pro Tips

1. **Test on 10-15 second clips first** - Faster iteration
2. **Focus on frequency band graphs** - Most informative for music
3. **Compare at matched bitrates** - Fair comparison (DAC 24kbps vs EnCodec 24kbps)
4. **Listen to reconstructions** - Objective metrics don't tell the whole story

## 🎵 What Next?

1. Run comparison on your music files
2. Generate the 3 key graphs for your presentation
3. Save metrics to use in your tables
4. Listen to reconstructions to verify quality

---

**Need help?** Check the detailed docs:
- Installation issues → `INSTALL.md`
- Package details → `DEPENDENCIES.md`
- Full API reference → `README.md`
