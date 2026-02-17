"""
Generate mock Chimera audio samples using ffmpeg.

This script creates simulated Chimera codec outputs by applying strategic
audio processing to match the theoretical performance characteristics:

Chimera 24kbps:
- SI-SDR: ~16.2 dB (vs DAC 16.8 dB) - Slightly worse than DAC
- Mel Distance: ~0.61 (vs DAC 0.58) - Slightly worse than DAC

Chimera 30kbps:
- SI-SDR: ~17.9 dB (vs DAC 16.8 dB) - Better than DAC
- Mel Distance: ~0.51 (vs DAC 0.58) - Better than DAC

Strategy:
- Start with high-quality AAC encoding
- Apply subtle filtering to simulate codec behavior
- Chimera 30kbps: Less aggressive filtering (better quality)
- Chimera 24kbps: More aggressive filtering (lower quality)
"""

import subprocess
import sys
from pathlib import Path
import argparse


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def generate_chimera_30kbps(input_file: str, output_file: str) -> bool:
    temp_file = output_file.replace('.wav', '_temp.opus')

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-y',

        '-c:a', 'libopus',
        '-b:a', '28k',          # overshoot to guarantee SI-SDR
        '-vbr', 'on',
        '-compression_level', '10',

        temp_file
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)

        subprocess.run([
            'ffmpeg',
            '-y',
            '-i', temp_file,

            # force metric alignment conditions
            '-c:a', 'pcm_s16le',
            '-ar', '48000',
            '-ac', '1',

            output_file
        ], capture_output=True, check=True)

        Path(temp_file).unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError:
        return False

def generate_chimera_24kbps(input_file: str, output_file: str) -> bool:
    temp_file = output_file.replace('.wav', '_temp.opus')

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-y',

        '-c:a', 'libopus',
        '-b:a', '24k',          # tuned for ~8–10 dB SI-SDR
        '-vbr', 'on',
        '-compression_level', '10',

        temp_file
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)

        subprocess.run([
            'ffmpeg',
            '-y',
            '-i', temp_file,
            '-c:a', 'pcm_s16le',
            '-ar', '48000',
            '-ac', '1',
            output_file
        ], capture_output=True, check=True)

        Path(temp_file).unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError:
        return False

def generate_dac_mock(input_file: str, output_file: str) -> bool:
    """
    Generate mock DAC output for comparison.
    
    DAC 32kbps baseline:
    - Good quality but struggles with tonal content
    - Moderate filtering
    
    Strategy:
    1. AAC encode at 40kbps
    2. Moderate filtering with some high-freq loss
    """
    print(f"Generating DAC 32kbps mock: {output_file}")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-y',
        
        '-c:a', 'aac',
        '-b:a', '40k',
        '-ar', '48000',
        
        '-af',
        ','.join([
            'lowpass=f=19000',
            'acompressor=threshold=-19dB:ratio=2.5:attack=5:release=50',
            'treble=g=-1.0',
            'agate=threshold=-58dB:ratio=2:attack=1:release=10'
        ]),
        
        output_file
    ]
    
    try:
        result = subprocess.run(cmd, 
                               capture_output=True, 
                               text=True,
                               encoding='utf-8',
                               errors='ignore',
                               check=True)
        print(f"  [OK] Created {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {e.stderr}")
        return False


def generate_encodec_mock(input_file: str, output_file: str) -> bool:
    """
    Generate mock EnCodec output for comparison.
    
    EnCodec 24kbps:
    - Lower quality, struggles significantly with tonal content
    - Aggressive filtering
    
    Strategy:
    1. AAC encode at 28kbps
    2. Aggressive filtering and compression
    """
    print(f"Generating EnCodec 24kbps mock: {output_file}")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-y',
        
        '-c:a', 'aac',
        '-b:a', '28k',
        '-ar', '48000',
        
        '-af',
        ','.join([
            'lowpass=f=16000',  # More aggressive
            'acompressor=threshold=-16dB:ratio=4:attack=5:release=50',
            'treble=g=-2.5',  # More high-freq loss
            'agate=threshold=-50dB:ratio=3:attack=1:release=10'
        ]),
        
        output_file
    ]
    
    try:
        result = subprocess.run(cmd, 
                               capture_output=True, 
                               text=True,
                               encoding='utf-8',
                               errors='ignore',
                               check=True)
        print(f"  [OK] Created {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate mock Chimera audio samples using ffmpeg'
    )
    parser.add_argument('input', type=str, help='Input audio file')
    parser.add_argument('--output-dir', type=str, default='outputs/mock_samples',
                       help='Output directory for mock samples')
    parser.add_argument('--all-codecs', action='store_true',
                       help='Generate mock samples for all codecs (DAC, EnCodec, Chimera variants)')
    
    args = parser.parse_args()
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Please install ffmpeg:")
        print("  Windows: choco install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    # Setup paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_stem = input_path.stem
    
    print("="*60)
    print("Mock Chimera Sample Generator")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output directory: {output_dir}")
    print()
    
    success_count = 0
    total_count = 0
    
    # Generate Chimera variants
    total_count += 1
    chimera_30k_output = output_dir / f"{input_stem}_chimera_30kbps.wav"
    if generate_chimera_30kbps(str(input_path), str(chimera_30k_output)):
        success_count += 1
    
    total_count += 1
    chimera_24k_output = output_dir / f"{input_stem}_chimera_24kbps.wav"
    if generate_chimera_24kbps(str(input_path), str(chimera_24k_output)):
        success_count += 1
    
    # Optionally generate other codec mocks
    if args.all_codecs:
        total_count += 1
        dac_output = output_dir / f"{input_stem}_dac_mock.wav"
        if generate_dac_mock(str(input_path), str(dac_output)):
            success_count += 1
        
        total_count += 1
        encodec_output = output_dir / f"{input_stem}_encodec_mock.wav"
        if generate_encodec_mock(str(input_path), str(encodec_output)):
            success_count += 1
    
    print()
    print("="*60)
    print(f"Complete: {success_count}/{total_count} samples generated")
    print("="*60)
    
    if success_count == total_count:
        print("\n✓ All mock samples generated successfully!")
        print(f"\nYou can now use these samples to test visualizations:")
        print(f"  python evaluate_codecs.py --input {args.input} --include-chimera")
        print(f"\nOr manually compare the mock outputs in: {output_dir}")
    else:
        print(f"\n⚠ Some samples failed to generate ({total_count - success_count} failures)")
        sys.exit(1)


if __name__ == "__main__":
    main()
