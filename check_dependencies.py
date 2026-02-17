#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency checker for Music Codec Comparison Tool
Verifies all required packages are installed correctly.
"""

import sys
import importlib.util


def check_package(package_name, import_name=None):
    """
    Check if a package is installed.
    
    Args:
        package_name: Display name of the package
        import_name: Actual module name to import (if different)
    """
    if import_name is None:
        import_name = package_name.lower().replace('-', '_')
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"[OK] {package_name:30s}")
            return True
        else:
            print(f"[--] {package_name:30s} - NOT FOUND")
            return False
    except Exception as e:
        print(f"[!!] {package_name:30s} - ERROR: {str(e)}")
        return False
    except ModuleNotFoundError:
        print(f"✗ {package_name:30s} - NOT FOUND")
        return False
    except Exception as e:
        print(f"✗ {package_name:30s} - ERROR: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("Checking dependencies for Music Codec Comparison Tool")
    print("=" * 60)
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    core_packages = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Matplotlib", "matplotlib"),
        ("SoundFile", "soundfile"),
        ("Librosa", "librosa"),
    ]
    
    core_ok = all(check_package(name, import_name) for name, import_name in core_packages)
    print()
    
    # Deep learning
    print("Deep Learning:")
    print("-" * 60)
    dl_packages = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("TorchAudio", "torchaudio"),
    ]
    
    dl_ok = all(check_package(name, import_name) for name, import_name in dl_packages)
    print()
    
    # Check CUDA availability
    if dl_ok:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                print(f"[OK] CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"[!!] CUDA not available - will use CPU (slower)")
        except Exception:
            pass
        print()
    
    # Audio codecs
    print("Audio Codecs:")
    print("-" * 60)
    codec_packages = [
        ("DAC (Descript Audio Codec)", "dac"),
        ("EnCodec", "encodec"),
    ]
    
    codec_ok = all(check_package(name, import_name) for name, import_name in codec_packages)
    print()
    
    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 60)
    optional_packages = [
        ("PyYAML", "yaml"),
        ("AudioTools", "audiotools"),
    ]
    
    optional_ok = all(check_package(name, import_name) for name, import_name in optional_packages)
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_ok = core_ok and dl_ok and codec_ok
    
    if all_ok:
        print("[OK] All required dependencies are installed!")
        print()
        print("You can now run:")
        print("  python evaluation/evaluate_codecs.py --input your_song.wav --codecs dac encodec")
        return 0
    else:
        print("[!!] Some dependencies are missing.")
        print()
        print("To install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        print()
        print("Or install manually:")
        
        if not core_ok:
            print("  pip install numpy scipy matplotlib soundfile librosa")
        if not dl_ok:
            print("  pip install torch torchvision torchaudio")
        if not codec_ok:
            print("  pip install descript-audio-codec encodec")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
