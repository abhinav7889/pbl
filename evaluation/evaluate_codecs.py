"""
Evaluation script for comparing neural audio codecs on music files.

Supports: DAC (Descript Audio Codec), EnCodec, SoundStream

Usage:
    python evaluate_codecs.py --input song.wav --codecs dac encodec soundstream --output-dir results/
"""

import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch


def load_audio(file_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if necessary.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        return audio, target_sr
    except ImportError:
        # Fallback to soundfile
        audio, sr = sf.read(file_path)
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != target_sr:
            print(f"Warning: Audio SR is {sr}Hz, expected {target_sr}Hz. Install librosa for automatic resampling.")
        
        return audio, sr


def encode_decode_dac(
    audio: np.ndarray,
    sr: int = 48000,
    model_type: str = "44khz",
    bitrate: Optional[int] = None
) -> np.ndarray:
    """
    Encode and decode audio using DAC (Descript Audio Codec).
    
    Args:
        audio: Input audio waveform
        sr: Sample rate of input audio
        model_type: DAC model type ("16khz", "24khz", "44khz")
        bitrate: Target bitrate in kbps (optional, uses model default if None)
        
    Returns:
        Reconstructed audio waveform (at original sample rate)
    """
    try:
        import dac
        import librosa
        
        # Map model type to expected sample rate
        model_sample_rates = {
            "16khz": 16000,
            "24khz": 24000,
            "44khz": 44100
        }
        
        target_sr = model_sample_rates.get(model_type, 44100)
        
        # Resample to target sample rate if needed
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio
        
        # Load DAC model
        model_path = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(model_path)
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Prepare audio
        audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            z, codes, latents, _, _ = model.encode(audio_tensor)
            
            # Optionally reduce bitrate by using fewer codebooks
            if bitrate:
                # Approximate number of codebooks needed for target bitrate
                # DAC uses ~1 kbps per codebook at 44kHz
                n_codebooks = min(bitrate, codes.shape[1])
                codes = codes[:, :n_codebooks, :]
            
            # Decode
            reconstruction = model.decode(z)
        
        # Convert back to numpy
        output = reconstruction.squeeze().cpu().numpy()
        
        # Resample back to original sample rate if needed
        if sr != target_sr:
            output = librosa.resample(output, orig_sr=target_sr, target_sr=sr)
        
        return output
        
    except ImportError:
        raise ImportError(
            "DAC requires descript-audio-codec and librosa. Install with: pip install descript-audio-codec librosa"
        )
    except Exception as e:
        raise RuntimeError(f"Error during DAC encoding/decoding: {str(e)}")


def encode_decode_encodec(
    audio: np.ndarray,
    sr: int = 48000,
    bandwidth: float = 24.0
) -> np.ndarray:
    """
    Encode and decode audio using EnCodec.
    
    Args:
        audio: Input audio waveform
        sr: Sample rate
        bandwidth: Target bandwidth in kbps (1.5, 3, 6, 12, 24)
        
    Returns:
        Reconstructed audio waveform
    """
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
        
        # Load model
        if sr == 48000:
            model = EncodecModel.encodec_model_48khz()
        elif sr == 24000:
            model = EncodecModel.encodec_model_24khz()
        else:
            print(f"Warning: EnCodec optimized for 24kHz or 48kHz, got {sr}Hz")
            model = EncodecModel.encodec_model_48khz()
        
        model.set_target_bandwidth(bandwidth)
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Prepare audio - EnCodec expects stereo (2 channels)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
        
        # Convert mono to stereo by duplicating the channel
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)  # [2, samples]
        
        audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 2, samples]
        
        # Encode and decode
        with torch.no_grad():
            encoded_frames = model.encode(audio_tensor)
            reconstruction = model.decode(encoded_frames)
        
        # Convert back to numpy and back to mono
        output = reconstruction.squeeze(0).cpu().numpy()  # [2, samples]
        
        # Average the two channels back to mono for consistency
        if output.ndim > 1 and output.shape[0] == 2:
            output = np.mean(output, axis=0)  # [samples]
        
        return output
        
    except ImportError:
        raise ImportError(
            "EnCodec is not installed. Install with: pip install encodec"
        )
    except Exception as e:
        raise RuntimeError(f"Error during EnCodec encoding/decoding: {str(e)}")


def encode_decode_soundstream(
    audio: np.ndarray,
    sr: int = 48000,
    target_bandwidth: float = 6.0
) -> np.ndarray:
    """
    Encode and decode audio using SoundStream.
    
    Args:
        audio: Input audio waveform
        sr: Sample rate (will be resampled to 16kHz for SoundStream)
        target_bandwidth: Target bandwidth in kbps (3, 6, 12, 18)
        
    Returns:
        Reconstructed audio waveform (resampled back to original sr)
    """
    try:
        from soundstream import from_pretrained
        import librosa
        
        # SoundStream expects 16kHz audio
        target_sr = 16000
        
        # Resample to 16kHz if needed
        if sr != target_sr:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_16k = audio
        
        # Load pre-trained SoundStream model
        audio_codec = from_pretrained()
        
        # Prepare audio - SoundStream expects shape [batch, channels, samples]
        audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).unsqueeze(0)
        
        # Encode and decode
        with torch.no_grad():
            # Encode to discrete codes
            quantized = audio_codec(audio_tensor, mode='encode')
            
            # Decode back to audio
            reconstruction = audio_codec(quantized, mode='decode')
        
        # Convert back to numpy
        output = reconstruction.squeeze().cpu().numpy()
        
        # Resample back to original sample rate
        if sr != target_sr:
            output = librosa.resample(output, orig_sr=target_sr, target_sr=sr)
        
        return output
        
    except ImportError as e:
        raise ImportError(
            "SoundStream requires soundstream and librosa. Install with: pip install soundstream librosa"
        )
    except Exception as e:
        raise RuntimeError(f"Error during SoundStream encoding/decoding: {str(e)}")


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Reference signal
        estimate: Estimated signal
        
    Returns:
        SI-SDR value in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Normalize
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Compute SI-SDR
    reference_energy = np.sum(reference ** 2) + 1e-8
    optimal_scaling = np.sum(reference * estimate) / reference_energy
    projection = optimal_scaling * reference
    
    noise = estimate - projection
    
    si_sdr = 10 * np.log10(
        (np.sum(projection ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    )
    
    return si_sdr


def compute_mel_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 48000,
    n_mels: int = 128
) -> float:
    """
    Compute Mel spectrogram distance.
    
    Args:
        reference: Reference signal
        estimate: Estimated signal
        sr: Sample rate
        n_mels: Number of mel bands
        
    Returns:
        Mean absolute error between mel spectrograms
    """
    try:
        import librosa
        
        # Ensure same length
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]
        
        # Compute mel spectrograms
        mel_ref = librosa.feature.melspectrogram(y=reference, sr=sr, n_mels=n_mels)
        mel_est = librosa.feature.melspectrogram(y=estimate, sr=sr, n_mels=n_mels)
        
        # Convert to dB
        mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
        mel_est_db = librosa.power_to_db(mel_est, ref=np.max)
        
        # Compute distance
        distance = np.mean(np.abs(mel_ref_db - mel_est_db))
        
        return distance
        
    except ImportError:
        print("Warning: librosa not available, skipping mel distance computation")
        return -1.0


def evaluate_codec(
    codec_name: str,
    original: np.ndarray,
    sr: int = 48000,
    **codec_kwargs
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate a codec on an audio file.
    
    Args:
        codec_name: Name of codec ('dac', 'encodec', 'soundstream')
        original: Original audio waveform
        sr: Sample rate
        **codec_kwargs: Additional arguments for specific codecs
        
    Returns:
        Tuple of (reconstructed_audio, metrics_dict)
    """
    print(f"\nEvaluating {codec_name.upper()}...")
    
    # Encode and decode
    if codec_name.lower() == 'dac':
        reconstruction = encode_decode_dac(original, sr, **codec_kwargs)
    elif codec_name.lower() == 'encodec':
        reconstruction = encode_decode_encodec(original, sr, **codec_kwargs)
    elif codec_name.lower() == 'soundstream':
        reconstruction = encode_decode_soundstream(original, sr, **codec_kwargs)
    else:
        raise ValueError(f"Unsupported codec: {codec_name}")
    
    # Compute metrics
    print("Computing metrics...")
    
    metrics = {}
    
    # SI-SDR
    metrics['SI-SDR (dB)'] = compute_si_sdr(original, reconstruction)
    print(f"  SI-SDR: {metrics['SI-SDR (dB)']:.2f} dB")
    
    # Mel Distance
    mel_dist = compute_mel_distance(original, reconstruction, sr)
    if mel_dist >= 0:
        metrics['Mel Distance'] = mel_dist
        print(f"  Mel Distance: {metrics['Mel Distance']:.4f}")
    
    # MSE
    min_len = min(len(original), len(reconstruction))
    mse = np.mean((original[:min_len] - reconstruction[:min_len]) ** 2)
    metrics['MSE'] = mse
    print(f"  MSE: {metrics['MSE']:.6f}")
    
    # MAE  
    mae = np.mean(np.abs(original[:min_len] - reconstruction[:min_len]))
    metrics['MAE'] = mae
    print(f"  MAE: {metrics['MAE']:.6f}")
    
    # Convert all numpy types to native Python types for JSON serialization
    metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in metrics.items()
    }
    
    return reconstruction, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate neural audio codecs')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--codecs', type=str, nargs='+', default=['dac', 'encodec'],
                       help='Codecs to evaluate (dac, encodec, soundstream)')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluations',
                       help='Output directory for results')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate')
    parser.add_argument('--dac-model', type=str, default='44khz',
                       choices=['16khz', '24khz', '44khz'], help='DAC model type')
    parser.add_argument('--encodec-bandwidth', type=float, default=24.0,
                       choices=[1.5, 3.0, 6.0, 12.0, 24.0], help='EnCodec bandwidth (kbps)')
    parser.add_argument('--soundstream-bandwidth', type=float, default=6.0,
                       choices=[3.0, 6.0, 12.0, 18.0], help='SoundStream bandwidth (kbps)')
    parser.add_argument('--include-chimera', action='store_true',
                       help='Include simulated Chimera codec (24kbps and 30kbps variants)')
    parser.add_argument('--artist', type=str, default=None,
                       help='Artist name for visualization titles')
    parser.add_argument('--title', type=str, default=None,
                       help='Song title for visualization titles')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    print(f"Loading audio from {args.input}...")
    original, sr = load_audio(args.input, target_sr=args.sr)
    print(f"Loaded {len(original)/sr:.2f} seconds of audio at {sr}Hz")
    
    # Save original
    input_name = Path(args.input).stem
    sf.write(output_dir / f"{input_name}_original.wav", original, sr)
    
    # Results storage
    all_results = {}
    all_reconstructions = {'Original': original}
    dac_reconstruction = None  # Store DAC reconstruction for Chimera simulation
    
    # Evaluate each codec
    for codec in args.codecs:
        codec_kwargs = {}
        
        if codec.lower() == 'dac':
            codec_kwargs['model_type'] = args.dac_model
        elif codec.lower() == 'encodec':
            codec_kwargs['bandwidth'] = args.encodec_bandwidth
        elif codec.lower() == 'soundstream':
            codec_kwargs['target_bandwidth'] = args.soundstream_bandwidth
        
        try:
            reconstruction, metrics = evaluate_codec(codec, original, sr, **codec_kwargs)
            
            # Save DAC reconstruction for Chimera simulation
            if codec.lower() == 'dac':
                dac_reconstruction = reconstruction.copy()
            
            # Save reconstruction
            output_file = output_dir / f"{input_name}_{codec}.wav"
            sf.write(output_file, reconstruction, sr)
            print(f"Saved reconstruction to {output_file}")
            
            # Store results
            all_results[codec] = metrics
            all_reconstructions[codec] = reconstruction
            
        except Exception as e:
            print(f"Error evaluating {codec}: {str(e)}")
            continue
    
    # Load Chimera files from output directory if requested
    if args.include_chimera:
        print("\n" + "="*60)
        print("Loading Project Chimera files")
        print("="*60)
        
        chimera_bitrates = ['24kbps', '30kbps']
        
        for bitrate in chimera_bitrates:
            chimera_name = f"chimera_{bitrate}"
            chimera_path = output_dir / f"{input_name}_{chimera_name}.wav"
            
            if chimera_path.exists():
                print(f"\nLoading Chimera {bitrate} from {chimera_path}")
                chimera_audio, chimera_sr = sf.read(chimera_path)
                
                # Convert stereo to mono if needed
                if chimera_audio.ndim > 1:
                    chimera_audio = np.mean(chimera_audio, axis=1) if chimera_audio.shape[1] == 2 else chimera_audio.mean(axis=1)
                
                # Compute metrics against original
                chimera_metrics = {}
                chimera_metrics['SI-SDR (dB)'] = compute_si_sdr(original, chimera_audio)
                print(f"  SI-SDR: {chimera_metrics['SI-SDR (dB)']:.2f} dB")
                
                mel_dist = compute_mel_distance(original, chimera_audio, sr)
                if mel_dist >= 0:
                    chimera_metrics['Mel Distance'] = mel_dist
                    print(f"  Mel Distance: {chimera_metrics['Mel Distance']:.4f}")
                
                min_len = min(len(original), len(chimera_audio))
                chimera_metrics['MSE'] = np.mean((original[:min_len] - chimera_audio[:min_len]) ** 2)
                print(f"  MSE: {chimera_metrics['MSE']:.6f}")
                
                chimera_metrics['MAE'] = np.mean(np.abs(original[:min_len] - chimera_audio[:min_len]))
                print(f"  MAE: {chimera_metrics['MAE']:.6f}")
                
                # Convert to native Python types
                chimera_metrics = {
                    key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                    for key, value in chimera_metrics.items()
                }
                
                # Store results
                all_results[chimera_name] = chimera_metrics
                all_reconstructions[chimera_name] = chimera_audio
            else:
                print(f"\nWarning: Chimera {bitrate} file not found at {chimera_path}")
                print(f"  Expected: {chimera_path}")
                print(f"  Run codec encoding first to generate Chimera files")
    
    # Save metrics to JSON
    metrics_file = output_dir / f"{input_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        # Add parent directory to path to find visualization module
        import sys
        from pathlib import Path as PathLib
        project_root = PathLib(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from visualization.music_comparison import (
            plot_comprehensive_analysis,
            plot_frequency_band_error
        )
        from visualization.naive_user_summary import plot_naive_user_summary
        
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Waveform comparison (zoom to 2-second window)
        duration = len(original) / sr
        mid_point = duration / 2
        time_range = (max(0, mid_point - 1), min(duration, mid_point + 1))
        
        reconstructions_only = {k: v for k, v in all_reconstructions.items() if k != 'Original'}
        
        # === COMPREHENSIVE MULTI-PANEL VISUALIZATION ===
        print("  Generating comprehensive analysis figure...")
        print("    (includes: spectrograms, errors, harmonics, freq bands, metrics, bitrate curves)")
        
        # Generate title based on artist/title inputs
        if args.artist and args.title:
            viz_title = f"Codec Analysis: {args.artist} - {args.title}"
        elif args.title:
            viz_title = f"Codec Analysis: {args.title}"
        else:
            viz_title = None  # Will use default or auto-generated title
        
        plot_comprehensive_analysis(
            original,
            reconstructions_only,
            all_results,
            sr=sr,
            time_range=time_range,
            include_theoretical=False,  # Real Chimera files used, no theoretical curves
            output_path=str(figures_dir / f"{input_name}_comprehensive_analysis.png"),
            title=viz_title,
            artist=args.artist,
            song_title=args.title
        )
        
        # === OPTIONAL: FREQUENCY BAND ERROR (SEPARATE) ===
        print("  Generating frequency band error plot...")
        plot_frequency_band_error(
            original,
            reconstructions_only,
            sr=sr,
            output_path=str(figures_dir / f"{input_name}_freq_bands.png")
        )
        
        # === NAIVE USER-FRIENDLY SUMMARY ===
        print("  Generating user-friendly summary...")
        plot_naive_user_summary(
            all_results,
            output_path=str(figures_dir / f"{input_name}_summary.png"),
            audio_name=input_name
        )
        
        print(f"\n✓ Saved all visualizations to {figures_dir}")
        print(f"  📊 Technical analysis: {input_name}_comprehensive_analysis.png")
        print(f"  👤 User-friendly summary: {input_name}_summary.png")
        print(f"  📈 Frequency bands: {input_name}_freq_bands.png")
        
    except Exception as e:
        import traceback
        print(f"Error generating visualizations: {str(e)}")
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
