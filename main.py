"""
Speech Signal Enhancement – Main Demo Script
=============================================
Demonstrates ALL enhancement methods on a synthetic test signal:

  Adaptive methods:
    1. LMS   – Least Mean Squares
    2. NLMS  – Normalised LMS
    3. RLS   – Recursive Least Squares
    4. Wiener Filter (frequency-domain, decision-directed)
    5. Spectral Subtraction

  Neural methods:
    6. BLSTM – Bidirectional LSTM mask estimator
    7. CRN   – Convolutional Recurrent Network

Usage
-----
    python main.py                    # demo with synthetic signal
    python main.py --input my.wav     # process your own file
    python main.py --compare          # plot comparison of all methods
    python main.py --snr 5            # choose input SNR level

Outputs are saved to  audio_samples/enhanced/  and  results/
"""

import os
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as ssig
import soundfile as sf
from pathlib import Path

from config import (SAMPLE_RATE, N_FFT, HOP_LENGTH, CLEAN_DIR, NOISY_DIR,
                    ENHANCED_DIR, RESULTS_DIR, configure_torch)
from src.adaptive  import (LMSFilter, NLMSFilter, RLSFilter,
                             WienerFilter, SpectralSubtraction)
from src.neural    import NeuralEnhancer
from src.evaluation import evaluate_all, print_metrics
from src.preprocessing import stft, normalise
from data.download_dataset import (generate_synthetic_speech,
                                    add_noise_at_snr, white_noise, pink_noise)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable GPU optimisations before any torch usage
configure_torch()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_wav(path: str, data: np.ndarray, sr: int = SAMPLE_RATE):
    data = np.clip(data, -1.0, 1.0)
    sf.write(path, data.astype(np.float32), sr)
    print(f"  Saved: {path}")


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal

    try:
        import librosa
        print(f"  Resampling {orig_sr} Hz -> {target_sr} Hz using librosa ...")
        return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        print(f"  Resampling {orig_sr} Hz -> {target_sr} Hz using scipy.signal.resample_poly ...")
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return ssig.resample_poly(signal, up, down).astype(np.float32)


def spectrogram_plot(ax, signal, title, sr=SAMPLE_RATE):
    mag, _ = stft(signal.astype(np.float32))
    ax.imshow(20 * np.log10(mag + 1e-9), origin="lower", aspect="auto",
              extent=[0, len(signal)/sr, 0, sr/2/1000],
              cmap="magma", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Freq (kHz)", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement pipeline
# ─────────────────────────────────────────────────────────────────────────────

def enhance_all(clean: np.ndarray, noisy: np.ndarray,
                reference_noise: np.ndarray) -> dict:
    """
    Run all enhancement methods and return dict of enhanced signals.
    """
    results = {}

    print("\n[1/7] LMS ...")
    results["LMS"]  = LMSFilter.denoise(noisy, reference_noise, order=32, mu=0.005)

    print("[2/7] NLMS ...")
    results["NLMS"] = NLMSFilter.denoise(noisy, reference_noise, order=32, mu=0.5)

    print("[3/7] RLS ...")
    results["RLS"]  = RLSFilter.denoise(noisy, reference_noise, order=32, lam=0.99)

    print("[4/7] Wiener Filter ...")
    wf              = WienerFilter(alpha=0.98, noise_frames=6)
    results["Wiener"] = wf.enhance(noisy)

    print("[5/7] Spectral Subtraction ...")
    ss              = SpectralSubtraction(alpha_over=2.0, beta=0.01)
    results["SpectralSub"] = ss.enhance(noisy)

    print("[6/7] BLSTM Neural ...")
    blstm           = NeuralEnhancer(model_type="blstm")
    results["BLSTM"] = blstm.enhance(noisy)

    print("[7/7] CRN Neural ...")
    crn             = NeuralEnhancer(model_type="crn")
    results["CRN"]  = crn.enhance(noisy)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(clean: np.ndarray, noisy: np.ndarray,
                    enhanced: dict, snr_in: float,
                    save_path: str = None):
    """
    Multi-panel figure: waveform + spectrogram for each method.
    """
    methods  = ["Noisy"] + list(enhanced.keys())
    signals  = [noisy]   + list(enhanced.values())
    methods  = ["Clean"] + methods
    signals  = [clean]   + signals

    fig = plt.figure(figsize=(20, 3 * len(methods)))
    gs  = gridspec.GridSpec(len(methods), 2, figure=fig,
                            wspace=0.3, hspace=0.8)
    t   = np.arange(len(clean)) / SAMPLE_RATE

    for i, (name, sig) in enumerate(zip(methods, signals)):
        N  = min(len(sig), len(clean))
        sig = sig[:N]

        # Waveform
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t[:N], sig, linewidth=0.4, color="steelblue")
        ax1.set_title(f"{name} – Waveform", fontsize=9)
        ax1.set_xlabel("Time (s)", fontsize=7); ax1.set_ylabel("Amp", fontsize=7)
        ax1.set_ylim(-1.1, 1.1)

        # Spectrogram
        ax2 = fig.add_subplot(gs[i, 1])
        spectrogram_plot(ax2, sig, f"{name} – Spectrogram")

    fig.suptitle(f"Speech Enhancement Comparison  (Input SNR = {snr_in} dB)",
                 fontsize=13, y=1.01)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table
# ─────────────────────────────────────────────────────────────────────────────

def metrics_table(clean: np.ndarray, enhanced_dict: dict, noisy: np.ndarray):
    import pandas as pd
    rows = []
    # Noisy baseline
    m = evaluate_all(clean, noisy)
    rows.append({"Method": "Noisy (input)", **m})
    # Each method
    for name, sig in enhanced_dict.items():
        N  = min(len(sig), len(clean))
        m  = evaluate_all(clean[:N], sig[:N])
        rows.append({"Method": name, **m})

    df = pd.DataFrame(rows).set_index("Method")
    print("\n" + "="*70)
    print("  OBJECTIVE METRICS COMPARISON")
    print("="*70)
    print(df.round(3).to_string())
    print("="*70)

    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df.to_csv(csv_path)
    print(f"  Saved to: {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Demo")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to noisy .wav file (if not provided, use synthetic)")
    parser.add_argument("--clean", type=str, default=None,
                        help="Path to corresponding clean .wav (for metrics)")
    parser.add_argument("--snr",   type=float, default=5.0,
                        help="Input SNR in dB for synthetic test")
    parser.add_argument("--noise", choices=["white", "pink", "babble"],
                        default="white", help="Noise type for synthetic test")
    parser.add_argument("--compare", action="store_true",
                        help="Show / save comparison plots")
    args = parser.parse_args()

    # ── Signal preparation ───────────────────────────────────────────────────
    if args.input:
        ext = Path(args.input).suffix.lower()
        wav_formats = ['.wav', '.flac', '.ogg', '.aiff', '.au']

        if ext not in wav_formats:
            # Auto-convert m4a / mp3 / mp4 / aac etc. using librosa + ffmpeg
            print(f"  Converting {ext} -> float32 PCM using librosa ...")
            try:
                import librosa
                noisy, sr = librosa.load(args.input, sr=None, mono=True)
                noisy = noisy.astype(np.float32)
            except Exception as e:
                print(f"\n  [ERROR] Could not read '{args.input}': {e}")
                print("  Fix options:")
                print("    1. Install ffmpeg:  winget install ffmpeg")
                print("    2. Convert manually: ffmpeg -i input.m4a output.wav")
                print("    3. Use a .wav file directly")
                sys.exit(1)
        else:
            noisy, sr = sf.read(args.input)
            if noisy.ndim > 1:
                noisy = noisy.mean(axis=1)
            noisy = noisy.astype(np.float32)

        # Resample to project sample rate if needed
        if sr != SAMPLE_RATE:
            noisy = resample_audio(noisy, sr, SAMPLE_RATE)

        if args.clean:
            clean_ext = Path(args.clean).suffix.lower()
            if clean_ext not in wav_formats:
                try:
                    import librosa
                    clean, csr = librosa.load(args.clean, sr=None, mono=True)
                    clean = clean.astype(np.float32)
                    if csr != SAMPLE_RATE:
                        clean = librosa.resample(clean, orig_sr=csr, target_sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"\n  [ERROR] Could not read '{args.clean}': {e}")
                    print("  Fix options:")
                    print("    1. Install ffmpeg:  winget install ffmpeg")
                    print("    2. Convert manually: ffmpeg -i input.m4a output.wav")
                    print("    3. Use a .wav file directly")
                    sys.exit(1)
            else:
                clean, csr = sf.read(args.clean)
                if clean.ndim > 1:
                    clean = clean.mean(axis=1)
                clean = clean.astype(np.float32)
                if csr != SAMPLE_RATE:
                    clean = resample_audio(clean, csr, SAMPLE_RATE)
        else:
            clean = None

        # Reference for adaptive filters: delayed version of noisy signal
        reference = np.roll(noisy, 32)
        reference[:32] = 0
    else:
        print(f"Generating synthetic test signal  (SNR={args.snr} dB, noise={args.noise}) ...")
        clean     = generate_synthetic_speech(duration=4.0)
        noise_fn  = {"white": white_noise, "pink": pink_noise,
                     "babble": lambda n: (np.random.randn(n)*0.5).astype(np.float32)}
        noise     = noise_fn[args.noise](len(clean)).astype(np.float32)
        noisy     = add_noise_at_snr(clean, noise, args.snr)
        reference = noise   # ideal reference (for demo purposes)

    print(f"\nSignal length: {len(noisy)/SAMPLE_RATE:.2f}s  |  SR: {SAMPLE_RATE} Hz")

    # Save inputs (only save clean if we have it)
    if clean is not None:
        save_wav(os.path.join(ENHANCED_DIR, "00_clean.wav"), clean)
    save_wav(os.path.join(ENHANCED_DIR, "00_noisy.wav"), noisy)

    # ── Run all methods ──────────────────────────────────────────────────────
    # When no clean reference available, pass noisy as dummy clean arg
    # (adaptive filters only use noisy + reference, not clean)
    enhanced = enhance_all(clean if clean is not None else noisy, noisy, reference)

    # Save enhanced signals
    for i, (name, sig) in enumerate(enhanced.items(), 1):
        save_wav(os.path.join(ENHANCED_DIR, f"{i:02d}_{name}.wav"), sig)

    # ── Metrics ──────────────────────────────────────────────────────────────
    if clean is not None:
        df = metrics_table(clean, enhanced, noisy)
    else:
        print("\n  [INFO] No clean reference provided -- skipping objective metrics.")
        print("     To get metrics, run:  python main.py --input noisy.wav --clean clean.wav")

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.compare:
        plot_comparison(
            clean if clean is not None else noisy,
            noisy, enhanced,
            snr_in=args.snr if clean is not None else 0,
            save_path=os.path.join(RESULTS_DIR, "comparison.png")
        )

    print("\n[DONE]  Enhanced files are in:", ENHANCED_DIR)


if __name__ == "__main__":
    main()
