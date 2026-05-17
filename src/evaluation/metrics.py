"""
Objective Speech Quality Metrics
==================================
Implements:

1. SNR  – Signal-to-Noise Ratio (dB)
2. SSNR – Segmental SNR           (dB)  (better correlates with perceptual quality)
3. PESQ – Perceptual Evaluation of Speech Quality (ITU-T P.862)
4. STOI – Short-Time Objective Intelligibility
5. LSD  – Log-Spectral Distortion

All functions accept 1-D float32 numpy arrays at the same sample rate.
"""

import numpy as np
import scipy.signal as ssig
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_TYPE


# ─────────────────────────────────────────────────────────────────────────────
# 1. SNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Global SNR (dB).
    SNR = 10 · log10 ( ||s||² / ||s − ŝ||² )
    """
    noise = clean - enhanced
    power_signal = np.mean(clean ** 2)
    power_noise  = np.mean(noise ** 2)
    if power_noise < 1e-12:
        return float("inf")
    return 10 * np.log10(power_signal / power_noise)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Segmental SNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_segmental_snr(clean: np.ndarray, enhanced: np.ndarray,
                           frame_len: int = 256, hop: int = 128,
                           min_snr: float = -10.0,
                           max_snr: float = 35.0) -> float:
    """
    Segmental SNR: average SNR over short frames.
    Values are clipped to [min_snr, max_snr] to reduce frame-level outliers.
    """
    N     = min(len(clean), len(enhanced))
    clean = clean[:N]; enhanced = enhanced[:N]
    snrs  = []
    for start in range(0, N - frame_len, hop):
        s = clean   [start: start + frame_len]
        e = enhanced[start: start + frame_len]
        n = s - e
        ps = np.mean(s ** 2)
        pn = np.mean(n ** 2)
        if ps < 1e-12 or pn < 1e-12:
            continue
        snrs.append(np.clip(10 * np.log10(ps / pn), min_snr, max_snr))
    return float(np.mean(snrs)) if snrs else -999.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. PESQ  (requires `pesq` package)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pesq(clean: np.ndarray, enhanced: np.ndarray,
                 sr: int = SAMPLE_RATE) -> float:
    """
    PESQ score (-0.5 … 4.5).  Requires `pip install pesq`.
    Returns NaN on failure.
    """
    try:
        from pesq import pesq
        mode = "wb" if sr == 16000 else "nb"
        return float(pesq(sr, clean, enhanced, mode))
    except ImportError:
        print("  [PESQ] Install with:  pip install pesq")
        return float("nan")
    except Exception as ex:
        print(f"  [PESQ] Error: {ex}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STOI  (requires `pystoi` package)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stoi(clean: np.ndarray, enhanced: np.ndarray,
                 sr: int = SAMPLE_RATE) -> float:
    """
    STOI score (0 … 1).  Requires `pip install pystoi`.
    """
    try:
        from pystoi import stoi
        return float(stoi(clean, enhanced, sr, extended=False))
    except ImportError:
        print("  [STOI] Install with:  pip install pystoi")
        return float("nan")
    except Exception as ex:
        print(f"  [STOI] Error: {ex}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Log-Spectral Distortion
# ─────────────────────────────────────────────────────────────────────────────

def compute_lsd(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Log-Spectral Distortion (dB).
    LSD = (1/T) Σ_t √[ (1/K) Σ_k (log P_clean(t,k) − log P_enh(t,k))² ]
    """
    N     = min(len(clean), len(enhanced))
    clean = clean[:N]; enhanced = enhanced[:N]

    _, _, Sc = ssig.stft(clean,    nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH,
                         window=WIN_TYPE)
    _, _, Se = ssig.stft(enhanced, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH,
                         window=WIN_TYPE)

    log_pc = np.log(np.abs(Sc) ** 2 + 1e-12)
    log_pe = np.log(np.abs(Se) ** 2 + 1e-12)
    frame_lsd = np.sqrt(np.mean((log_pc - log_pe) ** 2, axis=0))
    return float(np.mean(frame_lsd))


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(clean: np.ndarray, enhanced: np.ndarray,
                 sr: int = SAMPLE_RATE) -> dict:
    """
    Compute all metrics and return as a dict.
    """
    N     = min(len(clean), len(enhanced))
    clean = clean[:N].astype(np.float32)
    enhanced = enhanced[:N].astype(np.float32)

    return {
        "SNR_dB":         compute_snr(clean, enhanced),
        "SSNR_dB":        compute_segmental_snr(clean, enhanced),
        "PESQ":           compute_pesq(clean, enhanced, sr),
        "STOI":           compute_stoi(clean, enhanced, sr),
        "LSD_dB":         compute_lsd(clean, enhanced),
    }


def print_metrics(metrics: dict, label: str = ""):
    print(f"\n── Metrics {label} ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("─" * 40)
