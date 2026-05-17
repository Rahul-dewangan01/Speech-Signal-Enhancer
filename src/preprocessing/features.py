"""
Preprocessing & Feature Extraction
=====================================
Functions for:
  • STFT / ISTFT
  • Mel spectrogram
  • Log-Mel features
  • MFCC
  • Signal normalization
  • Frame-based energy / ZCR features
"""

import numpy as np
import scipy.signal as ssig
from typing import Optional, Tuple

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_TYPE, N_MELS


# ─────────────────────────────────────────────────────────────────────────────
# STFT / ISTFT
# ─────────────────────────────────────────────────────────────────────────────

def stft(x: np.ndarray, n_fft: int = N_FFT,
         hop: int = HOP_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (magnitude, phase) from STFT.
    magnitude: (F, T) float32
    phase    : (F, T) float32
    """
    _, _, S = ssig.stft(x.astype(np.float32), nperseg=n_fft,
                        noverlap=n_fft - hop, window=WIN_TYPE)
    return np.abs(S).astype(np.float32), np.angle(S).astype(np.float32)


def istft(magnitude: np.ndarray, phase: np.ndarray,
          n_fft: int = N_FFT, hop: int = HOP_LENGTH,
          length: Optional[int] = None) -> np.ndarray:
    """Reconstruct time-domain signal from magnitude + phase."""
    S = magnitude * np.exp(1j * phase)
    _, x = ssig.istft(S, nperseg=n_fft, noverlap=n_fft - hop, window=WIN_TYPE)
    if length is not None:
        x = x[:length]
    return x.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Mel filterbank
# ─────────────────────────────────────────────────────────────────────────────

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT,
                   sr: int = SAMPLE_RATE,
                   fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """Returns (n_mels, n_fft//2+1) mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    F = n_fft // 2 + 1
    filters = np.zeros((n_mels, F))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filters[m-1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filters[m-1, k] = (f_right - k) / (f_right - f_center)
    return filters.astype(np.float32)


def log_mel_spectrogram(x: np.ndarray, n_mels: int = N_MELS,
                         n_fft: int = N_FFT, hop: int = HOP_LENGTH,
                         sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (n_mels, T) log-Mel spectrogram."""
    mag, _ = stft(x, n_fft, hop)           # (F, T)
    fb     = mel_filterbank(n_mels, n_fft, sr)  # (n_mels, F)
    mel    = fb @ mag                        # (n_mels, T)
    return np.log1p(mel).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MFCC (from scratch — no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def mfcc(x: np.ndarray, n_mfcc: int = 13,
         n_mels: int = N_MELS, n_fft: int = N_FFT,
         hop: int = HOP_LENGTH, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Returns (n_mfcc, T) MFCC matrix.
    Steps: log-Mel → DCT type-II
    """
    log_mel = log_mel_spectrogram(x, n_mels, n_fft, hop, sr)  # (n_mels, T)
    # DCT-II: MFCC_m(t) = Σ_k  log_mel_k(t) · cos[π·m/M · (k + 0.5)]
    M, T = log_mel.shape
    m = np.arange(n_mfcc)[:, None]   # (n_mfcc, 1)
    k = np.arange(M)[None, :]        # (1, n_mels)
    dct = np.cos(np.pi * m * (k + 0.5) / M)   # (n_mfcc, n_mels)
    return (dct @ log_mel).astype(np.float32)  # (n_mfcc, T)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise(x: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalise a signal.
    method: "peak"  → divide by max abs
            "rms"   → divide by RMS
            "zscore"→ zero-mean, unit-variance
    """
    if method == "peak":
        return x / (np.max(np.abs(x)) + 1e-9)
    elif method == "rms":
        return x / (np.sqrt(np.mean(x**2)) + 1e-9)
    elif method == "zscore":
        return (x - np.mean(x)) / (np.std(x) + 1e-9)
    raise ValueError(f"Unknown normalisation method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level energy & ZCR (used for VAD)
# ─────────────────────────────────────────────────────────────────────────────

def frame_energy(x: np.ndarray, frame_len: int = 256,
                 hop: int = 128) -> np.ndarray:
    """Short-time energy per frame."""
    n_frames = (len(x) - frame_len) // hop + 1
    energy   = np.array([
        np.sum(x[i*hop: i*hop+frame_len] ** 2)
        for i in range(n_frames)
    ])
    return energy.astype(np.float32)


def zero_crossing_rate(x: np.ndarray, frame_len: int = 256,
                        hop: int = 128) -> np.ndarray:
    """Zero-crossing rate per frame."""
    n_frames = (len(x) - frame_len) // hop + 1
    zcr = np.array([
        np.sum(np.abs(np.diff(np.sign(x[i*hop: i*hop+frame_len])))) / (2 * frame_len)
        for i in range(n_frames)
    ])
    return zcr.astype(np.float32)
