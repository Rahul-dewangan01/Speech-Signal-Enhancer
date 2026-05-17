"""
Wiener Filter & Spectral Subtraction
======================================
Two classical frequency-domain enhancement methods.

──────────────────────────────────────────────
1. Wiener Filter (optimal linear estimator)
──────────────────────────────────────────────
In the STFT domain, the Wiener filter gain for each bin (k, l):

    H(k,l) = P_ss(k,l) / [P_ss(k,l) + P_nn(k,l)]
           = SNR_post(k,l) / [1 + SNR_post(k,l)]
           = 1 − 1/[1 + a·prior_SNR(k,l)]     (decision-directed)

where:
    P_ss = speech power spectrum
    P_nn = noise power spectrum
    Decision-directed prior SNR (Ephraim–Malah 1984):
        a_hat(k,l) = α · |Y_hat(k,l-1)|²/P_nn(k,l)
                   + (1-α) · max(posterior_SNR − 1, 0)

──────────────────────────────────────────────
2. Spectral Subtraction (Boll 1979)
──────────────────────────────────────────────
    |S_hat(k,l)|  = max( |Y(k,l)| − α_over · |N_hat(k,l)| , β · |Y(k,l)| )
    phase is preserved from noisy signal.

Parameters:
    α_over  : over-subtraction factor (> 1 reduces musical noise)
    β       : spectral floor factor   (prevents negative spectra)
"""

import numpy as np
import scipy.signal as sig
from config import N_FFT, HOP_LENGTH, WIN_TYPE, SAMPLE_RATE


# ── Helper: STFT / ISTFT wrappers ────────────────────────────────────────────

def _stft(x: np.ndarray, n_fft: int = N_FFT,
          hop: int = HOP_LENGTH) -> np.ndarray:
    _, _, S = sig.stft(x, fs=SAMPLE_RATE, nperseg=n_fft,
                       noverlap=n_fft - hop,
                       window=WIN_TYPE, padded=True)
    return S          # complex (F, T)


def _istft(S: np.ndarray, n_fft: int = N_FFT,
           hop: int = HOP_LENGTH, length: int = None) -> np.ndarray:
    _, x = sig.istft(S, fs=SAMPLE_RATE, nperseg=n_fft,
                     noverlap=n_fft - hop, window=WIN_TYPE)
    if length is not None:
        x = x[:length]
    return x.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Wiener Filter (Decision-Directed, Ephraim–Malah 1984)
# ─────────────────────────────────────────────────────────────────────────────

class WienerFilter:
    """
    Single-channel Wiener filter with decision-directed prior SNR estimation.

    Parameters
    ----------
    alpha       : Smoothing factor for decision-directed prior SNR (0.92–0.98).
    noise_frames: Number of initial frames used for noise PSD estimation.
    """

    def __init__(self, alpha: float = 0.98, noise_frames: int = 6):
        self.alpha        = alpha
        self.noise_frames = noise_frames

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        """
        Enhance noisy speech using Wiener filtering.

        Parameters
        ----------
        noisy : 1-D float32 noisy speech signal.

        Returns
        -------
        enhanced : 1-D float32 enhanced speech.
        """
        orig_len = len(noisy)
        Y = _stft(noisy)                          # (F, T) complex
        power_Y = np.abs(Y) ** 2                  # observed power

        # ── Noise PSD initialisation (avg first noise_frames) ────────────────
        noise_psd = np.mean(power_Y[:, :self.noise_frames], axis=1,
                            keepdims=True)         # (F, 1)

        # ── Recursive noise PSD update (min-statistics style approximation) ──
        noise_psd_smooth = noise_psd.copy()
        beta_n = 0.98

        F, T = Y.shape
        gain         = np.ones((F, T), dtype=np.float32)
        prior_snr    = np.ones((F, 1), dtype=np.float64)    # ξ(k,l)
        prev_enhanced_pow = noise_psd.copy()

        for l in range(T):
            # Posterior SNR  γ(k,l) = |Y|²/σ_n²
            post_snr = np.maximum(power_Y[:, l:l+1] / (noise_psd_smooth + 1e-12), 0.0)

            # Decision-directed prior SNR
            prior_snr = (self.alpha * prev_enhanced_pow / (noise_psd_smooth + 1e-12)
                         + (1 - self.alpha) * np.maximum(post_snr - 1, 0.0))
            prior_snr = np.maximum(prior_snr, 1e-3)

            # Wiener gain  H = ξ / (1 + ξ)
            H = prior_snr / (1.0 + prior_snr)
            gain[:, l] = H[:, 0]

            # Update enhanced power for next frame
            enhanced_pow = (H * np.abs(Y[:, l:l+1])) ** 2
            prev_enhanced_pow = enhanced_pow

            # Noise PSD update (only update if current frame is likely noise)
            if l > self.noise_frames:
                noise_psd_smooth = (beta_n * noise_psd_smooth
                                    + (1 - beta_n) * np.minimum(power_Y[:, l:l+1],
                                                                  noise_psd_smooth * 2))

        enhanced_stft = gain * Y
        return _istft(enhanced_stft, length=orig_len)

    def __repr__(self) -> str:
        return (f"WienerFilter(alpha={self.alpha}, "
                f"noise_frames={self.noise_frames})")


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Subtraction (Boll 1979)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralSubtraction:
    """
    Spectral subtraction with over-subtraction and spectral flooring.

    Parameters
    ----------
    alpha_over  : Over-subtraction factor α ≥ 1 (typical 1.0–2.0).
    beta        : Spectral floor β (typical 0.01–0.05).
    noise_frames: Number of leading frames for noise estimation.
    """

    def __init__(self, alpha_over: float = 2.0, beta: float = 0.01,
                 noise_frames: int = 6):
        self.alpha_over  = alpha_over
        self.beta        = beta
        self.noise_frames = noise_frames

    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        orig_len = len(noisy)
        Y   = _stft(noisy)
        mag = np.abs(Y)
        phase = np.angle(Y)

        # Noise magnitude estimate from first frames
        noise_mag = np.mean(mag[:, :self.noise_frames], axis=1,
                            keepdims=True)

        # Spectral subtraction
        mag_enhanced = (mag - self.alpha_over * noise_mag)

        # Spectral flooring: prevent negative / very small values
        floor = self.beta * mag
        mag_enhanced = np.maximum(mag_enhanced, floor)

        # Reconstruct with original phase
        S = mag_enhanced * np.exp(1j * phase)
        return _istft(S, length=orig_len)

    def __repr__(self) -> str:
        return (f"SpectralSubtraction(α={self.alpha_over}, β={self.beta})")
