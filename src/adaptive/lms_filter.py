"""
LMS (Least Mean Squares) Adaptive Filter
=========================================
The simplest and most widely used adaptive filter algorithm.

Theory
------
Given:
    d(n)  – desired signal (reference)
    x(n)  – input / noisy signal
    w(n)  – filter weights (tap vector)
    y(n)  = w^T(n) · x(n)   (filter output)
    e(n)  = d(n) − y(n)      (error signal)

Weight update rule (LMS):
    w(n+1) = w(n) + 2·μ·e(n)·x(n)

Convergence condition:  0 < μ < 1 / (N · λ_max)
where N is filter order and λ_max is the largest eigenvalue of Rxx.
"""

import numpy as np
from typing import Tuple


class LMSFilter:
    """
    Least Mean Squares adaptive filter.

    Parameters
    ----------
    order : int
        Number of filter taps (filter order M).
    mu : float
        Step size (learning rate).  Must satisfy  0 < μ < 1/(M·P_x)
        where P_x is the input power.
    """

    def __init__(self, order: int = 32, mu: float = 0.01):
        self.order = order
        self.mu    = mu
        self.reset()

    def reset(self):
        """Reset filter weights and buffer to zero."""
        self.w = np.zeros(self.order, dtype=np.float64)
        self._buf = np.zeros(self.order, dtype=np.float64)
        self._buf_idx = 0  # circular buffer write position

    # ── single sample ────────────────────────────────────────────────────────

    def _get_x(self) -> np.ndarray:
        """Return the current buffer contents in correct order (newest first)."""
        idx = self._buf_idx
        # Concatenate: [idx-1, idx-2, ..., 0, order-1, ..., idx]
        return np.concatenate([self._buf[idx:], self._buf[:idx]])[::-1]

    def update(self, x_n: float, d_n: float) -> Tuple[float, float]:
        """
        Process one sample.

        Parameters
        ----------
        x_n : float   Current input sample (noisy speech or reference noise).
        d_n : float   Desired signal sample.

        Returns
        -------
        y_n : float   Filter output.
        e_n : float   Error signal  e = d − y.
        """
        # Insert new sample into circular buffer
        self._buf[self._buf_idx] = x_n
        self._buf_idx = (self._buf_idx + 1) % self.order

        # Get the current buffer contents in order
        x_vec = self._get_x()

        # Filter output
        y_n = np.dot(self.w, x_vec)

        # Error
        e_n = d_n - y_n

        # LMS weight update
        self.w += 2.0 * self.mu * e_n * x_vec

        return float(y_n), float(e_n)

    # ── block processing ─────────────────────────────────────────────────────

    def run(self, x: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process arrays of samples.

        Parameters
        ----------
        x : (N,) input / reference signal.
        d : (N,) desired signal.

        Returns
        -------
        y : (N,) filter output.
        e : (N,) error signal (enhanced signal when used for noise cancellation).
        """
        assert len(x) == len(d), "x and d must have the same length."
        N = len(x)
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            y[n], e[n] = self.update(x[n], d[n])
        return y, e

    # ── Noise cancellation convenience wrapper ────────────────────────────────

    @classmethod
    def denoise(cls, noisy: np.ndarray, reference_noise: np.ndarray,
                order: int = 32, mu: float = 0.01) -> np.ndarray:
        """
        Adaptive noise cancellation.

        The filter tries to predict the noise correlated with `reference_noise`
        inside `noisy`; the error signal e(n) is the enhanced speech.

        Parameters
        ----------
        noisy           : Primary microphone signal  = speech + noise1
        reference_noise : Reference microphone signal ≈ noise only (no speech)
        """
        filt = cls(order=order, mu=mu)
        _, e = filt.run(reference_noise, noisy)
        return e

    @property
    def weights(self) -> np.ndarray:
        return self.w.copy()

    def __repr__(self) -> str:
        return (f"LMSFilter(order={self.order}, mu={self.mu}, "
                f"||w||={np.linalg.norm(self.w):.4f})")
