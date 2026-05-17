"""
NLMS (Normalised Least Mean Squares) Filter
=============================================
Improves on LMS by normalising the step size with input power, giving:
  • Faster convergence in non-stationary environments.
  • Constant misadjustment regardless of input power.

Weight update:
    w(n+1) = w(n) + [μ / (ε + ||x(n)||²)] · e(n) · x(n)

The effective step size is  μ_eff = μ / (ε + ||x||²),
so it adapts automatically to signal power.
"""

import numpy as np
from typing import Tuple


class NLMSFilter:
    """
    Normalised LMS adaptive filter.

    Parameters
    ----------
    order   : Filter order (number of taps).
    mu      : Normalised step size, 0 < μ < 2 for stability (typical 0.5).
    epsilon : Small constant to avoid division by zero.
    """

    def __init__(self, order: int = 32, mu: float = 0.5,
                 epsilon: float = 1e-8):
        self.order   = order
        self.mu      = mu
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.w     = np.zeros(self.order, dtype=np.float64)
        self._buf  = np.zeros(self.order, dtype=np.float64)
        self._buf_idx = 0
        self._power = 0.0  # running sum of squared buffer elements

    def _get_x(self) -> np.ndarray:
        """Return buffer contents in correct order (newest first)."""
        idx = self._buf_idx
        return np.concatenate([self._buf[idx:], self._buf[:idx]])[::-1]

    def update(self, x_n: float, d_n: float) -> Tuple[float, float]:
        # Remove oldest sample's contribution to power
        oldest_idx = self._buf_idx
        self._power -= self._buf[oldest_idx] ** 2

        # Insert new sample
        self._buf[oldest_idx] = x_n
        self._power += x_n ** 2
        self._buf_idx = (self._buf_idx + 1) % self.order

        x_vec = self._get_x()

        y_n   = np.dot(self.w, x_vec)
        e_n   = d_n - y_n

        # Normalised step (use tracked power instead of recomputing dot product)
        self.w += (self.mu / (self.epsilon + self._power)) * e_n * x_vec

        return float(y_n), float(e_n)

    def run(self, x: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(x)
        y, e = np.zeros(N), np.zeros(N)
        for n in range(N):
            y[n], e[n] = self.update(x[n], d[n])
        return y, e

    @classmethod
    def denoise(cls, noisy: np.ndarray, reference_noise: np.ndarray,
                order: int = 32, mu: float = 0.5) -> np.ndarray:
        """Adaptive noise cancellation using NLMS."""
        filt = cls(order=order, mu=mu)
        _, e = filt.run(reference_noise, noisy)
        return e

    def __repr__(self) -> str:
        return f"NLMSFilter(order={self.order}, mu={self.mu})"
