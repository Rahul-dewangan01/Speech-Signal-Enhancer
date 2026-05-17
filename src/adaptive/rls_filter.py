"""
RLS (Recursive Least Squares) Adaptive Filter
===============================================
Minimises the *exponentially weighted* least-squares cost:

    J(n) = Σ_{i=0}^{n}  λ^(n-i) · |e(i)|²

Recursive update (Matrix Inversion Lemma / Woodbury identity):
    k(n)  = [λ^{-1} P(n-1) x(n)] / [1 + λ^{-1} x^T(n) P(n-1) x(n)]
    e(n)  = d(n) − w^T(n-1) x(n)
    w(n)  = w(n-1) + k(n) · e(n)
    P(n)  = λ^{-1} [P(n-1) − k(n) x^T(n) P(n-1)]

where P(n) ≈ [R_xx(n)]^{-1} (inverse of the input correlation matrix).

Advantages over LMS:
  • Faster convergence (exploits 2nd-order statistics).
  • Handles correlated inputs well.
Disadvantages:
  • O(M²) complexity vs O(M) for LMS.
  • Numerical issues with λ close to 1.
"""

import numpy as np
from typing import Tuple


class RLSFilter:
    """
    Recursive Least Squares adaptive filter.

    Parameters
    ----------
    order  : Filter order M.
    lam    : Forgetting factor λ ∈ (0, 1].  Closer to 1 → slower forgetting.
    delta  : Initial value of P = delta * I (large → fast initial adaptation).
    """

    def __init__(self, order: int = 32, lam: float = 0.99, delta: float = 1.0):
        self.order = order
        self.lam   = lam
        self.delta = delta
        self.reset()

    def reset(self):
        self.w       = np.zeros(self.order, dtype=np.float64)
        self._buf    = np.zeros(self.order, dtype=np.float64)
        self._buf_idx = 0
        self.P       = self.delta * np.eye(self.order, dtype=np.float64)

    def _get_x(self) -> np.ndarray:
        """Return buffer contents in correct order (newest first)."""
        idx = self._buf_idx
        return np.concatenate([self._buf[idx:], self._buf[:idx]])[::-1]

    def update(self, x_n: float, d_n: float) -> Tuple[float, float]:
        # Insert new sample into circular buffer
        self._buf[self._buf_idx] = x_n
        self._buf_idx = (self._buf_idx + 1) % self.order
        x = self._get_x()

        # Gain vector
        Px  = self.P @ x
        denom = self.lam + x @ Px
        k   = Px / denom

        # A priori error
        y_n = float(self.w @ x)
        e_n = d_n - y_n

        # Weight update
        self.w += k * e_n

        # Covariance update
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lam
        # Enforce symmetry to prevent numerical drift over many iterations
        self.P = 0.5 * (self.P + self.P.T)

        return y_n, float(e_n)

    def run(self, x: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N     = len(x)
        y, e  = np.zeros(N), np.zeros(N)
        for n in range(N):
            y[n], e[n] = self.update(x[n], d[n])
        return y, e

    @classmethod
    def denoise(cls, noisy: np.ndarray, reference_noise: np.ndarray,
                order: int = 32, lam: float = 0.99) -> np.ndarray:
        filt = cls(order=order, lam=lam)
        _, e = filt.run(reference_noise, noisy)
        return e

    def __repr__(self) -> str:
        return f"RLSFilter(order={self.order}, λ={self.lam}, δ={self.delta})"
