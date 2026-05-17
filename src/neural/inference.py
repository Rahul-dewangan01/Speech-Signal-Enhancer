"""
Neural Inference Wrapper
=========================
Loads a trained model checkpoint and enhances arbitrary-length audio.
"""

import os
import sys
import numpy as np
import torch
import scipy.signal as ssig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (CHECKPOINT_DIR, SAMPLE_RATE, N_FFT, HOP_LENGTH,
                    WIN_TYPE, INPUT_FEATURES)
from src.neural.model import SpeechEnhancementNet, CRN


class NeuralEnhancer:
    """
    Convenience wrapper: load → enhance audio.

    Parameters
    ----------
    model_type : "blstm" or "crn"
    checkpoint : Path to .pt file (default: CHECKPOINT_DIR/blstm_best.pt)
    """

    def __init__(self, model_type: str = "blstm", checkpoint: str = None):
        self.model_type = model_type
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "crn":
            self.model = CRN()
        else:
            self.model = SpeechEnhancementNet()

        if checkpoint is None:
            checkpoint = os.path.join(CHECKPOINT_DIR, f"{model_type}_best.pt")

        if os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded checkpoint: {checkpoint}")
        else:
            print(f"  [WARN] No checkpoint found at {checkpoint}. "
                  "Using random weights (for demo only).")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def enhance(self, noisy: np.ndarray) -> np.ndarray:
        """
        Enhance a noisy 1-D float32 speech signal.

        Parameters
        ----------
        noisy : (N,) float32 array at SAMPLE_RATE Hz.

        Returns
        -------
        enhanced : (N,) float32 enhanced signal.
        """
        noisy = noisy.astype(np.float32)
        orig_len = len(noisy)

        # STFT
        _, _, S = ssig.stft(noisy, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
                            window=WIN_TYPE)
        mag   = np.abs(S).T.astype(np.float32)   # (T, F)
        phase = np.angle(S)                        # (F, T)

        log_mag = np.log1p(mag)

        # Model forward
        inp = torch.from_numpy(log_mag).unsqueeze(0).to(self.device)  # (1,T,F)
        if self.model_type == "crn":
            inp  = inp.unsqueeze(1)               # (1,1,T,F)
            mask = self.model(inp).squeeze(1).squeeze(0).cpu().numpy()
        else:
            mask = self.model(inp).squeeze(0).cpu().numpy()            # (T,F)

        enhanced_mag = (mask * mag).T            # (F, T)

        # ISTFT with original phase
        _, enhanced = ssig.istft(
            enhanced_mag * np.exp(1j * phase),
            nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH, window=WIN_TYPE)

        # Guard against ISTFT output being shorter than original signal
        if len(enhanced) < orig_len:
            enhanced = np.pad(enhanced, (0, orig_len - len(enhanced)))

        return enhanced[:orig_len].astype(np.float32)
