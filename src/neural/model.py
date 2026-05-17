"""
Neural Speech Enhancement Models
==================================
Two architectures are provided:

1. SpeechEnhancementNet  –  Bidirectional LSTM operating on log-power spectra.
2. CRN (Convolutional Recurrent Network) – CNN encoder + BLSTM + CNN decoder.
   Inspired by Tan & Wang (2018) "A Convolutional Recurrent Neural Network for
   Real-Time Speech Enhancement."

Both produce a real-valued *mask* M ∈ (0, 1) per time-frequency bin and apply
it to the noisy STFT magnitude:

    |S_hat(k,l)| = M(k,l) · |Y(k,l)|

Mask types used:
    IRM  (Ideal Ratio Mask):   M = P_ss / (P_ss + P_nn)
    IBM  (Ideal Binary Mask):  M = 1 if SNR_local > threshold, else 0
    cIRM (complex IRM):        complex-valued, used internally by CRN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import INPUT_FEATURES, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
# 1. Simple Bidirectional LSTM (BLSTM) Model
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEnhancementNet(nn.Module):
    """
    Bidirectional LSTM speech enhancement network.

    Architecture:
        Input  : (B, T, F) – log-power spectrogram frames
        BLSTM  : 3 stacked layers, hidden=256
        FC     : Linear projection + Sigmoid → IRM mask (B, T, F)
        Output : (B, T, F) real mask ∈ (0,1)

    Usage
    -----
        model = SpeechEnhancementNet()
        mask  = model(noisy_logpower)          # (B, T, F)
        enhanced_mag = mask * noisy_mag
    """

    def __init__(self, input_size: int = INPUT_FEATURES,
                 hidden_size: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Layer normalisation on input
        self.input_norm = nn.LayerNorm(input_size)

        # Stacked BiLSTM
        self.blstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Output projection: 2*hidden (bi) → input_size
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),   # IRM mask ∈ (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)  log-power spectrogram
        Returns mask: (B, T, F)
        """
        x = self.input_norm(x)
        out, _ = self.blstm(x)          # (B, T, 2*H)
        mask   = self.fc(out)           # (B, T, F)
        return mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRN – Convolutional Recurrent Network
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ELU"""
    def __init__(self, in_ch, out_ch, kernel=(2, 3), stride=(1, 2),
                 padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → ELU"""
    def __init__(self, in_ch, out_ch, kernel=(2, 3), stride=(1, 2),
                 padding=(1, 1), output_padding=(0, 1)):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride,
                                          padding=padding,
                                          output_padding=output_padding)
        self.bn     = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.elu(self.bn(self.deconv(x)))


class CRN(nn.Module):
    """
    Convolutional Recurrent Network for speech enhancement.

    Architecture (encoder–BLSTM–decoder with skip connections):
        Encoder : 5 × ConvBlock (freq downsampling via stride on freq axis)
        Middle  : 2 × BiLSTM
        Decoder : 5 × DeconvBlock (freq upsampling) + skip connections
        Output  : Sigmoid mask

    Input  : (B, 1, T, F) – single-channel magnitude spectrogram
    Output : (B, 1, T, F) – enhancement mask
    """

    def __init__(self, input_freq: int = INPUT_FEATURES,
                 channels: int = 64, lstm_hidden: int = 128):
        super().__init__()
        self.input_freq  = input_freq
        self.lstm_hidden = lstm_hidden

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(1,  16, kernel=(2,3), stride=(1,2), padding=(1,1))
        self.enc2 = ConvBlock(16, 32, kernel=(2,3), stride=(1,2), padding=(1,1))
        self.enc3 = ConvBlock(32, 64, kernel=(2,3), stride=(1,2), padding=(1,1))
        self.enc4 = ConvBlock(64, 128, kernel=(2,3), stride=(1,2), padding=(1,1))

        # Compute bottleneck freq dimension after 4 stride-2 convolutions
        # F → ceil(F/2) → … (4 times)
        bottleneck_f = input_freq
        for _ in range(4):
            bottleneck_f = (bottleneck_f + 2 - 3) // 2 + 1  # approx
        self.bottleneck_f = max(bottleneck_f, 1)

        # ── BiLSTM bottleneck ────────────────────────────────────────────────
        self.blstm = nn.LSTM(
            input_size   = 128 * self.bottleneck_f,
            hidden_size  = lstm_hidden,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout       = 0.2,
        )
        self.lstm_proj = nn.Linear(2 * lstm_hidden, 128 * self.bottleneck_f)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec4 = DeconvBlock(128 + 128, 64)
        self.dec3 = DeconvBlock(64  + 64,  32)
        self.dec2 = DeconvBlock(32  + 32,  16)
        self.dec1 = DeconvBlock(16  + 16,  1,
                                output_padding=(0,0))

        self.out_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T, F)
        Returns mask: (B, 1, T, F)
        """
        # ── Encoder ─────────────────────────────────────────────────────────
        e1 = self.enc1(x)    # (B,16,T,F/2)
        e2 = self.enc2(e1)   # (B,32,T,F/4)
        e3 = self.enc3(e2)   # (B,64,T,F/8)
        e4 = self.enc4(e3)   # (B,128,T,F/16)

        B, C, T, F = e4.shape
        # Reshape for LSTM: (B, T, C*F)
        r = e4.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        lstm_out, _ = self.blstm(r)         # (B, T, 2*hidden)
        r2 = self.lstm_proj(lstm_out)        # (B, T, C*F)
        r2 = r2.view(B, T, C, F).permute(0, 2, 1, 3)  # (B, C, T, F)

        # ── Decoder with skip connections ────────────────────────────────────
        d4 = self.dec4(torch.cat([r2, e4], dim=1))
        # Crop if needed to match encoder sizes
        d4 = d4[:, :, :e3.shape[2], :e3.shape[3]]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = d3[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = d2[:, :, :e1.shape[2], :e1.shape[3]]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        d1 = d1[:, :, :x.shape[2], :x.shape[3]]

        return self.out_sigmoid(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# IRM target computation (for training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_irm(clean_mag: torch.Tensor, noise_mag: torch.Tensor,
                eps: float = 1e-9) -> torch.Tensor:
    """
    Ideal Ratio Mask:   M = P_s / (P_s + P_n)
    Input:  clean_mag, noise_mag – magnitudes of same shape.
    Output: IRM mask ∈ (0, 1).
    """
    Ps = clean_mag ** 2
    Pn = noise_mag ** 2
    return Ps / (Ps + Pn + eps)


def compute_ibm(clean_mag: torch.Tensor, noise_mag: torch.Tensor,
                threshold_db: float = 0.0) -> torch.Tensor:
    """
    Ideal Binary Mask: M = 1 if local SNR > threshold, else 0.
    """
    local_snr = 20 * torch.log10(clean_mag / (noise_mag + 1e-9) + 1e-9)
    return (local_snr > threshold_db).float()
