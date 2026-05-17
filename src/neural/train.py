"""
Training Pipeline for Neural Speech Enhancement
================================================
Trains SpeechEnhancementNet (BLSTM) or CRN on paired (noisy, clean) audio.

Features:
  • IRM mask loss (MSE between predicted and ideal ratio mask)
  • Magnitude-domain reconstruction loss  (optional L1 on magnitude)
  • Learning-rate scheduler (ReduceLROnPlateau)
  • Early stopping
  • Checkpoint saving / resuming
  • TensorBoard-compatible metric logging to CSV

Usage
-----
    python -m src.neural.train --model blstm --epochs 50
    python -m src.neural.train --model crn   --epochs 100 --resume
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import scipy.signal as ssig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (CLEAN_DIR, NOISY_DIR, CHECKPOINT_DIR, RESULTS_DIR,
                    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_TYPE, INPUT_FEATURES,
                    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    PATIENCE, CLIP_GRAD_NORM, DURATION, configure_torch)
from src.neural.model import SpeechEnhancementNet, CRN, compute_irm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    """
    Paired noisy–clean speech dataset.

    Each item is a fixed-length STFT magnitude spectrogram pair.
    The noisy file naming convention (from download_dataset.py):
        <stem>_<noisetype>_snr<+dB>dB.wav

    Parameters
    ----------
    clean_dir  : Directory of clean .wav files.
    noisy_dir  : Directory of noisy .wav files.
    duration   : Clip length in seconds.
    n_fft      : FFT size.
    hop        : Hop length.
    """

    def __init__(self, clean_dir: str = CLEAN_DIR,
                 noisy_dir: str = NOISY_DIR,
                 duration: float = DURATION,
                 n_fft: int = N_FFT, hop: int = HOP_LENGTH,
                 split: str = "train", split_ratio: float = 0.8):
        super().__init__()
        self.n_fft    = n_fft
        self.hop      = hop
        self.clip_len = int(duration * SAMPLE_RATE)

        # Pair up files
        noisy_files = sorted(Path(noisy_dir).glob("*.wav"))
        self.pairs  = []
        for nf in noisy_files:
            # Derive clean stem: everything before the first noise type suffix
            parts     = nf.stem.rsplit("_", 2)          # [clean_stem, ntype, snr]
            clean_stem = "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
            clean_path = Path(clean_dir) / f"{clean_stem}.wav"
            if clean_path.exists():
                self.pairs.append((str(clean_path), str(nf)))

        # Train / val split
        n     = len(self.pairs)
        pivot = int(n * split_ratio)
        if split == "train":
            self.pairs = self.pairs[:pivot]
        else:
            self.pairs = self.pairs[pivot:]

    def _load_and_stft(self, path: str) -> np.ndarray:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        # Trim / pad to fixed length
        if len(data) >= self.clip_len:
            start = np.random.randint(0, len(data) - self.clip_len + 1)
            data  = data[start: start + self.clip_len]
        else:
            data  = np.pad(data, (0, self.clip_len - len(data)))
        _, _, S = ssig.stft(data, nperseg=self.n_fft,
                            noverlap=self.n_fft - self.hop,
                            window=WIN_TYPE)
        return np.abs(S).T.astype(np.float32)  # (T, F)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        clean_mag = self._load_and_stft(clean_path)
        noisy_mag = self._load_and_stft(noisy_path)
        # Clip to same T (may differ by 1 due to STFT)
        T = min(clean_mag.shape[0], noisy_mag.shape[0])
        clean_mag = clean_mag[:T]
        noisy_mag = noisy_mag[:T]

        # Log-magnitude input feature
        log_noisy = np.log1p(noisy_mag)

        return (torch.from_numpy(log_noisy),    # (T, F)  input
                torch.from_numpy(noisy_mag),     # (T, F)  for reconstruction loss
                torch.from_numpy(clean_mag))     # (T, F)  target clean


def collate_fn(batch):
    """Pad variable-length sequences to batch-max length."""
    log_noisy, noisy_mag, clean_mag = zip(*batch)
    T_max = max(x.shape[0] for x in log_noisy)
    F_dim = log_noisy[0].shape[1]

    def pad(tensors):
        out = torch.zeros(len(tensors), T_max, F_dim)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    return pad(log_noisy), pad(noisy_mag), pad(clean_mag)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────────────────────

class EnhancementLoss(nn.Module):
    """
    Combined loss:
        L = λ_mask · MSE(predicted_mask, IRM)
          + λ_mag  · L1(predicted_mag , clean_mag)

    Both terms are averaged only over non-padded frames.
    """

    def __init__(self, lambda_mask: float = 0.7, lambda_mag: float = 0.3):
        super().__init__()
        self.lm = lambda_mask
        self.ll = lambda_mag

    def forward(self, pred_mask, noisy_mag, clean_mag):
        # Compute noise magnitude as |noisy - clean|, clamped to non-negative.
        # This is a better approximation than (noisy_mag - clean_mag) which
        # can go negative and break IRM computation.
        noise_mag    = torch.clamp(noisy_mag - clean_mag, min=0.0)
        irm          = compute_irm(clean_mag, noise_mag)
        mask_loss    = F.mse_loss(pred_mask, irm)
        enhanced_mag = pred_mask * noisy_mag
        mag_loss     = F.l1_loss(enhanced_mag, clean_mag)
        return self.lm * mask_loss + self.ll * mag_loss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Handles the full train / validate / save loop.

    Parameters
    ----------
    model_type : "blstm" or "crn"
    resume     : Whether to resume from latest checkpoint.
    """

    def __init__(self, model_type: str = "blstm",
                 epochs: int = EPOCHS,
                 lr: float = LEARNING_RATE,
                 resume: bool = False):
        # Initialise GPU optimisations
        configure_torch()

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs     = epochs
        self.model_type = model_type

        # ── Model ────────────────────────────────────────────────────────────
        if model_type == "crn":
            self.model = CRN().to(self.device)
        else:
            self.model = SpeechEnhancementNet().to(self.device)
        print(f"Model: {model_type.upper()}  |  "
              f"Params: {self.model.count_parameters():,}  |  "
              f"Device: {self.device}")

        # ── Optimisation ─────────────────────────────────────────────────────
        self.criterion = EnhancementLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=lr, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5)

        # ── Data ─────────────────────────────────────────────────────────────
        train_ds = SpeechDataset(split="train")
        val_ds   = SpeechDataset(split="val")
        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                       shuffle=True, collate_fn=collate_fn,
                                       num_workers=2, pin_memory=True,
                                       persistent_workers=True,
                                       prefetch_factor=2)
        self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                     shuffle=False, collate_fn=collate_fn,
                                     num_workers=2, pin_memory=True,
                                     persistent_workers=True,
                                     prefetch_factor=2)

        # ── Logging ──────────────────────────────────────────────────────────
        self.log_path = os.path.join(RESULTS_DIR, f"training_log_{model_type}.csv")
        self.best_val_loss = float("inf")
        self.patience_ctr  = 0
        self.start_epoch   = 0

        if resume:
            self._load_checkpoint()

    def _ckpt_path(self, tag: str = "best") -> str:
        return os.path.join(CHECKPOINT_DIR, f"{self.model_type}_{tag}.pt")

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        torch.save({
            "epoch":      epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_loss":    val_loss,
        }, self._ckpt_path(tag))

    def _load_checkpoint(self):
        path = self._ckpt_path("last")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.start_epoch    = ckpt["epoch"] + 1
            self.best_val_loss  = ckpt.get("val_loss", float("inf"))
            print(f"  Resumed from epoch {self.start_epoch}")

    def _step(self, batch, train: bool) -> float:
        log_noisy, noisy_mag, clean_mag = [b.to(self.device) for b in batch]

        if self.model_type == "crn":
            # CRN expects (B, 1, T, F)
            inp  = log_noisy.unsqueeze(1)
            mask = self.model(inp).squeeze(1)
        else:
            mask = self.model(log_noisy)

        loss = self.criterion(mask, noisy_mag, clean_mag)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD_NORM)
            self.optimizer.step()

        return loss.item()

    def train(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for batch in self.train_loader:
                train_losses.append(self._step(batch, train=True))
            train_loss = np.mean(train_losses)

            # ── Validate ─────────────────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    val_losses.append(self._step(batch, train=False))
            val_loss = np.mean(val_losses) if val_losses else float("inf")

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"Epoch {epoch+1:3d}/{self.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.2e}  ({elapsed:.1f}s)")

            # ── Log ──────────────────────────────────────────────────────────
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, lr])

            # ── Save ─────────────────────────────────────────────────────────
            self._save_checkpoint(epoch, val_loss, "last")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, "best")
                self.patience_ctr  = 0
                print("  [*] Best model saved.")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}.")
                    break

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {CHECKPOINT_DIR}")
        print(f"Training log: {self.log_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["blstm", "crn"], default="blstm")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE)
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    trainer = Trainer(model_type=args.model, epochs=args.epochs,
                      lr=args.lr, resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
