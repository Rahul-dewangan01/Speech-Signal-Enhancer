"""
Demo Analysis Script
=====================
Run this as a regular Python script OR paste cells into a Jupyter notebook.
Demonstrates LMS learning curves, convergence analysis, and filter comparison.

Run:  python notebooks/demo_analysis.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — avoids display issues
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH
from src.adaptive import LMSFilter, NLMSFilter, RLSFilter, WienerFilter, SpectralSubtraction
from src.preprocessing import stft
from data.download_dataset import generate_synthetic_speech, add_noise_at_snr, white_noise

import os
RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── 1. Generate test signals ─────────────────────────────────────────────────
print("Generating test signals ...")
np.random.seed(42)
clean = generate_synthetic_speech(duration=3.0)
noise = white_noise(len(clean))
noisy = add_noise_at_snr(clean, noise, snr_db=5)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — LMS learning curves + frame SNR comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Running LMS convergence analysis ...")

mu_values  = [0.001, 0.005, 0.01, 0.05]
N_demo     = 5000
window     = 100
EPS        = 1e-12          # prevent log(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for mu in mu_values:
    filt   = LMSFilter(order=32, mu=mu)
    errors = np.zeros(N_demo)
    for n in range(N_demo):
        _, e     = filt.update(noise[n], noisy[n])
        errors[n] = e ** 2

    # Clip before smoothing so no zeros reach the log axis
    errors   = np.clip(errors, EPS, None)
    smoothed = np.convolve(errors, np.ones(window) / window, mode='valid')
    smoothed = np.clip(smoothed, EPS, None)     # guarantee strictly positive

    axes[0].semilogy(smoothed, label=f'μ={mu}', linewidth=1.5)

# Fix: set explicit finite y-limits so tight_layout never sees inf
axes[0].set_ylim(EPS * 10, 10.0)
axes[0].set_xlim(0, N_demo - window)
axes[0].set_xlabel('Sample index', fontsize=11)
axes[0].set_ylabel('MSE (log scale)', fontsize=11)
axes[0].set_title('LMS Learning Curves — Effect of Step Size μ', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, which='both')

# ─── Frame-level SNR comparison ───────────────────────────────────────────────
print("Comparing LMS / NLMS / RLS ...")
N_cmp      = len(clean)
frame_size = 256
filters = {
    'LMS (μ=0.01)': LMSFilter(order=32, mu=0.01),
    'NLMS (μ=0.5)': NLMSFilter(order=32, mu=0.5),
    'RLS (λ=0.99)': RLSFilter(order=32, lam=0.99),
}

snr_curves = {}
for name, filt in filters.items():
    _, e     = filt.run(noise[:N_cmp], noisy[:N_cmp])
    n_frames = N_cmp // frame_size
    snrs     = []
    for i in range(n_frames):
        s_f  = clean[i * frame_size:(i + 1) * frame_size]
        e_f  = e    [i * frame_size:(i + 1) * frame_size]
        n_f  = s_f - e_f
        ps   = np.mean(s_f ** 2)
        pn   = np.mean(n_f ** 2) + EPS
        snrs.append(float(np.clip(10 * np.log10(ps / pn), -10, 40)))
    snr_curves[name] = snrs

t_frames = np.arange(len(list(snr_curves.values())[0])) * frame_size / SAMPLE_RATE
for name, snrs in snr_curves.items():
    axes[1].plot(t_frames, snrs, label=name, linewidth=1.5)

axes[1].axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='Input SNR (5dB)')
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Frame SNR (dB)', fontsize=11)
axes[1].set_title('Algorithm Comparison — Frame SNR over Time', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-10, 40)

out1 = os.path.join(RESULTS_DIR, 'convergence_analysis.png')
plt.tight_layout()
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — LMS tap weight evolution
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting weight evolution ...")

filt2          = LMSFilter(order=8, mu=0.01)
N_wev          = 3000
weight_history = np.zeros((N_wev, 8))
for n in range(N_wev):
    filt2.update(noise[n], noisy[n])
    weight_history[n] = filt2.w.copy()

fig2, ax2 = plt.subplots(figsize=(12, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for tap in range(8):
    ax2.plot(weight_history[:, tap], alpha=0.8, linewidth=1,
             label=f'w[{tap}]', color=colors[tap])

ax2.set_xlabel('Sample', fontsize=11)
ax2.set_ylabel('Weight value', fontsize=11)
ax2.set_title('LMS Tap Weight Evolution (Order=8)', fontsize=12)
ax2.legend(ncol=4, fontsize=9)
ax2.grid(True, alpha=0.3)

out2 = os.path.join(RESULTS_DIR, 'weight_evolution.png')
plt.tight_layout()
plt.savefig(out2, dpi=120, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Spectrogram comparison (5 methods)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating spectrogram comparison ...")

wf           = WienerFilter(alpha=0.98)
enhanced_w   = wf.enhance(noisy)

ss           = SpectralSubtraction(alpha_over=2.0)
enhanced_ss  = ss.enhance(noisy)

lms_f        = LMSFilter(order=32, mu=0.01)
_, enh_lms   = lms_f.run(noise, noisy)

signals = {
    'Clean':       clean,
    'Noisy (5dB)': noisy,
    'LMS':         enh_lms,
    'Wiener':      enhanced_w,
    'Spec. Sub.':  enhanced_ss,
}

fig3, axes3 = plt.subplots(len(signals), 1, figsize=(14, 3 * len(signals)))
for i, (name, sig) in enumerate(signals.items()):
    N   = min(len(sig), len(clean))
    mag, _ = stft(sig[:N].astype(np.float32))
    # Clip to finite dB range before imshow — prevents colorbar overflow
    db  = np.clip(20 * np.log10(mag + 1e-9), -80, 0)
    axes3[i].imshow(db, origin='lower', aspect='auto',
                    extent=[0, N / SAMPLE_RATE, 0, SAMPLE_RATE / 2 / 1000],
                    cmap='magma', vmin=-80, vmax=0)
    axes3[i].set_ylabel(name, fontsize=10)
    axes3[i].set_xlabel('Time (s)' if i == len(signals) - 1 else '', fontsize=9)
    cb = plt.colorbar(axes3[i].images[0], ax=axes3[i])
    cb.set_label('dB', fontsize=8)

fig3.suptitle('Spectrogram Comparison — Classical Methods', fontsize=13)
out3 = os.path.join(RESULTS_DIR, 'spectrogram_comparison.png')
plt.tight_layout()
plt.savefig(out3, dpi=120, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — SNR vs step size μ (misadjustment trade-off)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting misadjustment vs convergence trade-off ...")

mu_list    = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
final_snrs = []

for mu in mu_list:
    filt   = LMSFilter(order=32, mu=mu)
    _, e   = filt.run(noise[:N_cmp], noisy[:N_cmp])
    # Global SNR of enhanced signal
    ps     = np.mean(clean[:N_cmp] ** 2)
    pn     = np.mean((clean[:N_cmp] - e) ** 2) + EPS
    snr    = float(np.clip(10 * np.log10(ps / pn), -10, 40))
    final_snrs.append(snr)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(mu_list, final_snrs, 'o-', color='steelblue', linewidth=2, markersize=7)
for mu, snr in zip(mu_list, final_snrs):
    ax4.annotate(f'{snr:.1f}dB', (mu, snr), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)
ax4.set_xscale('log')
ax4.set_xlabel('Step size μ (log scale)', fontsize=11)
ax4.set_ylabel('Output SNR (dB)', fontsize=11)
ax4.set_title('LMS: Output SNR vs Step Size μ\n(Misadjustment trade-off)', fontsize=12)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_ylim(min(final_snrs) - 3, max(final_snrs) + 3)

out4 = os.path.join(RESULTS_DIR, 'misadjustment_tradeoff.png')
plt.tight_layout()
plt.savefig(out4, dpi=120, bbox_inches='tight')
plt.close(fig4)
print(f"  Saved: {out4}")


print(f"\n[DONE] All 4 plots saved to {RESULTS_DIR}/")
print("    convergence_analysis.png")
print("    weight_evolution.png")
print("    spectrogram_comparison.png")
print("    misadjustment_tradeoff.png")
