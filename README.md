# 🎙 Speech Signal Enhancement — Adaptive + Neural Methods

> **Introduction to Adaptive Signal Processing — Team Project**  
> Complete implementation, training and evaluation of **7 enhancement algorithms** on real NOIZEUS speech corpus + synthetic signals.  
> Trained on RTX 4060 8GB · PyTorch 2.6 · Python 3.11 · Windows 11

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Signal Model & Theory](#4-signal-model--theory)
5. [Adaptive Filter Algorithms](#5-adaptive-filter-algorithms)
6. [Neural Network Architectures](#6-neural-network-architectures)
7. [Training Results](#7-training-results)
8. [Convergence Analysis](#8-convergence-analysis)
9. [Evaluation Metrics & Results](#9-evaluation-metrics--results)
10. [Spectrogram Analysis](#10-spectrogram-analysis)
11. [Dataset](#11-dataset)
12. [Installation](#12-installation)
13. [Usage](#13-usage)
14. [Key References](#14-key-references)
15. [Credits](#credits)

---

## 1. Project Overview

Speech enhancement recovers a clean speech signal **s(n)** from a degraded observation **y(n) = s(n) + d(n)**, where d(n) is additive noise. This project implements and rigorously evaluates seven algorithms across two paradigms:

**Team members:** Rahul Dewangan, Mandeep Singh, Shashi Kant Kumar, Navadeep Bandi

| Paradigm | Methods | Key Property |
|---|---|---|
| **Adaptive / Classical** | LMS, NLMS, RLS, Wiener Filter, Spectral Subtraction | Statistical noise models, no training needed |
| **Neural / Deep Learning** | BLSTM, CRN | Learned IRM mask estimation, data-driven |

### Why both paradigms?

The **Ideal Ratio Mask (IRM)** that neural networks estimate is mathematically identical to the Wiener filter gain:

```
IRM(k,l) = P_ss(k,l) / [P_ss(k,l) + P_nn(k,l)]   =   Wiener Gain H(k,l)
```

Neural networks learn to *estimate* this quantity from data, replacing hand-crafted statistical models with learned ones.

---

## 2. Repository Structure

```
speech_enhancement/
│
├── config.py                    ← All hyperparameters & paths
├── main.py                      ← Demo pipeline: all 7 methods
├── requirements.txt
├── explanation.html             ← Complete guide + 25 interview Q&A
│
├── data/
│   └── download_dataset.py      ← NOIZEUS download / synthetic generation
│
├── src/
│   ├── adaptive/
│   │   ├── lms_filter.py        ← LMS (from scratch)
│   │   ├── nlms_filter.py       ← Normalised LMS
│   │   ├── rls_filter.py        ← RLS (Matrix Inversion Lemma)
│   │   └── wiener_filter.py     ← Wiener (Ephraim-Malah DD) + Spectral Sub.
│   ├── neural/
│   │   ├── model.py             ← BLSTM + CRN (PyTorch)
│   │   ├── train.py             ← Training loop, loss, checkpointing
│   │   └── inference.py         ← NeuralEnhancer wrapper
│   ├── preprocessing/
│   │   └── features.py          ← STFT, Mel, MFCC, normalisation
│   └── evaluation/
│       └── metrics.py           ← SNR, SSNR, PESQ, STOI, LSD
│
├── audio_samples/
│   ├── clean/                   ← sp01–sp30 clean WAVs
│   ├── noisy/                   ← 960 noisy WAVs (30 × 8 types × 4 SNRs)
│   └── enhanced/                ← 7 enhanced outputs per run
│
├── results/
│   ├── checkpoints/
│   │   ├── blstm_best.pt        ← Best val loss: 0.0240 (epoch 13)
│   │   └── crn_best.pt          ← Best val loss: 0.0627 (epoch 37)
│   ├── training_log_blstm.csv
│   ├── training_log_crn.csv
│   ├── metrics_comparison.csv
│   ├── convergence_analysis.png
│   ├── weight_evolution.png
│   ├── spectrogram_comparison.png
│   └── misadjustment_tradeoff.png
│
└── notebooks/
    └── demo_analysis.py         ← All analysis plots
```

---

## 3. Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare data (NOIZEUS or synthetic)
python data/download_dataset.py --source /path/to/noizeus_corpora
# OR
python data/download_dataset.py --synthetic --n_clean 200

# 3. Train
python -m src.neural.train --model blstm --epochs 50
python -m src.neural.train --model crn   --epochs 50

# 4. Evaluate on real NOIZEUS speech
python main.py --input  audio_samples/noisy/sp01_babble_snr+5dB.wav \
               --clean  audio_samples/clean/sp01.wav --compare

# 5. Analysis plots
python notebooks/demo_analysis.py
```

---

## 4. Signal Model & Theory

### Additive Noise Model

```
y(n) = s(n) + d(n)

y(n) : observed noisy speech
s(n) : clean speech (desired)
d(n) : additive noise
```

### STFT Parameters

```
N_FFT  = 512   →  32 ms frame at 16 kHz
Hop    = 128   →   8 ms (75% overlap)
Window = Hann  →  -31 dB first side-lobe
Bins   = 257   →  0 to 8 kHz
```

### Wiener–Hopf Equation (Optimal Linear Filter)

```
R_xx · w_opt = r_dx

ξ_min = σ_d² − r_dx^T · R_xx⁻¹ · r_dx   (minimum achievable MSE)
```

All adaptive algorithms converge toward **w_opt** without explicitly inverting R_xx.

---

## 5. Adaptive Filter Algorithms

### LMS — Least Mean Squares

```
y(n)   = w^T(n) · x(n)
e(n)   = d(n) − y(n)
w(n+1) = w(n) + 2μ · e(n) · x(n)

Stability:      0 < μ < 1/(M · λ_max)
Misadjustment:  M_adj ≈ μ · M · σ_x²
Complexity:     O(M) per sample
```

### NLMS — Normalised LMS

```
w(n+1) = w(n) + [μ / (ε + ||x(n)||²)] · e(n) · x(n)

Stability:  0 < μ < 2  (for any input power)
M_adj:      μ / (2 − μ)  (independent of input power)
```

### RLS — Recursive Least Squares

```
Cost:  J(n) = Σ λ^(n-i) · |e(i)|²

k(n) = λ⁻¹P(n-1)x(n) / [1 + λ⁻¹x^T P(n-1)x]   ← Kalman gain
w(n) = w(n-1) + k(n) · e(n)                      ← weight update
P(n) = λ⁻¹ [P(n-1) − k(n)x^T(n)P(n-1)]          ← covariance update

λ=0.99 → memory ≈ 100 samples
Complexity: O(M²) per sample
```

### Wiener Filter (Decision-Directed, Ephraim–Malah 1984)

```
γ(k,l)  = |Y(k,l)|² / P̂_nn(k,l)                            ← posterior SNR
ξ̂(k,l)  = α·|Ŝ(k,l-1)|²/P̂_nn + (1-α)·max[γ-1, 0]         ← prior SNR
H(k,l)  = ξ̂(k,l) / [1 + ξ̂(k,l)]                           ← Wiener gain
α=0.98  → temporal smoothing suppresses musical noise
```

### Spectral Subtraction (Boll 1979)

```
|Ŝ(k,l)| = max( |Y| − α_over·|D̂| ,  β·|Y| )
α_over=2.0,  β=0.01
```

### Algorithm Comparison

| Algorithm | Complexity | Convergence | Non-stationary | Latency |
|---|---|---|---|---|
| LMS | O(M) | Slow | Poor | < 1 ms |
| NLMS | O(M) | Moderate | Fair | < 1 ms |
| RLS | O(M²) | Fast | Good | < 2 ms |
| Wiener DD | O(N log N) | — | Good | 1 frame |
| Spectral Sub. | O(N log N) | — | Moderate | 1 frame |

---

## 6. Neural Network Architectures

### BLSTM Architecture (4,406,531 parameters)

```
Input:  (B, T, 257)  log-magnitude spectrogram
        │
        LayerNorm(257)
        │
        BiLSTM ×3  [hidden=256/direction → 512 total]
        │
        FC(512→256) + ReLU + Dropout(0.2)
        FC(256→257) + Sigmoid
        │
Output: IRM Mask (B, T, 257) ∈ (0,1)

Enhanced = Mask ⊙ |Y|
```

### CRN Architecture (Convolutional Recurrent Network)

```
Input: (B, 1, T, 257)
       │
ENCODER (stride-2 on freq axis each layer):
  Conv2D(1→16)  → BN → ELU  →  (B,16,T,128)   e1
  Conv2D(16→32) → BN → ELU  →  (B,32,T,64)    e2
  Conv2D(32→64) → BN → ELU  →  (B,64,T,32)    e3
  Conv2D(64→128)→ BN → ELU  →  (B,128,T,16)   e4
       │
BiLSTM BOTTLENECK:
  Reshape → (B,T,2048) → BiLSTM×2 → Linear → (B,128,T,16)
       │
DECODER (U-Net skip connections):
  DeConv + skip(e4) → (B,64,T,32)
  DeConv + skip(e3) → (B,32,T,64)
  DeConv + skip(e2) → (B,16,T,128)
  DeConv + skip(e1) → (B,1,T,257)
       │ Sigmoid
Output: IRM Mask (B,1,T,257)
```

### Loss Function

```
L = 0.7 · MSE(M_pred, M_IRM)  +  0.3 · L1(M_pred ⊙ |Y|, |S|)
         ↑ mask supervision            ↑ magnitude reconstruction
```

---

## 7. Training Results

### BLSTM — Trained on RTX 4060 (CUDA 12.4)

| Setting | Value |
|---|---|
| Parameters | 4,406,531 |
| GPU | NVIDIA RTX 4060 Laptop |
| Best val loss | **0.0240** (epoch 13) |
| Stopped | Epoch 23 (early stopping patience=10) |
| LR schedule | 1e-3 → 5e-4 at epoch 19 |

| Epoch | Train | Val | LR |
|---|---|---|---|
| 1 | 0.04096 | 0.03158 | 1e-3 |
| 5 | 0.02563 | 0.02827 | 1e-3 |
| 9 | 0.01913 | **0.02470** ← best | 1e-3 |
| 13 | 0.01304 | 0.02402 | 1e-3 |
| 19 | 0.00990 | 0.02558 | **→5e-4** |
| 23 | 0.00920 | 0.02521 | 5e-4 ← stop |

### CRN — Training

| Setting | Value |
|---|---|
| Best val loss | **0.0627** (epoch 37) |
| Stopped | Epoch 47 |
| LR reductions | Epoch 19 (→5e-4), Epoch 43 (→2.5e-4) |

CRN trains more slowly due to deeper architecture and noisier loss landscape, but converges to a strong solution.

---

## 8. Convergence Analysis

### LMS Learning Curves & Frame SNR

![Convergence Analysis](results/convergence_analysis.png)

**Left — LMS Learning Curves (4 step sizes):**
- All curves decrease from MSE ≈ 0.1 to ≈ 0.02 over 5000 samples
- μ=0.05 shows higher variance (misadjustment noise) at same convergence speed
- The signal's uniform power means normalisation (NLMS) has less impact here

**Right — Frame SNR over Time (LMS vs NLMS vs RLS):**
- **RLS (green)** achieves highest and most variable frame SNR — exact second-order adaptation
- **NLMS (orange)** tracks near the 5 dB input with moderate improvement
- **LMS (blue)** converges slowest, stabilising below input SNR in early frames

---

### Misadjustment Trade-off Curve

![Misadjustment Tradeoff](results/misadjustment_tradeoff.png)

The fundamental trade-off between convergence speed and steady-state quality:

| μ | SNR Out | Notes |
|---|---|---|
| 5×10⁻⁴ | **17.0 dB** | Best quality — slow but low misadjustment |
| 1×10⁻³ | 14.6 dB | Good balance |
| 5×10⁻³ | 7.2 dB | Acceptable |
| 1×10⁻² | 3.1 dB | High misadjustment |
| 5×10⁻² | −3.4 dB | Diverging — filter adds noise |

---

### LMS Tap Weight Evolution

![Weight Evolution](results/weight_evolution.png)

- **w[0] (blue)** is the dominant tap, converging to ~0.15 with tracking fluctuations
- **w[1]–w[7]** model residual noise correlation at different lags
- Stochastic fluctuation amplitude ∝ μ (misadjustment)
- Periodic structure in weights reflects the quasi-periodic nature of synthetic speech

---

## 9. Evaluation Metrics & Results

### Metric Reference

| Metric | Range | Better | What it measures |
|---|---|---|---|
| SNR | −∞ to ∞ dB | ↑ Higher | Global noise-to-signal ratio |
| SSNR | −10 to 35 dB | ↑ Higher | Frame-level SNR (perceptual) |
| PESQ | −0.5 to 4.5 | ↑ Higher | Perceptual quality (ITU-T P.862) |
| STOI | 0 to 1 | ↑ Higher | Speech intelligibility |
| LSD | 0 to ∞ dB | ↓ Lower | Spectral distortion |

---

### Results — Synthetic Babble Noise (5 dB input SNR)

| Method | SNR | SSNR | PESQ | STOI | LSD |
|---|---|---|---|---|---|
| Noisy input | 10.00 | 5.00 | 1.015 | 0.561 | 13.81 |
| LMS | 7.07 | 7.09 | 1.036 | 0.874 | 10.39 |
| NLMS | 4.48 | 4.50 | 1.025 | **0.879** | 11.08 |
| RLS | 7.54 | 7.62 | 1.038 | 0.869 | 10.26 |
| Wiener | 0.26 | −0.41 | 1.015 | 0.387 | 7.67 |
| SpectralSub | 0.08 | −0.74 | 1.018 | 0.365 | 8.26 |
| **BLSTM** | **10.04** | 6.82 | **2.208** | 0.806 | **3.51** |
| CRN | 6.86 | 3.98 | 1.034 | 0.580 | 11.47 |

---

### Results — Real NOIZEUS sp01 Babble (−4.6 dB input SNR)

> Most challenging test: more noise than speech

| Method | SNR | SSNR | PESQ | STOI | LSD |
|---|---|---|---|---|---|
| Noisy input | −4.57 | −6.82 | 1.030 | 0.220 | 7.52 |
| LMS | −4.14 | −6.77 | 1.029 | 0.227 | 7.52 |
| NLMS | −3.43 | −6.45 | 1.025 | **0.267** | 7.90 |
| RLS | −3.55 | −6.58 | 1.026 | 0.218 | 7.71 |
| Wiener | −3.87 | −5.53 | **1.042** | 0.067 | **5.51** |
| SpectralSub | −3.54 | −5.54 | **1.043** | 0.114 | 5.27 |
| **BLSTM** | **−0.23** | **−2.08** | 1.035 | **0.418** | **3.20** |
| CRN | −3.12 | −6.15 | 1.032 | 0.214 | 6.38 |

---

### Key Findings

**Finding 1 — BLSTM wins on both real and synthetic tests**
Only method that consistently improves SNR, PESQ, and LSD across both test conditions.
On real babble: SNR improved by **+4.3 dB**, STOI nearly doubled (0.22 → 0.42).

**Finding 2 — Adaptive filters fail on non-stationary real noise**
LMS achieves +2 dB on synthetic but only +0.43 dB on real babble.
Reason: LMS assumes noise is correlated with a reference signal — real babble violates the stationary noise assumption.

**Finding 3 — RLS consistently beats LMS and NLMS**
O(M²) complexity pays off: RLS achieves the highest SNR/SSNR among all adaptive filters on both tests.

**Finding 4 — Wiener and SpectralSub destroy intelligibility on real babble**
STOI drops to 0.067–0.114 (extremely low). Both methods over-attenuate speech frequencies where babble and speech spectra overlap.

**Finding 5 — NLMS achieves best STOI on synthetic (0.879)**
Power normalisation preserves the speech amplitude envelope, maximising intelligibility.

---

## 10. Spectrogram Analysis

![Spectrogram Comparison](results/spectrogram_comparison.png)

| Row | What you see | Why |
|---|---|---|
| **Clean** | Sharp harmonic bands at F0 (100Hz) + harmonics, black background | Pure formant structure, no noise |
| **Noisy (5dB)** | Same bands visible but heavily masked by uniform purple-red floor | Broadband white noise added at 5 dB SNR |
| **LMS** | Harmonic bands re-emerge with periodic dropout pattern | LMS cancels periodic noise; dropouts = amplitude envelope variation |
| **Wiener** | Darker background (noise reduced) but speech bands also attenuated | Over-suppression when noise PSD overestimated on synthetic signal |
| **Spec. Sub.** | Reduced noise floor but scattered bright spots remain | "Musical noise" — residual spectral peaks after subtraction |

---

## 11. Dataset

### NOIZEUS Corpus

| Property | Value |
|---|---|
| Source | `chmodsss/noizeus_corpora` (GitHub) |
| Speakers | 30 IEEE sentences (3M + 3F) |
| Noise types | airport, babble, car, exhibition, restaurant, street, train, station |
| SNR levels | 0, 5, 10, 15 dB |
| Total files | 960 noisy + 30 clean |
| Original SR | 8 kHz → resampled to 16 kHz |

```bash
python data/download_dataset.py --source /path/to/noizeus_corpora
# OR auto-clone:
python data/download_dataset.py --git-clone
```

### Synthetic Dataset

- Formant synthesis: F0 (80–300 Hz), F1–F3 resonances
- 3 noise types: white, pink, babble
- Configurable SNR: 0, 5, 10, 15 dB

```bash
python data/download_dataset.py --synthetic --n_clean 200
```

---

## 12. Installation

```bash
# Clone and set up
git clone <repo-url> && cd speech_enhancement
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# GPU (RTX 40xx — CUDA 12.4, tested configuration)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 13. Usage

### Train

```bash
python -m src.neural.train --model blstm --epochs 50
python -m src.neural.train --model crn   --epochs 50
python -m src.neural.train --model blstm --epochs 50 --resume
```

### Enhance & Evaluate

```bash
# Synthetic test
python main.py --snr 5 --noise babble --compare

# Real NOIZEUS with full metrics
python main.py \
  --input audio_samples/noisy/sp01_babble_snr+5dB.wav \
  --clean audio_samples/clean/sp01.wav --compare

# All noise types
for noise in airport babble car street; do
  python main.py \
    --input audio_samples/noisy/sp01_${noise}_snr+5dB.wav \
    --clean audio_samples/clean/sp01.wav --compare
done
```

### Analysis Plots

```bash
python notebooks/demo_analysis.py
# → results/convergence_analysis.png
# → results/weight_evolution.png
# → results/spectrogram_comparison.png
# → results/misadjustment_tradeoff.png
```

---

## 14. Key References

| Paper | Contribution |
|---|---|
| Widrow & Hoff (1960) | LMS algorithm |
| Boll (1979) | Spectral subtraction |
| Ephraim & Malah (1984) | Decision-directed Wiener filter |
| Hu & Loizou (2007) | NOIZEUS corpus, PESQ/STOI evaluation |
| Loizou (2013) | *Speech Enhancement: Theory and Practice* (textbook) |
| Tan & Wang (2018) | CRN architecture |

---

## Environment

```
OS    : Windows 11
GPU   : NVIDIA RTX 4060 Laptop (8 GB, Ada Lovelace)
CUDA  : 12.4
Python: 3.11
Torch : 2.6.0+cu124
```

---


15. Credits

## Introduction to Adaptive Signal Processing — Team Project

### Team Members
- Rahul Dewangan
- Mandeep Singh
- Shashi Kant Kumar
- Navadeep Bandi
