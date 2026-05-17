"""
Central Configuration for Speech Signal Enhancement Project
===========================================================
Adaptive + Neural Methods
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
AUDIO_DIR      = os.path.join(BASE_DIR, "audio_samples")
CLEAN_DIR      = os.path.join(AUDIO_DIR, "clean")
NOISY_DIR      = os.path.join(AUDIO_DIR, "noisy")
ENHANCED_DIR   = os.path.join(AUDIO_DIR, "enhanced")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

for d in [DATA_DIR, RESULTS_DIR, AUDIO_DIR, CLEAN_DIR,
          NOISY_DIR, ENHANCED_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000          # Hz
FRAME_LENGTH  = 512            # samples  (~32 ms at 16 kHz)
HOP_LENGTH    = 128            # samples  (~8 ms)
N_FFT         = 512
N_MELS        = 80
DURATION      = 4.0            # seconds per training clip
WIN_TYPE      = "hann"

# ─── Noise ───────────────────────────────────────────────────────────────────
SNR_LEVELS_DB = [-5, 0, 5, 10, 15, 20]   # dB for evaluation
TRAIN_SNR_DB  = [0, 5, 10, 15]            # dB for training data

# ─── Adaptive Filters ────────────────────────────────────────────────────────
LMS_MU         = 0.01     # step size
NLMS_MU        = 0.5      # normalised step size
NLMS_EPSILON   = 1e-8     # stability constant
RLS_LAMBDA     = 0.99     # forgetting factor
RLS_DELTA      = 1.0      # initial covariance
FILTER_ORDER   = 32       # tap length for all adaptive filters

# ─── Neural Model ────────────────────────────────────────────────────────────
LSTM_HIDDEN    = 256
LSTM_LAYERS    = 3
CNN_CHANNELS   = [1, 16, 32, 64, 32, 16, 1]
DROPOUT        = 0.2
INPUT_FEATURES = N_FFT // 2 + 1   # 257 STFT bins

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 10          # early stopping
CLIP_GRAD_NORM = 5.0

# ─── Dataset (NOIZEUS-style) ─────────────────────────────────────────────────
NOIZEUS_URL    = "https://ecs.utdallas.edu/loizou/speech/noizeus.zip"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

NOISE_TYPES    = ["white", "pink", "babble", "car", "restaurant", "street"]


# ─── GPU / Torch (lazy init — only when torch is actually used) ──────────────
def configure_torch():
    """Call this once before training / inference to enable GPU optimisations."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
    except ImportError:
        pass
