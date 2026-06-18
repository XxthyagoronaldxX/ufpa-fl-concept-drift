import os
import torch

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42

# ── Federated Learning ───────────────────────────────────────────────────────
NUM_CLIENTS = 4  # 1 cliente por local (Location1..Location4)
NUM_ROUNDS = 48  # 7 warmup verão + 10 ciclos JJA↔DJF de CYCLE_LEN rodadas (8..47)
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ── Drift Detector ────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = 0.3
DRIFT_BURN_IN = 3
DRIFT_DELTA = 0.01

# ── Drift Corrector ───────────────────────────────────────────────────────────
DRIFT_BUFFER_PER_SEASON = 500

# ── Drift General ─────────────────────────────────────────────────────────────
DRIFT_ROUND = 3
CYCLE_LEN = 3

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEATURE_DIM = 9
TRAIN_FRACTION = 0.7  # split cronológico por janela sazonal
MAX_TRAIN_PER_CLIENT = 2000  # subsample por cliente/estação (None = sem cap)

# Eixos sazonais para o concept drift (hemisfério norte; o dataset é local).
SUMMER_MONTHS = [6, 7, 8]  # JJA — fase A
WINTER_MONTHS = [12, 1, 2]  # DJF — fase B

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
