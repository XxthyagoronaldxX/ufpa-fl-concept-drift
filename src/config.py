"""
config.py
─────────
Hiperparâmetros e constantes globais do experimento.
"""

import torch

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42

# ── Federated Learning ───────────────────────────────────────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS = 25
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ── Concept Drift ────────────────────────────────────────────────────────────
DRIFT_ROUND = 8  # rodada em que o drift começa
CYCLE_LEN = 4  # rodadas por fase no drift recorrente

# Detectores disponíveis:
#   "performance", "ks", "mean_shift", "composite"
DRIFT_DETECTOR_TYPE = "composite"
DRIFT_DETECTOR_POLICY = "any"  # "any" dispara com 1 detector; "majority" exige maioria
DRIFT_REFERENCE_SIZE = 4
DRIFT_WINDOW_SIZE = 3
DRIFT_MIN_DROP_PP = 8.0
DRIFT_KS_THRESHOLD = 0.35
DRIFT_MEAN_SHIFT_THRESHOLD = 0.18
DRIFT_DETECTOR_COOLDOWN = 2

# Corretores disponíveis:
#   "learning_rate", "epochs", "recent_replay", "severity_adaptive"
DRIFT_CORRECTOR_TYPE = "severity_adaptive"
DRIFT_CORRECTION_COOLDOWN = 4
DRIFT_LR_MULTIPLIER = 1.8
DRIFT_EXTRA_EPOCHS = 1
DRIFT_REPLAY_MEMORY_SIZE = 2
DRIFT_REPLAY_RATIO = 0.35

# ── Dataset ──────────────────────────────────────────────────────────────────
N_TRAIN = 2000  # amostras totais de treino (divididas entre clientes)
N_TEST = 500  # amostras de teste
FEATURE_DIM = 20
SPAM_RATIO = 0.4  # 40 % de spam

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
