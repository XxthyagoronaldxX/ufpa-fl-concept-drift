"""
config.py
─────────
Hiperparâmetros e constantes globais do experimento.
"""

import os

import torch

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42

# ── Federated Learning ───────────────────────────────────────────────────────
NUM_CLIENTS = 4  # 1 cliente por local (Location1..Location4)
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
DRIFT_REFERENCE_SIZE = 2
DRIFT_WINDOW_SIZE = 3
# Aumento mínimo de erro (em pontos de MAE×100, ou seja %) para sinalizar drift.
DRIFT_MIN_RISE_PP = 4.0
DRIFT_KS_THRESHOLD = 0.35
DRIFT_MEAN_SHIFT_THRESHOLD = 0.18
DRIFT_DETECTOR_COOLDOWN = 0

# Corretores disponíveis:
#   "learning_rate", "epochs", "recent_replay", "severity_adaptive"
DRIFT_CORRECTOR_TYPE = "severity_adaptive"
DRIFT_CORRECTION_COOLDOWN = 2
DRIFT_LR_MULTIPLIER = 3.0
DRIFT_EXTRA_EPOCHS = 3
DRIFT_REPLAY_MEMORY_SIZE = 8
DRIFT_REPLAY_RATIO = 0.5

# Concept Ensemble — mantém um snapshot de modelo por fase observada.
# Se a fingerprint da rodada atual estiver a uma distância L2 inferior ao
# limiar, recuperamos o expert daquela fase em vez de criar um novo.
DRIFT_ENSEMBLE_ENABLED = True
DRIFT_ENSEMBLE_THRESHOLD = 0.05

# Acurácia tolerante (ε-accuracy): fração de previsões com |ŷ − y| < tolerância.
# 0,10 = 10 p.p. na escala normalizada de Power; casa com o regime de MAE atual.
TOLERANCE_ACC = 0.10

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# 6 numéricas brutas + 4 sin/cos de direção (10m e 100m) + 5 derivadas físicas
# (v³ a 100m, densidade do ar, ρ·v³, hora sin/cos).
FEATURE_DIM = 15
TRAIN_FRACTION = 0.8  # split cronológico por janela sazonal

# Eixos sazonais para o concept drift (hemisfério norte; o dataset é local).
SUMMER_MONTHS = [6, 7, 8]  # JJA — fase A
WINTER_MONTHS = [12, 1, 2]  # DJF — fase B

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
