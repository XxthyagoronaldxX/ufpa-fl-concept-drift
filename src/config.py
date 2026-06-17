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
NUM_ROUNDS = 48  # 7 warmup verão + 10 ciclos JJA↔DJF de CYCLE_LEN rodadas (8..47)

TEST_RATE = 0.2  # Fração reservada para teste (split cronológico)
GLOBAL_ROUNDS = 20

LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ── Concept Drift ────────────────────────────────────────────────────────────
DRIFT_ROUND = 8  # rodada em que o drift começa
CYCLE_LEN = 4  # rodadas por fase no drift recorrente
# Detector por janela móvel: dispara quando MAE_atual > média(janela) × limiar.
DRIFT_WINDOW_SIZE = 5
DRIFT_THRESHOLD = 1.3

# SeasonalReplayBuffer: amostras máximas guardadas por estação, em cada cliente.
REPLAY_BUFFER_SIZE = 500
# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# 6 numéricas brutas + 4 sin/cos de direção (10m e 100m) + 5 derivadas físicas
# (v³ a 100m, densidade do ar, ρ·v³, hora sin/cos).
FEATURE_DIM = 8
TRAIN_FRACTION = 0.8  # split cronológico por janela sazonal
MAX_TRAIN_PER_CLIENT = 2000  # subsample por cliente/estação (None = sem cap)

# Eixos sazonais para o concept drift (hemisfério norte; o dataset é local).
SUMMER_MONTHS = [6, 7, 8]  # JJA — fase A
WINTER_MONTHS = [12, 1, 2]  # DJF — fase B

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
