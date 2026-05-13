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

# ── Non-IID e FL Adaptativo ───────────────────────────────────────────────────
DIRICHLET_ALPHA = 0.5  # concentração Dirichlet (menor = mais heterogêneo)
DRIFT_CLIENTS = 3  # clientes que observam o drift (de NUM_CLIENTS)
DETECTOR_WINDOW = 3  # janela (rodadas) para detecção de queda
DETECTOR_THRESHOLD = 4.0  # queda de acurácia (p.p.) para disparar alerta
BOOST_LR_FACTOR = 5.0  # multiplicador de LR após detecção de drift
BOOST_ROUNDS = 4  # rodadas com LR elevada após detecção

# ── Dataset ──────────────────────────────────────────────────────────────────
N_TRAIN = 2000  # amostras totais de treino (divididas entre clientes)
N_TEST = 500  # amostras de teste
FEATURE_DIM = 20
SPAM_RATIO = 0.4  # 40 % de spam

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
