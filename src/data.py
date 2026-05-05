"""
data.py
───────
Geração de dados sintéticos de e-mail (spam/ham) e utilitários de dataset.

Features (20 dimensões):
  Idx  Nome               Descrição
  ─── ──────────────────  ────────────────────────────────────────────────
   0  word_free           Frequência de "free"
   1  word_win            Frequência de "win"
   2  word_prize          Frequência de "prize"
   3  word_click          Frequência de "click here"
   4  word_offer          Frequência de "limited offer"
   5  word_crypto         Frequência de "crypto/bitcoin"
   6  word_investment     Frequência de "investment opportunity"
   7  word_profit         Frequência de "profit/return"
   8  word_urgent         Frequência de "urgent/act now"
   9  word_verify         Frequência de "verify your account"
  10  num_links           Quantidade de links no corpo
  11  num_exclamation     Quantidade de "!"
  12  email_length        Comprimento normalizado do e-mail
  13  caps_ratio          Proporção de letras maiúsculas
  14  has_unsubscribe     Possui link de descadastro (0/1)
  15  reply_to_diff       Reply-To diferente do remetente (0/1)
  16  html_heavy          E-mail com muito HTML/CSS inline (0/1)
  17  num_images          Quantidade de imagens embutidas
  18  sender_known        Remetente está na lista de contatos (0/1)
  19  subject_all_caps    Assunto todo em maiúsculas (0/1)

Perfis de spam:
  Fase A — clássico : abusa das features 0–4  (free/win/prize)
  Fase B — moderno  : abusa das features 5–9  (crypto/phishing), estrutura
                      propositalmente próxima ao ham para forçar o modelo a
                      aprender as novas palavras-chave, tornando o drift visível.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from config import FEATURE_DIM, SPAM_RATIO

# ── Nomes das features ───────────────────────────────────────────────────────
FEATURE_NAMES = [
    "word_free",
    "word_win",
    "word_prize",
    "word_click",
    "word_offer",
    "word_crypto",
    "word_investment",
    "word_profit",
    "word_urgent",
    "word_verify",
    "num_links",
    "num_exclamation",
    "email_length",
    "caps_ratio",
    "has_unsubscribe",
    "reply_to_diff",
    "html_heavy",
    "num_images",
    "sender_known",
    "subject_all_caps",
]


# ── Geradores de features brutas ─────────────────────────────────────────────


def _ham_features(n: int) -> np.ndarray:
    """Features de e-mails legítimos (ham)."""
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    x[:, 0:10] = np.random.exponential(0.05, (n, 10))  # palavras-spam raras
    x[:, 10] = np.random.poisson(1.5, n)  # poucos links
    x[:, 11] = np.random.poisson(0.5, n)  # raramente "!"
    x[:, 12] = np.clip(np.random.normal(0.50, 0.15, n), 0, 1)
    x[:, 13] = np.clip(np.random.normal(0.05, 0.03, n), 0, 1)
    x[:, 14] = np.random.binomial(1, 0.70, n)  # geralmente tem unsubscribe
    x[:, 15] = np.random.binomial(1, 0.05, n)
    x[:, 16] = np.random.binomial(1, 0.30, n)
    x[:, 17] = np.random.poisson(0.8, n)
    x[:, 18] = np.random.binomial(1, 0.80, n)  # remetente conhecido
    x[:, 19] = np.random.binomial(1, 0.02, n)
    return x


def _spam_phase_a(n: int) -> np.ndarray:
    """Spam clássico — abusa de free/win/prize/click/offer (features 0–4)."""
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    x[:, 0:5] = np.random.exponential(1.2, (n, 5))  # palavras clássicas altas
    x[:, 5:10] = np.random.exponential(0.05, (n, 5))  # palavras modernas baixas
    x[:, 10] = np.random.poisson(8, n)
    x[:, 11] = np.random.poisson(5, n)
    x[:, 12] = np.clip(np.random.normal(0.70, 0.15, n), 0, 1)
    x[:, 13] = np.clip(np.random.normal(0.35, 0.10, n), 0, 1)
    x[:, 14] = np.random.binomial(1, 0.20, n)
    x[:, 15] = np.random.binomial(1, 0.75, n)
    x[:, 16] = np.random.binomial(1, 0.85, n)
    x[:, 17] = np.random.poisson(5, n)
    x[:, 18] = np.random.binomial(1, 0.05, n)
    x[:, 19] = np.random.binomial(1, 0.70, n)
    return x


def _spam_phase_b(n: int) -> np.ndarray:
    """Spam moderno furtivo — crypto/phishing (features 5–9 altas).

    Estrutura (features 10-19) propositalmente próxima ao ham para forçar
    o modelo a aprender as novas palavras-chave, tornando o drift visível.
    """
    x = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    x[:, 0:5] = np.random.exponential(0.05, (n, 5))  # palavras clássicas baixas
    x[:, 5:10] = np.random.exponential(1.2, (n, 5))  # palavras modernas altas
    x[:, 10] = np.random.poisson(1.8, n)  # poucos links (≈ ham=1.5)
    x[:, 11] = np.random.poisson(0.6, n)  # raramente "!" (≈ ham=0.5)
    x[:, 12] = np.clip(np.random.normal(0.52, 0.15, n), 0, 1)
    x[:, 13] = np.clip(np.random.normal(0.07, 0.03, n), 0, 1)
    x[:, 14] = np.random.binomial(1, 0.60, n)
    x[:, 15] = np.random.binomial(1, 0.35, n)
    x[:, 16] = np.random.binomial(1, 0.35, n)
    x[:, 17] = np.random.poisson(0.9, n)
    x[:, 18] = np.random.binomial(1, 0.40, n)
    x[:, 19] = np.random.binomial(1, 0.10, n)
    return x


# ── Normalização global ──────────────────────────────────────────────────────
# Calculada uma única vez a partir de amostras representativas de todas as
# fases, garantindo escala consistente entre treino e teste.

_ref_X = np.vstack(
    [
        _ham_features(3000),
        _spam_phase_a(2000),
        _spam_phase_b(2000),
    ]
).astype(np.float32)

GLOBAL_MIN = _ref_X.min(axis=0)
GLOBAL_MAX = _ref_X.max(axis=0)
del _ref_X


def _normalize(X: np.ndarray) -> np.ndarray:
    return (X - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + 1e-8)


# ── Fábrica de datasets ──────────────────────────────────────────────────────


def make_dataset(n: int, spam_phase: str = "A", alpha: float = 0.0) -> TensorDataset:
    """Gera um TensorDataset de e-mails sintéticos.

    Args:
        n:          Número total de amostras.
        spam_phase: "A" | "B" | "mixed"
        alpha:      Proporção de spam fase B quando spam_phase="mixed".
                    0.0 → 100 % fase A;  1.0 → 100 % fase B.
    """
    n_spam = int(n * SPAM_RATIO)
    n_ham = n - n_spam

    ham_x = _ham_features(n_ham)

    if spam_phase == "A":
        spam_x = _spam_phase_a(n_spam)
    elif spam_phase == "B":
        spam_x = _spam_phase_b(n_spam)
    else:  # "mixed"
        n_b = int(n_spam * alpha)
        n_a = n_spam - n_b
        parts = []
        if n_a > 0:
            parts.append(_spam_phase_a(n_a))
        if n_b > 0:
            parts.append(_spam_phase_b(n_b))
        spam_x = np.concatenate(parts)

    X = np.vstack([ham_x, spam_x]).astype(np.float32)
    y = np.array([0] * n_ham + [1] * n_spam, dtype=np.int64)

    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    X = _normalize(X).clip(0, 1)

    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def split_iid(dataset: TensorDataset, num_clients: int) -> list:
    """Divide o dataset de forma IID entre os clientes."""
    X, y = dataset.tensors
    n = len(y)
    idx = np.random.permutation(n)
    chunk = n // num_clients
    return [
        TensorDataset(
            X[idx[i * chunk : (i + 1) * chunk]],
            y[idx[i * chunk : (i + 1) * chunk]],
        )
        for i in range(num_clients)
    ]
