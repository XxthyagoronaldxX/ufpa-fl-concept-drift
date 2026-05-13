"""
federated.py
────────────
Núcleo do Federated Learning: treino local, agregação FedAvg, avaliação
e detector de concept drift baseado em janela deslizante.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from config import BATCH_SIZE, DEVICE

# ── Detector de Concept Drift ─────────────────────────────────────────────────


class DriftDetector:
    """Detector de drift baseado em queda de acurácia em janela deslizante.

    Emite alerta quando a média das últimas `window` rodadas cai mais que
    `threshold` pontos percentuais em relação ao pico de acurácia observado.

    Args:
        window:    número de rodadas recentes a considerar.
        threshold: queda mínima (p.p.) para disparar o alerta.
    """

    def __init__(self, window: int = 3, threshold: float = 4.0):
        self.window = window
        self.threshold = threshold
        self._history: list[float] = []
        self._peak: float = 0.0

    def update(self, acc: float) -> bool:
        """Registra acurácia e retorna True se drift for detectado."""
        self._history.append(acc)
        self._peak = max(self._peak, acc)

        if len(self._history) < self.window:
            return False

        recent_avg = float(np.mean(self._history[-self.window :]))
        return (self._peak - recent_avg) > self.threshold

    def reset_peak(self) -> None:
        """Reinicia o pico de referência após adaptação ao novo padrão."""
        if self._history:
            self._peak = self._history[-1]


def local_train(global_model: nn.Module, dataset: TensorDataset, epochs: int, lr: float) -> tuple[dict, int]:
    """Treina uma cópia do modelo global nos dados locais de um cliente.

    Returns:
        (state_dict, n_samples) — pesos atualizados e tamanho do dataset local.
    """
    model = deepcopy(global_model).to(DEVICE)
    model.train()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

    return model.state_dict(), len(dataset)


def fed_avg(global_weights: dict, client_updates: list) -> dict:
    """Agrega os pesos dos clientes usando a média ponderada (FedAvg).

    Args:
        global_weights:  state_dict do modelo global (usado como template).
        client_updates:  lista de (state_dict, n_samples) de cada cliente.

    Returns:
        Novo state_dict agregado.
    """
    total = sum(sz for _, sz in client_updates)
    agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_weights.items()}

    for weights, sz in client_updates:
        weight = sz / total
        for k in agg:
            agg[k] += weights[k].float() * weight

    return agg


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset) -> tuple[float, float]:
    """Avalia o modelo em um dataset.

    Returns:
        (accuracy %, f1-score %) — ambos em percentual.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds, all_labels = [], []
    for X_b, y_b in loader:
        preds = model(X_b.to(DEVICE)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = 100.0 * f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1
