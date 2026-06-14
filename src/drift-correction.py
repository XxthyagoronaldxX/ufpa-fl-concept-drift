"""
drift-correction.py
───────────────────
Estratégias de correção/adaptação após detecção de concept drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
from torch.utils.data import TensorDataset

from config import (
    DRIFT_CORRECTION_COOLDOWN,
    DRIFT_CORRECTOR_TYPE,
    DRIFT_ENSEMBLE_THRESHOLD,
    DRIFT_EXTRA_EPOCHS,
    DRIFT_LR_MULTIPLIER,
    DRIFT_REPLAY_MEMORY_SIZE,
    DRIFT_REPLAY_RATIO,
    LEARNING_RATE,
    LOCAL_EPOCHS,
)


@dataclass
class CorrectionState:
    active: bool
    learning_rate: float
    local_epochs: int
    replay_dataset: TensorDataset | None = None
    replay_ratio: float = 0.0
    remaining_rounds: int = 0
    message: str = ""


def _fingerprint(datasets: list[TensorDataset]) -> np.ndarray:
    """Resume um snapshot de clientes como vetor médio de features.

    Permite identificar a fase atual (verão/inverno/...) sem rótulos explícitos:
    snapshots da mesma fase têm fingerprints próximos em distância L2.
    """
    arrays = [ds.tensors[0].detach().cpu().numpy() for ds in datasets]
    stacked = np.concatenate(arrays, axis=0).astype(np.float32)
    return stacked.mean(axis=0)


class BaseDriftCorrector:
    name = "base"

    def __init__(self, base_lr: float = LEARNING_RATE, base_epochs: int = LOCAL_EPOCHS, cooldown_rounds: int = DRIFT_CORRECTION_COOLDOWN):
        self.base_lr = base_lr
        self.base_epochs = base_epochs
        self.cooldown_rounds = cooldown_rounds
        self.remaining_rounds = 0
        # Memória como (fingerprint, datasets) — permite escolher snapshot da fase atual.
        self.memory: deque[tuple[np.ndarray, list[TensorDataset]]] = deque(maxlen=DRIFT_REPLAY_MEMORY_SIZE)
        self.active_severity = "baixa"

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        raise NotImplementedError

    def remember(self, client_datasets: list[TensorDataset]) -> None:
        # Congela o buffer enquanto a correção está ativa: preserva o snapshot
        # da fase pré-drift para servir como replay anti-esquecimento.
        if self.remaining_rounds > 0:
            return
        fingerprint = _fingerprint(client_datasets)
        self.memory.append((fingerprint, client_datasets))

    def apply_replay(self, client_datasets: list[TensorDataset], replay_ratio: float = DRIFT_REPLAY_RATIO) -> list[TensorDataset]:
        if not self.memory or replay_ratio <= 0:
            return client_datasets

        # Replay consciente da fase: escolhe o snapshot com fingerprint mais
        # parecida com a rodada atual. Em drift recorrente, isso recupera o
        # snapshot da MESMA fase (verão↔verão, inverno↔inverno).
        current_fp = _fingerprint(client_datasets)
        replay_clients = min(
            self.memory,
            key=lambda entry: float(np.linalg.norm(current_fp - entry[0])),
        )[1]

        mixed = []
        for current, replay in zip(client_datasets, replay_clients):
            cur_x, cur_y = current.tensors
            rep_x, rep_y = replay.tensors
            take = min(len(rep_y), max(1, int(len(cur_y) * replay_ratio)))
            mixed.append(
                TensorDataset(
                    torch.cat([cur_x, rep_x[:take]], dim=0),
                    torch.cat([cur_y, rep_y[:take]], dim=0),
                )
            )
        return mixed

    def _tick(self) -> bool:
        if self.remaining_rounds <= 0:
            return False
        self.remaining_rounds -= 1
        return True


class AdaptiveLearningRateCorrector(BaseDriftCorrector):
    name = "learning_rate"

    def __init__(self, lr_multiplier: float = DRIFT_LR_MULTIPLIER, **kwargs):
        super().__init__(**kwargs)
        self.lr_multiplier = lr_multiplier

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        if detection_result.detected:
            self.remaining_rounds = self.cooldown_rounds
        active = self._tick()
        lr = self.base_lr * self.lr_multiplier if active else self.base_lr
        message = f"LR adaptativo={lr:.4f}" if active else ""
        return CorrectionState(active, lr, self.base_epochs, remaining_rounds=self.remaining_rounds, message=message)


class AdaptiveEpochCorrector(BaseDriftCorrector):
    name = "epochs"

    def __init__(self, extra_epochs: int = DRIFT_EXTRA_EPOCHS, **kwargs):
        super().__init__(**kwargs)
        self.extra_epochs = extra_epochs

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        if detection_result.detected:
            self.remaining_rounds = self.cooldown_rounds
        active = self._tick()
        epochs = self.base_epochs + self.extra_epochs if active else self.base_epochs
        message = f"épocas adaptativas={epochs}" if active else ""
        return CorrectionState(active, self.base_lr, epochs, remaining_rounds=self.remaining_rounds, message=message)


class RecentReplayCorrector(BaseDriftCorrector):
    name = "recent_replay"

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        if detection_result.detected:
            self.remaining_rounds = self.cooldown_rounds
        active = self._tick()
        message = "replay de dados recentes ativo" if active and self.memory else ""
        replay_ratio = DRIFT_REPLAY_RATIO if active else 0.0
        return CorrectionState(active, self.base_lr, self.base_epochs, current_dataset if active else None, replay_ratio, self.remaining_rounds, message)


class SeverityBasedCorrector(BaseDriftCorrector):
    name = "severity_adaptive"

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        if detection_result.detected:
            self.remaining_rounds = self.cooldown_rounds
            self.active_severity = detection_result.severity

        active = self._tick()
        if not active:
            return CorrectionState(False, self.base_lr, self.base_epochs, remaining_rounds=0)

        severity = self.active_severity
        if severity == "alta":
            lr = self.base_lr * max(DRIFT_LR_MULTIPLIER, 2.2)
            epochs = self.base_epochs + max(DRIFT_EXTRA_EPOCHS, 2)
            replay_ratio = min(0.50, DRIFT_REPLAY_RATIO + 0.15)
        elif severity == "média":
            lr = self.base_lr * DRIFT_LR_MULTIPLIER
            epochs = self.base_epochs + DRIFT_EXTRA_EPOCHS
            replay_ratio = DRIFT_REPLAY_RATIO
        else:
            lr = self.base_lr * 1.3
            epochs = self.base_epochs
            replay_ratio = DRIFT_REPLAY_RATIO / 2

        replay = current_dataset if self.memory and replay_ratio > 0 else None
        message = f"correção {severity}: LR={lr:.4f}, épocas={epochs}, replay={replay_ratio:.2f}"
        return CorrectionState(True, lr, epochs, replay, replay_ratio, self.remaining_rounds, message)

    def apply_replay(self, client_datasets: list[TensorDataset], replay_ratio: float = DRIFT_REPLAY_RATIO) -> list[TensorDataset]:
        if self.remaining_rounds > 0:
            return super().apply_replay(client_datasets, replay_ratio)
        return client_datasets


def build_drift_corrector(corrector_type: str = DRIFT_CORRECTOR_TYPE) -> BaseDriftCorrector:
    if corrector_type == "learning_rate":
        return AdaptiveLearningRateCorrector()
    if corrector_type == "epochs":
        return AdaptiveEpochCorrector()
    if corrector_type == "recent_replay":
        return RecentReplayCorrector()
    if corrector_type == "severity_adaptive":
        return SeverityBasedCorrector()
    raise ValueError(f"Corretor de drift desconhecido: {corrector_type}")


class ConceptEnsemble:
    """Mantém um snapshot de modelo por fase de concept observada.

    A cada rodada, a fingerprint dos dados atuais é comparada com as fingerprints
    armazenadas. Se houver match (distância L2 < threshold), o snapshot
    correspondente é recuperado para servir de warm-start; caso contrário, um
    novo concept é criado. Após o treino, o snapshot do concept ativo é
    atualizado com os pesos da rodada.

    Resolve o problema estrutural do drift recorrente: o FedAvg aplicado a um
    único modelo força um compromisso entre concepts incompatíveis (verão vs.
    inverno). Com snapshots por concept, cada fase mantém seu próprio expert.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._snapshots: list[tuple[np.ndarray, dict]] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    def match(self, fingerprint: np.ndarray) -> tuple[int, float] | None:
        """Retorna (índice, distância) do snapshot mais próximo, ou None se vazio."""
        if not self._snapshots:
            return None
        distances = [float(np.linalg.norm(fingerprint - fp)) for fp, _ in self._snapshots]
        idx = int(np.argmin(distances))
        return idx, distances[idx]

    def select_or_create(self, fingerprint: np.ndarray, current_state: dict) -> tuple[int, bool]:
        """Encontra concept compatível ou registra um novo.

        Retorna (concept_id, was_created). Não modifica `current_state`; o
        caller decide se faz `model.load_state_dict(...)` quando o concept já
        existia (warm-start) ou se mantém o estado atual quando criado.
        """
        match = self.match(fingerprint)
        if match is not None and match[1] < self.threshold:
            return match[0], False

        # Novo concept: clona o estado atual como ponto de partida do expert.
        cloned = {k: v.detach().clone() for k, v in current_state.items()}
        self._snapshots.append((fingerprint.copy(), cloned))
        return len(self._snapshots) - 1, True

    def get_state(self, concept_id: int) -> dict:
        return self._snapshots[concept_id][1]

    def save(self, concept_id: int, state: dict) -> None:
        """Atualiza o snapshot do concept com os pesos pós-treino."""
        fingerprint, _ = self._snapshots[concept_id]
        cloned = {k: v.detach().clone() for k, v in state.items()}
        self._snapshots[concept_id] = (fingerprint, cloned)


def build_concept_ensemble(threshold: float = DRIFT_ENSEMBLE_THRESHOLD) -> ConceptEnsemble:
    return ConceptEnsemble(threshold=threshold)
