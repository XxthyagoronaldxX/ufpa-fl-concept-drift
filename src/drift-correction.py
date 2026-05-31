"""
drift-correction.py
───────────────────
Estratégias de correção/adaptação após detecção de concept drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch
from torch.utils.data import TensorDataset

from config import (
    DRIFT_CORRECTION_COOLDOWN,
    DRIFT_CORRECTOR_TYPE,
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


class BaseDriftCorrector:
    name = "base"

    def __init__(self, base_lr: float = LEARNING_RATE, base_epochs: int = LOCAL_EPOCHS, cooldown_rounds: int = DRIFT_CORRECTION_COOLDOWN):
        self.base_lr = base_lr
        self.base_epochs = base_epochs
        self.cooldown_rounds = cooldown_rounds
        self.remaining_rounds = 0
        self.memory = deque(maxlen=DRIFT_REPLAY_MEMORY_SIZE)
        self.active_severity = "baixa"

    def update(self, detection_result, current_dataset: TensorDataset | None = None) -> CorrectionState:
        raise NotImplementedError

    def remember(self, client_datasets: list[TensorDataset]) -> None:
        self.memory.append(client_datasets)

    def apply_replay(self, client_datasets: list[TensorDataset], replay_ratio: float = DRIFT_REPLAY_RATIO) -> list[TensorDataset]:
        if not self.memory or replay_ratio <= 0:
            return client_datasets

        replay_clients = self.memory[-1]
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
