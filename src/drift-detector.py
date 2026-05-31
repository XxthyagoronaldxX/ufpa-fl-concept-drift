"""
drift-detector.py
─────────────────
Detectores de concept drift para o experimento de spam federado.

Os detectores compartilham uma interface simples: recebem a rodada, as métricas
de desempenho e, quando disponível, o dataset da rodada. O resultado informa se
houve drift, a severidade e a mensagem que será exibida no log do experimento.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np
from torch.utils.data import TensorDataset

try:
    from scipy.stats import ks_2samp
except Exception:  # scipy não é dependência obrigatória do projeto
    ks_2samp = None

from config import (
    DRIFT_DETECTOR_COOLDOWN,
    DRIFT_DETECTOR_POLICY,
    DRIFT_DETECTOR_TYPE,
    DRIFT_KS_THRESHOLD,
    DRIFT_MEAN_SHIFT_THRESHOLD,
    DRIFT_MIN_DROP_PP,
    DRIFT_REFERENCE_SIZE,
    DRIFT_WINDOW_SIZE,
)


@dataclass
class DriftDetectionResult:
    detected: bool
    round_id: int
    detector_name: str
    severity: str = "none"
    score: float = 0.0
    message: str = ""


class BaseDriftDetector:
    name = "base"

    def update(self, round_id: int, accuracy: float, f1: float, dataset: TensorDataset | None = None) -> DriftDetectionResult:
        raise NotImplementedError


def _severity(score: float, low: float, medium: float) -> str:
    if score >= medium:
        return "alta"
    if score >= low:
        return "média"
    return "baixa"


def _features_from_dataset(dataset: TensorDataset | None) -> np.ndarray | None:
    if dataset is None:
        return None
    x = dataset.tensors[0]
    return x.detach().cpu().numpy().astype(np.float32)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if ks_2samp is not None:
        return float(ks_2samp(a, b).statistic)

    values = np.sort(np.unique(np.concatenate([a, b])))
    if len(values) == 0:
        return 0.0
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / max(1, len(a))
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / max(1, len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


class PerformanceDropDetector(BaseDriftDetector):
    """Detecta drift quando a média recente da métrica cai frente à referência."""

    name = "performance"

    def __init__(self, reference_size: int = DRIFT_REFERENCE_SIZE, window_size: int = DRIFT_WINDOW_SIZE, min_drop_pp: float = DRIFT_MIN_DROP_PP, metric: str = "f1"):
        self.reference_size = reference_size
        self.window_size = window_size
        self.min_drop_pp = min_drop_pp
        self.metric = metric
        self.values = []

    def update(self, round_id: int, accuracy: float, f1: float, dataset: TensorDataset | None = None) -> DriftDetectionResult:
        value = f1 if self.metric == "f1" else accuracy
        self.values.append(float(value))

        needed = self.reference_size + self.window_size
        if len(self.values) < needed:
            return DriftDetectionResult(False, round_id, self.name, message="coletando referência de desempenho")

        reference = np.mean(self.values[: self.reference_size])
        recent = np.mean(self.values[-self.window_size :])
        drop = max(0.0, reference - recent)
        detected = drop >= self.min_drop_pp
        severity = _severity(drop, self.min_drop_pp, self.min_drop_pp * 1.8) if detected else "none"
        message = f"queda de {self.metric.upper()}={drop:.1f} p.p. (ref={reference:.1f}, recente={recent:.1f})"
        return DriftDetectionResult(detected, round_id, self.name, severity, drop, message)


class FeatureKSTestDetector(BaseDriftDetector):
    """Compara distribuições das features via estatística Kolmogorov-Smirnov."""

    name = "ks_features"

    def __init__(self, reference_size: int = DRIFT_REFERENCE_SIZE, threshold: float = DRIFT_KS_THRESHOLD):
        self.reference_size = reference_size
        self.threshold = threshold
        self.reference_batches = []

    def update(self, round_id: int, accuracy: float, f1: float, dataset: TensorDataset | None = None) -> DriftDetectionResult:
        x = _features_from_dataset(dataset)
        if x is None:
            return DriftDetectionResult(False, round_id, self.name, message="dataset indisponível")

        if len(self.reference_batches) < self.reference_size:
            self.reference_batches.append(x)
            return DriftDetectionResult(False, round_id, self.name, message="coletando referência de features")

        reference = np.vstack(self.reference_batches)
        stats = [_ks_statistic(reference[:, i], x[:, i]) for i in range(x.shape[1])]
        score = float(np.max(stats))
        detected = score >= self.threshold
        severity = _severity(score, self.threshold, self.threshold * 1.6) if detected else "none"
        message = f"KS máximo={score:.3f} (limiar={self.threshold:.3f})"
        return DriftDetectionResult(detected, round_id, self.name, severity, score, message)


class MeanShiftDetector(BaseDriftDetector):
    """Detecta deslocamento global pela distância entre médias das features."""

    name = "mean_shift"

    def __init__(self, reference_size: int = DRIFT_REFERENCE_SIZE, threshold: float = DRIFT_MEAN_SHIFT_THRESHOLD):
        self.reference_size = reference_size
        self.threshold = threshold
        self.reference_batches = []

    def update(self, round_id: int, accuracy: float, f1: float, dataset: TensorDataset | None = None) -> DriftDetectionResult:
        x = _features_from_dataset(dataset)
        if x is None:
            return DriftDetectionResult(False, round_id, self.name, message="dataset indisponível")

        if len(self.reference_batches) < self.reference_size:
            self.reference_batches.append(x)
            return DriftDetectionResult(False, round_id, self.name, message="coletando referência de médias")

        reference = np.vstack(self.reference_batches)
        score = float(np.linalg.norm(x.mean(axis=0) - reference.mean(axis=0)) / np.sqrt(x.shape[1]))
        detected = score >= self.threshold
        severity = _severity(score, self.threshold, self.threshold * 1.7) if detected else "none"
        message = f"deslocamento médio={score:.3f} (limiar={self.threshold:.3f})"
        return DriftDetectionResult(detected, round_id, self.name, severity, score, message)


class CompositeDriftDetector(BaseDriftDetector):
    """Combina múltiplos detectores por política 'any' ou 'majority'."""

    name = "composite"

    def __init__(self, detectors: Iterable[BaseDriftDetector], policy: str = DRIFT_DETECTOR_POLICY, cooldown_rounds: int = DRIFT_DETECTOR_COOLDOWN):
        self.detectors = list(detectors)
        self.policy = policy
        self.cooldown_rounds = cooldown_rounds
        self.cooldown = 0
        self.last_results = deque(maxlen=8)

    def update(self, round_id: int, accuracy: float, f1: float, dataset: TensorDataset | None = None) -> DriftDetectionResult:
        results = [detector.update(round_id, accuracy, f1, dataset) for detector in self.detectors]
        self.last_results.extend(results)
        votes = [result for result in results if result.detected]

        if self.cooldown > 0:
            self.cooldown -= 1
            messages = "; ".join(result.message for result in results)
            return DriftDetectionResult(False, round_id, self.name, message=f"cooldown do detector; {messages}")

        if self.policy == "majority":
            detected = len(votes) >= (len(results) // 2 + 1)
        else:
            detected = len(votes) > 0

        if not detected:
            messages = "; ".join(result.message for result in results)
            return DriftDetectionResult(False, round_id, self.name, message=messages)

        strongest = max(votes, key=lambda result: result.score)
        self.cooldown = self.cooldown_rounds
        names = ", ".join(result.detector_name for result in votes)
        message = f"detectado por {names}; principal: {strongest.message}"
        return DriftDetectionResult(True, round_id, self.name, strongest.severity, strongest.score, message)


def build_drift_detector(detector_type: str = DRIFT_DETECTOR_TYPE) -> BaseDriftDetector:
    if detector_type == "performance":
        return PerformanceDropDetector()
    if detector_type == "ks":
        return FeatureKSTestDetector()
    if detector_type == "mean_shift":
        return MeanShiftDetector()
    if detector_type == "composite":
        return CompositeDriftDetector(
            [
                PerformanceDropDetector(),
                FeatureKSTestDetector(),
                MeanShiftDetector(),
            ]
        )
    raise ValueError(f"Detector de drift desconhecido: {detector_type}")
