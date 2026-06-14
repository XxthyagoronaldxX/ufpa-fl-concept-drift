"""
scenarios.py
────────────
Definição dos cenários de Federated Learning e execução por rodada,
aplicados a previsão de potência eólica.

Cenários (eixo de drift = sazonal):
  1. FL Padrão       — distribuição estacionária (somente verão)
  2. Drift Recorrente — alterna verão ↔ inverno em ciclos de CYCLE_LEN rodadas

A alternância recorrente é o único padrão fiel à natureza cíclica das estações
do ano; cenários de drift súbito ou unidirecional gradual não condizem com a
realidade meteorológica adotada como eixo de drift.

Cada cliente federado corresponde a uma das 4 localizações do dataset
(Location1..Location4) — federação não-IID. O dataset de teste de cada rodada
acompanha a distribuição vigente, tornando visível o impacto do drift.
"""

import importlib.util
import os
import sys

from config import (
    CYCLE_LEN,
    DEVICE,
    DRIFT_ENSEMBLE_ENABLED,
    DRIFT_ROUND,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_ROUNDS,
    SUMMER_MONTHS,
    WINTER_MONTHS,
)
from data import build_seasonal_pools
from federated_service import FederatedService
from model import WindPowerMLP


def _load_local_module(module_name: str, filename: str):
    """Carrega módulos locais cujos arquivos têm hífen no nome."""
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_drift_detector_module = _load_local_module("drift_detector", "drift-detector.py")
_drift_correction_module = _load_local_module("drift_correction", "drift-correction.py")

build_drift_detector = _drift_detector_module.build_drift_detector
build_drift_corrector = _drift_correction_module.build_drift_corrector
build_concept_ensemble = _drift_correction_module.build_concept_ensemble
_fingerprint = _drift_correction_module._fingerprint


# ── Preparação dos pools de dados ────────────────────────────────────────────


def build_data_pools() -> dict:
    """Pré-gera os pools de treino/teste (verão e inverno).

    A chave "clients_A" representa o pool de verão (JJA) e "clients_B" o pool
    de inverno (DJF). Os nomes A/B são mantidos como rótulos neutros para
    compatibilidade com os seletores do experimento.
    """
    return build_seasonal_pools(SUMMER_MONTHS, WINTER_MONTHS)


# ── Seletores de dados por rodada ────────────────────────────────────────────


def make_standard_fns(pools: dict):
    """Cenário 1 — sem drift: somente verão (distribuição estacionária)."""

    def get_train(rnd):
        return pools["clients_A"]

    def get_test(rnd):
        return pools["test_A"]

    return get_train, get_test


def make_recurrent_fns(pools: dict):
    """Cenário 2 — drift recorrente: alterna verão ↔ inverno em ciclos de CYCLE_LEN."""

    def _phase(rnd: int) -> str:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        return "B" if pos < CYCLE_LEN else "A"

    def get_train(rnd):
        if rnd < DRIFT_ROUND:
            return pools["clients_A"]
        return pools["clients_B"] if _phase(rnd) == "B" else pools["clients_A"]

    def get_test(rnd):
        if rnd < DRIFT_ROUND:
            return pools["test_A"]
        return pools["test_B"] if _phase(rnd) == "B" else pools["test_A"]

    return get_train, get_test


# ── Log por rodada ───────────────────────────────────────────────────────────


def _drift_note(scenario: str, rnd: int) -> str:
    """Retorna uma nota descritiva sobre o drift na rodada atual."""
    if "Recorrente" in scenario and rnd >= DRIFT_ROUND:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        phase = "inverno" if pos < CYCLE_LEN else "verão"
        return f"ciclo — fase {phase}"
    return ""


# ── Loop principal de FL ─────────────────────────────────────────────────────


def run_scenario(name: str, get_train_fn, get_test_fn, enable_correction: bool = True) -> dict:
    """Executa NUM_ROUNDS de FL e retorna históricos por rodada.

    Args:
        name:              Identificador do cenário (usado no log).
        get_train_fn:      callable(rnd) → list[TensorDataset]
        get_test_fn:       callable(rnd) → TensorDataset
        enable_correction: aplica correções adaptativas quando um drift é detectado.

    Returns:
        dict com chaves:
            - "mae", "rmse":  listas (uma entrada por rodada, em p.p. de Power).
            - "events":       lista de dicts com telemetria por rodada
              (`detected`, `severity`, `correction_active`, `lr`, `epochs`,
              `replay_ratio`, `phase`).
    """
    model = WindPowerMLP().to(DEVICE)
    mae_hist, rmse_hist, acc_hist, events = [], [], [], []
    detector = build_drift_detector()
    corrector = build_drift_corrector()
    ensemble = build_concept_ensemble() if (enable_correction and DRIFT_ENSEMBLE_ENABLED) else None
    correction_state = _drift_correction_module.CorrectionState(False, LEARNING_RATE, LOCAL_EPOCHS)

    print(f"\n{'═' * 64}")
    print(f"  Cenário: {name}")
    print(f"{'═' * 64}")
    print(f"  {'Rodada':>7}  │  {'MAE':>7}  │  {'RMSE':>7}  │  {'R²':>6}  │  {'Acur':>6}  │  Observação")
    print(f"  {'-' * 92}")

    for rnd in range(1, NUM_ROUNDS + 1):
        # Treino local + agregação FedAvg
        client_datasets = get_train_fn(rnd)

        # Concept ensemble: warm-start no expert da fase atual (se houver match)
        concept_id = None
        concept_created = False
        if ensemble is not None:
            fingerprint = _fingerprint(client_datasets)
            concept_id, concept_created = ensemble.select_or_create(fingerprint, model.state_dict())
            if not concept_created:
                model.load_state_dict(ensemble.get_state(concept_id))

        train_datasets = corrector.apply_replay(client_datasets, correction_state.replay_ratio) if enable_correction and correction_state.active else client_datasets
        updates = [FederatedService.local_train(model, ds, correction_state.local_epochs, correction_state.learning_rate) for ds in train_datasets]
        model.load_state_dict(FederatedService.fed_avg(model.state_dict(), updates))

        if ensemble is not None and concept_id is not None:
            ensemble.save(concept_id, model.state_dict())

        # Avaliação no dataset de teste da rodada atual
        test_dataset = get_test_fn(rnd)
        mae, rmse, r2, acc = FederatedService.evaluate(model, test_dataset)
        mae_hist.append(mae)
        rmse_hist.append(rmse)
        acc_hist.append(acc)

        detection = detector.update(rnd, mae, rmse, test_dataset)
        next_correction_state = corrector.update(detection) if enable_correction else _drift_correction_module.CorrectionState(False, LEARNING_RATE, LOCAL_EPOCHS)
        corrector.remember(client_datasets)

        events.append(
            {
                "round": rnd,
                "detected": bool(detection.detected),
                "severity": detection.severity if detection.detected else "none",
                "correction_active": bool(correction_state.active),
                "lr": float(correction_state.learning_rate),
                "epochs": int(correction_state.local_epochs),
                "replay_ratio": float(correction_state.replay_ratio),
                "phase": _drift_note(name, rnd),
            }
        )

        notes = [_drift_note(name, rnd)]
        if ensemble is not None and concept_id is not None:
            tag = f"NOVO concept #{concept_id}" if concept_created else f"concept #{concept_id}"
            notes.append(tag)
        if detection.detected:
            notes.append(f"DETECTADO {detection.severity}: {detection.message}")
        if enable_correction and correction_state.active:
            notes.append(correction_state.message)
        note = " | ".join(note for note in notes if note)
        print(f"  {rnd:>7d}  │  {mae:>6.2f}%  │  {rmse:>6.2f}%  │  {r2:>6.3f}  │  acc={acc * 100:>5.1f}%  │  {note}")
        correction_state = next_correction_state

    return {"mae": mae_hist, "rmse": rmse_hist, "acc": acc_hist, "events": events}
