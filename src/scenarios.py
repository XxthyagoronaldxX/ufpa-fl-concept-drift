"""
scenarios.py
────────────
Definição dos quatro cenários de Federated Learning e execução por rodada.

Cenários:
  1. FL Padrão       — distribuição estacionária (sem drift)
  2. Drift Súbito    — spammers mudam de tática abruptamente na rodada DRIFT_ROUND
  3. Drift Gradual   — transição progressiva para novas táticas de spam
  4. Drift Recorrente — padrões de spam alternam ciclicamente (A → B → A → B …)

O dataset de teste acompanha a distribuição vigente em cada rodada, tornando
visível o impacto do drift: o modelo treinado na fase A falha ao ser avaliado
na fase B (spam furtivo).
"""

import importlib.util
import os
import sys

from config import DRIFT_ROUND, NUM_ROUNDS, CYCLE_LEN, N_TRAIN, N_TEST, NUM_CLIENTS, DEVICE, LEARNING_RATE, LOCAL_EPOCHS
from data import make_dataset, split_iid
from model import SpamMLP
from federated import local_train, fed_avg, evaluate


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

# ── Preparação dos pools de dados ────────────────────────────────────────────


def build_data_pools() -> dict:
    """Pré-gera todos os datasets de treino e teste necessários.

    Returns:
        Dicionário com as chaves:
        - "clients_A", "clients_B": listas de datasets por cliente
        - "gradual_train": dict {rodada: lista de datasets}
        - "test_A", "test_B": datasets de teste fixos
        - "gradual_test": dict {rodada: dataset}
    """
    n_drift_rounds = NUM_ROUNDS - DRIFT_ROUND + 1

    clients_a = split_iid(make_dataset(N_TRAIN, spam_phase="A"), NUM_CLIENTS)
    clients_b = split_iid(make_dataset(N_TRAIN, spam_phase="B"), NUM_CLIENTS)

    gradual_train = {
        rnd: split_iid(
            make_dataset(N_TRAIN, spam_phase="mixed", alpha=min(1.0, (rnd - DRIFT_ROUND + 1) / n_drift_rounds)),
            NUM_CLIENTS,
        )
        for rnd in range(DRIFT_ROUND, NUM_ROUNDS + 1)
    }

    test_a = make_dataset(N_TEST, spam_phase="A")
    test_b = make_dataset(N_TEST, spam_phase="B")
    gradual_test = {rnd: make_dataset(N_TEST, spam_phase="mixed", alpha=min(1.0, (rnd - DRIFT_ROUND + 1) / n_drift_rounds)) for rnd in range(DRIFT_ROUND, NUM_ROUNDS + 1)}

    return {
        "clients_A": clients_a,
        "clients_B": clients_b,
        "test_A": test_a,
        "test_B": test_b,
        "gradual_train": gradual_train,
        "gradual_test": gradual_test,
    }


# ── Seletores de dados por rodada ────────────────────────────────────────────


def make_standard_fns(pools: dict):
    """Cenário 1 — sem drift: distribuição A em todas as rodadas."""

    def get_train(rnd):
        return pools["clients_A"]

    def get_test(rnd):
        return pools["test_A"]

    return get_train, get_test


def make_sudden_fns(pools: dict):
    """Cenário 2 — drift súbito: muda de A para B na rodada DRIFT_ROUND."""

    def get_train(rnd):
        return pools["clients_A"] if rnd < DRIFT_ROUND else pools["clients_B"]

    def get_test(rnd):
        return pools["test_A"] if rnd < DRIFT_ROUND else pools["test_B"]

    return get_train, get_test


def make_gradual_fns(pools: dict):
    """Cenário 3 — drift gradual: mistura A→B progressivamente."""

    def get_train(rnd):
        return pools["clients_A"] if rnd < DRIFT_ROUND else pools["gradual_train"][rnd]

    def get_test(rnd):
        return pools["test_A"] if rnd < DRIFT_ROUND else pools["gradual_test"][rnd]

    return get_train, get_test


def make_recurrent_fns(pools: dict):
    """Cenário 4 — drift recorrente: alterna A ↔ B em ciclos de CYCLE_LEN rodadas."""

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
    if "Súbito" in scenario and rnd == DRIFT_ROUND:
        return "◄ DRIFT SÚBITO — spammers mudam para crypto/phishing"
    if "Gradual" in scenario and rnd >= DRIFT_ROUND:
        alpha = min(1.0, (rnd - DRIFT_ROUND + 1) / (NUM_ROUNDS - DRIFT_ROUND + 1))
        return f"gradual α={alpha:.2f} ({int(alpha * 100)}% fase B)"
    if "Recorrente" in scenario and rnd >= DRIFT_ROUND:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        phase = "B (moderno)" if pos < CYCLE_LEN else "A (clássico)"
        return f"ciclo — fase {phase}"
    return ""


# ── Loop principal de FL ─────────────────────────────────────────────────────


def run_scenario(name: str, get_train_fn, get_test_fn, enable_correction: bool = True) -> tuple[list, list]:
    """Executa NUM_ROUNDS de FL e retorna históricos de acurácia e F1.

    Args:
        name:              Identificador do cenário (usado no log).
        get_train_fn:      callable(rnd) → list[TensorDataset]
        get_test_fn:       callable(rnd) → TensorDataset
        enable_correction: aplica correções adaptativas quando um drift é detectado.

    Returns:
        (acc_history, f1_history) — listas com um valor por rodada.
    """
    model = SpamMLP().to(DEVICE)
    acc_hist, f1_hist = [], []
    detector = build_drift_detector()
    corrector = build_drift_corrector()
    correction_state = _drift_correction_module.CorrectionState(False, LEARNING_RATE, LOCAL_EPOCHS)

    print(f"\n{'═' * 64}")
    print(f"  Cenário: {name}")
    print(f"{'═' * 64}")
    print(f"  {'Rodada':>7}  │  {'Acc':>7}  │  {'F1':>7}  │  Observação")
    print(f"  {'-' * 82}")

    for rnd in range(1, NUM_ROUNDS + 1):
        # Treino local + agregação FedAvg
        client_datasets = get_train_fn(rnd)
        train_datasets = corrector.apply_replay(client_datasets, correction_state.replay_ratio) if enable_correction and correction_state.active else client_datasets
        updates = [local_train(model, ds, correction_state.local_epochs, correction_state.learning_rate) for ds in train_datasets]
        model.load_state_dict(fed_avg(model.state_dict(), updates))

        # Avaliação no dataset de teste da rodada atual
        test_dataset = get_test_fn(rnd)
        acc, f1 = evaluate(model, test_dataset)
        acc_hist.append(acc)
        f1_hist.append(f1)

        detection = detector.update(rnd, acc, f1, test_dataset)
        next_correction_state = corrector.update(detection) if enable_correction else _drift_correction_module.CorrectionState(False, LEARNING_RATE, LOCAL_EPOCHS)
        corrector.remember(client_datasets)

        notes = [_drift_note(name, rnd)]
        if detection.detected:
            notes.append(f"DETECTADO {detection.severity}: {detection.message}")
        if enable_correction and correction_state.active:
            notes.append(correction_state.message)
        note = " | ".join(note for note in notes if note)
        print(f"  {rnd:>7d}  │  {acc:>6.1f}%  │  {f1:>6.1f}%  │  {note}")
        correction_state = next_correction_state

    return acc_hist, f1_hist
