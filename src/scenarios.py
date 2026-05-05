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

from torch.utils.data import TensorDataset

from config import DRIFT_ROUND, NUM_ROUNDS, CYCLE_LEN, N_TRAIN, N_TEST, NUM_CLIENTS
from data import make_dataset, split_iid
from model import SpamMLP
from federated import local_train, fed_avg, evaluate
from config import DEVICE, LEARNING_RATE, LOCAL_EPOCHS

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

    clients_A = split_iid(make_dataset(N_TRAIN, spam_phase="A"), NUM_CLIENTS)
    clients_B = split_iid(make_dataset(N_TRAIN, spam_phase="B"), NUM_CLIENTS)

    gradual_train = {
        rnd: split_iid(
            make_dataset(N_TRAIN, spam_phase="mixed", alpha=min(1.0, (rnd - DRIFT_ROUND + 1) / n_drift_rounds)),
            NUM_CLIENTS,
        )
        for rnd in range(DRIFT_ROUND, NUM_ROUNDS + 1)
    }

    test_A = make_dataset(N_TEST, spam_phase="A")
    test_B = make_dataset(N_TEST, spam_phase="B")
    gradual_test = {rnd: make_dataset(N_TEST, spam_phase="mixed", alpha=min(1.0, (rnd - DRIFT_ROUND + 1) / n_drift_rounds)) for rnd in range(DRIFT_ROUND, NUM_ROUNDS + 1)}

    return {
        "clients_A": clients_A,
        "clients_B": clients_B,
        "gradual_train": gradual_train,
        "test_A": test_A,
        "test_B": test_B,
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


def run_scenario(
    name: str,
    get_train_fn,
    get_test_fn,
) -> tuple[list, list]:
    """Executa NUM_ROUNDS de FL e retorna históricos de acurácia e F1.

    Args:
        name:         Identificador do cenário (usado no log).
        get_train_fn: callable(rnd) → list[TensorDataset]
        get_test_fn:  callable(rnd) → TensorDataset

    Returns:
        (acc_history, f1_history) — listas com um valor por rodada.
    """
    model = SpamMLP().to(DEVICE)
    acc_hist, f1_hist = [], []

    print(f"\n{'═' * 64}")
    print(f"  Cenário: {name}")
    print(f"{'═' * 64}")
    print(f"  {'Rodada':>7}  │  {'Acc':>7}  │  {'F1':>7}  │  Observação")
    print(f"  {'-' * 58}")

    for rnd in range(1, NUM_ROUNDS + 1):
        # Treino local + agregação FedAvg
        client_datasets = get_train_fn(rnd)
        updates = [local_train(model, ds, LOCAL_EPOCHS, LEARNING_RATE) for ds in client_datasets]
        model.load_state_dict(fed_avg(model.state_dict(), updates))

        # Avaliação no dataset de teste da rodada atual
        acc, f1 = evaluate(model, get_test_fn(rnd))
        acc_hist.append(acc)
        f1_hist.append(f1)

        note = _drift_note(name, rnd)
        print(f"  {rnd:>7d}  │  {acc:>6.1f}%  │  {f1:>6.1f}%  │  {note}")

    return acc_hist, f1_hist
