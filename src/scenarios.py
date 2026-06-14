"""
scenarios.py
────────────
Definição dos cenários de Federated Learning e execução por rodada,
aplicados a previsão de potência eólica.

Cenários (eixo de drift = sazonal):
  1. FL Padrão          — distribuição estacionária (somente verão)
  2. FL Drift Recorrente — alterna verão ↔ inverno em ciclos de CYCLE_LEN
                            rodadas, usando SeasonalReplayBuffer (mistura
                            50% de amostras da estação oposta a cada batch).

O detector observacional permanece ativo nos dois cenários apenas para
sinalizar no log quando uma queda de desempenho é detectada.
"""

from config import (
    CYCLE_LEN,
    DEVICE,
    DRIFT_ROUND,
    DRIFT_THRESHOLD,
    DRIFT_WINDOW_SIZE,
    NUM_CLIENTS,
    NUM_ROUNDS,
    REPLAY_BUFFER_SIZE,
    SUMMER_MONTHS,
    WINTER_MONTHS,
)
from data import build_seasonal_pools
from drift_detector import DetectorDeDrift
from federated_service import FederatedService
from model import WindPowerMLP
from seasonal_replay_buffer import SeasonalReplayBuffer
from torch.utils.data import ConcatDataset

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
    """Cenário 1 — sem drift: somente verão (distribuição estacionária).

    O teste é feito no pool combinado (verão + inverno) para expor o quanto
    o modelo treinado só em verão falha em generalizar para o inverno.
    """

    def get_train(rnd):
        return pools["clients_A"]

    def get_test(rnd):
        return pools["test_combined"]

    return get_train, get_test


def make_recurrent_fns(pools: dict):
    """Cenário 2 — drift recorrente: alterna verão ↔ inverno em ciclos de CYCLE_LEN.

    O teste também é feito no pool combinado: assim, o ganho do replay sobre
    o catastrophic forgetting fica mensurável — o modelo é cobrado nas duas
    estações a cada rodada, não só na corrente.
    """

    def _phase(rnd: int) -> str:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        return "B" if pos < CYCLE_LEN else "A"

    def get_train(rnd):
        if rnd < DRIFT_ROUND:
            return pools["clients_A"]
        return pools["clients_B"] if _phase(rnd) == "B" else pools["clients_A"]

    def get_test(rnd):
        return pools["test_combined"]

    return get_train, get_test


# ── Log por rodada ───────────────────────────────────────────────────────────


def _drift_note(scenario: str, rnd: int) -> str:
    """Retorna uma nota descritiva sobre o drift na rodada atual."""
    if "Recorrente" in scenario and rnd >= DRIFT_ROUND:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        phase = "inverno" if pos < CYCLE_LEN else "verão"
        return f"ciclo — fase {phase}"
    return ""


def _season_for_round(scenario: str, rnd: int) -> str:
    """Mapeia a rodada para 'verao' ou 'inverno' (chaves do replay buffer)."""
    if "Recorrente" in scenario and rnd >= DRIFT_ROUND:
        pos = (rnd - DRIFT_ROUND) % (2 * CYCLE_LEN)
        return "inverno" if pos < CYCLE_LEN else "verao"
    return "verao"


# ── Loop principal de FL ─────────────────────────────────────────────────────


def run_scenario(name: str, get_train_fn, get_test_fn, use_replay: bool = False) -> dict:
    """Executa NUM_ROUNDS de FL com FedAvg + Adam e retorna o histórico de MAE.

    Args:
        name:         Identificador do cenário (usado no log).
        get_train_fn: callable(rnd) → list[TensorDataset]
        get_test_fn:  callable(rnd) → TensorDataset
        use_replay:   se True, cada cliente usa um SeasonalReplayBuffer que
                      mistura amostras da estação oposta nos batches de treino.

    Returns:
        dict com chave "mae": lista (uma entrada por rodada, em p.p. de Power).
    """
    model = WindPowerMLP().to(DEVICE)
    mae_hist = []
    detector = DetectorDeDrift(tamanho_janela=DRIFT_WINDOW_SIZE, limiar_alerta=DRIFT_THRESHOLD)
    replay_buffers = [SeasonalReplayBuffer(REPLAY_BUFFER_SIZE) for _ in range(NUM_CLIENTS)] if use_replay else None
    prev_season: str | None = None

    print(f"\n{'═' * 64}")
    print(f"  Cenário: {name}")
    print(f"{'═' * 64}")
    print(f"  {'Rodada':>7}  │  {'MAE':>7}  │  Fase")
    print(f"  {'-' * 60}")

    for rnd in range(1, NUM_ROUNDS + 1):
        client_datasets = get_train_fn(rnd)
        season = _season_for_round(name, rnd)

        if use_replay:
            # Em transições de fase (incluindo a primeira rodada), preenche o buffer
            # de cada cliente com amostras da nova estação (fill-up).
            if season != prev_season:
                for i, ds in enumerate(client_datasets):
                    replay_buffers[i].add_dataset(ds, season)

            updates = []
            for i, ds in enumerate(client_datasets):
                buffer_ds = replay_buffers[i].get_dataset()
                train_ds = ConcatDataset([ds, buffer_ds]) if buffer_ds is not None else ds
                updates.append(FederatedService.local_train(model, train_ds))
        else:
            updates = [FederatedService.local_train(model, ds) for ds in client_datasets]

        prev_season = season
        model.load_state_dict(FederatedService.fed_avg(model.state_dict(), updates))

        test_dataset = get_test_fn(rnd)
        mae = FederatedService.evaluate(model, test_dataset)
        mae_hist.append(mae)

        note = _drift_note(name, rnd)
        if detector.atualizar_e_checar(mae):
            alerta = f"[ALERTA] Concept Drift detectado na rodada {rnd}!"
            note = f"{note} | {alerta}" if note else alerta
        print(f"  {rnd:>7d}  │  {mae:>6.2f}%  │  {note}")

    return {"mae": mae_hist}
