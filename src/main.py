"""
main.py
───────
Federated Learning com Concept Drift — Previsão de Geração de Energia Eólica

Cenários comparados (eixo de drift = sazonal):
  1. FL Padrão          — distribuição estacionária (somente verão)
  2. FL Drift Recorrente — alterna verão ↔ inverno em ciclos de CYCLE_LEN rodadas

Dataset: 4 locais reais (Location1..Location4), hora-a-hora 2017–2021.
         Atributos: temperatura, umidade, ponto de orvalho, velocidade/rajada
         de vento (10 m e 100 m), direção sin/cos. Alvo: Power ∈ [0, 1].
Modelo:  MLP regressor (saída sigmoid).
FL:      FedAvg (McMahan et al. 2017), 1 cliente por local (não-IID).
"""

import sys
import os
import warnings
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from config import SEED, DRIFT_ROUND, NUM_CLIENTS, FEATURE_DIM, DEVICE
from scenarios import (
    build_data_pools,
    make_standard_fns,
    make_recurrent_fns,
    run_scenario,
)
from visualization import plot_results, plot_separated_results, plot_correction_treatment, plot_accuracy, print_summary

warnings.filterwarnings("ignore")

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("=" * 68)
    print("  Federated Learning com Concept Drift — Wind Power Forecasting")
    print(f"  Dataset real | {NUM_CLIENTS} locais | {FEATURE_DIM} features | regressão")
    print("=" * 68)
    print(f"[INFO] Dispositivo: {DEVICE}")

    pools = build_data_pools()
    n_train_per_client = len(pools["clients_A"][0])
    n_test = len(pools["test_A"])
    print(f"[INFO] Pools sazonais carregados ({n_train_per_client} amostras de treino/cliente, {n_test} de teste por estação).")

    histories = {
        "FL Padrão": run_scenario("FL Padrão", *make_standard_fns(pools), enable_correction=False),
        "Recorrente sem correção": run_scenario("FL Drift Recorrente — sem correção", *make_recurrent_fns(pools), enable_correction=False),
        "Recorrente com correção": run_scenario("FL Drift Recorrente — com correção", *make_recurrent_fns(pools), enable_correction=True),
    }

    print_summary(histories, DRIFT_ROUND)
    plot_results(histories, DRIFT_ROUND)
    plot_separated_results(histories, DRIFT_ROUND)
    plot_correction_treatment(histories, DRIFT_ROUND)
    plot_accuracy(histories, DRIFT_ROUND)


if __name__ == "__main__":
    main()
