"""
main.py
───────
Federated Learning com Concept Drift — Detecção de Spam em E-mails

Quatro cenários comparados:
  1. FL Padrão          — distribuição estacionária (sem drift)
  2. FL Drift Súbito    — spammers mudam táticas abruptamente na rodada DRIFT_ROUND
  3. FL Drift Gradual   — transição progressiva para novas táticas de spam
  4. FL Drift Recorrente — padrões de spam alternam ciclicamente (A → B → A → B …)

Dataset: sintético — 20 features por e-mail, classificação binária spam/ham
Modelo:  MLP (Multi-Layer Perceptron)
FL:      FedAvg (McMahan et al. 2017)
"""

import sys
import os
import warnings
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from config import SEED, DRIFT_ROUND, N_TRAIN, N_TEST, DEVICE
from scenarios import (
    build_data_pools,
    make_standard_fns,
    make_sudden_fns,
    make_gradual_fns,
    make_recurrent_fns,
    run_scenario,
)
from visualization import plot_results, plot_separated_results, print_summary

warnings.filterwarnings("ignore")

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print("=" * 68)
    print("  Federated Learning com Concept Drift — Spam de E-mail")
    print("  Dataset sintético | 20 features | spam/ham")
    print("=" * 68)
    print(f"[INFO] Dispositivo: {DEVICE}")

    pools = build_data_pools()
    print(f"[INFO] Datasets gerados ({N_TRAIN} treino / {N_TEST} teste por fase).")

    histories = {
        "FL Padrão": run_scenario("FL Padrão", *make_standard_fns(pools), enable_correction=False),
        "Súbito sem correção": run_scenario("FL Drift Súbito — sem correção", *make_sudden_fns(pools), enable_correction=False),
        "Súbito com correção": run_scenario("FL Drift Súbito — com correção", *make_sudden_fns(pools), enable_correction=True),
        "Gradual sem correção": run_scenario("FL Drift Gradual — sem correção", *make_gradual_fns(pools), enable_correction=False),
        "Gradual com correção": run_scenario("FL Drift Gradual — com correção", *make_gradual_fns(pools), enable_correction=True),
        "Recorrente sem correção": run_scenario("FL Drift Recorrente — sem correção", *make_recurrent_fns(pools), enable_correction=False),
        "Recorrente com correção": run_scenario("FL Drift Recorrente — com correção", *make_recurrent_fns(pools), enable_correction=True),
    }

    print_summary(histories, DRIFT_ROUND)
    plot_results(histories, DRIFT_ROUND)
    plot_separated_results(histories, DRIFT_ROUND)


if __name__ == "__main__":
    main()
