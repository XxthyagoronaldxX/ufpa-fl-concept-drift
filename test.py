"""
test.py
───────
Sanidade do pipeline de dados eólicos (substitui o antigo stub do KaggleHub).

Carrega os 4 CSVs locais via src/data.py e imprime as formas dos pools sazonais
e do split por cliente, para validar que o pipeline está pronto para o treino.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DATA_DIR, FEATURE_DIM, NUM_CLIENTS, SUMMER_MONTHS, WINTER_MONTHS  # noqa: E402
from data import client_pool, load_locations, pooled_test  # noqa: E402


def main() -> None:
    print(f"[INFO] DATA_DIR = {DATA_DIR}")
    locations = load_locations()
    print(f"[INFO] Locais carregados: {sorted(locations.keys())}")
    for loc_id, payload in locations.items():
        x = payload["X"]
        y = payload["y"]
        print(f"  Location{loc_id}: X={tuple(x.shape)} y={tuple(y.shape)} " f"X∈[{x.min():.3f},{x.max():.3f}] y∈[{y.min():.3f},{y.max():.3f}]")

    print(f"\n[INFO] Verão (meses {SUMMER_MONTHS}) — clientes:")
    summer_clients = client_pool(SUMMER_MONTHS, split="train")
    summer_test = pooled_test(SUMMER_MONTHS)
    for i, ds in enumerate(summer_clients, start=1):
        x, y = ds.tensors
        assert x.dtype == torch.float32 and y.dtype == torch.float32
        assert x.shape[1] == FEATURE_DIM and y.shape[1] == 1
        print(f"  cliente {i}: X={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"  pooled test verão: X={tuple(summer_test.tensors[0].shape)}")

    print(f"\n[INFO] Inverno (meses {WINTER_MONTHS}) — clientes:")
    winter_clients = client_pool(WINTER_MONTHS, split="train")
    winter_test = pooled_test(WINTER_MONTHS)
    for i, ds in enumerate(winter_clients, start=1):
        x, y = ds.tensors
        print(f"  cliente {i}: X={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"  pooled test inverno: X={tuple(winter_test.tensors[0].shape)}")

    assert len(summer_clients) == NUM_CLIENTS == len(winter_clients)
    print("\n[OK] Pipeline de dados pronto para treino federado.")


if __name__ == "__main__":
    main()
