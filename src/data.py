"""
data.py
───────
Carregamento dos CSVs reais de geração de energia eólica e construção de
pools federados por estação do ano.

Dataset (4 locais, hora-a-hora, 2017–2021):
  Time, temperature_2m, relativehumidity_2m, dewpoint_2m,
  windspeed_10m, windspeed_100m, winddirection_10m, winddirection_100m,
  windgusts_10m, Power (alvo, normalizado em [0, 1]).

Engenharia de atributos (10 colunas finais):
  temperature_2m, relativehumidity_2m, dewpoint_2m,
  windspeed_10m, windspeed_100m, windgusts_10m,
  winddir_10m_sin, winddir_10m_cos, winddir_100m_sin, winddir_100m_cos.

Cada cliente federado corresponde a um local (Location1..Location4),
caracterizando uma federação não-IID. Concept drift é simulado pelo eixo
sazonal (verão JJA ↔ inverno DJF), conforme o artigo em
[data/readme-artig.md](data/readme-artig.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from config import DATA_DIR, FEATURE_DIM, NUM_CLIENTS, TRAIN_FRACTION

FEATURE_NAMES = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "windgusts_10m",
    "winddir_10m_sin",
    "winddir_10m_cos",
    "winddir_100m_sin",
    "winddir_100m_cos",
]
TARGET_NAME = "Power"

_RAW_NUMERIC = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "windgusts_10m",
]


@dataclass(frozen=True)
class _Slice:
    X: np.ndarray  # [N, FEATURE_DIM] float32 normalizado em [0,1]
    y: np.ndarray  # [N, 1]           float32 já normalizado pelo dataset


_loaded_locations: dict[int, dict[str, np.ndarray]] | None = None
_feature_min: np.ndarray | None = None
_feature_max: np.ndarray | None = None


def _engineer(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Aplica sin/cos nas direções de vento e devolve (X bruto, y)."""
    deg10 = np.deg2rad(df["winddirection_10m"].to_numpy(dtype=np.float32))
    deg100 = np.deg2rad(df["winddirection_100m"].to_numpy(dtype=np.float32))

    columns = [df[name].to_numpy(dtype=np.float32) for name in _RAW_NUMERIC]
    columns.extend([np.sin(deg10), np.cos(deg10), np.sin(deg100), np.cos(deg100)])

    X = np.stack(columns, axis=1).astype(np.float32)
    y = df[TARGET_NAME].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y


def _load_all(data_dir: str) -> dict[int, dict[str, np.ndarray]]:
    """Lê os 4 CSVs, faz engenharia de atributos e indexa pelo mês.

    Retorna: {loc_id: {"X": [N,FEATURE_DIM], "y": [N,1], "month": [N]}}.
    Também ajusta os min/max globais (pool combinado) usados para escalar X.
    """
    global _feature_min, _feature_max

    locations: dict[int, dict[str, np.ndarray]] = {}
    pooled: list[np.ndarray] = []
    for loc_id in range(1, NUM_CLIENTS + 1):
        path = os.path.join(data_dir, f"Location{loc_id}.csv")
        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"])
        month = df["Time"].dt.month.to_numpy(dtype=np.int8)
        X_raw, y = _engineer(df)
        locations[loc_id] = {"X_raw": X_raw, "y": y, "month": month}
        pooled.append(X_raw)

    pool = np.vstack(pooled)
    _feature_min = pool.min(axis=0)
    _feature_max = pool.max(axis=0)
    span = (_feature_max - _feature_min) + 1e-8

    for loc in locations.values():
        scaled = (loc["X_raw"] - _feature_min) / span
        loc["X"] = np.clip(scaled, 0.0, 1.0).astype(np.float32)
        del loc["X_raw"]

    if locations[1]["X"].shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Esperado FEATURE_DIM={FEATURE_DIM}, obtido {locations[1]['X'].shape[1]}")

    return locations


def load_locations(data_dir: str = DATA_DIR) -> dict[int, dict[str, np.ndarray]]:
    """Carrega (uma vez) os dados de todas as localizações."""
    global _loaded_locations
    if _loaded_locations is None:
        _loaded_locations = _load_all(data_dir)
    return _loaded_locations


def _location_seasonal_slice(loc_id: int, months: list[int], split: str) -> _Slice:
    """Recorta um cliente pelos meses pedidos e devolve treino ou teste.

    Split é cronológico (80% inicial = treino, 20% final = teste) dentro da
    janela sazonal, evitando vazamento temporal.
    """
    locations = load_locations()
    loc = locations[loc_id]
    mask = np.isin(loc["month"], months)
    X = loc["X"][mask]
    y = loc["y"][mask]

    n = len(y)
    if n == 0:
        return _Slice(np.empty((0, FEATURE_DIM), dtype=np.float32), np.empty((0, 1), dtype=np.float32))

    cut = int(n * TRAIN_FRACTION)
    if split == "train":
        return _Slice(X[:cut], y[:cut])
    if split == "test":
        return _Slice(X[cut:], y[cut:])
    raise ValueError(f"split inválido: {split!r}")


def _to_tensor_dataset(slc: _Slice) -> TensorDataset:
    return TensorDataset(torch.from_numpy(slc.X), torch.from_numpy(slc.y))


def client_pool(months: list[int], split: str = "train") -> list[TensorDataset]:
    """Lista de TensorDatasets, um por cliente (= local), filtrado por meses."""
    return [_to_tensor_dataset(_location_seasonal_slice(loc_id, months, split)) for loc_id in range(1, NUM_CLIENTS + 1)]


def pooled_test(months: list[int]) -> TensorDataset:
    """Concatena o split de teste de todos os locais para a estação dada."""
    slices = [_location_seasonal_slice(loc_id, months, "test") for loc_id in range(1, NUM_CLIENTS + 1)]
    X = np.vstack([s.X for s in slices])
    y = np.vstack([s.y for s in slices])
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def build_seasonal_pools(months_a: list[int], months_b: list[int]) -> dict:
    """Pré-gera pools de treino e teste para os cenários sazonais.

    Layout:
      - "clients_A", "clients_B": list[TensorDataset]   (1 por cliente/local)
      - "test_A", "test_B"      : TensorDataset         (pool de todos os locais)
    """
    return {
        "clients_A": client_pool(months_a, split="train"),
        "clients_B": client_pool(months_b, split="train"),
        "test_A": pooled_test(months_a),
        "test_B": pooled_test(months_b),
    }
