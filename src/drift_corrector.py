"""
seasonal_replay_buffer.py
─────────────────────────
Buffer de replay sazonal: armazena pares (x, y) por estação ('verao',
'inverno') e expõe o conteúdo acumulado como um TensorDataset, usado para
augmentar o dataset de treino do cliente via ConcatDataset.

Estratégia (Case 3): cada cliente FL mantém o seu próprio buffer (preserva
o isolamento da federação). Política **fill-up**: o buffer recebe amostras
até atingir `max_size_per_season` e então congela — exposições futuras à
mesma estação não substituem o que já está guardado. Isso preserva uma
referência estável de cada concept ao longo das rodadas.

Substitui o catastrophic forgetting causado pelo concept drift recorrente:
ao concatenar o buffer com os dados atuais, o cliente vê amostras das duas
estações em cada época de treino local.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


class SeasonalReplayBuffer:
    def __init__(self, max_size_per_season: int = 500):
        """
        Inicializa o buffer de replay sazonal.

        Args:
            max_size_per_season (int): O limite máximo de amostras (pares x, y)
                que podem ser armazenadas para CADA estação individualmente.
                Por exemplo, se definido como 500, o buffer guardará até 500
                amostras de 'verao' e até 500 de 'inverno', totalizando 1000.
        """
        self.max_size = max_size_per_season
        self.buffer: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            "verao": [],
            "inverno": [],
        }

    def add_dataset(self, dataset: TensorDataset, season: str) -> None:
        """Empurra o dataset de treino atual para o buffer da estação dada,
        respeitando a capacidade (fill-up: para de aceitar quando enche)."""
        season = season.lower()
        bucket = self.buffer[season]
        remaining = self.max_size - len(bucket)
        if remaining <= 0:
            return

        for i in range(min(len(dataset), remaining)):
            x, y = dataset[i]
            bucket.append((x.detach().clone(), y.detach().clone()))

    def get_dataset(self) -> TensorDataset | None:
        """Concatena todos os pares guardados (de ambas as estações) em um
        único TensorDataset. Retorna None se o buffer estiver vazio."""
        all_x: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []

        for samples in self.buffer.values():
            for x, y in samples:
                all_x.append(x)
                all_y.append(y)

        if not all_x:
            return None

        X = torch.stack(all_x)
        Y = torch.stack(all_y)

        return TensorDataset(X, Y)

    def size(self) -> int:
        return sum(len(v) for v in self.buffer.values())
