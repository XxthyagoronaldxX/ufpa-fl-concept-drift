import random
from torch.utils.data import TensorDataset
from typing import Optional
import torch


class SeasonalReplayBuffer:
    def __init__(self, max_samples_per_context: int = 500):
        self.max_samples = max_samples_per_context
        self.buffer: dict[int, TensorDataset] = {}

    def add_history(self, dataset: TensorDataset, context_id: int):
        current_size = len(dataset)
        sample_size = min(self.max_samples, current_size)

        # 1. Escolhe índices aleatórios do dataset atual
        indexes = random.sample(range(current_size), sample_size)

        # 2. Extrai e fatia os tensores DIRETAMENTE usando os índices sorteados
        x_new = dataset.tensors[0][indexes]
        y_new = dataset.tensors[1][indexes]

        if context_id in self.buffer:
            # 3. Recupera as matrizes antigas que já estavam no buffer
            dataset_old = self.buffer[context_id]
            x_old = dataset_old.tensors[0]
            y_old = dataset_old.tensors[1]

            # 4. Concatena tudo de forma nativa e matemática
            x_concat = torch.cat([x_old, x_new], dim=0)
            y_concat = torch.cat([y_old, y_new], dim=0)

            # 5. Corta o excesso mantendo apenas as amostras mais recentes (do final da matriz)
            if len(x_concat) > self.max_samples:
                x_concat = x_concat[-self.max_samples :]
                y_concat = y_concat[-self.max_samples :]

            self.buffer[context_id] = TensorDataset(x_concat, y_concat)
        else:
            self.buffer[context_id] = TensorDataset(x_new, y_new)

    def recovery_history(self) -> Optional[list[TensorDataset]]:
        if not self.buffer:
            return None

        datasets_to_replay: list[TensorDataset] = []

        for _, ds in self.buffer.items():
            datasets_to_replay.append(ds)

        if not datasets_to_replay:
            return None

        return datasets_to_replay
