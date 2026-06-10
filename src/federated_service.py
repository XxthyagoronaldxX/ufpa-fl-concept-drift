import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from config import BATCH_SIZE, DEVICE


class FederatedService:
    @staticmethod
    def local_train(global_model: nn.Module, dataset: TensorDataset, epochs: int, lr: float) -> tuple[dict, int]:
        # CRIA UMA CÓPIA DO MODELO GLOBAL PARA TREINAR LOCALMENTE
        # Simula o envio do modelo para o cliente, que treina e retorna os pesos atualizados.
        model = deepcopy(global_model).to(DEVICE)

        # O modelo é colocado em modo de treino, e o otimizador e critério são configurados.
        model.train()

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(x_b), y_b).backward()
                optimizer.step()

        return model.state_dict(), len(dataset)

    @staticmethod
    def fed_avg(global_weights: dict, client_updates: list) -> dict:
        total = sum(sz for _, sz in client_updates)
        agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_weights.items()}

        for weights, sz in client_updates:
            weight = sz / total
            for k in agg:
                agg[k] += weights[k].float() * weight

        return agg

    @staticmethod
    @torch.no_grad()
    def evaluate(model: nn.Module, dataset: TensorDataset) -> tuple[float, float]:
        model.eval()
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        for x_b, y_b in loader:
            preds = model(x_b.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

        acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100.0 * f1_score(all_labels, all_preds, zero_division=0)
        return acc, f1
