import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

from config import BATCH_SIZE, DEVICE, LEARNING_RATE, LOCAL_EPOCHS


class FederatedService:
    @staticmethod
    def local_train(global_model: nn.Module, dataset: TensorDataset) -> tuple[dict, int]:
        # Cópia local do modelo global. Simula envio ao cliente; o cliente treina e
        # devolve os pesos atualizados.
        model = deepcopy(global_model).to(DEVICE)
        model.train()

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

        for _ in range(LOCAL_EPOCHS):
            for x_b, y_b in loader:
                x_b = x_b.to(DEVICE)

                y_b = y_b.to(DEVICE).float()

                optimizer.zero_grad()

                predictions = model(x_b)

                loss = criterion(predictions, y_b)

                loss.backward()

                optimizer.step()

        return model.state_dict(), len(dataset)

    @staticmethod
    def fed_avg_v2(local_weights: list[dict], local_sizes: list[int]) -> dict:
        total_size = sum(local_sizes)
        aggregated_weights = deepcopy(local_weights[0])

        # Zera os pesos para começar a soma
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        # Soma ponderada dos pesos de todos os clientes
        for i in range(len(local_weights)):
            client_weight = local_sizes[i] / total_size

            for key in aggregated_weights.keys():
                aggregated_weights[key] += local_weights[i][key] * client_weight

        return aggregated_weights

    @staticmethod
    def fed_avg(global_weights: dict, client_updates: list) -> dict:
        total = sum(size for _, size in client_updates)
        agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_weights.items()}

        for weights, size in client_updates:
            weight = size / total
            for k in agg:
                agg[k] += weights[k].float() * weight

        return agg

    @staticmethod
    @torch.no_grad()
    def evaluate(model: nn.Module, dataset: TensorDataset) -> float:
        model.eval()

        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        y_true_all = []
        y_pred_all = []

        for x_b, y_b in loader:
            x_b = x_b.to(DEVICE)

            predictions = model(x_b)

            y_pred_all.append(predictions.cpu().numpy())
            y_true_all.append(y_b.cpu().numpy())

        y_pred_all = np.vstack(y_pred_all).reshape(-1)
        y_true_all = np.vstack(y_true_all).reshape(-1)

        return mean_absolute_error(y_true_all, y_pred_all)
