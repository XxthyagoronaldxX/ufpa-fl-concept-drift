import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import BATCH_SIZE, DEVICE


class FederatedService:
    @staticmethod
    def local_train(global_model: nn.Module, dataset: TensorDataset, epochs: int, lr: float) -> tuple[dict, int]:
        # Cópia local do modelo global. Simula envio ao cliente; o cliente treina e
        # devolve os pesos atualizados.
        model = deepcopy(global_model).to(DEVICE)
        model.train()

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            for x_b, y_b in loader:
                x_b = x_b.to(DEVICE)
                y_b = y_b.to(DEVICE).float()
                optimizer.zero_grad()
                criterion(model(x_b), y_b).backward()
                optimizer.step()

        return model.state_dict(), len(dataset)

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
    def evaluate(model: nn.Module, dataset: TensorDataset) -> tuple[float, float, float]:
        """Retorna (MAE, RMSE, R²) em escala percentual para MAE/RMSE.

        MAE e RMSE são multiplicados por 100 para virarem "p.p. de Power".
        R² fica na escala original (sem multiplicar) — pode ser negativo se o
        modelo for pior que prever a média.
        """
        model.eval()
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        for x_b, y_b in loader:
            preds = model(x_b.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_b.numpy())

        y_pred = np.vstack(all_preds).reshape(-1)
        y_true = np.vstack(all_labels).reshape(-1)

        mae = 100.0 * mean_absolute_error(y_true, y_pred)
        rmse = 100.0 * float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
        return mae, rmse, r2
