import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

from config import BATCH_SIZE, DEVICE, LOCAL_EPOCHS


class FederatedService:
    @staticmethod
    def local_train(global_model: nn.Module, dataset) -> tuple[dict, int]:
        # Cópia local do modelo global. Simula envio ao cliente; o cliente treina e
        # devolve os pesos atualizados.
        model = deepcopy(global_model).to(DEVICE)
        model.train()

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for _ in range(LOCAL_EPOCHS):
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
    def evaluate(model: nn.Module, dataset: TensorDataset) -> float:
        """Retorna o MAE em escala percentual ("p.p. de Power")."""
        model.eval()
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        for x_b, y_b in loader:
            preds = model(x_b.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_b.numpy())

        y_pred = np.vstack(all_preds).reshape(-1)
        y_true = np.vstack(all_labels).reshape(-1)

        return 100.0 * mean_absolute_error(y_true, y_pred)
