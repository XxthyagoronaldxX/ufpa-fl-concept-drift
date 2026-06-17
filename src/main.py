import sys
import os
import warnings
import numpy as np
import torch

from data_model import ClientModel
from data_service import DataService
from model import WindPowerMLP
from scenario_service import ScenarioService
from visualization_service import VisualizationService

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from config import SEED, DEVICE

warnings.filterwarnings("ignore")

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    data_paths = ["data\\Location1.csv", "data\\Location2.csv", "data\\Location3.csv", "data\\Location4.csv"]

    clients = []

    for i, path in enumerate(data_paths):
        client = ClientModel(id=i)
        client.data = DataService.find_all_data_by(path)
        clients.append(client)

    datasets_per_round, test_dataset_global = DataService.get_dataset_by(clients)

    global_model = WindPowerMLP().to(DEVICE)
    mae_history = ScenarioService.run(datasets_per_round, test_dataset_global, global_model)
    global_model = WindPowerMLP().to(DEVICE)
    mae_history_with_correction = ScenarioService.run_with_correction(datasets_per_round, test_dataset_global, global_model)

    VisualizationService.plot_comparison(mae_history, mae_history_with_correction)


if __name__ == "__main__":
    main()
