import torch.nn as nn
from torch.utils.data import ConcatDataset, TensorDataset
from drift_detector import PageHinkleyDetector
from federated_service import FederatedService
from seasonal_replay_buffer import SeasonalReplayBuffer
from utils.dataset_util import DatasetUtil


class ScenarioService:
    @staticmethod
    def run(datasets_per_round: list[list[TensorDataset]], test_dataset: TensorDataset, global_model: nn.Module) -> list[float]:
        mae_history = []

        for round in range(len(datasets_per_round)):
            print(f"Rodada {round + 1}/{len(datasets_per_round)}")

            local_weights = []
            local_sizes = []

            client_dataset_this_month = datasets_per_round[round]

            for _, client_dataset in enumerate(client_dataset_this_month):
                weight, size = FederatedService.local_train(global_model, client_dataset)

                local_weights.append(weight)
                local_sizes.append(size)

            global_model.load_state_dict(FederatedService.fed_avg_v2(local_weights, local_sizes))
            mae = FederatedService.evaluate(global_model, test_dataset)
            mae_history.append(mae)

        return mae_history

    @staticmethod
    def run_with_correction(datasets_per_round: list[list[TensorDataset]], test_dataset: TensorDataset, global_model: nn.Module) -> list[float]:
        drift_detector = PageHinkleyDetector(threshold=0.1, delta=0.005, burn_in=3)
        drift_corrector = SeasonalReplayBuffer(max_samples_per_context=200)
        mae_history = []

        for round in range(len(datasets_per_round)):
            print(f"Rodada {round + 1}/{len(datasets_per_round)}")

            local_weights = []
            local_sizes = []

            client_dataset_this_month = datasets_per_round[round]

            for _, client_dataset in enumerate(client_dataset_this_month):
                weight, size = FederatedService.local_train(global_model, client_dataset)
                local_weights.append(weight)
                local_sizes.append(size)

            global_model.load_state_dict(FederatedService.fed_avg_v2(local_weights, local_sizes))
            mae = FederatedService.evaluate(global_model, test_dataset)
            mae_history.append(mae)

            has_drift = drift_detector.update_and_check(mae)

            if has_drift:
                print("  → Drift detectado! Iniciando correção com amostras históricas...")

                old_dataset = drift_corrector.recovery_history()

                if old_dataset is not None:
                    corrective_dataset = ConcatDataset([*client_dataset_this_month, old_dataset])

                    corrective_tensor = DatasetUtil.cast_to_tensor_dataset(corrective_dataset)

                    corrective_weights, _ = FederatedService.local_train(global_model, corrective_tensor)

                    global_model.load_state_dict(corrective_weights)

                    mae_corrected = FederatedService.evaluate(global_model, test_dataset)

                    mae_history[-1] = mae_corrected

            dataset_this_month = ConcatDataset(client_dataset_this_month)

            month_id = (round % 12) + 1

            drift_corrector.add_history(DatasetUtil.cast_to_tensor_dataset(dataset_this_month), month_id)

        return mae_history
