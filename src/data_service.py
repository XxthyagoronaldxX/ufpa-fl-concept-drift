from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from config import TEST_RATE
from data_model import ClientModel, DataModel
from csv import DictReader
from numpy import array
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset


class DataService:
    @staticmethod
    def find_all_data_by(path: str) -> list[DataModel]:
        data = []

        with open(path, mode="r", encoding="utf-8") as arquivo_csv:
            reader = DictReader(arquivo_csv)

            for line in reader:
                data_model = DataModel(
                    time=line["Time"],
                    temperature_2m=float(line["temperature_2m"]),
                    relativehumidity_2m=float(line["relativehumidity_2m"]),
                    dewpoint_2m=float(line["dewpoint_2m"]),
                    windspeed_10m=float(line["windspeed_10m"]),
                    windspeed_100m=float(line["windspeed_100m"]),
                    winddirection_10m=float(line["winddirection_10m"]),
                    winddirection_100m=float(line["winddirection_100m"]),
                    windgusts_10m=float(line["windgusts_10m"]),
                    power=float(line["Power"]),
                )

                data.append(data_model)

        return data

    @staticmethod
    def get_test_scaler_by(x_test_global_list, y_test_global_list, x_train_scaler_list) -> tuple[TensorDataset, StandardScaler]:
        x_test_global_raw = np.vstack(x_test_global_list)
        y_test_global_raw = np.vstack(y_test_global_list)

        x_train_scaler_raw = np.vstack(x_train_scaler_list)

        scaler = StandardScaler()

        # Aprende a escala usando APENAS dados do passado (Treino) de todos os clientes
        scaler.fit(x_train_scaler_raw)

        # Aplica a escala no Super Dataset de Teste
        x_test_global_scaled = scaler.transform(x_test_global_raw)

        x_test_tensor = torch.tensor(x_test_global_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_global_raw, dtype=torch.float32)

        return TensorDataset(x_test_tensor, y_test_tensor), scaler

    @staticmethod
    def get_dataset_by(client_list: list[ClientModel]) -> tuple[list[list[TensorDataset]], TensorDataset]:
        raw_data_per_month = defaultdict(lambda: {i: [] for i in range(len(client_list))})

        x_test_global_list = []
        y_test_global_list = []
        x_train_to_scaler = []

        for index, client in enumerate(client_list):
            train_limit = int(len(client.data) * (1 - TEST_RATE))
            train_model_list = client.data[:train_limit]
            test_model_list = client.data[train_limit:]

            # 2.1 Envia os últimos 20% para o Teste Global no Servidor
            for test_model in test_model_list:
                x_test_global_list.append(test_model.to_feature_list())
                y_test_global_list.append([test_model.power])

            # 2.2 Agrupa os primeiros 80% por Ano-Mês
            for train_model in train_model_list:
                month_year = train_model.time[:7]  # Pega "2017-01" de "2017-01-02 00:00:00"

                features = train_model.to_feature_list()
                target = [train_model.power]

                raw_data_per_month[month_year][index].append((features, target))
                x_train_to_scaler.append(features)

        # 3. Normalização (Escala)
        scaler = StandardScaler()
        scaler.fit(x_train_to_scaler)
        # ... (Criar o test_loader_global aqui exatamente como fazíamos antes, usando o scaler.transform) ...
        x_test_global_scaled = scaler.transform(x_test_global_list)

        x_test_tensor = torch.tensor(x_test_global_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_global_list, dtype=torch.float32)
        test_tensor = TensorDataset(x_test_tensor, y_test_tensor)

        # 4. Criando a estrutura final de Datasets Mês a Mês
        sorted_months = sorted(raw_data_per_month.keys())
        datasets_per_round = []

        for mes in sorted_months:
            clients_dataset_this_month = []

            for cliente_idx in range(4):
                client_data = raw_data_per_month[mes][cliente_idx]

                if len(client_data) > 0:
                    x_raw = [item[0] for item in client_data]
                    y_raw = [item[1] for item in client_data]

                    # Aplica a escala
                    x_scaled = scaler.transform(x_raw)

                    # Converte para Tensores
                    dataset_cliente = TensorDataset(torch.tensor(x_scaled, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32))

                    clients_dataset_this_month.append(dataset_cliente)

            datasets_per_round.append(clients_dataset_this_month)

        return datasets_per_round, test_tensor

    @staticmethod
    def get_train_test_by(data_list: list[DataModel]) -> tuple[array, array, array, array]:
        x_raw = []
        y_raw = []

        for data in data_list:
            features = [
                data.temperature_2m,
                data.relativehumidity_2m,
                data.dewpoint_2m,
                data.windspeed_10m,
                data.windspeed_100m,
                data.winddirection_10m,
                data.winddirection_100m,
                data.windgusts_10m,
            ]
            target = [data.power]

            x_raw.append(features)
            y_raw.append(target)

        x_np = array(x_raw)
        y_np = array(y_raw)

        x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=TEST_RATE, shuffle=False)

        return x_train, y_train, x_test, y_test
