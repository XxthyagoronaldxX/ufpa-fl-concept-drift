from torch.utils.data import ConcatDataset, TensorDataset
import torch


class DatasetUtil:
    @staticmethod
    def cast_to_tensor_dataset(dataset: ConcatDataset) -> TensorDataset:
        xs = []
        ys = []

        for _, item in enumerate(dataset):
            x, y = item
            xs.append(x)
            ys.append(y)

        x_tensor = torch.stack(xs)
        y_tensor = torch.stack(ys)

        return TensorDataset(x_tensor, y_tensor)
