import torch

from efficient.char_cnn_lstm import CharCNNLSTM
from efficient.torch_model_base import TorchModelBase


# TODO
class Trainer(TorchModelBase):
    def __init__(self, **model_kwargs):
        super().__init__(**model_kwargs)

    def build_graph(self, **kwargs):
        return CharCNNLSTM(**kwargs)

    def build_dataset(self, contents, labels):
        return Dataset(contents, labels)

    def fit(self):
        pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self, contents, labels):
        self.contents = contents
        self.labels = labels
