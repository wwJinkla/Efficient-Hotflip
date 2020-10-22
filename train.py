import random
from typing import List

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

    def fit(self, X, y):
        # TODO: one hot encodes labels
        pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self, contents: List[str], labels: List[int]):
        self.contents = contents
        self.labels = labels
        assert len(self.contents) == len(self.labels)

    def __getitem__(self, batchsize=3):
        batch_contents = random.choices(self.contents, k=batchsize)
        batch_labels = random.choices(self.labels, k=batchsize)
        return (batch_contents, batch_labels)

    def __len__(self):
        return len(self.labels)
