import random
from typing import List

import torch
from torch import nn

from efficient.char_cnn_lstm import CharCNNLSTM
from efficient.torch_model_base import TorchModelBase


class Trainer(TorchModelBase):
    def __init__(self, vocab, char_embed_size, embed_size, hidden_size, **model_kwargs):
        super().__init__(**model_kwargs)
        self.vocab = vocab
        self.char_embed_size = char_embed_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.loss = nn.CrossEntropyLoss()

    def build_graph(self):
        return CharCNNLSTM(
            embed_size=self.embed_size,
            char_embed_size=self.char_embed_size,
            hidden_size=self.hidden_size,
            vocab=self.vocab,
        )

    def build_dataset(self, contents, labels):
        return Dataset(contents, labels)

    def fit(self, contents, labels):
        # TODO: validation set, loss, accuracy
        # TODO: the model doesn't converge
        # TODO: checkpoints

        self.dataset = self.build_dataset(contents, labels)
        self.model = self.build_graph()
        self.optimizer = self.build_optimizer()

        self.optimizer.zero_grad()
        self.model.train()
        for iter_step in range(1, self.max_iter + 1):
            total_losses = 0.0

            for batch_step in range(1, 20):
                sentences, labels = self.dataset[self.batch_size]

                pred = self.model(sentences)
                losses = self.loss(pred, labels)
                losses.backward()
                total_losses += losses.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print("iter:", iter_step, "loss:", total_losses)

        return

    def predict(self):
        # TODO: test accuracy and save predicted output
        raise NotImplementedError


class Dataset(torch.utils.data.Dataset):
    def __init__(self, contents: List[str], labels: List[int]):
        self.contents = contents
        self.labels = labels
        assert len(self.contents) == len(self.labels)

    def __getitem__(self, batchsize=3):
        batch_contents = random.choices(self.contents, k=batchsize)
        batch_labels = random.choices(self.labels, k=batchsize)
        batch_labels = torch.tensor(batch_labels) - 1  # original labels are [1,2,3,4]

        return batch_contents, batch_labels

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    from efficient.utils import read_corpus, read_labels
    from efficient.vocab import Vocab

    vocab = Vocab.load("data/vocab.json")
    train_contents = read_corpus("data/train_content.txt")
    train_labels = read_labels("data/train_label.txt")
    trainer = Trainer(
        vocab,
        char_embed_size=25,
        embed_size=100,
        hidden_size=100,
        batch_size=100,
        eta=0.01,
    )
    trainer.fit(train_contents, train_labels)
