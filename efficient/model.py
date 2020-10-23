import random
from collections import Counter
from typing import List
from itertools import cycle
import torch
from torch import nn

from efficient.char_cnn_lstm import CharCNNLSTM
from efficient.torch_model_base import TorchModelBase


class CharCNNLSTMModel(TorchModelBase):
    def __init__(
        self,
        vocab,
        char_embed_size,
        embed_size,
        hidden_size,
        max_word_length,
        val_batch_size,
        **model_kwargs,
    ):
        super().__init__(**model_kwargs)
        self.vocab = vocab
        self.char_embed_size = char_embed_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.max_word_length = max_word_length
        self.val_batch_size = val_batch_size
        self.loss = nn.CrossEntropyLoss()
        self.model = self.build_graph()

    def build_graph(self):
        return CharCNNLSTM(
            embed_size=self.embed_size,
            char_embed_size=self.char_embed_size,
            hidden_size=self.hidden_size,
            max_word_length=self.max_word_length,
            vocab=self.vocab,
            device=self.device
        )

    def build_dataset(self, contents, labels):
        return Dataset(contents, labels)

    def fit(self, contents, labels):
        # TODO: validation set, loss, accuracy
        # TODO: checkpoints
        (
            train_contents,
            train_labels,
            val_contents,
            val_labels,
        ) = self._build_validation_split(contents, labels)
        self.train_set = self.build_dataset(train_contents, train_labels)
        self.val_set = self.build_dataset(val_contents, val_labels)

        self.optimizer = self.build_optimizer()

        self.optimizer.zero_grad()
        self.model.to(self.device)
        self.model.train()

        best_accuracy = 0
        for iter_step in range(1, self.max_iter + 1):
            total_losses = 0.0

            for batch_step in range(1, 20):
                contents, labels = self.train_set[self.batch_size]
                labels = labels.to(self.device)
                pred = self.model(contents)
                losses = self.loss(pred, labels)
                losses.backward()
                total_losses += losses.item()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Only on last batch step
            predicted_labels = torch.argmax(pred, dim=1)
            accuracy = float((predicted_labels == labels).float().mean())

            val_contents, val_labels = self.val_set[self.val_batch_size]
            val_labels = val_labels.to(self.device)
            val_losses, val_accuracy = self.predict(val_contents, val_labels)
            print(
                "Iter:",
                iter_step,
                "train loss:",
                losses.item(),
                "train acc:",
                accuracy,
                "sampled val loss:",
                val_losses,
                "sampled val acc:",
                val_accuracy,
            )
            if iter_step % 10 == 0:
                model_path = f"checkpoints/model_iter_{iter_step}.pkl"
                torch.save(self.model.state_dict(), model_path)
                print("Model saved to:", model_path)

                val_accuracies = []
                # This will drop few examples
                for batch_index in range(0, len(self.val_set), self.val_batch_size):
                    batch_contents, batch_labels = self.val_set[self.val_batch_size]
                    _, accuracy = self.predict(batch_contents, batch_labels)
                    val_accuracies.append(accuracy)

                total_val_accuracy = sum(val_accuracies)/len(val_accuracies)

                if total_val_accuracy > best_accuracy:
                    best_accuracy = total_val_accuracy
                    best_model_path = f"checkpoints/best_model.pkl"
                    torch.save(self.model.state_dict(), best_model_path)
                    print(
                        "Best model saved to:",
                        best_model_path,
                        "with total validation accuracy:",
                        best_accuracy,
                    )

    def predict(self, contents, labels):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            # TODO: smaller batch if OOM
            pred = self.model(contents)
            labels = labels.to(self.device)
            losses = self.loss(pred, labels).item()
            predicted_labels = torch.argmax(pred, dim=1)
            accuracy = float((predicted_labels == labels).float().mean())
        self.model.train()
        return losses, accuracy

    @staticmethod
    def _build_validation_split(contents, labels, validation_fraction=0.1):
        """Perform a static split for training and validation sets
        """

        assert len(contents) == len(labels)
        all_train = list(zip(contents, labels))
        random.seed(0)
        random.shuffle(all_train)
        split_index = int(len(all_train) * validation_fraction)

        train = all_train[split_index:]
        val = all_train[:split_index]

        train_contents = [c for c, l in train]
        train_labels = [l for c, l in train]

        val_contents = [c for c, l in val]
        val_labels = [l for c, l in val]

        print(
            "train set distribution:",
            sorted(Counter(train_labels).most_common(), key=lambda x: x[0]),
        )
        print(
            "val set distribution:",
            sorted(Counter(val_labels).most_common(), key=lambda x: x[0]),
        )

        return train_contents, train_labels, val_contents, val_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, contents: List[str], labels: List[int]):
        self.contents = contents
        self.labels = labels
        assert len(self.contents) == len(self.labels)
        self.data_generator = iter(cycle(list(zip(self.contents, self.labels))))

    def __getitem__(self, batch_size=3):
        batch_data = [next(self.data_generator) for _ in range(batch_size)]
        batch_contents = [c for c, l in batch_data]
        batch_labels = [l for c, l in batch_data]
        batch_labels = torch.tensor(batch_labels) - 1  # original labels are [1,2,3,4]

        return batch_contents, batch_labels

    def __len__(self):
        return len(self.labels)
