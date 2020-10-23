import torch

from efficient.model import CharCNNLSTMModel, Dataset
from efficient.utils import read_corpus, read_labels
from efficient.vocab import Vocab


def infer(model_path, vocab_path, test_contents_path, test_label_path, **model_config):
    vocab = Vocab.load(vocab_path)
    test_contents = read_corpus(test_contents_path)
    test_labels = read_labels(test_label_path)
    test_dataset = Dataset(test_contents, test_labels)
    contents, labels = test_dataset[len(test_dataset)]

    predictor = CharCNNLSTMModel(vocab, **model_config)
    predictor.model.load_state_dict(torch.load(model_path))
    _, accuracy = predictor.predict(contents, labels)
    print("test accuracy:", accuracy)


if __name__ == "__main__":
    vocab_path = "data/vocab.json"
    test_contents_path = "data/test_content.txt"
    test_label_path = "data/test_label.txt"
    model_path = "checkpoints/model_iter_50.pkl"

    model_config = dict(
        char_embed_size=25,
        embed_size=100,
        hidden_size=100,
        max_word_length=30,
        batch_size=100,
        eta=0.01,
        max_grad_norm=1,
        max_iter=500,
    )
    infer(model_path, vocab_path, test_contents_path, test_label_path, **model_config)
