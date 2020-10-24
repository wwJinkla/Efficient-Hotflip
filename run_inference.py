import torch

from efficient.model import CharCNNLSTMModel, Dataset
from efficient.utils import read_corpus, read_labels
from efficient.vocab import Vocab


def infer(model_path, vocab_path, test_contents_path, test_label_path, **model_config):
    vocab = Vocab.load(vocab_path)
    test_contents = read_corpus(test_contents_path)
    test_labels = read_labels(test_label_path)
    test_dataset = Dataset(
        test_contents, test_labels, vocab, model_config.get("max_word_length"), "cpu"
    )
    predictor = CharCNNLSTMModel(vocab, **model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )

    batch_size = 20
    accuracies = []
    # This will drop the last few examples (<= 19)
    for batch_index in range(0, len(test_labels), batch_size):
        batch_contents, batch_labels, batch_content_lengths = test_dataset[batch_size]
        _, accuracy = predictor.predict(
            batch_contents, batch_labels, batch_content_lengths
        )
        accuracies.append(accuracy)

    print("test accuracy:", sum(accuracies) / len(accuracies))


if __name__ == "__main__":
    vocab_path = "data/vocab.json"
    test_contents_path = "data/test_content.txt"
    test_label_path = "data/test_label.txt"
    model_path = "checkpoints/long/best_model.pkl"

    model_config = dict(
        char_embed_size=25,
        embed_size=500,
        hidden_size=500,
        max_word_length=30,
        batch_size=100,
        eta=0.001,
        max_grad_norm=1,
        max_iter=500,
        val_batch_size=500,
    )
    infer(model_path, vocab_path, test_contents_path, test_label_path, **model_config)
