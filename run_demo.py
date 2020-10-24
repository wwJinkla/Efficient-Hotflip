import random

import torch

from efficient.model import CharCNNLSTMModel, Dataset
from efficient.utils import read_corpus, read_labels, read_raw_corpus
from efficient.vocab import Vocab

index_mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def demo(
    model_path,
    vocab_path,
    test_contents_path,
    test_label_path,
    num_examples=10,
    **model_config
):
    vocab = Vocab.load(vocab_path)

    predictor = CharCNNLSTMModel(vocab, **model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )

    test_contents = read_corpus(test_contents_path)
    test_labels = read_labels(test_label_path)
    test_raw_contents = read_raw_corpus(test_contents_path)

    test_data = list(zip(test_contents, test_labels, test_raw_contents))
    demo_data = random.choices(test_data, k=num_examples)

    demo_contents = [c for c, l, r in demo_data]
    demo_labels = [l for c, l, r in demo_data]
    demo_raw_contents = [r for c, l, r in demo_data]

    demo_dataset = Dataset(
        demo_contents, demo_labels, vocab, model_config.get("max_word_length"), "cpu"
    )

    demo_contents, demo_labels, demo_contents_lengths = demo_dataset[len(demo_dataset)]

    with torch.no_grad():
        pred = predictor.model(demo_contents, demo_contents_lengths)
        predicted_labels = torch.argmax(pred, dim=1)

    for content, gt, pr in zip(demo_raw_contents, demo_labels, predicted_labels):
        print("Content:", content)
        print("Predicted category:", index_mapping[int(pr)])
        print("Ground truth category:", index_mapping[int(gt)])
        print("\n")


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
    demo(
        model_path,
        vocab_path,
        test_contents_path,
        test_label_path,
        num_examples=10,
        **model_config
    )
