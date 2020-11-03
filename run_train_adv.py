import torch

from efficient.model import CharCNNLSTMModel
from efficient.utils import read_corpus, read_labels
from efficient.vocab import Vocab


def adv_train(
    vocab_path,
    train_contents_path,
    train_label_path,
    adv_train_contents_path,
    adv_train_label_path,
    model_path,
    model_output_path,
    **model_config,
):
    vocab = Vocab.load(vocab_path)
    train_contents = read_corpus(train_contents_path)
    train_labels = read_labels(train_label_path)

    adv_train_contents = read_corpus(adv_train_contents_path)
    adv_train_labels = read_labels(adv_train_label_path)

    contents = train_contents + adv_train_contents
    labels = train_labels + adv_train_labels

    model = CharCNNLSTMModel(vocab, **model_config)
    model.model.load_state_dict(torch.load(model_path))
    model.fit(contents, labels, model_output_path)


if __name__ == "__main__":

    vocab_path = "data/vocab.json"
    train_contents_path = "data/train_content.txt"
    train_label_path = "data/train_label.txt"
    model_config = dict(
        char_embed_size=25,
        embed_size=500,
        hidden_size=500,
        max_word_length=30,
        batch_size=100,
        eta=0.0001,  # fine-tuning, an order smaller
        max_grad_norm=1,
        max_iter=300,
        val_batch_size=100,
    )
    model_path = "checkpoints/case_aware/best_model.pkl"

    # for attack in ["random_flip", "random_insert", "random_delete", "random_mix"]:
    for attack in ["greedy_flip"]:

        print(attack)
        adv_train_contents_path = f"data/adversary/{attack}_train_content.txt"
        adv_train_label_path = f"data/adversary/{attack}_train_label.txt"
        model_output_path = f"checkpoints/{attack}"
        adv_train(
            vocab_path,
            train_contents_path,
            train_label_path,
            adv_train_contents_path,
            adv_train_label_path,
            model_path,
            model_output_path,
            **model_config,
        )
