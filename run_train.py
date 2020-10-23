from efficient.model import CharCNNLSTMModel
from efficient.utils import read_corpus, read_labels
from efficient.vocab import Vocab


def train(vocab_path, train_contents_path, train_label_path, **model_config):
    vocab = Vocab.load(vocab_path)
    train_contents = read_corpus(train_contents_path)
    train_labels = read_labels(train_label_path)
    model = CharCNNLSTMModel(vocab, **model_config)
    model.fit(train_contents, train_labels)


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
        eta=0.001,
        max_grad_norm=1,
        max_iter=1000,
        val_batch_size=100,
    )
    train(vocab_path, train_contents_path, train_label_path, **model_config)
