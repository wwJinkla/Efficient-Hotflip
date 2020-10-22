import sys

from model.utils import read_corpus
from model.vocab import Vocab

sys.path.append("../")


if __name__ == "__main__":
    data = read_corpus("train_content.txt")
    vocab = Vocab.build(data, vocab_size=30000, freq_cutoff=2)
    vocab.save("vocab.json")
