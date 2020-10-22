import sys

from efficient.utils import read_corpus
from efficient.vocab import Vocab

sys.path.append("../")


if __name__ == "__main__":
    data = read_corpus("train_content.txt")
    vocab = Vocab.build(data, vocab_size=10000, freq_cutoff=30)
    vocab.save("vocab.json")
    print("# of characters:", len(vocab.src.char2id))
