import sys

from efficient.utils import read_corpus
from efficient.vocab import Vocab

sys.path.append("../")  # isort: skip


if __name__ == "__main__":
    vocab = Vocab.build()
    vocab.save("vocab.json")
    print("# of characters:", len(vocab.src.char2id))
