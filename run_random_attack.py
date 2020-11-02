import random

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from efficient.model import CharCNNLSTMModel, Dataset
from efficient.utils import read_corpus, read_labels
from efficient.vocab import Vocab


def setup(vocab_path, model_path, contents_path, label_path, model_config):
    vocab = Vocab.load(vocab_path)
    predictor = CharCNNLSTMModel(vocab, **model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )
    model = predictor.model
    model.to(device)
    model_embedding = model.model_embeddings
    test_contents = read_corpus(contents_path)
    test_labels = read_labels(label_path)

    if dataset == "train":
        # only sample 30000  examples to do adversaril training
        sampled_idx = random.choices(range(len(test_contents)), k=30000)
    else:
        sampled_idx = range(len(test_contents))

    return (
        vocab,
        predictor,
        device,
        model,
        model_embedding,
        test_contents,
        test_labels,
        sampled_idx,
    )


def get_data(test_contents, test_labels, idx, vocab, model):
    batch_data = [(test_contents[idx], test_labels[idx])]
    batch_contents = [c for c, l in batch_data]
    batch_labels = [l for c, l in batch_data]
    batch_labels = torch.tensor(batch_labels) - 1
    batch_contents_lengths = [len(s) for s in batch_contents]
    batch_contents = vocab.src.to_input_tensor_char(
        batch_contents, max_word_length=model.max_word_length, device=model.device
    )
    batch_contents = batch_contents.to(device=model.device)
    batch_labels = batch_labels.to(device=model.device)
    candidates = torch.where(
        (batch_contents != 0) ^ (batch_contents != 1) ^ (batch_contents != 2)
    )
    candidates = list(zip(*[(i).tolist() for i in candidates]))
    # Remove <s> and <\s>
    candidates = [
        cand
        for cand in candidates
        if cand[0] != 0 and cand[0] != batch_contents_lengths[0] - 1
    ]
    return batch_contents, batch_labels, batch_contents_lengths, candidates


def forward(x_emb, model_embedding, model, batch_contents_lengths):
    sent_len, batch_size, max_word_len, _ = x_emb.shape
    view_shape = (
        sent_len * batch_size,
        max_word_len,
        model_embedding.char_embed_size,
    )
    x_reshaped = x_emb.view(view_shape).permute(0, 2, 1)
    x_conv_out = model_embedding.CNN(x_reshaped)
    x_highway = model_embedding.Highway(x_conv_out)
    x_word_emb = model_embedding.Dropout(x_highway)
    x_word_emb = x_word_emb.view(sent_len, batch_size, model_embedding.embed_size)
    X_packed = pack_padded_sequence(
        x_word_emb, batch_contents_lengths, enforce_sorted=False
    )
    enc_hiddens, (last_hidden, last_cell) = model.encoder(X_packed)
    (enc_hiddens, _) = pad_packed_sequence(enc_hiddens)
    enc_hiddens = enc_hiddens.permute(1, 0, 2)
    init_decoder_hidden = model.h_projection(
        torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    )
    init_decoder_cell = model.c_projection(
        torch.cat((last_cell[0], last_cell[1]), dim=1)
    )
    dec_init_state = (init_decoder_hidden, init_decoder_cell)

    logits = model.final_layer(dec_init_state[0])
    return logits


def index_to_sentence(batch_contents, vocab):
    sent = []
    for word_tensor in batch_contents:
        word = [
            vocab.src.id2char[int(char_index)]
            for char_index in word_tensor[0]
            if int(char_index) not in [0, 1, 2]
        ]

        sent.append("".join(word))
    sentence = " ".join(sent[1:-1])  # remove <s> and <\s>
    return sentence


def flip(
    batch_contents,
    batch_labels,
    model_embedding,
    model,
    batch_contents_lengths,
    predictor,
    position,
):
    i, j, k = position
    x_emb = model_embedding.CharEmbedding(batch_contents)
    one_char = x_emb[i, j, k].clone()
    x_emb[i, j, k] = one_char
    one_char.retain_grad()

    # Computation graph except CharEmbedding
    logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
    predicted_labels = torch.argmax(logits, dim=1)
    losses = predictor.loss(logits, batch_labels)
    losses.backward()
    gradients = one_char.grad

    with torch.no_grad():
        char_embedding = model_embedding.CharEmbedding.weight.data
        # First order approximation
        L = torch.matmul((char_embedding - one_char), gradients)
        adv_i = torch.argmax(L)
        batch_contents[i, j, k] = adv_i

    return batch_contents


def insert(
    batch_contents,
    batch_labels,
    model_embedding,
    model,
    batch_contents_lengths,
    vocab,
    predictor,
    position,
):
    char_embedding = model_embedding.CharEmbedding.weight.data
    a_emb = char_embedding[vocab.src.char2id["a"]]

    i, j, k = position
    x_emb = model_embedding.CharEmbedding(batch_contents)
    # move all characters after k-th one-slot to one position righter
    x_emb[i, j, k + 1 :] = x_emb[i, j, k:-1].clone()

    insertion_emb = a_emb.clone().requires_grad_(True)
    x_emb[i, j, k] = insertion_emb
    insertion_emb.retain_grad()

    logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
    predicted_labels = torch.argmax(logits, dim=1)
    losses = predictor.loss(logits, batch_labels)
    losses.backward()
    gradients = insertion_emb.grad

    with torch.no_grad():
        L = torch.matmul((char_embedding - insertion_emb), gradients)
        adv_i = torch.argmax(L, dim=0)
        batch_contents[i, j, k + 1 :] = batch_contents[i, j, k:-1].clone()
        batch_contents[i, j, k] = adv_i

    return batch_contents


def delete(batch_contents, position, vocab):
    i, j, k = position
    batch_contents[i, j, k:-1] = batch_contents[i, j, k + 1 :].clone()
    batch_contents[i, j, -1] = vocab.src.char2id["<pad>"]
    return batch_contents


def random_flip(
    contents_path,
    label_path,
    model_path,
    model_config,
    vocab_path,
    budget,
    content_output_path,
    label_output_path,
    dataset,
):
    (
        vocab,
        predictor,
        device,
        model,
        model_embedding,
        test_contents,
        test_labels,
        sampled_idx,
    ) = setup(vocab_path, model_path, contents_path, label_path, model_config)

    for idx in tqdm(sampled_idx):
        batch_contents, batch_labels, batch_contents_lengths, candidates = get_data(
            test_contents, test_labels, idx, vocab, model
        )

        for position in random.choices(
            candidates, k=batch_contents_lengths[0] // budget
        ):
            batch_contents = flip(
                batch_contents,
                batch_labels,
                model_embedding,
                model,
                batch_contents_lengths,
                predictor,
                position,
            )

        sentence = index_to_sentence(batch_contents, vocab)

        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def random_insert(
    contents_path,
    label_path,
    model_path,
    model_config,
    vocab_path,
    budget,
    content_output_path,
    label_output_path,
    dataset,
):
    (
        vocab,
        predictor,
        device,
        model,
        model_embedding,
        test_contents,
        test_labels,
        sampled_idx,
    ) = setup(vocab_path, model_path, contents_path, label_path, model_config)

    for idx in tqdm(sampled_idx):

        batch_contents, batch_labels, batch_contents_lengths, candidates = get_data(
            test_contents, test_labels, idx, vocab, model
        )
        for position in random.choices(
            candidates, k=batch_contents_lengths[0] // budget
        ):
            batch_contents = insert(
                batch_contents,
                batch_labels,
                model_embedding,
                model,
                batch_contents_lengths,
                vocab,
                predictor,
                position,
            )
        sentence = index_to_sentence(batch_contents, vocab)

        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def random_delete(
    contents_path,
    label_path,
    model_path,
    model_config,
    vocab_path,
    budget,
    content_output_path,
    label_output_path,
    dataset,
):
    (
        vocab,
        predictor,
        device,
        model,
        model_embedding,
        test_contents,
        test_labels,
        sampled_idx,
    ) = setup(vocab_path, model_path, contents_path, label_path, model_config)

    for idx in tqdm(sampled_idx):
        batch_contents, batch_labels, batch_contents_lengths, candidates = get_data(
            test_contents, test_labels, idx, vocab, model
        )
        for position in random.choices(
            candidates, k=batch_contents_lengths[0] // budget
        ):
            batch_contents = delete(batch_contents, position, vocab)
        sentence = index_to_sentence(batch_contents, vocab)
        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def random_mix(
    contents_path,
    label_path,
    model_path,
    model_config,
    vocab_path,
    budget,
    content_output_path,
    label_output_path,
    dataset,
):
    (
        vocab,
        predictor,
        device,
        model,
        model_embedding,
        test_contents,
        test_labels,
        sampled_idx,
    ) = setup(vocab_path, model_path, contents_path, label_path, model_config)

    for idx in tqdm(sampled_idx):

        batch_contents, batch_labels, batch_contents_lengths, candidates = get_data(
            test_contents, test_labels, idx, vocab, model
        )

        for position in random.choices(
            candidates, k=batch_contents_lengths[0] // budget
        ):
            operation = random.choice(["flip", "insert", "delete"])
            if operation == "flip":
                batch_contents = flip(
                    batch_contents,
                    batch_labels,
                    model_embedding,
                    model,
                    batch_contents_lengths,
                    predictor,
                    position,
                )
            elif operation == "insert":
                batch_contents = insert(
                    batch_contents,
                    batch_labels,
                    model_embedding,
                    model,
                    batch_contents_lengths,
                    vocab,
                    predictor,
                    position,
                )
            else:
                batch_contents = delete(batch_contents, position, vocab)

        sentence = index_to_sentence(batch_contents, vocab)
        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


if __name__ == "__main__":
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
    model_path = "checkpoints/case_aware/best_model.pkl"
    vocab_path = "data/vocab.json"
    budget = 5
    operation2func = {
        "flip": random_flip,
        "insert": random_insert,
        "delete": random_delete,
        "mix": random_mix,
    }
    for operation in ["mix"]:
        for dataset in ["test", "train"]:
            print(operation, dataset)
            contents_path = f"data/{dataset}_content.txt"
            label_path = f"data/{dataset}_label.txt"
            content_output_path = (
                f"data/adversary/random_{operation}_{dataset}_content.txt"
            )
            label_output_path = f"data/adversary/random_{operation}_{dataset}_label.txt"

            operation2func[operation](
                contents_path,
                label_path,
                model_path,
                model_config,
                vocab_path,
                budget,
                content_output_path,
                label_output_path,
                dataset,
            )
