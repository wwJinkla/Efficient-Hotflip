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
    candidates = set(
        [
            cand
            for cand in candidates
            if cand[0] != 0 and cand[0] != batch_contents_lengths[0] - 1
        ]
    )
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


def best_flip(
    batch_contents,
    adv_label,
    model_embedding,
    model,
    batch_contents_lengths,
    predictor,
    candidates,
):
    """find best position to flip according to the first order approximation of loss increases
    """

    x_emb = model_embedding.CharEmbedding(batch_contents)
    char_embedding = model_embedding.CharEmbedding.weight.data
    x_emb.retain_grad()

    logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
    predicted_labels = torch.argmax(logits, dim=1)
    losses = predictor.loss(logits, adv_label)
    losses.backward()
    gradients = x_emb.grad

    with torch.no_grad():
        x_emb.unsqueeze_(3)  # [sent_len, 1, word_len, 1, char_emb_size]
        gradients.unsqueeze_(3)  # [sent_len, 1, word_len, 1, char_emb_size]

        char_embedding_ = char_embedding.clone()
        char_embedding_.unsqueeze_(0)
        char_embedding_.unsqueeze_(0)
        char_embedding_.unsqueeze_(0)  # [1, 1, 1, vocab_size, char_emb_size]

        difference = (
            char_embedding_ - x_emb
        )  # [sent_len, 1, word_len, vocab_size, char_emb_size]
        L = (difference * gradients).sum(dim=-1)  # [sent_len, 1, word_len, vocab_size]
        values, indexes = torch.min(L, dim=-1)  # [sent_len, 1, word_len]

        loss, adv_index, position = min(
            [
                (float(values[i, j, k]), int(indexes[i, j, k]), (i, j, k))
                for i, j, k in candidates
            ],
            key=lambda x: x[0],
        )
        i, j, k = position
        batch_contents[i, j, k] = adv_index
        candidates.remove(position)  # prevent same position being flip again

    return batch_contents, candidates


def greedy_flip(
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
        labels = [0, 1, 2, 3]
        labels.remove(int(batch_labels))
        adv_label = torch.tensor(
            [random.choice(labels)], device=device
        )  # minimize loss w.r.t to this label

        for _ in range(batch_contents_lengths[0] // budget):

            batch_contents, candidates = best_flip(
                batch_contents,
                adv_label,
                model_embedding,
                model,
                batch_contents_lengths,
                predictor,
                candidates,
            )

        sentence = index_to_sentence(batch_contents, vocab)

        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def best_insert(
    batch_contents,
    adv_label,
    model_embedding,
    model,
    batch_contents_lengths,
    vocab,
    predictor,
    candidates,
):
    insert_idx = random.randint(
        0, len(vocab.src) - 1
    )  # pick a random character to insert

    x_emb = model_embedding.CharEmbedding(batch_contents)  # [sent_len, 1, word_len, 25]
    x_emb = torch.Tensor.repeat_interleave(
        x_emb, len(candidates), dim=1
    )  # [sent_len, len(candidates), word_len, 25]

    char_embedding = model_embedding.CharEmbedding.weight.data.clone()
    a_emb = char_embedding[insert_idx]

    for row, position in enumerate(candidates):
        i, j, k = position
        x_emb[i, row, k + 1 :] = x_emb[i, row, k:-1].clone()
        x_emb[i, row, k] = a_emb.clone()

    # Brute-force search over all positions
    with torch.no_grad():

        logits = forward(
            x_emb, model_embedding, model, batch_contents_lengths * len(candidates)
        )
        best_cand = torch.argmax(logits[:, adv_label])
        i, j, k = candidates[best_cand]

        batch_contents[i, j, k + 1 :] = batch_contents[i, j, k:-1].clone()
        batch_contents[i, j, k] = insert_idx

    del candidates[best_cand]

    return batch_contents, candidates


def greedy_insert(
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
        candidates = random.choices(
            list(candidates), k=150
        )  # limit search space to prevent OOM
        candidates = sorted(list(candidates))
        labels = [0, 1, 2, 3]
        labels.remove(int(batch_labels))

        # Find the adversary label that has largest logits if the original prediction is correct
        x_emb = model_embedding.CharEmbedding(batch_contents)
        logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
        if torch.argsort(logits)[:, -1] == batch_labels:
            adv_label = torch.argsort(logits)[:, -2].to(device=device)
        else:
            adv_label = torch.tensor([random.choice(labels)]).to(device=device)

        for _ in range(batch_contents_lengths[0] // budget):

            batch_contents, candidates = best_insert(
                batch_contents,
                adv_label,
                model_embedding,
                model,
                batch_contents_lengths,
                vocab,
                predictor,
                candidates,
            )

        sentence = index_to_sentence(batch_contents, vocab)

        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def best_delete(
    batch_contents,
    adv_label,
    model_embedding,
    model,
    batch_contents_lengths,
    vocab,
    predictor,
    candidates,
):
    x_emb = model_embedding.CharEmbedding(batch_contents)
    x_emb = torch.Tensor.repeat_interleave(x_emb, len(candidates), dim=1)
    char_embedding = model_embedding.CharEmbedding.weight.data.clone()

    for row, position in enumerate(candidates):
        i, j, k = position
        x_emb[i, row, k - 1 : -1] = x_emb[i, row, k:].clone()

    # Brute-force search over all positions
    with torch.no_grad():
        logits = forward(
            x_emb, model_embedding, model, batch_contents_lengths * len(candidates)
        )
        best_cand = torch.argmax(logits[:, adv_label])
        i, j, k = candidates[best_cand]
        batch_contents[i, j, k - 1 : -1] = batch_contents[i, j, k:].clone()

    del candidates[best_cand]

    return batch_contents, candidates


def greedy_delete(
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
        candidates = random.choices(
            list(candidates), k=150
        )  # limit search space to prevent OOM
        candidates = sorted(list(candidates))
        labels = [0, 1, 2, 3]
        labels.remove(int(batch_labels))

        # Find the adversary label that has largest logits if the original prediction is correct
        x_emb = model_embedding.CharEmbedding(batch_contents)
        logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
        if torch.argsort(logits)[:, -1] == batch_labels:
            adv_label = torch.argsort(logits)[:, -2].to(device=device)
        else:
            adv_label = torch.tensor([random.choice(labels)]).to(device=device)

        for _ in range(batch_contents_lengths[0] // budget):

            batch_contents, candidates = best_delete(
                batch_contents,
                adv_label,
                model_embedding,
                model,
                batch_contents_lengths,
                vocab,
                predictor,
                candidates,
            )

        sentence = index_to_sentence(batch_contents, vocab)

        with open(content_output_path, "a") as f:
            f.write(sentence + "\n")

        with open(label_output_path, "a") as f:
            f.write(str(int(batch_labels) + 1) + "\n")  # one-indexing


def greedy_mix(
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
        candidates = random.choices(
            list(candidates), k=150
        )  # limit search space to prevent OOM
        candidates = sorted(list(candidates))
        labels = [0, 1, 2, 3]
        labels.remove(int(batch_labels))

        # Find the adversary label that has largest logits if the original prediction is correct
        x_emb = model_embedding.CharEmbedding(batch_contents)
        logits = forward(x_emb, model_embedding, model, batch_contents_lengths)
        if torch.argsort(logits)[:, -1] == batch_labels:
            adv_label = torch.argsort(logits)[:, -2].to(device=device)
        else:
            adv_label = torch.tensor([random.choice(labels)]).to(device=device)

        for _ in range(batch_contents_lengths[0] // budget):
            operation = random.choice(["flip", "insert", "delete"])
            if operation == "flip":
                batch_contents, candidates = best_flip(
                    batch_contents,
                    adv_label,
                    model_embedding,
                    model,
                    batch_contents_lengths,
                    predictor,
                    candidates,
                )
            elif operation == "insert":
                batch_contents, candidates = best_insert(
                    batch_contents,
                    adv_label,
                    model_embedding,
                    model,
                    batch_contents_lengths,
                    vocab,
                    predictor,
                    candidates,
                )

            else:
                batch_contents, candidates = best_delete(
                    batch_contents,
                    adv_label,
                    model_embedding,
                    model,
                    batch_contents_lengths,
                    vocab,
                    predictor,
                    candidates,
                )

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
        "flip": greedy_flip,
        "insert": greedy_insert,
        "delete": greedy_delete,
        "mix": greedy_mix,
    }
    # for operation in ["insert"]:
    #     for dataset in ["test"]:

    for operation, dataset in [
        ("mix", "test"),
        ("insert", "test"),
        ("delete", "train"),
        ("mix", "train"),
        ("insert", "train"),
    ]:

        print(operation, dataset)
        contents_path = f"data/{dataset}_content.txt"
        label_path = f"data/{dataset}_label.txt"
        content_output_path = f"data/adversary/greedy_{operation}_{dataset}_content.txt"
        label_output_path = f"data/adversary/greedy_{operation}_{dataset}_label.txt"

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
