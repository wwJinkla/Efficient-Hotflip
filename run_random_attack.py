import random

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from efficient.model import CharCNNLSTMModel, Dataset
from efficient.utils import read_corpus, read_labels, read_raw_corpus
from efficient.vocab import Vocab


def random_flip(
    test_contents_path,
    test_label_path,
    model_path,
    model_config,
    vocab_path,
    budget,
    output_path,
):
    index_mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    vocab = Vocab.load(vocab_path)
    predictor = CharCNNLSTMModel(vocab, **model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))
    )
    model = predictor.model
    model_embedding = model.model_embeddings
    test_contents = read_corpus(test_contents_path)
    test_labels = read_labels(test_label_path)
    test_raw_contents = read_raw_corpus(test_contents_path)

    for idx in range(len(test_contents)):
        if idx % 100 == 0:
            print(idx)
        batch_data = [(test_contents[idx], test_labels[idx])]
        batch_contents = [c for c, l in batch_data]
        batch_labels = [l for c, l in batch_data]
        batch_labels = torch.tensor(batch_labels) - 1
        batch_contents_lengths = [len(s) for s in batch_contents]
        batch_contents = vocab.src.to_input_tensor_char(
            batch_contents, max_word_length=model.max_word_length, device=model.device
        )

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

        for i, j, k in random.choices(
            candidates, k=batch_contents_lengths[0] // budget
        ):
            source_char_index = int(batch_contents[i, j, k])
            x_emb = model_embedding.CharEmbedding(batch_contents)
            one_char = x_emb[i, j, k].clone()
            x_emb[i, j, k] = one_char
            one_char.retain_grad()

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
            x_word_emb = x_word_emb.view(
                sent_len, batch_size, model_embedding.embed_size
            )
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
            predicted_labels = torch.argmax(logits, dim=1)
            losses = predictor.loss(logits, batch_labels)
            losses.backward()
            gradients = one_char.grad

            with torch.no_grad():
                char_embedding = model_embedding.CharEmbedding.weight.data
                char_embedding.shape
                L = torch.matmul((char_embedding - one_char), gradients)
                adv_i = torch.argmax(L)
                batch_contents[i, j, k] = adv_i

        sent = []
        for word_tensor in batch_contents:
            word = [
                vocab.src.id2char[int(char_index)]
                for char_index in word_tensor[0]
                if int(char_index) not in [0, 1, 2]
            ]
            sent.append("".join(word))
        sentence = " ".join(sent)

        with open(output_path, "a") as f:
            f.write(sentence + "\n")


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
    model_path = "checkpoints/long/best_model.pkl"
    vocab_path = "data/vocab.json"
    budget = 5

    operation = "flip"
    for dataset in ["test", "train"]:
        print(operation, dataset)
        contents_path = f"data/{dataset}_content.txt"
        label_path = f"data/{dataset}_label.txt"
        output_path = f"data/adversary/random_{operation}_{dataset}.txt"

        random_flip(
            contents_path,
            label_path,
            model_path,
            model_config,
            vocab_path,
            budget,
            output_path,
        )
